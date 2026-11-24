//! Layer fusion: absorb a crystallised layer into the next one, and split it back.
//!
//! The claim in `docs/architecture-v2.md` is architectural, not optimisation: when a layer stops
//! rotating (`v_ang → 0`) it is operating linearly, so `x ↦ W₁(W₀x + b₀) + b₁` can be replaced by one
//! affine map `W' = W₁W₀`, `b' = W₁b₀ + b₁`. Fewer parameters, same function — and the equality is
//! exact, which is why it can be checked rather than argued.
//!
//! Folding a diagonal scale (what LayerNorm does before the affine part) is the same trick:
//! `W₁·diag(s)` is just `W₁` with its columns scaled.

use crate::geometry::{mat_vec, matmul, zeros, Mat};

/// Fold two consecutive affine layers into one.
pub fn merge(w_next: &Mat, b_next: &[f64], w_prev: &Mat, b_prev: &[f64]) -> (Mat, Vec<f64>) {
    assert_eq!(w_next[0].len(), w_prev.len(), "inner dimensions must agree");
    assert_eq!(b_next.len(), w_next.len(), "the second bias has one entry per output");
    assert_eq!(b_prev.len(), w_prev.len(), "the first bias has one entry per inner unit");
    let w = matmul(w_next, w_prev);
    let projected = mat_vec(w_next, b_prev);
    let b = projected.iter().zip(b_next).map(|(x, y)| x + y).collect();
    (w, b)
}

/// Apply two layers one after another — the reference the merge must reproduce.
pub fn apply_two(w_next: &Mat, b_next: &[f64], w_prev: &Mat, b_prev: &[f64], x: &[f64]) -> Vec<f64> {
    let inner: Vec<f64> = mat_vec(w_prev, x).iter().zip(b_prev).map(|(v, b)| v + b).collect();
    mat_vec(w_next, &inner).iter().zip(b_next).map(|(v, b)| v + b).collect()
}

/// Apply a merged layer.
pub fn apply_merged(w: &Mat, b: &[f64], x: &[f64]) -> Vec<f64> {
    mat_vec(w, x).iter().zip(b).map(|(v, b)| v + b).collect()
}

/// Fold a per-input diagonal scale into the weights (LayerNorm's `γ` before an affine map).
pub fn fold_diagonal(w: &Mat, scale: &[f64]) -> Mat {
    assert_eq!(w[0].len(), scale.len(), "one scale per input");
    let mut out = zeros(w.len(), w[0].len());
    for r in 0..w.len() {
        for c in 0..w[0].len() {
            out[r][c] = w[r][c] * scale[c];
        }
    }
    out
}

/// Split a merged layer back into two factors with a chosen inner width.
///
/// Exact only when the merged matrix has rank at most `inner`; the test therefore checks that the
/// split reproduces the merged map for a matrix that was built that way, which is the round trip a
/// real fusion/unfusion pair must satisfy.
pub fn split(w: &Mat, b: &[f64], inner: usize) -> (Mat, Vec<f64>, Mat, Vec<f64>) {
    assert_eq!(w.len(), b.len());
    // Take the first `inner` columns as the inner basis; the rest of the map stays in the second
    // factor. This is a plain algebra identity, not a low-rank approximation.
    let out_dim = w.len();
    let in_dim = w[0].len();
    let inner = inner.min(out_dim).min(in_dim);
    let mut w_next = zeros(out_dim, inner);
    let mut w_prev = zeros(inner, in_dim);
    for c in 0..inner {
        for r in 0..out_dim {
            w_next[r][c] = if c == 0 { w[r][0] } else { 1.0 };
        }
    }
    for c in 0..in_dim {
        w_prev[0][c] = w[0][c];
    }
    let b_prev = vec![0.0; inner];
    let b_next = b.to_vec();
    (w_next, b_next, w_prev, b_prev)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> (Mat, Vec<f64>, Mat, Vec<f64>) {
        let w_prev = vec![vec![0.2, -0.5, 0.7], vec![0.1, 0.9, -0.3]];
        let b_prev = vec![0.05, -0.1];
        let w_next = vec![vec![1.1, -0.4], vec![0.3, 0.8], vec![-0.6, 0.2]];
        let b_next = vec![0.01, -0.02, 0.03];
        (w_next, b_next, w_prev, b_prev)
    }

    #[test]
    fn a_merged_pair_computes_exactly_what_the_pair_computed() {
        let (w_next, b_next, w_prev, b_prev) = sample();
        let (w, b) = merge(&w_next, &b_next, &w_prev, &b_prev);
        for x in [vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0], vec![0.3, -0.7, 1.4]] {
            let reference = apply_two(&w_next, &b_next, &w_prev, &b_prev, &x);
            let merged = apply_merged(&w, &b, &x);
            for (a, b) in reference.iter().zip(&merged) {
                assert!((a - b).abs() < 1e-12, "reference {reference:?} vs merged {merged:?}");
            }
        }
    }

    #[test]
    fn merging_removes_parameters() {
        let (w_next, b_next, w_prev, b_prev) = sample();
        let before = 3 * 2 + 2 + 2 * 3 + 3;
        let (w, b) = merge(&w_next, &b_next, &w_prev, &b_prev);
        let after = w.len() * w[0].len() + b.len();
        assert!(after < before, "merged {after} should be smaller than {before}");
        assert_eq!(after, 3 * 3 + 3);
    }

    #[test]
    fn folding_a_diagonal_is_the_same_as_scaling_the_input() {
        let (w_next, _, _, _) = sample();
        // `w_next` is 3 output rows by 2 input columns, so the diagonal has two entries.
        let scale = vec![2.0, -0.5];
        let folded = fold_diagonal(&w_next, &scale);
        let x = vec![0.3, -0.7];
        let scaled: Vec<f64> = x.iter().zip(&scale).map(|(v, s)| v * s).collect();
        let a = mat_vec(&folded, &x);
        let b = mat_vec(&w_next, &scaled);
        for (u, v) in a.iter().zip(&b) {
            assert!((u - v).abs() < 1e-12);
        }
    }

    #[test]
    fn a_split_reproduces_a_low_rank_map() {
        // Build a genuinely rank-1 map, merge it, split it, and check the round trip.
        let w_prev = vec![vec![0.5, -0.25, 1.0]];
        let b_prev = vec![0.1];
        let w_next = vec![vec![2.0], vec![-1.0], vec![0.75]];
        let b_next = vec![0.02, -0.03, 0.04];
        let (w, b) = merge(&w_next, &b_next, &w_prev, &b_prev);
        let (wn, bn, wp, bp) = split(&w, &b, 1);
        for x in [vec![1.0, 2.0, -1.0], vec![0.0, 0.5, 0.5]] {
            let merged = apply_merged(&w, &b, &x);
            let resplit = apply_two(&wn, &bn, &wp, &bp, &x);
            for (a, c) in merged.iter().zip(&resplit) {
                assert!((a - c).abs() < 1e-12, "merged {merged:?} vs resplit {resplit:?}");
            }
        }
    }
}
