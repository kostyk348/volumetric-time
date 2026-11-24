//! Geometry of one update step: the radial and angular parts, and what rank it leaves behind.
//!
//! The notebook computed `Δw` on the flattened parameter vector and split it into "radial" and
//! "angular" parts by hand. That split has no meaning for a matrix: the same linear map written in
//! another basis gives a different vector, so its "angle" changes. What is basis-independent is the
//! pair `(skew, sym)` of `W₀ᵀ·ΔW`:
//!
//! ```text
//! A     = W₀ᵀ · (W₁ − W₀)
//! v_ang = ‖skew(A)‖_F / ‖W₀‖²_F      (rotation of features — what "own time" integrates)
//! v_rad = ‖sym(A)‖_F  / ‖W₀‖²_F      (stretching — what weight decay and scaling do)
//! ```
//!
//! Left-multiplying both `W₀` and `W₁` by an orthogonal `Q` leaves `A` unchanged, because
//! `(QW₀)ᵀ(QW₁ − QW₀) = W₀ᵀ(W₁ − W₀)`. That is the property the test below pins down, and it is the
//! reason this quantity may be compared across layers of different width — unlike `‖Δw‖`.

pub type Mat = Vec<Vec<f64>>;

pub fn zeros(rows: usize, cols: usize) -> Mat {
    vec![vec![0.0; cols]; rows]
}

pub fn transpose(a: &Mat) -> Mat {
    let mut out = zeros(a[0].len(), a.len());
    for (r, row) in a.iter().enumerate() {
        for (c, value) in row.iter().enumerate() {
            out[c][r] = *value;
        }
    }
    out
}

pub fn matmul(a: &Mat, b: &Mat) -> Mat {
    let mut out = zeros(a.len(), b[0].len());
    for (r, row) in a.iter().enumerate() {
        for (k, value) in row.iter().enumerate() {
            if *value == 0.0 {
                continue;
            }
            for c in 0..b[0].len() {
                out[r][c] += value * b[k][c];
            }
        }
    }
    out
}

pub fn mat_vec(a: &Mat, x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum()).collect()
}

pub fn frobenius(a: &Mat) -> f64 {
    a.iter().flat_map(|row| row.iter()).map(|v| v * v).sum::<f64>().sqrt()
}

pub fn subtract(a: &Mat, b: &Mat) -> Mat {
    let mut out = zeros(a.len(), a[0].len());
    for r in 0..a.len() {
        for c in 0..a[0].len() {
            out[r][c] = a[r][c] - b[r][c];
        }
    }
    out
}

/// Eigendecomposition of a symmetric matrix by the cyclic Jacobi method.
///
/// Small, dependency-free and exact to machine precision on the sizes an analysis script sees
/// (a layer is never diagonalised as a whole; the Gram matrix of a layer is at most a few hundred
/// wide, and even that is done offline).
pub fn symmetric_eigenvalues(a: &Mat) -> Vec<f64> {
    let n = a.len();
    let mut m = a.clone();
    for sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[p][q] * m[p][q];
            }
        }
        if off < 1e-26 * (1.0 + frobenius(&m)) || sweep == 99 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p][q];
                if apq.abs() < 1e-18 {
                    continue;
                }
                let theta = (m[q][q] - m[p][p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt()).max(1e-300);
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let mkp = m[k][p];
                    let mkq = m[k][q];
                    m[k][p] = c * mkp - s * mkq;
                    m[k][q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[p][k];
                    let mqk = m[q][k];
                    m[p][k] = c * mpk - s * mqk;
                    m[q][k] = s * mpk + c * mqk;
                }
            }
        }
    }
    (0..n).map(|i| m[i][i]).collect()
}

/// Effective rank: `exp` of the entropy of the normalised spectrum. A flat spectrum of width `n`
/// gives `n`; a single dominant direction gives `1`. This is the macro-level "structure" signal.
pub fn effective_rank(a: &Mat) -> f64 {
    let gram = matmul(&transpose(a), a);
    let eigenvalues: Vec<f64> = symmetric_eigenvalues(&gram).into_iter().filter(|v| *v > 0.0).collect();
    let total: f64 = eigenvalues.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let entropy: f64 = eigenvalues
        .iter()
        .map(|v| {
            let p = v / total;
            -p * p.ln()
        })
        .sum();
    entropy.exp()
}

/// Spectral norm, the largest singular value — the cost of a layer in a Lipschitz bound.
pub fn spectral_norm(a: &Mat) -> f64 {
    let gram = matmul(&transpose(a), a);
    symmetric_eigenvalues(&gram).into_iter().fold(0.0_f64, |acc, v| acc.max(v)).sqrt()
}

#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    /// Rotation of feature directions, scale-free.
    pub v_ang: f64,
    /// Stretching along existing directions, scale-free.
    pub v_rad: f64,
    /// Share of the movement that is rotation: `v_ang / (v_ang + v_rad)`, zero when nothing moved.
    pub rotation_share: f64,
    /// Effective rank before and after the step.
    pub rank_before: f64,
    pub rank_after: f64,
    /// How much the rank collapsed in this step, in percent of `rank_before`.
    pub rank_collapse_pct: f64,
}

/// Analyse one step of one layer.
pub fn analyse(w0: &Mat, w1: &Mat) -> Update {
    let scale = frobenius(w0).powi(2).max(1e-300);
    let delta = subtract(w1, w0);
    let a = matmul(&transpose(w0), &delta);
    let at = transpose(&a);
    let mut skew = zeros(a.len(), a[0].len());
    let mut sym = zeros(a.len(), a[0].len());
    for r in 0..a.len() {
        for c in 0..a[0].len() {
            skew[r][c] = (a[r][c] - at[r][c]) / 2.0;
            sym[r][c] = (a[r][c] + at[r][c]) / 2.0;
        }
    }
    let v_ang = frobenius(&skew) / scale;
    let v_rad = frobenius(&sym) / scale;
    let total = v_ang + v_rad;
    let rank_before = effective_rank(w0);
    let rank_after = effective_rank(w1);
    Update {
        v_ang,
        v_rad,
        rotation_share: if total > 0.0 { v_ang / total } else { 0.0 },
        rank_before,
        rank_after,
        rank_collapse_pct: if rank_before > 0.0 { (1.0 - rank_after / rank_before) * 100.0 } else { 0.0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(n: usize) -> Mat {
        let mut m = zeros(n, n);
        for i in 0..n {
            m[i][i] = 1.0;
        }
        m
    }

    fn rotation(angle: f64) -> Mat {
        vec![vec![angle.cos(), -angle.sin()], vec![angle.sin(), angle.cos()]]
    }

    #[test]
    fn a_small_rotation_reads_as_rotation() {
        let w0 = identity(2);
        let w1 = matmul(&rotation(0.01), &w0);
        let update = analyse(&w0, &w1);
        assert!(update.v_ang > 0.0);
        assert!(update.rotation_share > 0.95, "rotation share {} should be near one", update.rotation_share);
    }

    #[test]
    fn a_pure_scaling_reads_as_stretching() {
        let w0 = identity(2);
        let mut w1 = identity(2);
        w1[0][0] = 1.01;
        w1[1][1] = 1.01;
        let update = analyse(&w0, &w1);
        assert!(update.v_ang < 1e-12, "no rotation: v_ang = {}", update.v_ang);
        assert!(update.v_rad > 0.0);
        assert!(update.rotation_share < 1e-12);
    }

    #[test]
    fn the_angular_part_does_not_depend_on_the_basis() {
        // Left-multiplication by an orthogonal matrix re-expresses the same step in another basis.
        let w0 = vec![vec![0.9, 0.3, -0.2], vec![0.1, 1.2, 0.4], vec![0.0, -0.5, 0.8]];
        let w1 = vec![vec![0.85, 0.33, -0.25], vec![0.15, 1.1, 0.5], vec![0.05, -0.45, 0.9]];
        let q = rotation(0.3);
        let mut q3 = identity(3);
        for (r, row) in q.iter().enumerate() {
            for (c, value) in row.iter().enumerate() {
                q3[r][c] = *value;
            }
        }
        q3[2][2] = 1.0;
        let left = analyse(&w0, &w1);
        let right = analyse(&matmul(&q3, &w0), &matmul(&q3, &w1));
        assert!((left.v_ang - right.v_ang).abs() < 1e-12, "{} vs {}", left.v_ang, right.v_ang);
        assert!((left.v_rad - right.v_rad).abs() < 1e-12);
    }

    #[test]
    fn effective_rank_counts_the_directions_that_matter() {
        assert!((effective_rank(&identity(4)) - 4.0).abs() < 1e-9);
        let mut rank_one = zeros(4, 4);
        rank_one[0][0] = 3.0;
        assert!((effective_rank(&rank_one) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn spectral_norm_of_a_diagonal_is_its_largest_entry() {
        let mut m = zeros(3, 3);
        m[0][0] = 2.0;
        m[1][1] = -5.0;
        m[2][2] = 1.0;
        assert!((spectral_norm(&m) - 5.0).abs() < 1e-9);
    }
}
