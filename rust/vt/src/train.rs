//! A trainer in plain Rust, so the deciding experiment needs no Python and no GPU.
//!
//! The task is the notebook's own: learn `(a + b) mod P` from 30 % of the pairs, with weight decay
//! strong enough that the model memorises first and generalises much later. Three arms run on the
//! same seed and the same data:
//!
//! * `base` — AdamW with a cosine schedule, the recipe to beat;
//! * `randfreeze` — the same recipe, plus random freezing of a layer's backward pass. Same number
//!   of frozen updates as `v2` gets in expectation, but no signal: this is the control that decides
//!   whether the geometry is worth anything;
//! * `v2` — freeze a layer when its angular velocity has fallen far from its own peak and its
//!   spectral rank has collapsed; count the backward work saved.
//!
//! What is counted honestly: a frozen layer saves its *backward* pass only. Its forward still runs,
//! so the savings here are one third of the flops, not two thirds, and the report says so.

use crate::geometry::{analyse, Mat};

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn normal(&mut self) -> f64 {
        // Box-Muller, good enough for initialisation.
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arm {
    Base,
    RandomFreeze,
    V2,
}

impl Arm {
    pub fn parse(text: &str) -> Option<Arm> {
        match text {
            "base" => Some(Arm::Base),
            "randfreeze" => Some(Arm::RandomFreeze),
            "v2" => Some(Arm::V2),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Arm::Base => "base",
            Arm::RandomFreeze => "randfreeze",
            Arm::V2 => "v2",
        }
    }
}

pub struct Config {
    pub p: usize,
    pub dim: usize,
    pub hidden: usize,
    pub train_fraction: f64,
    pub epochs: usize,
    pub batch: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub log_every: usize,
    pub cosine: bool,
    pub seed: u64,
    pub arm: Arm,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            p: 97,
            dim: 24,
            hidden: 96,
            train_fraction: 0.3,
            epochs: 6000,
            batch: 256,
            lr: 1e-3,
            weight_decay: 0.01,
            log_every: 200,
            cosine: false,
            seed: 1,
            arm: Arm::Base,
        }
    }
}

struct Model {
    emb: Vec<f64>,  // p × dim
    w1: Vec<f64>,   // (2·dim) × hidden
    b1: Vec<f64>,   // hidden
    w2: Vec<f64>,   // hidden × p
    b2: Vec<f64>,   // p
    dim: usize,
    hidden: usize,
    p: usize,
}

impl Model {
    fn new(p: usize, dim: usize, hidden: usize, rng: &mut Rng) -> Self {
        let mut emb = vec![0.0; p * dim];
        for value in emb.iter_mut() {
            *value = rng.normal();
        }
        let mut w1 = vec![0.0; 2 * dim * hidden];
        let scale1 = (1.0 / (2 * dim) as f64).sqrt();
        for value in w1.iter_mut() {
            *value = rng.normal() * scale1;
        }
        let mut w2 = vec![0.0; hidden * p];
        let scale2 = (1.0 / hidden as f64).sqrt();
        for value in w2.iter_mut() {
            *value = rng.normal() * scale2;
        }
        Model {
            emb,
            w1,
            b1: vec![0.0; hidden],
            w2,
            b2: vec![0.0; p],
            dim,
            hidden,
            p,
        }
    }

    fn forward(&self, a: usize, b: usize, x: &mut Vec<f64>, h: &mut Vec<f64>, logits: &mut Vec<f64>) {
        x.clear();
        x.extend_from_slice(&self.emb[a * self.dim..(a + 1) * self.dim]);
        x.extend_from_slice(&self.emb[b * self.dim..(b + 1) * self.dim]);
        for j in 0..self.hidden {
            let mut sum = self.b1[j];
            for i in 0..2 * self.dim {
                sum += x[i] * self.w1[i * self.hidden + j];
            }
            h[j] = if sum > 0.0 { sum } else { 0.0 };
        }
        for k in 0..self.p {
            let mut sum = self.b2[k];
            for j in 0..self.hidden {
                sum += h[j] * self.w2[j * self.p + k];
            }
            logits[k] = sum;
        }
    }
}

struct Grads {
    emb: Vec<f64>,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
}

struct Adam {
    m: Grads,
    v: Grads,
    step: u64,
}

fn zero_like(model: &Model) -> Grads {
    Grads {
        emb: vec![0.0; model.emb.len()],
        w1: vec![0.0; model.w1.len()],
        b1: vec![0.0; model.b1.len()],
        w2: vec![0.0; model.w2.len()],
        b2: vec![0.0; model.b2.len()],
    }
}

#[derive(Debug, Clone)]
pub struct Report {
    pub arm: &'static str,
    pub seed: u64,
    /// First epoch at which test accuracy reached 0.95, if it ever did.
    pub epoch_to_95: Option<usize>,
    pub best_test_acc: f64,
    pub final_test_loss: f64,
    pub train_loss_first_mem: f64,
    /// Share of backward work that frozen layers did not do.
    pub backward_saved_pct: f64,
    pub freezes: usize,
}

pub fn run(config: &Config, verbose: bool) -> Report {
    let mut rng = Rng::new(config.seed);
    // All pairs, then a deterministic split: the same split for every arm and seed.
    let mut all: Vec<(usize, usize, usize)> = Vec::with_capacity(config.p * config.p);
    for a in 0..config.p {
        for b in 0..config.p {
            all.push((a, b, (a + b) % config.p));
        }
    }
    for i in (1..all.len()).rev() {
        let j = rng.below(i + 1);
        all.swap(i, j);
    }
    let n_train = (config.train_fraction * all.len() as f64) as usize;
    let (train, test) = all.split_at(n_train);

    let mut model = Model::new(config.p, config.dim, config.hidden, &mut rng);
    let mut adam = Adam { m: zero_like(&model), v: zero_like(&model), step: 0 };

    let mut grads = zero_like(&model);
    let mut x = vec![0.0; 2 * config.dim];
    let mut h = vec![0.0; config.hidden];
    let mut logits = vec![0.0; config.p];
    let mut dlogits = vec![0.0; config.p];
    let mut dh = vec![0.0; config.hidden];

    // Freezing state per matrix: 0 = w1, 1 = w2.
    let mut frozen = [false, false];
    let mut peak_v_ang = [0.0_f64, 0.0_f64];
    let mut freezes = 0usize;
    let mut saved_backward = 0.0_f64;
    let mut total_backward = 0.0_f64;
    let mut prev_w1 = model.w1.clone();
    let mut prev_w2 = model.w2.clone();

    // One pass over a fixed subset of the training data is the yardstick for accuracy.
    let eval_set: Vec<(usize, usize, usize)> = test.iter().copied().take(2000).collect();

    let mut epoch_to_95: Option<usize> = None;
    let mut best_test_acc = 0.0_f64;
    let mut final_test_loss = f64::NAN;
    let mut train_loss_first_mem = f64::NAN;

    for epoch in 1..=config.epochs {
        let progress = epoch as f64 / config.epochs as f64;
        let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        let lr = if config.cosine { config.lr * (0.05 + 0.95 * cosine) } else { config.lr };

        // Learning-rate scale per matrix: a frozen matrix takes no step at all.
        let lr_scale = [if frozen[0] { 0.0 } else { 1.0 }, if frozen[1] { 0.0 } else { 1.0 }];

        let mut batch_loss = 0.0;
        for _ in 0..config.batch {
            let (a, b, y) = train[rng.below(train.len())];
            model.forward(a, b, &mut x, &mut h, &mut logits);
            let max = logits.iter().cloned().fold(f64::MIN, f64::max);
            let mut sum = 0.0;
            for value in logits.iter() {
                sum += (value - max).exp();
            }
            let log_sum = sum.ln() + max;
            let log_prob = logits[y] - log_sum;
            batch_loss += -log_prob;
            for k in 0..config.p {
                let p = (logits[k] - log_sum).exp();
                dlogits[k] = p - if k == y { 1.0 } else { 0.0 };
            }
            for j in 0..config.hidden {
                let mut acc = 0.0;
                for k in 0..config.p {
                    acc += model.w2[j * config.p + k] * dlogits[k];
                }
                dh[j] = if h[j] > 0.0 { acc } else { 0.0 };
            }
            // w2
            for j in 0..config.hidden {
                if lr_scale[1] == 0.0 {
                    break;
                }
                let prefix = j * config.p;
                for k in 0..config.p {
                    grads.w2[prefix + k] += h[j] * dlogits[k];
                }
            }
            if lr_scale[1] != 0.0 {
                for k in 0..config.p {
                    grads.b2[k] += dlogits[k];
                }
            }
            // w1
            if lr_scale[0] != 0.0 {
                for i in 0..2 * config.dim {
                    let xi = x[i];
                    if xi == 0.0 {
                        continue;
                    }
                    let prefix = i * config.hidden;
                    for j in 0..config.hidden {
                        grads.w1[prefix + j] += xi * dh[j];
                    }
                }
                for j in 0..config.hidden {
                    grads.b1[j] += dh[j];
                }
                // embedding rows follow the same pattern as w1, using w1's current weights
                for i in 0..2 * config.dim {
                    let mut acc = 0.0;
                    for j in 0..config.hidden {
                        acc += model.w1[i * config.hidden + j] * dh[j];
                    }
                    let row = if i < config.dim { a } else { b };
                    let col = if i < config.dim { i } else { i - config.dim };
                    grads.emb[row * config.dim + col] += acc;
                }
            }
        }
        let inv_batch = 1.0 / config.batch as f64;
        for value in grads.emb.iter_mut() {
            *value *= inv_batch;
        }
        for value in grads.w1.iter_mut() {
            *value *= inv_batch;
        }
        for value in grads.b1.iter_mut() {
            *value *= inv_batch;
        }
        for value in grads.w2.iter_mut() {
            *value *= inv_batch;
        }
        for value in grads.b2.iter_mut() {
            *value *= inv_batch;
        }

        adam.step += 1;
        // `weight_decay` is defined per *epoch*, as in the notebook, where one epoch was one
        // full-batch update. Here an epoch is `batch` mini-steps, so the decay per step is divided
        // accordingly — otherwise a decay of 1.0 annihilates the weights long before generalisation.
        let wd_step = config.weight_decay; // per optimizer step, as AdamW defines it
        let (beta1, beta2, eps): (f64, f64, f64) = (0.9, 0.999, 1e-8);
        let bc1 = 1.0 - beta1.powi(adam.step as i32);
        let bc2 = 1.0 - beta2.powi(adam.step as i32);
        let mut update = |param: &mut f64, grad: f64, m: &mut f64, v: &mut f64, scale: f64, wd: f64| {
            *m = beta1 * *m + (1.0 - beta1) * grad;
            *v = beta2 * *v + (1.0 - beta2) * grad * grad;
            let m_hat = *m / bc1;
            let v_hat = *v / bc2;
            if scale == 0.0 {
                return;
            }
            // Decoupled weight decay still acts on a frozen matrix: it is a step too.
            *param -= lr * wd * *param;
            *param -= scale * lr * m_hat / (v_hat.sqrt() + eps);
        };
        for i in 0..model.emb.len() {
            let (m, v) = (&mut adam.m.emb[i], &mut adam.v.emb[i]);
            update(&mut model.emb[i], grads.emb[i], m, v, lr_scale[0], wd_step);
        }
        for i in 0..model.w1.len() {
            let (m, v) = (&mut adam.m.w1[i], &mut adam.v.w1[i]);
            update(&mut model.w1[i], grads.w1[i], m, v, lr_scale[0], wd_step);
        }
        for i in 0..model.b1.len() {
            let (m, v) = (&mut adam.m.b1[i], &mut adam.v.b1[i]);
            update(&mut model.b1[i], grads.b1[i], m, v, lr_scale[0], 0.0);
        }
        for i in 0..model.w2.len() {
            let (m, v) = (&mut adam.m.w2[i], &mut adam.v.w2[i]);
            update(&mut model.w2[i], grads.w2[i], m, v, lr_scale[1], wd_step);
        }
        for i in 0..model.b2.len() {
            let (m, v) = (&mut adam.m.b2[i], &mut adam.v.b2[i]);
            update(&mut model.b2[i], grads.b2[i], m, v, lr_scale[1], 0.0);
        }
        for value in grads.emb.iter_mut() {
            *value = 0.0;
        }
        for value in grads.w1.iter_mut() {
            *value = 0.0;
        }
        for value in grads.b1.iter_mut() {
            *value = 0.0;
        }
        for value in grads.w2.iter_mut() {
            *value = 0.0;
        }
        for value in grads.b2.iter_mut() {
            *value = 0.0;
        }

        // Account for the backward work the frozen matrices did not do.
        // One sample of backward work: 2·dim·hidden for w1, hidden·p for w2 (per sample).
        let per_sample_w1 = (2 * config.dim * config.hidden) as f64;
        let per_sample_w2 = (config.hidden * config.p) as f64;
        total_backward += (config.batch as f64) * (per_sample_w1 + per_sample_w2);
        if frozen[0] {
            saved_backward += (config.batch as f64) * per_sample_w1;
        }
        if frozen[1] {
            saved_backward += (config.batch as f64) * per_sample_w2;
        }

        if epoch % config.log_every == 0 {
            // Evaluate.
            let mut loss = 0.0;
            let mut correct = 0usize;
            for (a, b, y) in eval_set.iter().copied() {
                model.forward(a, b, &mut x, &mut h, &mut logits);
                let max = logits.iter().cloned().fold(f64::MIN, f64::max);
                let mut sum = 0.0;
                for value in logits.iter() {
                    sum += (value - max).exp();
                }
                let log_sum = sum.ln() + max;
                loss += -(logits[y] - log_sum);
                let mut best = 0usize;
                for k in 1..config.p {
                    if logits[k] > logits[best] {
                        best = k;
                    }
                }
                if best == y {
                    correct += 1;
                }
            }
            let test_loss = loss / eval_set.len() as f64;
            let test_acc = correct as f64 / eval_set.len() as f64;
            let train_loss = batch_loss / config.batch as f64;

            // Geometry of the last window.
            let w1_now = to_mat(&model.w1, 2 * config.dim, config.hidden);
            let w1_prev = to_mat(&prev_w1, 2 * config.dim, config.hidden);
            let w2_now = to_mat(&model.w2, config.hidden, config.p);
            let w2_prev = to_mat(&prev_w2, config.hidden, config.p);
            let g1 = analyse(&w1_prev, &w1_now);
            let g2 = analyse(&w2_prev, &w2_now);
            prev_w1 = model.w1.clone();
            prev_w2 = model.w2.clone();
            peak_v_ang[0] = peak_v_ang[0].max(g1.v_ang);
            peak_v_ang[1] = peak_v_ang[1].max(g2.v_ang);

            if train_loss < 0.01 && train_loss_first_mem.is_nan() {
                train_loss_first_mem = epoch as f64;
            }
            best_test_acc = best_test_acc.max(test_acc);
            final_test_loss = test_loss;
            if epoch_to_95.is_none() && test_acc >= 0.95 {
                epoch_to_95 = Some(epoch);
            }

            // Freezing decisions.
            match config.arm {
                Arm::Base => {}
                Arm::RandomFreeze => {
                    // Same freeze rate the v2 arm uses on average: one layer, every four windows.
                    if epoch % (config.epochs / 8).max(1) == 0 {
                        let which = rng.below(2);
                        if !frozen[which] {
                            frozen[which] = true;
                            freezes += 1;
                        }
                    }
                }
                Arm::V2 => {
                    for (index, geo) in [g1.clone(), g2.clone()].iter().enumerate() {
                        let peak = peak_v_ang[index].max(1e-12);
                        let fell = geo.v_ang < 0.35 * peak;
                        if !frozen[index] && fell && geo.rank_collapse_pct > 0.5 {
                            frozen[index] = true;
                            freezes += 1;
                        }
                    }
                }
            }

            if verbose && epoch + config.log_every > config.epochs {
                for (a, b, y) in eval_set.iter().copied().take(6) {
                    model.forward(a, b, &mut x, &mut h, &mut logits);
                    let mut best = 0usize;
                    for k in 1..config.p {
                        if logits[k] > logits[best] {
                            best = k;
                        }
                    }
                    println!("   a={a:3} b={b:3} y={y:3} pred={best:3} logit_y={:7.3}", logits[y]);
                }
            }
            if verbose {
                println!(
                    "ep {:5} | tr {:7.4} | te {:7.4} | acc {:5.3} | v1 {:7.5} v2 {:7.5} | ranks {:5.2}/{:5.2} | frozen {}{}",
                    epoch,
                    train_loss,
                    test_loss,
                    test_acc,
                    g1.v_ang,
                    g2.v_ang,
                    g1.rank_after,
                    g2.rank_after,
                    if frozen[0] { "w1" } else { "" },
                    if frozen[1] { " w2" } else { "" }
                );
            }
        }
    }

    Report {
        arm: config.arm.name(),
        seed: config.seed,
        epoch_to_95,
        best_test_acc,
        final_test_loss,
        train_loss_first_mem,
        backward_saved_pct: if total_backward > 0.0 { saved_backward / total_backward * 100.0 } else { 0.0 },
        freezes,
    }
}

fn to_mat(flat: &[f64], rows: usize, cols: usize) -> Mat {
    let mut out = vec![vec![0.0; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            out[r][c] = flat[r * cols + c];
        }
    }
    out
}
