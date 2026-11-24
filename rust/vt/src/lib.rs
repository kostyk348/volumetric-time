//! The three mechanisms of `docs/architecture-v2.md`, as code with invariants as tests.
//!
//! Nothing here trains a network: training needs autodiff, and that belongs in the notebook.
//! What lives here is everything that can be *checked* without a GPU — the geometry of an update,
//! the arithmetic of a hard compute budget, and the exactness of layer fusion. If one of these is
//! wrong, the architecture above it is wrong, and this is the cheap place to find out.
//!
//! Matrices are row-major `Vec<Vec<f64>>`.

pub mod fusion;
pub mod geometry;
pub mod plan;
