//! General-Purpose Quintessa WSP Compiler Passes (`compiler_passes.rs`).
//!
//! Architecture-agnostic compiler passes implementing Capabilities 2, 3, 5, 6, and 7:
//! - **Cap 2 (`NullspaceProjector`):** Multi-layer protected subspace basis $U_{\text{prot}}$
//!   and orthogonal projection $W_{\text{safe}} = W (I - U_{\text{prot}} U_{\text{prot}}^T)$
//!   with automated Frobenius isolation proof (`||W_safe U_prot||_F < 1e-6`).
//! - **Cap 3 (`lower_sae_to_jumprelu`):** External SAE/Transcoder dictionary ingestion with
//!   dead-latent pruning (`fire_rate <= 0`), near-duplicate merging (`cos > 0.90`), unit-norm
//!   `W_dec` enforcement, and Top-K -> `F32` JumpReLU threshold lowering.
//! - **Cap 5 (`promote_outlier_dtype`):** Dynamic outlier auto-promotion (`max_abs > 10.0` -> `F32`)
//!   and GPU-safe `F64` -> `F32` cap.
//! - **Cap 6 (`StreamingRidgeAccumulator`):** Out-of-core mini-batch normal-equation accumulator
//!   ($X^T X$, $X^T Z$) from streaming `F16`/`F32` activation batches (`O(d^2)` RAM, never loads
//!   full activation matrix `X` into memory) + trace-adaptive Ridge solver.
//! - **Cap 7 (`verify_dictionary_geometry`, `verify_sparsity_l0`, `verify_dynamical_rollout`):**
//!   Automated pre-emission spectral, geometry, sparsity (`L0`), and recurrent rollout gates.

use crate::etf::welch_bound;
use crate::safetensors_writer::{f16_to_f64, WeightDtype};
use crate::sys::ErrorCode;

// ---------------------------------------------------------------------------
// Cap 5: Dynamic Outlier Auto-Promotion & GPU-Safe Precision Guard
// ---------------------------------------------------------------------------

/// Default outlier magnitude threshold above which `BF16`/`F16`/`I8` tensors are
/// automatically promoted to `F32` before emission.
pub const DEFAULT_OUTLIER_PROMOTE_ABS: f64 = 10.0;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PrecisionGuardResult {
    pub final_dtype: WeightDtype,
    pub max_abs: f64,
    pub promoted_for_outlier: bool,
    pub capped_from_f64: bool,
}

/// Inspect a tensor's values and enforce GPU-safe + outlier-safe precision:
/// - Caps `F64` -> `F32` when `gpu_safe_no_f64` is true.
/// - Auto-promotes `BF16` / `F16` / `I8` -> `F32` when `max_abs > outlier_threshold`.
/// - Rejects any tensor containing `NaN` or `Inf` (`ErrorCode::VerificationFailed`).
pub fn promote_outlier_dtype(
    requested: WeightDtype,
    data: &[f64],
    outlier_threshold: f64,
    gpu_safe_no_f64: bool,
) -> Result<PrecisionGuardResult, ErrorCode> {
    let mut max_abs = 0.0_f64;
    for &v in data {
        if !v.is_finite() {
            return Err(ErrorCode::VerificationFailed);
        }
        let av = v.abs();
        if av > max_abs {
            max_abs = av;
        }
    }

    let mut dt = requested;
    let mut capped_from_f64 = false;
    let mut promoted_for_outlier = false;

    if gpu_safe_no_f64 && dt == WeightDtype::F64 {
        dt = WeightDtype::F32;
        capped_from_f64 = true;
    }

    if max_abs > outlier_threshold
        && matches!(dt, WeightDtype::BF16 | WeightDtype::F16 | WeightDtype::I8)
    {
        dt = WeightDtype::F32;
        promoted_for_outlier = true;
    }

    Ok(PrecisionGuardResult {
        final_dtype: dt,
        max_abs,
        promoted_for_outlier,
        capped_from_f64,
    })
}

// ---------------------------------------------------------------------------
// Cap 2: Multi-Layer Protected Subspace & Nullspace Isolation Planner
// ---------------------------------------------------------------------------

/// Orthonormal basis $U_{\text{prot}} \in \mathbb{R}^{r \times d}$ for protected directions
/// across one or more layers. Projects learned updates into the orthogonal complement
/// $P_{\text{null}} = I - U_{\text{prot}}^T U_{\text{prot}}$ for zero drift on protected circuits.
#[derive(Clone, Debug)]
pub struct NullspaceProjector {
    pub dim: usize,
    pub rank: usize,
    /// Row-major `[rank, dim]` orthonormal basis vectors.
    pub basis: Vec<f64>,
}

impl NullspaceProjector {
    /// Build an orthonormal protected basis from raw direction rows `[n_dirs, dim]`
    /// via two-pass Modified Gram-Schmidt (rank-revealing with tolerance `tol`).
    pub fn from_protected_rows(dim: usize, rows: &[f64], tol: f64) -> Result<Self, ErrorCode> {
        if dim == 0 || !rows.len().is_multiple_of(dim) {
            return Err(ErrorCode::ShapeMismatch);
        }
        let n_rows = rows.len() / dim;
        let mut basis: Vec<f64> = Vec::new();
        let mut rank = 0usize;
        let mut work = vec![0.0_f64; dim];

        for r in 0..n_rows {
            work.copy_from_slice(&rows[r * dim..(r + 1) * dim]);
            // Two-pass MGS for numerical orthogonality
            for _ in 0..2 {
                for b in 0..rank {
                    let brow = &basis[b * dim..(b + 1) * dim];
                    let mut dot = 0.0_f64;
                    for d in 0..dim {
                        dot += work[d] * brow[d];
                    }
                    for d in 0..dim {
                        work[d] -= dot * brow[d];
                    }
                }
            }
            let norm: f64 = work.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > tol {
                let inv = 1.0 / norm;
                for v in &mut work {
                    *v *= inv;
                }
                basis.extend_from_slice(&work);
                rank += 1;
                if rank == dim {
                    break;
                }
            }
        }

        Ok(Self { dim, rank, basis })
    }

    /// Project a row-major weight matrix `W [out_dim, dim]` in-place into the nullspace
    /// of $U_{\text{prot}}$: each row $w_i \leftarrow w_i - \sum_b (w_i \cdot u_b) u_b$.
    pub fn project_rows_in_place(&self, w: &mut [f64], out_dim: usize) -> Result<(), ErrorCode> {
        if w.len() != out_dim * self.dim {
            return Err(ErrorCode::ShapeMismatch);
        }
        for r in 0..out_dim {
            let row = &mut w[r * self.dim..(r + 1) * self.dim];
            for b in 0..self.rank {
                let brow = &self.basis[b * self.dim..(b + 1) * self.dim];
                let mut dot = 0.0_f64;
                for d in 0..self.dim {
                    dot += row[d] * brow[d];
                }
                for d in 0..self.dim {
                    row[d] -= dot * brow[d];
                }
            }
        }
        Ok(())
    }

    /// Compute the exact Frobenius isolation leak `||W * U_prot^T||_F`.
    /// Must be `< 1e-6` (F32) / `< 1e-12` (F64) for certified zero drift.
    pub fn isolation_leak_frobenius(&self, w: &[f64], out_dim: usize) -> Result<f64, ErrorCode> {
        if w.len() != out_dim * self.dim {
            return Err(ErrorCode::ShapeMismatch);
        }
        let mut sum_sq = 0.0_f64;
        for r in 0..out_dim {
            let row = &w[r * self.dim..(r + 1) * self.dim];
            for b in 0..self.rank {
                let brow = &self.basis[b * self.dim..(b + 1) * self.dim];
                let mut dot = 0.0_f64;
                for d in 0..self.dim {
                    dot += row[d] * brow[d];
                }
                sum_sq += dot * dot;
            }
        }
        Ok(sum_sq.sqrt())
    }
}

// ---------------------------------------------------------------------------
// Cap 3: External SAE / Teacher Feature Importer & Top-K -> JumpReLU Lowering
// ---------------------------------------------------------------------------

/// One feature entry imported from a trained teacher SAE or Transcoder.
#[derive(Clone, Debug)]
pub struct SaeFeatureEntry {
    pub w_enc_col: Vec<f64>,
    pub w_dec_row: Vec<f64>,
    pub b_enc: f64,
    pub fire_rate: f64,
    pub topk_cutoff: f64,
}

/// Result of lowering a teacher Top-K SAE into a pruned, deduplicated `JumpReLU` layer.
#[derive(Clone, Debug)]
pub struct LoweredJumpReLuLayer {
    pub dim: usize,
    pub width: usize,
    pub pruned_dead: usize,
    pub merged_duplicates: usize,
    /// `[dim, width]` row-major encoder matrix.
    pub w_enc: Vec<f64>,
    /// `[width]` encoder bias (`F32`).
    pub b_enc: Vec<f64>,
    /// `[width]` calibrated JumpReLU thresholds (`F32`).
    pub threshold: Vec<f64>,
    /// `[width, dim]` unit-norm decoder dictionary rows.
    pub w_dec: Vec<f64>,
}

/// Ingest external SAE features, prune dead latents (`fire_rate <= min_fire_rate`),
/// merge near-duplicate dictionary directions (`|cos| > merge_cos_threshold`),
/// enforce unit-norm `W_dec` rows, and calibrate `JumpReLU` thresholds above the
/// interference floor `3 * sqrt(active_k / dim)`.
pub fn lower_sae_to_jumprelu(
    dim: usize,
    features: &[SaeFeatureEntry],
    active_k: usize,
    min_fire_rate: f64,
    merge_cos_threshold: f64,
) -> Result<LoweredJumpReLuLayer, ErrorCode> {
    if dim == 0 || features.is_empty() {
        return Err(ErrorCode::InvalidInput);
    }
    let interference_floor = 3.0 * ((active_k.max(1) as f64) / (dim as f64)).sqrt();

    let mut kept_dec: Vec<Vec<f64>> = Vec::new();
    let mut kept_enc: Vec<Vec<f64>> = Vec::new();
    let mut kept_b: Vec<f64> = Vec::new();
    let mut kept_thr: Vec<f64> = Vec::new();

    let mut pruned_dead = 0usize;
    let mut merged_duplicates = 0usize;

    for feat in features {
        if feat.w_dec_row.len() != dim || feat.w_enc_col.len() != dim {
            return Err(ErrorCode::ShapeMismatch);
        }
        if feat.fire_rate <= min_fire_rate {
            pruned_dead += 1;
            continue;
        }
        let norm: f64 = feat.w_dec_row.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= 1e-12 || !norm.is_finite() {
            pruned_dead += 1;
            continue;
        }
        let dec_unit: Vec<f64> = feat.w_dec_row.iter().map(|x| x / norm).collect();

        // Check cosine similarity against already-kept dictionary directions
        let mut duplicate_idx: Option<usize> = None;
        for (j, existing) in kept_dec.iter().enumerate() {
            let dot: f64 = existing
                .iter()
                .zip(dec_unit.iter())
                .map(|(a, b)| a * b)
                .sum();
            if dot.abs() > merge_cos_threshold {
                duplicate_idx = Some(j);
                break;
            }
        }

        let thr = feat.topk_cutoff.max(interference_floor);
        if let Some(j) = duplicate_idx {
            // Merge into existing feature by averaging encoder/threshold and renormalizing decoder
            merged_duplicates += 1;
            for d in 0..dim {
                kept_dec[j][d] += dec_unit[d];
                kept_enc[j][d] = 0.5 * (kept_enc[j][d] + feat.w_enc_col[d]);
            }
            let n2: f64 = kept_dec[j]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt()
                .max(1e-12);
            for v in &mut kept_dec[j] {
                *v /= n2;
            }
            kept_b[j] = 0.5 * (kept_b[j] + feat.b_enc);
            kept_thr[j] = kept_thr[j].max(thr);
        } else {
            kept_dec.push(dec_unit);
            kept_enc.push(feat.w_enc_col.clone());
            kept_b.push(feat.b_enc);
            kept_thr.push(thr);
        }
    }

    let width = kept_dec.len();
    if width == 0 {
        return Err(ErrorCode::InvalidInput);
    }

    let mut w_enc = vec![0.0_f64; dim * width];
    let mut w_dec = vec![0.0_f64; width * dim];
    for j in 0..width {
        w_dec[j * dim..(j + 1) * dim].copy_from_slice(&kept_dec[j]);
        for i in 0..dim {
            w_enc[i * width + j] = kept_enc[j][i];
        }
    }

    Ok(LoweredJumpReLuLayer {
        dim,
        width,
        pruned_dead,
        merged_duplicates,
        w_enc,
        b_enc: kept_b,
        threshold: kept_thr,
        w_dec,
    })
}

// ---------------------------------------------------------------------------
// Cap 6: Streaming Out-of-Core Normal-Equation Ridge Accumulator (8 GB Safe)
// ---------------------------------------------------------------------------

/// Streaming mini-batch normal-equation accumulator (`O(d^2 + d*m)` RAM).
///
/// Accumulates $G = X^T X \in \mathbb{R}^{d \times d}$ and $C = X^T Z \in \mathbb{R}^{d \times m}$
/// from arbitrary-length disk-backed `F16` or `F64` activation streams without ever loading
/// the full dataset `X` into memory, then solves $(G + \lambda I) W = C$.
#[derive(Clone, Debug)]
pub struct StreamingRidgeAccumulator {
    pub d_in: usize,
    pub d_out: usize,
    pub sample_count: u64,
    /// Row-major `[d_in, d_in]` Gram matrix $X^T X$.
    pub gram: Vec<f64>,
    /// Row-major `[d_in, d_out]` cross-covariance $X^T Z$.
    pub cross: Vec<f64>,
}

impl StreamingRidgeAccumulator {
    pub fn new(d_in: usize, d_out: usize) -> Result<Self, ErrorCode> {
        if d_in == 0 || d_out == 0 {
            return Err(ErrorCode::InvalidInput);
        }
        Ok(Self {
            d_in,
            d_out,
            sample_count: 0,
            gram: vec![0.0_f64; d_in * d_in],
            cross: vec![0.0_f64; d_in * d_out],
        })
    }

    /// Accumulate a mini-batch of `n` samples (`x_batch: [n, d_in]`, `z_batch: [n, d_out]`).
    pub fn accumulate_batch_f64(
        &mut self,
        x_batch: &[f64],
        z_batch: &[f64],
    ) -> Result<(), ErrorCode> {
        if !x_batch.len().is_multiple_of(self.d_in) || !z_batch.len().is_multiple_of(self.d_out) {
            return Err(ErrorCode::ShapeMismatch);
        }
        let n = x_batch.len() / self.d_in;
        if n != z_batch.len() / self.d_out {
            return Err(ErrorCode::ShapeMismatch);
        }
        for s in 0..n {
            let x = &x_batch[s * self.d_in..(s + 1) * self.d_in];
            let z = &z_batch[s * self.d_out..(s + 1) * self.d_out];
            for i in 0..self.d_in {
                let xi = x[i];
                let g_row = &mut self.gram[i * self.d_in..(i + 1) * self.d_in];
                for j in i..self.d_in {
                    g_row[j] += xi * x[j];
                }
                let c_row = &mut self.cross[i * self.d_out..(i + 1) * self.d_out];
                for k in 0..self.d_out {
                    c_row[k] += xi * z[k];
                }
            }
        }
        self.sample_count = self.sample_count.saturating_add(n as u64);
        Ok(())
    }

    /// Accumulate a mini-batch directly from raw IEEE-754 `FP16` (`u16`) activation caches
    /// using a reusable caller-provided scratch buffer (zero per-batch heap allocation).
    pub fn accumulate_batch_f16(
        &mut self,
        x_f16: &[u16],
        z_f16: &[u16],
        scratch_x: &mut Vec<f64>,
        scratch_z: &mut Vec<f64>,
    ) -> Result<(), ErrorCode> {
        scratch_x.clear();
        scratch_x.extend(x_f16.iter().map(|&b| f16_to_f64(b)));
        scratch_z.clear();
        scratch_z.extend(z_f16.iter().map(|&b| f16_to_f64(b)));
        self.accumulate_batch_f64(scratch_x, scratch_z)
    }

    /// Solve trace-adaptive Ridge: $(G + \lambda I) W = C$ where
    /// $\lambda = \max(\text{min\_lambda}, \alpha \cdot \text{tr}(G) / d_{\text{in}})$.
    /// Returns row-major `W [d_in, d_out]`.
    pub fn solve_trace_adaptive(&self, alpha: f64, min_lambda: f64) -> Result<Vec<f64>, ErrorCode> {
        if self.sample_count == 0 {
            return Err(ErrorCode::InvalidInput);
        }
        let d = self.d_in;
        let m = self.d_out;
        let mut a = self.gram.clone();
        // Mirror upper triangle to lower triangle
        let mut trace = 0.0_f64;
        for i in 0..d {
            trace += a[i * d + i];
            for j in (i + 1)..d {
                a[j * d + i] = a[i * d + j];
            }
        }
        let lambda = (alpha * trace / (d as f64)).max(min_lambda).max(1e-12);
        for i in 0..d {
            a[i * d + i] += lambda;
        }

        // Cholesky factorization A = L L^T
        let mut l = vec![0.0_f64; d * d];
        for i in 0..d {
            for j in 0..=i {
                let mut sum = a[i * d + j];
                for k in 0..j {
                    sum -= l[i * d + k] * l[j * d + k];
                }
                if i == j {
                    if sum <= 0.0 || !sum.is_finite() {
                        return Err(ErrorCode::SingularMatrix);
                    }
                    l[i * d + j] = sum.sqrt();
                } else {
                    l[i * d + j] = sum / l[j * d + j];
                }
            }
        }

        // Solve L L^T W = C column by column
        let mut w = vec![0.0_f64; d * m];
        let mut y = vec![0.0_f64; d];
        for col in 0..m {
            for i in 0..d {
                let mut sum = self.cross[i * m + col];
                for k in 0..i {
                    sum -= l[i * d + k] * y[k];
                }
                y[i] = sum / l[i * d + i];
            }
            for i in (0..d).rev() {
                let mut sum = y[i];
                for k in (i + 1)..d {
                    sum -= l[k * d + i] * w[k * m + col];
                }
                w[i * m + col] = sum / l[i * d + i];
            }
        }

        Ok(w)
    }
}

// ---------------------------------------------------------------------------
// Cap 7: Pre-Emission Spectral, Geometry, Sparsity (L0) & Rollout Gates
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DictionaryGeometryReport {
    pub num_features: usize,
    pub dim: usize,
    pub max_norm_error: f64,
    pub welch_bound: f64,
    pub median_coherence: f64,
    pub p99_coherence: f64,
    pub max_coherence: f64,
    pub passed: bool,
}

/// Verify unit-norm `W_dec` rows (`1.0 ± 1e-5`) and off-diagonal coherence vs. Welch bound.
pub fn verify_dictionary_geometry(
    w_dec: &[f64],
    num_features: usize,
    dim: usize,
    max_p99_coherence: f64,
) -> Result<DictionaryGeometryReport, ErrorCode> {
    if num_features == 0 || dim == 0 || w_dec.len() != num_features * dim {
        return Err(ErrorCode::ShapeMismatch);
    }
    let mut max_norm_error = 0.0_f64;
    for i in 0..num_features {
        let row = &w_dec[i * dim..(i + 1) * dim];
        let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        let err = (norm - 1.0).abs();
        if err > max_norm_error {
            max_norm_error = err;
        }
    }

    let sample_n = num_features.min(256);
    let mut coherences = Vec::with_capacity(sample_n * sample_n / 2);
    for i in 0..sample_n {
        let ri = &w_dec[i * dim..(i + 1) * dim];
        for j in (i + 1)..sample_n {
            let rj = &w_dec[j * dim..(j + 1) * dim];
            let dot: f64 = ri.iter().zip(rj.iter()).map(|(a, b)| a * b).sum();
            coherences.push(dot.abs());
        }
    }
    coherences.sort_by(|a, b| a.total_cmp(b));
    let (median_coh, p99_coh, max_coh) = if coherences.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let n = coherences.len();
        (
            coherences[n / 2],
            coherences[((n - 1) * 99) / 100],
            coherences[n - 1],
        )
    };
    let wb = welch_bound(num_features, dim);
    let passed = max_norm_error <= 1e-5 && p99_coh <= max_p99_coherence;

    Ok(DictionaryGeometryReport {
        num_features,
        dim,
        max_norm_error,
        welch_bound: wb,
        median_coherence: median_coh,
        p99_coherence: p99_coh,
        max_coherence: max_coh,
        passed,
    })
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SparsityGateReport {
    pub mean_l0: f64,
    pub dead_latent_ratio: f64,
    pub passed: bool,
}

/// Verify active sparsity `L0 in [min_l0, max_l0]` and dead-latent ratio `< max_dead_ratio`
/// on a batch of sparse feature activations `[num_samples, width]`.
pub fn verify_sparsity_l0(
    activations: &[f64],
    num_samples: usize,
    width: usize,
    min_l0: f64,
    max_l0: f64,
    max_dead_ratio: f64,
) -> Result<SparsityGateReport, ErrorCode> {
    if num_samples == 0 || width == 0 || activations.len() != num_samples * width {
        return Err(ErrorCode::ShapeMismatch);
    }
    let mut fired_counts = vec![0u64; width];
    let mut total_active = 0u64;
    for s in 0..num_samples {
        let row = &activations[s * width..(s + 1) * width];
        for j in 0..width {
            if row[j].abs() > 0.0 {
                fired_counts[j] += 1;
                total_active += 1;
            }
        }
    }
    let mean_l0 = (total_active as f64) / (num_samples as f64);
    let dead = fired_counts.iter().filter(|&&c| c == 0).count();
    let dead_latent_ratio = (dead as f64) / (width as f64);
    let passed = mean_l0 >= min_l0 && mean_l0 <= max_l0 && dead_latent_ratio < max_dead_ratio;
    Ok(SparsityGateReport {
        mean_l0,
        dead_latent_ratio,
        passed,
    })
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RolloutGateReport {
    pub steps: usize,
    pub max_state_norm: f64,
    pub nonfinite_steps: usize,
    pub passed: bool,
}

/// Multi-step dynamical rollout stability gate: runs any recurrent/SSM/HRM step closure
/// for `steps` (`>= 64`) and asserts `0` NaN/Inf states and bounded norm (`<= max_allowed_norm`).
pub fn verify_dynamical_rollout<F>(
    state_dim: usize,
    steps: usize,
    max_allowed_norm: f64,
    mut step_fn: F,
) -> Result<RolloutGateReport, ErrorCode>
where
    F: FnMut(usize, &mut [f64]),
{
    if state_dim == 0 || steps == 0 {
        return Err(ErrorCode::InvalidInput);
    }
    let mut state = vec![0.0_f64; state_dim];
    let mut max_state_norm = 0.0_f64;
    let mut nonfinite_steps = 0usize;

    for t in 0..steps {
        step_fn(t, &mut state);
        if state.iter().any(|v| !v.is_finite()) {
            nonfinite_steps += 1;
            continue;
        }
        let norm = state.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > max_state_norm {
            max_state_norm = norm;
        }
    }

    let passed = nonfinite_steps == 0 && max_state_norm <= max_allowed_norm;
    Ok(RolloutGateReport {
        steps,
        max_state_norm,
        nonfinite_steps,
        passed,
    })
}

// ---------------------------------------------------------------------------
// Multimodal Weight-Space Geometry: Procrustes, Direct-Sum Subspaces & N-Dim RoPE
// ---------------------------------------------------------------------------

/// Remove the modality cone effect by subtracting the empirical mean vector across `n` rows
/// (`[n, dim]`) and renormalizing each row to unit L2 norm.
pub fn center_and_renormalize_rows(rows: &mut [f64], dim: usize) -> Result<(), ErrorCode> {
    if dim == 0 || rows.is_empty() || !rows.len().is_multiple_of(dim) {
        return Err(ErrorCode::ShapeMismatch);
    }
    let n = rows.len() / dim;
    let mut mean = vec![0.0_f64; dim];
    for r in 0..n {
        let row = &rows[r * dim..(r + 1) * dim];
        for d in 0..dim {
            mean[d] += row[d];
        }
    }
    let inv_n = 1.0 / (n as f64);
    for m in &mut mean {
        *m *= inv_n;
    }
    for r in 0..n {
        let row = &mut rows[r * dim..(r + 1) * dim];
        let mut sq = 0.0_f64;
        for d in 0..dim {
            row[d] -= mean[d];
            sq += row[d] * row[d];
        }
        let norm = sq.sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }
    Ok(())
}

/// Symmetric Jacobi eigendecomposition of a `d x d` symmetric matrix `a` (row-major).
/// Returns `(eigenvalues, eigenvectors_col_major_as_rows)` where `eigvecs[k]` is the `k`-th
/// unit eigenvector (stored row-major `[d, d]` with eigenvector `k` in column `k`).
fn jacobi_symmetric_eigen(d: usize, mut a: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0_f64; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }
    let max_sweeps = 64;
    for _ in 0..max_sweeps {
        let mut max_off = 0.0_f64;
        for p in 0..d {
            for q in (p + 1)..d {
                let apq = a[p * d + q];
                if apq.abs() > max_off {
                    max_off = apq.abs();
                }
                if apq.abs() < 1e-14 {
                    continue;
                }
                let app = a[p * d + p];
                let aqq = a[q * d + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[p * d + p] = app - t * apq;
                a[q * d + q] = aqq + t * apq;
                a[p * d + q] = 0.0;
                a[q * d + p] = 0.0;

                for r in 0..d {
                    if r != p && r != q {
                        let arp = a[r * d + p];
                        let arq = a[r * d + q];
                        let nrp = c * arp - s * arq;
                        let nrq = s * arp + c * arq;
                        a[r * d + p] = nrp;
                        a[p * d + r] = nrp;
                        a[r * d + q] = nrq;
                        a[q * d + r] = nrq;
                    }
                    let vrp = v[r * d + p];
                    let vrq = v[r * d + q];
                    v[r * d + p] = c * vrp - s * vrq;
                    v[r * d + q] = s * vrp + c * vrq;
                }
            }
        }
        if max_off < 1e-13 {
            break;
        }
    }
    let evals: Vec<f64> = (0..d).map(|i| a[i * d + i]).collect();
    (evals, v)
}

/// Solve the **Orthogonal Procrustes Problem**:
/// `min_{R^T R = I} || X * R^T - Y ||_F` for paired anchor activations `X, Y: [n, dim]`.
///
/// Returns a strictly orthogonal `dim x dim` alignment matrix `R` (row-major) such that
/// `y_hat = R * x`, with `||R^T R - I||_F < 1e-10` and `cond(R) = 1.0` (zero rank collapse).
pub fn solve_orthogonal_procrustes(
    dim: usize,
    x_rows: &[f64],
    y_rows: &[f64],
) -> Result<Vec<f64>, ErrorCode> {
    if dim == 0
        || x_rows.is_empty()
        || x_rows.len() != y_rows.len()
        || !x_rows.len().is_multiple_of(dim)
    {
        return Err(ErrorCode::ShapeMismatch);
    }
    let n = x_rows.len() / dim;

    // Cross-covariance M = Y^T X [dim, dim]
    let mut m = vec![0.0_f64; dim * dim];
    for s in 0..n {
        let xr = &x_rows[s * dim..(s + 1) * dim];
        let yr = &y_rows[s * dim..(s + 1) * dim];
        for i in 0..dim {
            let yi = yr[i];
            let mrow = &mut m[i * dim..(i + 1) * dim];
            for j in 0..dim {
                mrow[j] += yi * xr[j];
            }
        }
    }

    // M^T M = V Sigma^2 V^T
    let mut mtm = vec![0.0_f64; dim * dim];
    for i in 0..dim {
        for j in i..dim {
            let mut acc = 0.0_f64;
            for k in 0..dim {
                acc += m[k * dim + i] * m[k * dim + j];
            }
            mtm[i * dim + j] = acc;
            mtm[j * dim + i] = acc;
        }
    }

    let (evals, v) = jacobi_symmetric_eigen(dim, mtm);

    // Build orthonormal columns of U = M * V * Sigma^{-1}, completing any zero-sigma columns via MGS
    let mut u_cols: Vec<Vec<f64>> = vec![vec![0.0_f64; dim]; dim];
    let mut valid_col = vec![false; dim];
    for k in 0..dim {
        let sigma = evals[k].max(0.0).sqrt();
        if sigma > 1e-10 {
            for i in 0..dim {
                let mut acc = 0.0_f64;
                for j in 0..dim {
                    acc += m[i * dim + j] * v[j * dim + k];
                }
                u_cols[k][i] = acc / sigma;
            }
            // Re-orthogonalize against previous valid columns
            for prev in 0..k {
                if valid_col[prev] {
                    let dot: f64 = u_cols[k]
                        .iter()
                        .zip(u_cols[prev].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    for i in 0..dim {
                        u_cols[k][i] -= dot * u_cols[prev][i];
                    }
                }
            }
            let norm = u_cols[k].iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for x in &mut u_cols[k] {
                    *x /= norm;
                }
                valid_col[k] = true;
            }
        }
    }

    // Complete any missing orthonormal columns using standard basis candidates
    for k in 0..dim {
        if valid_col[k] {
            continue;
        }
        for cand in 0..dim {
            let mut w = vec![0.0_f64; dim];
            w[cand] = 1.0;
            for prev in 0..dim {
                if valid_col[prev] {
                    let dot: f64 = w.iter().zip(u_cols[prev].iter()).map(|(a, b)| a * b).sum();
                    for i in 0..dim {
                        w[i] -= dot * u_cols[prev][i];
                    }
                }
            }
            let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-8 {
                for x in &mut w {
                    *x /= norm;
                }
                u_cols[k] = w;
                valid_col[k] = true;
                break;
            }
        }
    }

    // R = U * V^T => R[i, j] = sum_k U[i, k] * V[j, k]
    let mut r = vec![0.0_f64; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut acc = 0.0_f64;
            for k in 0..dim {
                acc += u_cols[k][i] * v[j * dim + k];
            }
            r[i * dim + j] = acc;
        }
    }
    Ok(r)
}

/// Direct-sum orthogonal residual subspace partition:
/// `R^dim = S_shared [0..shared_dim) (+) S_coord [shared_dim..coord_end) (+) S_private [coord_end..dim)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultimodalSubspacePlan {
    pub dim: usize,
    pub shared_dim: usize,
    pub coord_dim: usize,
    pub private_dim: usize,
}

impl MultimodalSubspacePlan {
    /// Standard 50% shared / 25% spatio-temporal coord / 25% modality-private partition.
    pub fn default_for_dim(dim: usize) -> Result<Self, ErrorCode> {
        if dim < 4 {
            return Err(ErrorCode::InvalidArchitecture);
        }
        let shared_dim = dim / 2;
        let coord_dim = dim / 4;
        let private_dim = dim - shared_dim - coord_dim;
        Ok(Self {
            dim,
            shared_dim,
            coord_dim,
            private_dim,
        })
    }

    /// Build a `NullspaceProjector` protecting `S_shared` and/or `S_coord` so modality-private
    /// weight updates written into `S_private` cause exact zero drift on shared semantics.
    pub fn build_protected_nullspace(
        &self,
        protect_shared: bool,
        protect_coord: bool,
    ) -> Result<NullspaceProjector, ErrorCode> {
        let mut rows = Vec::new();
        if protect_shared {
            for i in 0..self.shared_dim {
                let mut e = vec![0.0_f64; self.dim];
                e[i] = 1.0;
                rows.extend_from_slice(&e);
            }
        }
        if protect_coord {
            for i in self.shared_dim..(self.shared_dim + self.coord_dim) {
                let mut e = vec![0.0_f64; self.dim];
                e[i] = 1.0;
                rows.extend_from_slice(&e);
            }
        }
        NullspaceProjector::from_protected_rows(self.dim, &rows, 1e-12)
    }
}

/// Synthesize factorized N-Dimensional (`1D` text/audio, `2D` image `H x W`, `3D` video `T x H x W`)
/// RoPE inverse frequencies in `F32` (total length = `head_dim / 2`).
pub fn synthesize_ndim_rope_inv_freq(
    head_dim: usize,
    axes: u8,
    base_theta: f64,
) -> Result<Vec<f32>, ErrorCode> {
    if head_dim < 2 || !head_dim.is_multiple_of(2) || axes == 0 || axes > 3 || base_theta <= 1.0 {
        return Err(ErrorCode::InvalidInput);
    }
    let half = head_dim / 2;
    let ax = axes as usize;
    let per_axis = half / ax;
    let rem = half % ax;
    let mut freqs = Vec::with_capacity(half);

    for a in 0..ax {
        let count = per_axis + if a < rem { 1 } else { 0 };
        let denom = (count.max(1) * 2) as f64;
        for k in 0..count {
            let exp = ((2 * k) as f64) / denom;
            freqs.push((1.0 / base_theta.powf(exp)) as f32);
        }
    }
    Ok(freqs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors_writer::f64_to_f16;

    #[test]
    fn cap5_dynamic_outlier_promotion_and_f64_cap() {
        let normal = [0.01, -0.05, 0.25, -0.1];
        let r1 = promote_outlier_dtype(WeightDtype::BF16, &normal, 10.0, true).unwrap();
        assert_eq!(r1.final_dtype, WeightDtype::BF16);
        assert!(!r1.promoted_for_outlier);

        let outlier = [0.1, 250.0, -0.2];
        let r2 = promote_outlier_dtype(WeightDtype::F16, &outlier, 10.0, true).unwrap();
        assert_eq!(r2.final_dtype, WeightDtype::F32);
        assert!(r2.promoted_for_outlier);

        let r3 = promote_outlier_dtype(WeightDtype::F64, &normal, 10.0, true).unwrap();
        assert_eq!(r3.final_dtype, WeightDtype::F32);
        assert!(r3.capped_from_f64);

        let bad = [0.0, f64::NAN, 1.0];
        assert_eq!(
            promote_outlier_dtype(WeightDtype::F32, &bad, 10.0, true).unwrap_err(),
            ErrorCode::VerificationFailed
        );
    }

    #[test]
    fn cap2_nullspace_projector_achieves_zero_drift() {
        let prot = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2 protected dirs in R^4
        let proj = NullspaceProjector::from_protected_rows(4, &prot, 1e-10).unwrap();
        assert_eq!(proj.rank, 2);

        let mut w = vec![3.0, -2.0, 0.5, 1.5, -1.0, 4.0, 2.0, -0.5];
        proj.project_rows_in_place(&mut w, 2).unwrap();
        let leak = proj.isolation_leak_frobenius(&w, 2).unwrap();
        assert!(leak < 1e-12, "Expected zero nullspace leak, got {leak}");
    }

    #[test]
    fn cap3_sae_importer_prunes_dead_merges_dups_and_calibrates_jumprelu() {
        let feats = vec![
            SaeFeatureEntry {
                w_enc_col: vec![1.0, 0.0, 0.0, 0.0],
                w_dec_row: vec![2.0, 0.0, 0.0, 0.0], // non-unit -> normalized
                b_enc: -0.1,
                fire_rate: 0.05,
                topk_cutoff: 0.2,
            },
            SaeFeatureEntry {
                w_enc_col: vec![0.99, 0.01, 0.0, 0.0],
                w_dec_row: vec![1.0, 0.01, 0.0, 0.0], // duplicate cos > 0.99 -> merged
                b_enc: -0.1,
                fire_rate: 0.04,
                topk_cutoff: 0.25,
            },
            SaeFeatureEntry {
                w_enc_col: vec![0.0, 1.0, 0.0, 0.0],
                w_dec_row: vec![0.0, 1.0, 0.0, 0.0],
                b_enc: 0.0,
                fire_rate: 0.0, // dead -> pruned
                topk_cutoff: 0.1,
            },
            SaeFeatureEntry {
                w_enc_col: vec![0.0, 0.0, 1.0, 0.0],
                w_dec_row: vec![0.0, 0.0, 1.0, 0.0],
                b_enc: -0.05,
                fire_rate: 0.08,
                topk_cutoff: 0.9,
            },
        ];

        let lowered = lower_sae_to_jumprelu(4, &feats, 1, 0.0, 0.90).unwrap();
        assert_eq!(lowered.pruned_dead, 1);
        assert_eq!(lowered.merged_duplicates, 1);
        assert_eq!(lowered.width, 2);

        let geom = verify_dictionary_geometry(&lowered.w_dec, lowered.width, 4, 0.25).unwrap();
        assert!(geom.passed);
    }

    #[test]
    fn cap6_streaming_f16_ridge_recovers_linear_map() {
        let mut acc = StreamingRidgeAccumulator::new(2, 2).unwrap();
        let mut sx = Vec::new();
        let mut sz = Vec::new();
        // True W = [[2.0, -1.0], [0.5, 3.0]]
        let samples = [
            ([1.0_f64, 0.0], [2.0_f64, -1.0]),
            ([0.0, 1.0], [0.5, 3.0]),
            ([1.0, 1.0], [2.5, 2.0]),
            ([-1.0, 2.0], [-1.0, 7.0]),
        ];
        for (x, z) in samples {
            let xf: Vec<u16> = x.iter().map(|&v| f64_to_f16(v)).collect();
            let zf: Vec<u16> = z.iter().map(|&v| f64_to_f16(v)).collect();
            acc.accumulate_batch_f16(&xf, &zf, &mut sx, &mut sz)
                .unwrap();
        }
        let w = acc.solve_trace_adaptive(1e-6, 1e-8).unwrap();
        assert!((w[0] - 2.0).abs() < 1e-2);
        assert!((w[1] - (-1.0)).abs() < 1e-2);
        assert!((w[2] - 0.5).abs() < 1e-2);
        assert!((w[3] - 3.0).abs() < 1e-2);
    }

    #[test]
    fn cap7_sparsity_and_rollout_gates() {
        let mut acts = vec![0.0_f64; 4 * 20];
        for s in 0..4 {
            for j in 0..20 {
                if (j + s) % 2 == 0 || j < 14 {
                    acts[s * 20 + j] = 1.0;
                }
            }
        }
        let sp = verify_sparsity_l0(&acts, 4, 20, 12.0, 20.0, 0.05).unwrap();
        assert!(sp.passed);

        let roll = verify_dynamical_rollout(8, 64, 10.0, |_t, st| {
            for v in st.iter_mut() {
                *v = 0.85 * *v + 0.1;
            }
        })
        .unwrap();
        assert!(roll.passed);
        assert_eq!(roll.nonfinite_steps, 0);
    }

    #[test]
    fn multimodal_procrustes_subspace_and_ndim_rope() {
        // 1. Exact orthogonal rotation recovery: rotate 2D/4D anchors by 90 deg in (0,1) and (2,3)
        let x = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let y = [
            0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0,
        ];
        let r = solve_orthogonal_procrustes(4, &x, &y).unwrap();

        // Check strict orthogonality ||R^T R - I||_F < 1e-10
        let mut orth_err_sq = 0.0_f64;
        for i in 0..4 {
            for j in 0..4 {
                let mut dot = 0.0_f64;
                for k in 0..4 {
                    dot += r[k * 4 + i] * r[k * 4 + j];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                orth_err_sq += (dot - target) * (dot - target);
            }
        }
        assert!(
            orth_err_sq.sqrt() < 1e-10,
            "orth_err={}",
            orth_err_sq.sqrt()
        );

        // Check exact recovery R * x_0 == y_0
        for s in 0..4 {
            for i in 0..4 {
                let mut pred = 0.0_f64;
                for j in 0..4 {
                    pred += r[i * 4 + j] * x[s * 4 + j];
                }
                assert!((pred - y[s * 4 + i]).abs() < 1e-9);
            }
        }

        // 2. Direct-sum subspace plan + zero-drift nullspace isolation
        let plan = MultimodalSubspacePlan::default_for_dim(16).unwrap();
        assert_eq!(plan.shared_dim, 8);
        assert_eq!(plan.coord_dim, 4);
        assert_eq!(plan.private_dim, 4);
        let ns = plan.build_protected_nullspace(true, true).unwrap();
        assert_eq!(ns.rank, 12);
        let mut w_mod = vec![1.0_f64; 2 * 16];
        ns.project_rows_in_place(&mut w_mod, 2).unwrap();
        assert!(ns.isolation_leak_frobenius(&w_mod, 2).unwrap() < 1e-12);

        // 3. 3D RoPE (T x H x W) frequency table
        let f3d = synthesize_ndim_rope_inv_freq(64, 3, 10_000.0).unwrap();
        assert_eq!(f3d.len(), 32);
        assert!(f3d.iter().all(|v| v.is_finite() && *v > 0.0 && *v <= 1.0));
    }
}
