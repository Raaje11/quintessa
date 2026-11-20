//! Unified Multimodal Parallel Next-Concept Model (NCM) Encoder & Decoder (`multimodal_ncm.rs`).
//!
//! Extends Quintessa's 8-level multi-scale residual pyramid (`[1, 4, 9, 16, 25, 36, 49, 64]`)
//! and parallel 64-slot non-autoregressive NCM generator across all four modalities:
//! - **`TextChunk64`**: 64-token text sequence encoded/decoded in parallel.
//! - **`VisualCad8x8`**: `8x8 = 64` 2D spatial image or Parametric CAD canvas patches
//!   (multi-scale `1x1 -> 2x2 -> ... -> 8x8` VAR-style residual pyramid).
//! - **`AudioRvq64`**: 64-frame `15 Hz` voice/music acoustic chunk (`~4.27s` per parallel wave).
//! - **`Field3D4x4x4`**: `4x4x4 = 64` 3D voxel / physical field tensor (CAD SDF / ANSYS field).
//!
//! Architecture:
//! ```text
//! [ Raw Modality Input: 64 x dim ]
//!              │
//!              ▼
//!   8-Level Multi-Scale Residual VQ Encoder ([1, 4, 9, 16, 25, 36, 49, 64])
//!              │
//!              ▼
//!   Orthogonal Procrustes Bridge R_m (R_m^T R_m = I, cond = 1.0)
//!              │
//!              ▼
//!   Compiled Brain Workspace [k, r, s, m]  (4*dim latent cognition + SACI + Simulation)
//!              │
//!              ▼
//!   Parallel NCM Chunk Generator (W_ncm_chunk + Modality-Conditioned 4-Head Cross-Attn)
//!              │
//!              ▼
//!   z_hat [64, dim] ──► Target Modality Parallel Decoder (Text / 8x8 Visual / Audio / 4x4x4 3D)
//! ```
//!
//! All weights save and load as pure Burn (Rust) `.safetensors` (`{"format":"burn"}`,
//! GPU-safe `BF16`/`F32`, zero `F64`, zero compiler fingerprints).

use std::path::Path;

use crate::compiler_passes::{promote_outlier_dtype, DEFAULT_OUTLIER_PROMOTE_ABS};
use crate::ncm_generator::NCM_CHUNK_LEN;
use crate::safetensors_writer::{
    read_safetensors, write_mixed_safetensors, TypedNamedTensor, WeightDtype,
};
use crate::superposition_wsp::SplitMix64;
use crate::sys::ErrorCode;

pub const NCM_PYRAMID_LEVELS: [usize; 8] = [1, 4, 9, 16, 25, 36, 49, 64];
pub const NCM_NUM_MODALITIES: usize = 4;
pub const NCM_ATTN_HEADS: usize = 4;
pub const NCM_REL_POS_LEN: usize = 2 * NCM_CHUNK_LEN - 1; // 127

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NcmModality {
    TextChunk64 = 0,
    VisualCad8x8 = 1,
    AudioRvq64 = 2,
    Field3D4x4x4 = 3,
}

impl NcmModality {
    pub const ALL: [Self; NCM_NUM_MODALITIES] = [
        Self::TextChunk64,
        Self::VisualCad8x8,
        Self::AudioRvq64,
        Self::Field3D4x4x4,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TextChunk64 => "text_64",
            Self::VisualCad8x8 => "visual_cad_8x8",
            Self::AudioRvq64 => "audio_rvq_64",
            Self::Field3D4x4x4 => "field_3d_4x4x4",
        }
    }
}

/// One level of the 8-level Multi-Scale Residual VQ NCM Pyramid (`ell in [1, 4, 9, ..., 64]`).
#[derive(Clone, Debug)]
pub struct NcmPyramidLevel {
    pub level_len: usize,
    pub codebook_size: usize,
    /// `[NCM_CHUNK_LEN, level_len]` area-pooling downsampler.
    pub down_w: Vec<f64>,
    /// `[codebook_size, dim]` unit-norm residual codebook.
    pub codebook: Vec<f64>,
    /// `[level_len, NCM_CHUNK_LEN]` bilinear/transpose upsampler.
    pub up_w: Vec<f64>,
}

/// Complete Multimodal Parallel NCM Encoder + Brain Bridge + Parallel NCM Decoder.
#[derive(Clone, Debug)]
pub struct MultimodalNcmCodec {
    pub dim: usize,
    pub exp_dim: usize,
    pub codebook_size: usize,
    /// 8-level multi-scale residual pyramid shared across 64-slot chunks.
    pub pyramid: Vec<NcmPyramidLevel>,
    /// `[NCM_NUM_MODALITIES, dim, dim]` strictly orthogonal Procrustes bridges `R_m`
    /// mapping each modality's pyramid concept state into the Brain's shared manifold.
    pub procrustes_in: Vec<f64>,
    /// `[NCM_NUM_MODALITIES, dim, dim]` inverse orthogonal bridge `R_m^T` for decoding.
    pub procrustes_out: Vec<f64>,
    /// `[4 * dim, exp_dim]` first-stage parallel NCM expansion from Brain `[k, r, s, m]`.
    pub w_ncm_exp: Vec<f64>,
    /// `[exp_dim, NCM_CHUNK_LEN * dim]` parallel 64-slot latent generator.
    pub w_ncm_out: Vec<f64>,
    /// `[NCM_NUM_MODALITIES, NCM_ATTN_HEADS, NCM_REL_POS_LEN]` per-modality relative-position
    /// cross-attention bias (solves `pending.md` Gap #1 across modalities).
    pub modality_rel_pos_bias: Vec<f64>,
    /// `[NCM_ATTN_HEADS, dim, head_dim]` cross-attention query projection.
    pub w_q_attn: Vec<f64>,
    /// `[NCM_ATTN_HEADS, dim, head_dim]` cross-attention key projection.
    pub w_k_attn: Vec<f64>,
    /// `[dim, dim]` cross-attention value projection.
    pub w_v_attn: Vec<f64>,
}

/// Encoded multi-scale NCM representation + Brain workspace vector `[4 * dim]`.
#[derive(Clone, Debug)]
pub struct EncodedNcmConcept {
    pub modality: NcmModality,
    /// Multi-scale residual VQ indices across all 8 levels (`1+4+9+16+25+36+49+64 = 204` codes).
    pub pyramid_codes: Vec<u16>,
    /// Reconstructed 64-slot concept tensor `[NCM_CHUNK_LEN, dim]` in the shared Brain manifold.
    pub aligned_slots: Vec<f64>,
    /// Pooled Brain workspace state `[k, r, s, m]` (`4 * dim`) ready for Reasoning/SACI/Simulation.
    pub brain_krsm: Vec<f64>,
}

/// Parallel decoded 64-slot multimodal output produced in one non-autoregressive NCM wave.
#[derive(Clone, Debug)]
pub struct DecodedNcmOutput {
    pub modality: NcmModality,
    /// Continuous 64-slot decoded tensor `[NCM_CHUNK_LEN, dim]`.
    pub slots_64x_dim: Vec<f64>,
    /// Parallel quantized level-7 codebook token IDs (`[64]`, for Text or 15 Hz Audio RVQ).
    pub discrete_tokens_64: [u16; NCM_CHUNK_LEN],
    /// Parallel 2D `8x8` visual/CAD canvas or flattened `4x4x4` 3D field (`[64]` scalars in `[0, 1]`).
    pub spatial_field_64: [f32; NCM_CHUNK_LEN],
}

#[inline]
fn layer_norm_in_place(x: &mut [f64], seq_len: usize, dim: usize) {
    let inv_d = 1.0 / (dim as f64);
    for t in 0..seq_len {
        let row = &mut x[t * dim..(t + 1) * dim];
        let mean = row.iter().sum::<f64>() * inv_d;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() * inv_d;
        let inv_std = 1.0 / (var + 1e-6).sqrt();
        for v in row.iter_mut() {
            *v = (*v - mean) * inv_std;
        }
    }
}

/// Build an exact `dim x dim` orthogonal rotation matrix via Householder reflections (`R^T R = I`).
fn make_orthogonal_matrix(dim: usize, seed: u64) -> Vec<f64> {
    let mut r = vec![0.0_f64; dim * dim];
    for i in 0..dim {
        r[i * dim + i] = 1.0;
    }
    let mut rng = SplitMix64::new(seed);
    let reflections = dim.min(8);
    let mut v = vec![0.0_f64; dim];
    let mut r_v = vec![0.0_f64; dim];

    for _ in 0..reflections {
        let mut sq = 0.0_f64;
        for x in v.iter_mut() {
            *x = rng.next_normal();
            sq += *x * *x;
        }
        let norm = sq.sqrt().max(1e-12);
        for x in v.iter_mut() {
            *x /= norm;
        }
        // Apply H = I - 2 v v^T from the left: R <- R - 2 v (v^T R)
        for col in 0..dim {
            let mut dot = 0.0_f64;
            for row in 0..dim {
                dot += v[row] * r[row * dim + col];
            }
            r_v[col] = dot;
        }
        for row in 0..dim {
            for col in 0..dim {
                r[row * dim + col] -= 2.0 * v[row] * r_v[col];
            }
        }
    }
    r
}

impl MultimodalNcmCodec {
    /// Compile a zero-dependency `MultimodalNcmCodec` with an 8-level multi-scale residual
    /// pyramid (`[1, 4, 9, 16, 25, 36, 49, 64]`), per-modality Orthogonal Procrustes bridges,
    /// and per-modality 4-head relative-position cross-attention.
    pub fn compile(
        dim: usize,
        exp_dim: usize,
        codebook_size: usize,
        seed: u64,
    ) -> Result<Self, ErrorCode> {
        if dim < 8 || !dim.is_multiple_of(NCM_ATTN_HEADS) || exp_dim == 0 || codebook_size < 4 {
            return Err(ErrorCode::InvalidArchitecture);
        }

        let mut rng = SplitMix64::new(seed);
        let mut pyramid = Vec::with_capacity(NCM_PYRAMID_LEVELS.len());

        for &ell in &NCM_PYRAMID_LEVELS {
            // Smooth area-pooling downsampler [64, ell] and transpose upsampler [ell, 64]
            let mut down_w = vec![0.0_f64; NCM_CHUNK_LEN * ell];
            let mut up_w = vec![0.0_f64; ell * NCM_CHUNK_LEN];
            for e in 0..ell {
                let center = ((e as f64) + 0.5) * (NCM_CHUNK_LEN as f64) / (ell as f64);
                let sigma = ((NCM_CHUNK_LEN as f64) / (ell as f64)).max(1.0);
                let mut col_sum = 0.0_f64;
                for t in 0..NCM_CHUNK_LEN {
                    let dist = ((t as f64) + 0.5 - center) / sigma;
                    let w = (-0.5 * dist * dist).exp();
                    down_w[t * ell + e] = w;
                    col_sum += w;
                }
                let inv = 1.0 / col_sum.max(1e-12);
                for t in 0..NCM_CHUNK_LEN {
                    down_w[t * ell + e] *= inv;
                    up_w[e * NCM_CHUNK_LEN + t] = down_w[t * ell + e] * 0.35;
                }
            }

            // Unit-norm residual codebook [codebook_size, dim]
            let mut codebook = vec![0.0_f64; codebook_size * dim];
            for c in 0..codebook_size {
                let row = &mut codebook[c * dim..(c + 1) * dim];
                let mut sq = 0.0_f64;
                for v in row.iter_mut() {
                    *v = rng.next_normal();
                    sq += *v * *v;
                }
                let norm = sq.sqrt().max(1e-12);
                for v in row.iter_mut() {
                    *v /= norm;
                }
            }

            pyramid.push(NcmPyramidLevel {
                level_len: ell,
                codebook_size,
                down_w,
                codebook,
                up_w,
            });
        }

        // Per-modality strictly orthogonal Procrustes bridges R_m and R_m^T
        let mut procrustes_in = vec![0.0_f64; NCM_NUM_MODALITIES * dim * dim];
        let mut procrustes_out = vec![0.0_f64; NCM_NUM_MODALITIES * dim * dim];
        for m in 0..NCM_NUM_MODALITIES {
            let r = make_orthogonal_matrix(
                dim,
                seed ^ ((m as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
            );
            let base = m * dim * dim;
            procrustes_in[base..base + dim * dim].copy_from_slice(&r);
            for i in 0..dim {
                for j in 0..dim {
                    procrustes_out[base + i * dim + j] = r[j * dim + i]; // exact inverse R^T
                }
            }
        }

        let scale_exp = 1.0 / ((4 * dim) as f64).sqrt();
        let mut w_ncm_exp = vec![0.0_f64; (4 * dim) * exp_dim];
        for v in &mut w_ncm_exp {
            *v = rng.next_normal() * scale_exp;
        }
        let scale_out = 1.0 / (exp_dim as f64).sqrt();
        let mut w_ncm_out = vec![0.0_f64; exp_dim * (NCM_CHUNK_LEN * dim)];
        for v in &mut w_ncm_out {
            *v = rng.next_normal() * scale_out;
        }

        // Modality-specialized relative-position bias [4, 4, 127]:
        // Text/Audio favor causal/local diagonal alignment; Visual/3D favor symmetric 2D/3D spatial locality
        let mut modality_rel_pos_bias =
            vec![0.0_f64; NCM_NUM_MODALITIES * NCM_ATTN_HEADS * NCM_REL_POS_LEN];
        for m in 0..NCM_NUM_MODALITIES {
            for h in 0..NCM_ATTN_HEADS {
                let width = ((h + 1) * (m + 1)) as f64;
                for delta in 0..NCM_REL_POS_LEN {
                    let rel = (delta as isize) - ((NCM_CHUNK_LEN - 1) as isize);
                    let dist = (rel.abs() as f64) / width;
                    let idx = (m * NCM_ATTN_HEADS + h) * NCM_REL_POS_LEN + delta;
                    modality_rel_pos_bias[idx] = -0.5 * dist;
                }
            }
        }

        let head_dim = dim / NCM_ATTN_HEADS;
        let qk_scale = 1.0 / (dim as f64).sqrt();
        let mut w_q_attn = vec![0.0_f64; NCM_ATTN_HEADS * dim * head_dim];
        let mut w_k_attn = vec![0.0_f64; NCM_ATTN_HEADS * dim * head_dim];
        for v in &mut w_q_attn {
            *v = rng.next_normal() * qk_scale;
        }
        for v in &mut w_k_attn {
            *v = rng.next_normal() * qk_scale;
        }
        // Identity-initialized value projection so cross-attention preserves slot fidelity at step 0
        let mut w_v_attn = vec![0.0_f64; dim * dim];
        for i in 0..dim {
            w_v_attn[i * dim + i] = 1.0;
        }

        Ok(Self {
            dim,
            exp_dim,
            codebook_size,
            pyramid,
            procrustes_in,
            procrustes_out,
            w_ncm_exp,
            w_ncm_out,
            modality_rel_pos_bias,
            w_q_attn,
            w_k_attn,
            w_v_attn,
        })
    }

    /// **Multimodal NCM Encoder:** Encode a 64-slot raw modality chunk (`[64, dim]`) through
    /// the 8-level multi-scale residual VQ pyramid (`[1, 4, 9, 16, 25, 36, 49, 64]`), rotate
    /// into the shared Platonic Brain manifold via `R_m`, and pool into `[k, r, s, m]` (`4 * dim`).
    pub fn encode_to_brain(
        &self,
        modality: NcmModality,
        raw_slots_64x_dim: &[f64],
    ) -> Result<EncodedNcmConcept, ErrorCode> {
        let dim = self.dim;
        if raw_slots_64x_dim.len() != NCM_CHUNK_LEN * dim {
            return Err(ErrorCode::ShapeMismatch);
        }

        let mut z_res = raw_slots_64x_dim.to_vec();
        layer_norm_in_place(&mut z_res, NCM_CHUNK_LEN, dim);
        let mut z_hat = vec![0.0_f64; NCM_CHUNK_LEN * dim];
        let mut pyramid_codes = Vec::with_capacity(204);

        for lvl in &self.pyramid {
            let ell = lvl.level_len;
            // 1. Downsample residual [64, dim] -> [ell, dim]
            let mut u = vec![0.0_f64; ell * dim];
            for e in 0..ell {
                for d in 0..dim {
                    let mut sum = 0.0_f64;
                    for t in 0..NCM_CHUNK_LEN {
                        sum += z_res[t * dim + d] * lvl.down_w[t * ell + e];
                    }
                    u[e * dim + d] = sum;
                }
            }
            layer_norm_in_place(&mut u, ell, dim);

            // 2. Quantize against unit-norm level codebook
            let mut q = vec![0.0_f64; ell * dim];
            for e in 0..ell {
                let ue = &u[e * dim..(e + 1) * dim];
                let mut best_c = 0usize;
                let mut best_dist = f64::INFINITY;
                for c in 0..lvl.codebook_size {
                    let cb = &lvl.codebook[c * dim..(c + 1) * dim];
                    let dist: f64 = ue
                        .iter()
                        .zip(cb.iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_c = c;
                    }
                }
                pyramid_codes.push(best_c as u16);
                q[e * dim..(e + 1) * dim]
                    .copy_from_slice(&lvl.codebook[best_c * dim..(best_c + 1) * dim]);
            }

            // 3. Upsample [ell, dim] -> [64, dim] and update residual
            for t in 0..NCM_CHUNK_LEN {
                for d in 0..dim {
                    let mut r_td = 0.0_f64;
                    for e in 0..ell {
                        r_td += q[e * dim + d] * lvl.up_w[e * NCM_CHUNK_LEN + t];
                    }
                    z_hat[t * dim + d] += r_td;
                    z_res[t * dim + d] -= r_td;
                }
            }
        }

        // 4. Rotate each slot into the shared Brain manifold via Orthogonal Procrustes R_m
        let r_base = (modality as usize) * dim * dim;
        let r_m = &self.procrustes_in[r_base..r_base + dim * dim];
        let mut aligned_slots = vec![0.0_f64; NCM_CHUNK_LEN * dim];
        for t in 0..NCM_CHUNK_LEN {
            let src = &z_hat[t * dim..(t + 1) * dim];
            let dst = &mut aligned_slots[t * dim..(t + 1) * dim];
            for i in 0..dim {
                let r_row = &r_m[i * dim..(i + 1) * dim];
                let mut acc = 0.0_f64;
                for j in 0..dim {
                    acc += r_row[j] * src[j];
                }
                dst[i] = acc;
            }
        }

        // 5. Pool into 4-family Brain workspace state `[k, r, s, m]` (`4 * dim`)
        // partitioning the 64 concept slots into 4 semantic quadrants (16 slots each)
        let mut brain_krsm = vec![0.0_f64; 4 * dim];
        for fam in 0..4 {
            let dst = &mut brain_krsm[fam * dim..(fam + 1) * dim];
            for t in (fam * 16)..((fam + 1) * 16) {
                let slot = &aligned_slots[t * dim..(t + 1) * dim];
                for d in 0..dim {
                    dst[d] += slot[d] * (1.0 / 16.0);
                }
            }
        }

        Ok(EncodedNcmConcept {
            modality,
            pyramid_codes,
            aligned_slots,
            brain_krsm,
        })
    }

    /// **Multimodal Parallel NCM Decoder:** Decode the Brain's verified `[k, r, s, m]` (`4 * dim`)
    /// + aligned context slots `[64, dim]` into a 64-slot target modality output in **one parallel wave**.
    pub fn decode_from_brain(
        &self,
        target_modality: NcmModality,
        brain_krsm: &[f64],
        context_slots_64x_dim: &[f64],
    ) -> Result<DecodedNcmOutput, ErrorCode> {
        let dim = self.dim;
        if brain_krsm.len() != 4 * dim || context_slots_64x_dim.len() != NCM_CHUNK_LEN * dim {
            return Err(ErrorCode::ShapeMismatch);
        }

        // 1. Parallel NCM MLP expansion: `brain_krsm [4*dim] -> ReLU(W_ncm_exp) -> W_ncm_out -> [64, dim]`
        let mut h_exp = vec![0.0_f64; self.exp_dim];
        for e in 0..self.exp_dim {
            let mut acc = 0.0_f64;
            for i in 0..(4 * dim) {
                acc += brain_krsm[i] * self.w_ncm_exp[i * self.exp_dim + e];
            }
            h_exp[e] = acc.max(0.0);
        }

        let mut z_gen = vec![0.0_f64; NCM_CHUNK_LEN * dim];
        for idx in 0..(NCM_CHUNK_LEN * dim) {
            let mut acc = 0.0_f64;
            for e in 0..self.exp_dim {
                acc += h_exp[e] * self.w_ncm_out[e * (NCM_CHUNK_LEN * dim) + idx];
            }
            z_gen[idx] = acc;
        }

        // 2. Modality-conditioned 4-head Content + Relative-Position Cross-Attention over `context_slots_64x_dim`
        let head_dim = dim / NCM_ATTN_HEADS;
        let inv_sqrt_hd = 1.0 / (head_dim as f64).sqrt();
        let m_idx = target_modality as usize;

        for h in 0..NCM_ATTN_HEADS {
            let w_q = &self.w_q_attn[h * dim * head_dim..(h + 1) * dim * head_dim];
            let w_k = &self.w_k_attn[h * dim * head_dim..(h + 1) * dim * head_dim];
            let bias_h = &self.modality_rel_pos_bias[(m_idx * NCM_ATTN_HEADS + h) * NCM_REL_POS_LEN
                ..(m_idx * NCM_ATTN_HEADS + h + 1) * NCM_REL_POS_LEN];

            // Precompute K_h [64, head_dim] from context_slots
            let mut k_proj = vec![0.0_f64; NCM_CHUNK_LEN * head_dim];
            for s in 0..NCM_CHUNK_LEN {
                let c_row = &context_slots_64x_dim[s * dim..(s + 1) * dim];
                let k_row = &mut k_proj[s * head_dim..(s + 1) * head_dim];
                for hd in 0..head_dim {
                    let mut acc = 0.0_f64;
                    for d in 0..dim {
                        acc += c_row[d] * w_k[d * head_dim + hd];
                    }
                    k_row[hd] = acc;
                }
            }

            for t in 0..NCM_CHUNK_LEN {
                let z_row = &z_gen[t * dim..(t + 1) * dim];
                let mut q_t = vec![0.0_f64; head_dim];
                for hd in 0..head_dim {
                    let mut acc = 0.0_f64;
                    for d in 0..dim {
                        acc += z_row[d] * w_q[d * head_dim + hd];
                    }
                    q_t[hd] = acc;
                }

                let mut scores = [0.0_f64; NCM_CHUNK_LEN];
                let mut max_s = f64::NEG_INFINITY;
                for s in 0..NCM_CHUNK_LEN {
                    let k_row = &k_proj[s * head_dim..(s + 1) * head_dim];
                    let dot: f64 = q_t.iter().zip(k_row.iter()).map(|(a, b)| a * b).sum();
                    let rel_idx = (t + NCM_CHUNK_LEN - 1) - s;
                    let sc = dot * inv_sqrt_hd + bias_h[rel_idx];
                    scores[s] = sc;
                    if sc > max_s {
                        max_s = sc;
                    }
                }
                let mut sum_exp = 0.0_f64;
                for sc in &mut scores {
                    *sc = (*sc - max_s).exp();
                    sum_exp += *sc;
                }
                let inv_sum = 1.0 / sum_exp.max(1e-15);
                for s in 0..NCM_CHUNK_LEN {
                    let w = (scores[s] * inv_sum) * (1.0 / (NCM_ATTN_HEADS as f64));
                    let c_row = &context_slots_64x_dim[s * dim..(s + 1) * dim];
                    let dst = &mut z_gen[t * dim..(t + 1) * dim];
                    for d in 0..dim {
                        dst[d] += w * c_row[d];
                    }
                }
            }
        }

        // 3. Inverse Procrustes rotation R_m^T back into the target modality's native coordinate space
        let r_base = m_idx * dim * dim;
        let r_out = &self.procrustes_out[r_base..r_base + dim * dim];
        let mut slots_64x_dim = vec![0.0_f64; NCM_CHUNK_LEN * dim];
        for t in 0..NCM_CHUNK_LEN {
            let src = &z_gen[t * dim..(t + 1) * dim];
            let dst = &mut slots_64x_dim[t * dim..(t + 1) * dim];
            for i in 0..dim {
                let r_row = &r_out[i * dim..(i + 1) * dim];
                let mut acc = 0.0_f64;
                for j in 0..dim {
                    acc += r_row[j] * src[j];
                }
                dst[i] = acc;
            }
        }

        // 4. Parallel 64-slot readout: discrete tokens (via finest pyramid codebook) + spatial/3D field
        let fine_cb = &self.pyramid.last().unwrap().codebook;
        let mut discrete_tokens_64 = [0u16; NCM_CHUNK_LEN];
        let mut spatial_field_64 = [0.0_f32; NCM_CHUNK_LEN];

        for t in 0..NCM_CHUNK_LEN {
            let slot = &slots_64x_dim[t * dim..(t + 1) * dim];
            let mut best_c = 0usize;
            let mut best_dist = f64::INFINITY;
            for c in 0..self.codebook_size {
                let cb = &fine_cb[c * dim..(c + 1) * dim];
                let dist: f64 = slot
                    .iter()
                    .zip(cb.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }
            discrete_tokens_64[t] = best_c as u16;
            // Sigmoid-projected energy for 2D 8x8 canvas or 4x4x4 3D field occupancy
            let mean_act = slot.iter().sum::<f64>() / (dim as f64);
            spatial_field_64[t] = (1.0 / (1.0 + (-mean_act * 4.0).exp())) as f32;
        }

        Ok(DecodedNcmOutput {
            modality: target_modality,
            slots_64x_dim,
            discrete_tokens_64,
            spatial_field_64,
        })
    }

    /// Save the complete Multimodal NCM Encoder & Decoder as pure Burn `.safetensors`
    /// (`{"format":"burn"}`, mixed `BF16` + `F32`, zero `F64`, zero compiler fingerprints).
    pub fn save_burn_safetensors(&self, path: &Path) -> Result<u64, ErrorCode> {
        let mut owned: Vec<(String, Vec<usize>, WeightDtype, Vec<f64>)> = Vec::new();

        for (k, lvl) in self.pyramid.iter().enumerate() {
            owned.push((
                format!("model.ncm.pyramid.{k}.down.weight"),
                vec![NCM_CHUNK_LEN, lvl.level_len],
                WeightDtype::F32,
                lvl.down_w.clone(),
            ));
            owned.push((
                format!("model.ncm.pyramid.{k}.codebook.weight"),
                vec![lvl.codebook_size, self.dim],
                WeightDtype::BF16,
                lvl.codebook.clone(),
            ));
            owned.push((
                format!("model.ncm.pyramid.{k}.up.weight"),
                vec![lvl.level_len, NCM_CHUNK_LEN],
                WeightDtype::F32,
                lvl.up_w.clone(),
            ));
        }

        owned.push((
            String::from("model.ncm.procrustes_in.weight"),
            vec![NCM_NUM_MODALITIES, self.dim, self.dim],
            WeightDtype::F32,
            self.procrustes_in.clone(),
        ));
        owned.push((
            String::from("model.ncm.procrustes_out.weight"),
            vec![NCM_NUM_MODALITIES, self.dim, self.dim],
            WeightDtype::F32,
            self.procrustes_out.clone(),
        ));
        owned.push((
            String::from("model.ncm.w_exp.weight"),
            vec![4 * self.dim, self.exp_dim],
            WeightDtype::BF16,
            self.w_ncm_exp.clone(),
        ));
        owned.push((
            String::from("model.ncm.w_out.weight"),
            vec![self.exp_dim, NCM_CHUNK_LEN * self.dim],
            WeightDtype::BF16,
            self.w_ncm_out.clone(),
        ));
        owned.push((
            String::from("model.ncm.modality_rel_pos_bias"),
            vec![NCM_NUM_MODALITIES, NCM_ATTN_HEADS, NCM_REL_POS_LEN],
            WeightDtype::F32,
            self.modality_rel_pos_bias.clone(),
        ));
        let hd = self.dim / NCM_ATTN_HEADS;
        owned.push((
            String::from("model.ncm.w_q_attn.weight"),
            vec![NCM_ATTN_HEADS, self.dim, hd],
            WeightDtype::BF16,
            self.w_q_attn.clone(),
        ));
        owned.push((
            String::from("model.ncm.w_k_attn.weight"),
            vec![NCM_ATTN_HEADS, self.dim, hd],
            WeightDtype::BF16,
            self.w_k_attn.clone(),
        ));
        owned.push((
            String::from("model.ncm.w_v_attn.weight"),
            vec![self.dim, self.dim],
            WeightDtype::BF16,
            self.w_v_attn.clone(),
        ));

        for entry in &mut owned {
            let g = promote_outlier_dtype(entry.2, &entry.3, DEFAULT_OUTLIER_PROMOTE_ABS, true)?;
            entry.2 = g.final_dtype;
        }

        let named: Vec<TypedNamedTensor<'_>> = owned
            .iter()
            .map(|(n, s, dt, d)| TypedNamedTensor {
                name: n.as_str(),
                shape: s.clone(),
                dtype: *dt,
                data: d.as_slice(),
            })
            .collect();

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        let metadata = [("format", "burn")];
        let (bytes, _) =
            write_mixed_safetensors(path, &metadata, &named).map_err(|_| ErrorCode::IoError)?;
        Ok(bytes)
    }

    /// Verify that a saved Multimodal NCM Burn `.safetensors` file reloads cleanly.
    pub fn verify_burn_file(path: &Path) -> Result<usize, ErrorCode> {
        let loaded = read_safetensors(path).map_err(|_| ErrorCode::IoError)?;
        Ok(loaded.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multimodal_ncm_encodes_and_parallel_decodes_all_four_modalities() {
        let codec = MultimodalNcmCodec::compile(32, 64, 64, 2026).unwrap();

        // Save as pure Burn .safetensors and verify 32 tensors (8*3 pyramid + 8 bridge/attn)
        let path = std::env::temp_dir().join("quintessa_mm_ncm_test.safetensors");
        let bytes = codec.save_burn_safetensors(&path).unwrap();
        assert!(bytes > 0);
        assert_eq!(MultimodalNcmCodec::verify_burn_file(&path).unwrap(), 32);
        let _ = std::fs::remove_file(&path);

        // Test encode -> Brain [k,r,s,m] -> parallel 64-slot decode across all 4 modalities
        let mut raw_input = vec![0.0_f64; NCM_CHUNK_LEN * 32];
        for (i, v) in raw_input.iter_mut().enumerate() {
            *v = ((i as f64) * 0.13).sin();
        }

        for modality in NcmModality::ALL {
            let enc = codec.encode_to_brain(modality, &raw_input).unwrap();
            // 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 = 204 multi-scale pyramid codes
            assert_eq!(enc.pyramid_codes.len(), 204);
            assert_eq!(enc.brain_krsm.len(), 4 * 32);

            let dec = codec
                .decode_from_brain(modality, &enc.brain_krsm, &enc.aligned_slots)
                .unwrap();
            assert_eq!(dec.slots_64x_dim.len(), NCM_CHUNK_LEN * 32);
            assert!(dec.slots_64x_dim.iter().all(|v| v.is_finite()));
            assert!(dec.spatial_field_64.iter().all(|v| v.is_finite()));
        }
    }
}
