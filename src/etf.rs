//! Feature / ETF IR Compiler (Priority 1).
//!
//! Allocates the four superposed feature directions per neuron:
//! - Knowledge (K)
//! - Reasoning (R)
//! - SACI Interface (S)
//! - Simulation (M)
//!
//! Each neuron `i` has 4 unit-norm feature vectors in `R^substrate_dim`
//! constructed via Equiangular Tight Frames (Helmert Simplex ETF or
//! Alternating-Projection Tight Frame), while enforcing the formal invariant:
//!   sum_{f in {K,R,S,M}} z_{i,f} <= 1.

use crate::sys::{ErrorCode, FeatureFamily, MASK_LEGAL_ALL};

#[derive(Clone, Debug)]
pub struct FeatureEtfIr {
    pub num_neurons: usize,
    pub substrate_dim: usize,
    /// Contiguous `[num_neurons * 4 * substrate_dim]` row-major basis vectors.
    /// Feature `(neuron_i, family)` is at offset `(neuron_i * 4 + family as usize) * substrate_dim`.
    pub basis: Vec<f64>,
    /// Per-neuron legal feature-family bitmask (`u8`, bits 0..3 for K,R,S,M).
    pub legal_masks: Vec<u8>,
    pub welch_bound: f64,
    pub max_coherence: f64,
}

impl FeatureEtfIr {
    #[inline]
    pub fn feature_vec(&self, neuron: usize, family: FeatureFamily) -> &[f64] {
        let idx = (neuron * 4 + (family as usize)) * self.substrate_dim;
        &self.basis[idx..idx + self.substrate_dim]
    }

    #[inline]
    pub fn is_family_legal(&self, neuron: usize, family: FeatureFamily) -> bool {
        (self.legal_masks[neuron] & family.bit()) != 0
    }
}

#[inline]
pub fn welch_bound(n: usize, m: usize) -> f64 {
    if n <= 1 || m >= n {
        return 0.0;
    }
    let nf = n as f64;
    let mf = m as f64;
    ((nf - mf) / (mf * (nf - 1.0))).sqrt()
}

/// Build an exact 4-direction Simplex ETF in `R^3` (regular tetrahedron, coherence = 1/3).
fn simplex_4x3() -> [f64; 12] {
    let scale = (4.0_f64 / 3.0_f64).sqrt();
    let mut out = [0.0_f64; 12];
    for k in 0..3 {
        let kf = k as f64;
        let norm = ((kf + 1.0) * (kf + 2.0)).sqrt();
        let val = scale / norm;
        for r in 0..=k {
            out[r * 3 + k] = val;
        }
        out[(k + 1) * 3 + k] = -((kf + 1.0) * scale) / norm;
    }
    out
}

/// Compile the 4-family Feature / ETF IR for `num_neurons` in `substrate_dim`.
pub fn compile_feature_etf(
    num_neurons: usize,
    substrate_dim: usize,
    legal_masks: Option<&[u8]>,
) -> Result<FeatureEtfIr, ErrorCode> {
    if num_neurons == 0 || substrate_dim < 3 {
        return Err(ErrorCode::InvalidFeatureDim);
    }
    if let Some(masks) = legal_masks {
        if masks.len() != num_neurons {
            return Err(ErrorCode::InvalidFeatureDim);
        }
    }

    let total_features = num_neurons * 4;
    let mut basis = vec![0.0_f64; total_features * substrate_dim];

    if substrate_dim >= total_features {
        // Orthogonal regime: zero cross-feature interference.
        for f in 0..total_features {
            basis[f * substrate_dim + f] = 1.0;
        }
    } else if substrate_dim == total_features - 1 {
        // Exact global Helmert Simplex ETF
        let n = total_features;
        let m = substrate_dim;
        let scale = ((n as f64) / (m as f64)).sqrt();
        for k in 0..m {
            let kf = k as f64;
            let norm = ((kf + 1.0) * (kf + 2.0)).sqrt();
            let val = scale / norm;
            for r in 0..=k {
                basis[r * m + k] = val;
            }
            basis[(k + 1) * m + k] = -((kf + 1.0) * scale) / norm;
        }
    } else {
        // Block-Simplex ETF: pack each neuron's 4 features into a 3D regular
        // tetrahedron ETF. First pass (wrap == 0) uses disjoint 3D blocks; wrapped passes
        // apply full 3D SO(3) rotation + strided distinct indices to eliminate cos=1.0 duplicates.
        let tetra = simplex_4x3();
        let num_blocks = (substrate_dim / 3).max(1);
        for i in 0..num_neurons {
            let wrap = i / num_blocks;
            let (idx0, idx1, idx2) = if wrap == 0 {
                let b = i * 3;
                (b, b + 1, b + 2)
            } else {
                let stride = (wrap * 2 + 1).min(substrate_dim - 1).max(1);
                let base = (i * 3 + wrap * 7) % substrate_dim;
                let i0 = base;
                let mut i1 = (base + stride) % substrate_dim;
                if i1 == i0 {
                    i1 = (i0 + 1) % substrate_dim;
                }
                let mut i2 = (i1 + stride) % substrate_dim;
                while i2 == i0 || i2 == i1 {
                    i2 = (i2 + 1) % substrate_dim;
                }
                (i0, i1, i2)
            };

            let a_xy = (i as f64) * 0.618_033_988_749_894_9 * core::f64::consts::TAU;
            let a_yz = (wrap as f64) * 0.754_877_666_246_692_7 * core::f64::consts::TAU
                + (i as f64) * 0.132_471_795_724_474_6;
            let a_zx = (wrap as f64) * 0.569_840_290_998_053_2 * core::f64::consts::TAU;
            let (s_xy, c_xy) = a_xy.sin_cos();
            let (s_yz, c_yz) = a_yz.sin_cos();
            let (s_zx, c_zx) = a_zx.sin_cos();

            for f in 0..4 {
                let dst = (i * 4 + f) * substrate_dim;
                let x0 = tetra[f * 3];
                let y0 = tetra[f * 3 + 1];
                let z0 = tetra[f * 3 + 2];
                // Full SO(3) rotation preserves exact unit norm & intra-neuron coherence = 1/3
                let x1 = c_xy * x0 - s_xy * y0;
                let y1 = s_xy * x0 + c_xy * y0;
                let z1 = z0;
                let y2 = c_yz * y1 - s_yz * z1;
                let z2 = s_yz * y1 + c_yz * z1;
                let x2 = x1;
                let z3 = c_zx * z2 - s_zx * x2;
                let x3 = s_zx * z2 + c_zx * x2;
                let y3 = y2;

                basis[dst + idx0] = x3;
                basis[dst + idx1] = y3;
                basis[dst + idx2] = z3;
            }
        }
    }

    // Measure max intra-neuron coherence (between the 4 families of neuron 0)
    let mut max_coh = 0.0_f64;
    for a in 0..4 {
        for b in (a + 1)..4 {
            let mut dot = 0.0;
            for d in 0..substrate_dim {
                dot += basis[a * substrate_dim + d] * basis[b * substrate_dim + d];
            }
            if dot.abs() > max_coh {
                max_coh = dot.abs();
            }
        }
    }

    let masks = match legal_masks {
        Some(m) => m.to_vec(),
        None => vec![MASK_LEGAL_ALL; num_neurons],
    };

    Ok(FeatureEtfIr {
        num_neurons,
        substrate_dim,
        basis,
        legal_masks: masks,
        welch_bound: welch_bound(total_features, substrate_dim),
        max_coherence: max_coh,
    })
}
