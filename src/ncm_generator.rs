//! Parallel Next-Concept Model (NCM) Generator & `ada_vq` Chunk Decoder.
//!
//! Replaces autoregressive next-token prediction with true non-autoregressive NCM:
//! `Verified Workspace State [k, r, s, m] -> W_ncm_chunk -> z_hat [L x D] -> ada_vq Decoder -> L tokens in parallel`.

use crate::bpk_loader::BpkArchive;
use crate::sys::ErrorCode;

pub const NCM_CHUNK_LEN: usize = 64;
const VQ_NUM_HEADS: usize = 4;
const VQ_LEVEL_LENS: [usize; 8] = [1, 4, 9, 16, 25, 36, 49, 64];

#[inline]
fn gelu(x: f64) -> f64 {
    const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

fn layer_norm(x: &[f64], gamma: &[f64], beta: &[f64], seq_len: usize, dim: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; seq_len * dim];
    let inv_d = 1.0 / (dim as f64);
    for t in 0..seq_len {
        let xi = &x[t * dim..(t + 1) * dim];
        let yi = &mut out[t * dim..(t + 1) * dim];
        let mean = xi.iter().sum::<f64>() * inv_d;
        let var = xi
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f64>()
            * inv_d;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        for d in 0..dim {
            yi[d] = gamma[d] * (xi[d] - mean) * inv_std + beta[d];
        }
    }
    out
}

fn param_free_layer_norm(x: &mut [f64], seq_len: usize, dim: usize) {
    let inv_d = 1.0 / (dim as f64);
    for t in 0..seq_len {
        let xi = &mut x[t * dim..(t + 1) * dim];
        let mean = xi.iter().sum::<f64>() * inv_d;
        let var = xi
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f64>()
            * inv_d;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        for v in xi.iter_mut() {
            *v = (*v - mean) * inv_std;
        }
    }
}

#[derive(Clone, Debug)]
pub struct VqEncoderLevel {
    pub level_len: usize,
    pub down_w: Vec<f64>,   // [seq_len, level_len]
    pub down_b: Vec<f64>,   // [level_len]
    pub codebook: Vec<f64>, // [codebook_size, dim]
    pub codebook_size: usize,
    pub up_w: Vec<f64>,   // [level_len, seq_len]
    pub up_b: Vec<f64>,   // [seq_len]
    pub proj_w: Vec<f64>, // [dim, dim]
    pub proj_b: Vec<f64>, // [dim]
}

#[derive(Clone, Debug)]
pub struct VqEncoderWeights {
    pub dim: usize,
    pub ln_enc_gamma: Vec<f64>,
    pub ln_enc_beta: Vec<f64>,
    pub levels: Vec<VqEncoderLevel>,
}

impl VqEncoderWeights {
    pub fn from_bpk(bpk: &BpkArchive, dim: usize) -> Result<Self, ErrorCode> {
        let (_, ln_enc_gamma) = bpk.read_f32_as_f64("ln_enc.gamma")?;
        let (_, ln_enc_beta) = bpk.read_f32_as_f64("ln_enc.beta")?;
        let mut levels = Vec::with_capacity(VQ_LEVEL_LENS.len());
        for (k, &level_len) in VQ_LEVEL_LENS.iter().enumerate() {
            let (_, down_w) = bpk.read_f32_as_f64(&format!("downsamplers.{k}.weight"))?;
            let (_, down_b) = bpk.read_f32_as_f64(&format!("downsamplers.{k}.bias"))?;
            let (cb_shape, codebook) = bpk.read_f32_as_f64(&format!("codebooks.{k}.weight"))?;
            let (_, up_w) = bpk.read_f32_as_f64(&format!("upsamplers.{k}.weight"))?;
            let (_, up_b) = bpk.read_f32_as_f64(&format!("upsamplers.{k}.bias"))?;
            let (_, proj_w) = bpk.read_f32_as_f64(&format!("res_proj.{k}.weight"))?;
            let (_, proj_b) = bpk.read_f32_as_f64(&format!("res_proj.{k}.bias"))?;
            if cb_shape.len() != 2 || cb_shape[1] != dim {
                return Err(ErrorCode::ShapeMismatch);
            }
            levels.push(VqEncoderLevel {
                level_len,
                down_w,
                down_b,
                codebook,
                codebook_size: cb_shape[0],
                up_w,
                up_b,
                proj_w,
                proj_b,
            });
        }
        Ok(Self {
            dim,
            ln_enc_gamma,
            ln_enc_beta,
            levels,
        })
    }

    /// Encode `tokens` (`seq_len <= NCM_CHUNK_LEN`) into `ada_vq`'s `z_hat` `[NCM_CHUNK_LEN, dim]`.
    pub fn encode_chunk(&self, tokens: &[u16], tok_emb: &[f64], vocab_size: usize) -> Vec<f64> {
        let seq_len = tokens.len();
        let dim = self.dim;
        let mut x_emb = vec![0.0_f64; seq_len * dim];
        for (t, &tok) in tokens.iter().enumerate() {
            let tid = tok as usize;
            if tid < vocab_size {
                x_emb[t * dim..(t + 1) * dim].copy_from_slice(&tok_emb[tid * dim..(tid + 1) * dim]);
            }
        }
        let z_init = layer_norm(&x_emb, &self.ln_enc_gamma, &self.ln_enc_beta, seq_len, dim);
        let mut z_hat = vec![0.0_f64; seq_len * dim];
        let mut z_res = z_init;

        for lvl in &self.levels {
            let ell = lvl.level_len;
            // u = (z_res^T @ down_w + down_b)^T -> [ell, dim]
            let mut u = vec![0.0_f64; ell * dim];
            for e in 0..ell {
                let b = lvl.down_b[e];
                for d in 0..dim {
                    let mut sum = b;
                    for t in 0..seq_len {
                        sum += z_res[t * dim + d] * lvl.down_w[t * ell + e];
                    }
                    u[e * dim + d] = sum;
                }
            }
            param_free_layer_norm(&mut u, ell, dim);

            // Nearest-neighbor codebook lookup per position e in 0..ell
            let mut q = vec![0.0_f64; ell * dim];
            for e in 0..ell {
                let ue = &u[e * dim..(e + 1) * dim];
                let mut best_idx = 0usize;
                let mut best_dist = f64::INFINITY;
                for c in 0..lvl.codebook_size {
                    let cb = &lvl.codebook[c * dim..(c + 1) * dim];
                    let dist: f64 = ue
                        .iter()
                        .zip(cb.iter())
                        .map(|(&a, &b)| {
                            let d = a - b;
                            d * d
                        })
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = c;
                    }
                }
                q[e * dim..(e + 1) * dim]
                    .copy_from_slice(&lvl.codebook[best_idx * dim..(best_idx + 1) * dim]);
            }

            // r = (q^T @ up_w + up_b)^T -> [seq_len, dim]
            let mut r = vec![0.0_f64; seq_len * dim];
            for t in 0..seq_len {
                let b = lvl.up_b[t];
                for d in 0..dim {
                    let mut sum = b;
                    for e in 0..ell {
                        sum += q[e * dim + d] * lvl.up_w[e * seq_len + t];
                    }
                    r[t * dim + d] = sum;
                }
            }

            // delta_z = r @ proj_w + proj_b -> [seq_len, dim]
            for t in 0..seq_len {
                let rt = &r[t * dim..(t + 1) * dim];
                for j in 0..dim {
                    let mut sum = lvl.proj_b[j];
                    for (i, &rv) in rt.iter().enumerate() {
                        sum += rv * lvl.proj_w[i * dim + j];
                    }
                    z_hat[t * dim + j] += sum;
                    z_res[t * dim + j] -= sum;
                }
            }
        }
        z_hat
    }
}

#[derive(Clone, Debug)]
pub struct VqDecoderWeights {
    pub dim: usize,
    pub vocab_size: usize,
    pub ln_dec_gamma: Vec<f64>,
    pub ln_dec_beta: Vec<f64>,
    pub tok_emb: Vec<f64>,   // [vocab_size, dim]
    pub head_bias: Vec<f64>, // [vocab_size]
    pub dec1: TransformerDecBlock,
    pub dec2: TransformerDecBlock,
}

#[derive(Clone, Debug)]
pub struct TransformerDecBlock {
    pub ln1_gamma: Vec<f64>,
    pub ln1_beta: Vec<f64>,
    pub wq: Vec<f64>,
    pub bq: Vec<f64>,
    pub wk: Vec<f64>,
    pub bk: Vec<f64>,
    pub wv: Vec<f64>,
    pub bv: Vec<f64>,
    pub wo: Vec<f64>,
    pub bo: Vec<f64>,
    pub ln2_gamma: Vec<f64>,
    pub ln2_beta: Vec<f64>,
    pub ff1_w: Vec<f64>,
    pub ff1_b: Vec<f64>,
    pub ff2_w: Vec<f64>,
    pub ff2_b: Vec<f64>,
    pub ff_dim: usize,
}

impl TransformerDecBlock {
    pub fn from_bpk(bpk: &BpkArchive, prefix: &str, dim: usize) -> Result<Self, ErrorCode> {
        let (_, ln1_gamma) = bpk.read_f32_as_f64(&format!("{prefix}.ln1.gamma"))?;
        let (_, ln1_beta) = bpk.read_f32_as_f64(&format!("{prefix}.ln1.beta"))?;
        let (_, wq) = bpk.read_f32_as_f64(&format!("{prefix}.wq.weight"))?;
        let (_, bq) = bpk.read_f32_as_f64(&format!("{prefix}.wq.bias"))?;
        let (_, wk) = bpk.read_f32_as_f64(&format!("{prefix}.wk.weight"))?;
        let (_, bk) = bpk.read_f32_as_f64(&format!("{prefix}.wk.bias"))?;
        let (_, wv) = bpk.read_f32_as_f64(&format!("{prefix}.wv.weight"))?;
        let (_, bv) = bpk.read_f32_as_f64(&format!("{prefix}.wv.bias"))?;
        let (_, wo) = bpk.read_f32_as_f64(&format!("{prefix}.wo.weight"))?;
        let (_, bo) = bpk.read_f32_as_f64(&format!("{prefix}.wo.bias"))?;
        let (_, ln2_gamma) = bpk.read_f32_as_f64(&format!("{prefix}.ln2.gamma"))?;
        let (_, ln2_beta) = bpk.read_f32_as_f64(&format!("{prefix}.ln2.beta"))?;
        let (ff1_shape, ff1_w) = bpk.read_f32_as_f64(&format!("{prefix}.ff1.weight"))?;
        let (_, ff1_b) = bpk.read_f32_as_f64(&format!("{prefix}.ff1.bias"))?;
        let (_, ff2_w) = bpk.read_f32_as_f64(&format!("{prefix}.ff2.weight"))?;
        let (_, ff2_b) = bpk.read_f32_as_f64(&format!("{prefix}.ff2.bias"))?;

        if ff1_shape.len() != 2 || ff1_shape[0] != dim {
            return Err(ErrorCode::ShapeMismatch);
        }
        Ok(Self {
            ln1_gamma,
            ln1_beta,
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            wo,
            bo,
            ln2_gamma,
            ln2_beta,
            ff1_w,
            ff1_b,
            ff2_w,
            ff2_b,
            ff_dim: ff1_shape[1],
        })
    }

    /// Parallel 4-head self-attention + GELU FFN with pre-LayerNorm over `[seq_len, dim]`.
    pub fn forward(&self, x: &[f64], seq_len: usize, dim: usize) -> Vec<f64> {
        let proj = |inp: &[f64], w: &[f64], b: &[f64], d_in: usize, d_out: usize| -> Vec<f64> {
            let mut out = vec![0.0_f64; seq_len * d_out];
            for t in 0..seq_len {
                let xi = &inp[t * d_in..(t + 1) * d_in];
                let yi = &mut out[t * d_out..(t + 1) * d_out];
                yi.copy_from_slice(b);
                for (i, &xv) in xi.iter().enumerate() {
                    let row = &w[i * d_out..(i + 1) * d_out];
                    for (j, yv) in yi.iter_mut().enumerate() {
                        *yv += xv * row[j];
                    }
                }
            }
            out
        };

        let xn1 = layer_norm(x, &self.ln1_gamma, &self.ln1_beta, seq_len, dim);
        let q = proj(&xn1, &self.wq, &self.bq, dim, dim);
        let k = proj(&xn1, &self.wk, &self.bk, dim, dim);
        let v = proj(&xn1, &self.wv, &self.bv, dim, dim);

        let dh = dim / VQ_NUM_HEADS;
        let inv_sqrt_dh = 1.0 / (dh as f64).sqrt();
        let mut attn_out = vec![0.0_f64; seq_len * dim];
        let mut scores = vec![0.0_f64; seq_len];

        for h in 0..VQ_NUM_HEADS {
            let hoff = h * dh;
            for t in 0..seq_len {
                let qt = &q[t * dim + hoff..t * dim + hoff + dh];
                let mut max_s = f64::NEG_INFINITY;
                for s in 0..seq_len {
                    let ks = &k[s * dim + hoff..s * dim + hoff + dh];
                    let dot: f64 = qt.iter().zip(ks.iter()).map(|(a, b)| a * b).sum();
                    let sc = dot * inv_sqrt_dh;
                    scores[s] = sc;
                    if sc > max_s {
                        max_s = sc;
                    }
                }
                let mut sum_exp = 0.0;
                for sc in &mut scores {
                    *sc = (*sc - max_s).exp();
                    sum_exp += *sc;
                }
                let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                let out_t = &mut attn_out[t * dim + hoff..t * dim + hoff + dh];
                for s in 0..seq_len {
                    let w = scores[s] * inv_sum;
                    let vs = &v[s * dim + hoff..s * dim + hoff + dh];
                    for d in 0..dh {
                        out_t[d] += w * vs[d];
                    }
                }
            }
        }

        let o_proj = proj(&attn_out, &self.wo, &self.bo, dim, dim);
        let mut res1: Vec<f64> = x.iter().zip(o_proj.iter()).map(|(a, b)| a + b).collect();

        let xn2 = layer_norm(&res1, &self.ln2_gamma, &self.ln2_beta, seq_len, dim);
        let mut ff1 = proj(&xn2, &self.ff1_w, &self.ff1_b, dim, self.ff_dim);
        for val in &mut ff1 {
            *val = gelu(*val);
        }
        let ff2 = proj(&ff1, &self.ff2_w, &self.ff2_b, self.ff_dim, dim);
        for (r, &f) in res1.iter_mut().zip(ff2.iter()) {
            *r += f;
        }
        res1
    }
}

impl VqDecoderWeights {
    pub fn from_bpk(bpk: &BpkArchive) -> Result<Self, ErrorCode> {
        let (_, ln_dec_gamma) = bpk.read_f32_as_f64("ln_dec.gamma")?;
        let (_, ln_dec_beta) = bpk.read_f32_as_f64("ln_dec.beta")?;
        let (emb_shape, tok_emb) = bpk.read_f32_as_f64("tok_emb.weight")?;
        if emb_shape.len() != 2 {
            return Err(ErrorCode::ShapeMismatch);
        }
        let vocab_size = emb_shape[0];
        let dim = emb_shape[1];
        let (_, head_bias) = bpk.read_f32_as_f64("head_bias")?;
        if head_bias.len() != vocab_size {
            return Err(ErrorCode::ShapeMismatch);
        }
        let dec1 = TransformerDecBlock::from_bpk(bpk, "dec1", dim)?;
        let dec2 = TransformerDecBlock::from_bpk(bpk, "dec2", dim)?;

        Ok(Self {
            dim,
            vocab_size,
            ln_dec_gamma,
            ln_dec_beta,
            tok_emb,
            head_bias,
            dec1,
            dec2,
        })
    }

    /// Decode a parallel `[seq_len, dim]` latent concept chunk `z_hat` into `seq_len` token IDs.
    pub fn decode_chunk_parallel(&self, z_hat: &[f64], seq_len: usize) -> Vec<u16> {
        let x_ln = layer_norm(
            z_hat,
            &self.ln_dec_gamma,
            &self.ln_dec_beta,
            seq_len,
            self.dim,
        );
        let h1 = self.dec1.forward(&x_ln, seq_len, self.dim);
        let h2 = self.dec2.forward(&h1, seq_len, self.dim);
        let inv_sqrt_d = 1.0 / (self.dim as f64).sqrt();

        let mut out_tokens = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let ht = &h2[t * self.dim..(t + 1) * self.dim];
            let mut best_id = 0u16;
            let mut best_logit = f64::NEG_INFINITY;
            for v in 0..self.vocab_size {
                let row = &self.tok_emb[v * self.dim..(v + 1) * self.dim];
                let dot: f64 = ht.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
                let logit = dot * inv_sqrt_d + self.head_bias[v];
                if logit > best_logit {
                    best_logit = logit;
                    best_id = v as u16;
                }
            }
            out_tokens.push(best_id);
        }
        out_tokens
    }
}

#[derive(Clone, Debug)]
pub struct ParallelNcmGenerator {
    pub ws_dim: usize,
    pub exp_dim: usize,
    pub dom_exp_dim: usize,
    pub num_domains: usize,
    pub seq_len: usize,
    pub dim: usize,
    /// Optional fixed non-linear expansion matrix `[ws_dim + num_domains, exp_dim]` (empty if linear).
    pub w_expand: Vec<f64>,
    /// Optional per-domain residual expansion matrix `[ws_dim, dom_exp_dim]`.
    pub w_dom_expand: Vec<f64>,
    /// Per-domain latent centroids `[num_domains, seq_len * dim]` (empty if uncentered).
    pub domain_means: Vec<f64>,
    /// WSP-written parallel concept-chunk map: `[in_dim, seq_len * dim]` where `in_dim = exp_dim.max(ws_dim)`.
    pub w_ncm_chunk: Vec<f64>,
    /// Per-domain isolated residual WSP heads `[num_domains, dom_exp_dim, seq_len * dim]`.
    pub w_ncm_domain: Vec<f64>,
    /// Per-domain cross-position copy/routing matrices `[num_domains, seq_len, seq_len]`.
    pub a_copy: Vec<f64>,
    /// Shared embedding-to-VQ-latent alignment `[dim, dim]`.
    pub w_val: Vec<f64>,
    /// Coarse-to-fine stage: `coarse_len` (e.g. 16) and projection `[exp_dim, coarse_len * dim]`.
    pub coarse_len: usize,
    pub w_coarse: Vec<f64>,
    /// Variable-length 4-head content + relative-position cross-attention projections:
    /// `w_q_attn: [num_domains, 4, dim, dh]` (or `[4, dim, dh]`),
    /// `w_k_heads: [4, dim, dh]`,
    /// `rel_pos_bias: [num_domains, 4, 127]` (or `[4, 127]`),
    /// `w_v_attn: [num_domains, dim, dim]` (or `[dim, dim]`).
    pub w_q_attn: Vec<f64>,
    pub w_k_heads: Vec<f64>,
    pub rel_pos_bias: Vec<f64>,
    pub w_v_attn: Vec<f64>,
}

fn invert_sym_pd(n: usize, a: &mut [f64]) -> Result<Vec<f64>, ErrorCode> {
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        let mut max_v = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > max_v {
                max_v = v;
                pivot = r;
            }
        }
        if max_v < 1e-12 {
            return Err(ErrorCode::SingularMatrix);
        }
        if pivot != col {
            for c in 0..n {
                a.swap(col * n + c, pivot * n + c);
                inv.swap(col * n + c, pivot * n + c);
            }
        }
        let diag = a[col * n + col];
        for c in 0..n {
            a[col * n + c] /= diag;
            inv[col * n + c] /= diag;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r * n + col];
            if f != 0.0 {
                for c in 0..n {
                    a[r * n + c] -= f * a[col * n + c];
                    inv[r * n + c] -= f * inv[col * n + c];
                }
            }
        }
    }
    Ok(inv)
}

fn solve_ridge_matrix(
    feats: &[Vec<f64>],
    targets: &[Vec<f64>],
    in_dim: usize,
    out_dim: usize,
    lambda: f64,
) -> Result<Vec<f64>, ErrorCode> {
    if feats.is_empty() || feats.len() != targets.len() || lambda <= 0.0 {
        return Err(ErrorCode::InvalidInput);
    }
    for (x, y) in feats.iter().zip(targets.iter()) {
        if x.len() != in_dim || y.len() != out_dim {
            return Err(ErrorCode::ShapeMismatch);
        }
    }

    let n_samples = feats.len();
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(n_samples.max(1));

    let (mut xtx, xty) = if n_threads > 1 && n_samples >= 64 {
        let chunk_sz = n_samples.div_ceil(n_threads);
        let partials: Vec<(Vec<f64>, Vec<f64>)> = std::thread::scope(|s| {
            let mut handles = Vec::new();
            for c_idx in 0..n_threads {
                let start = c_idx * chunk_sz;
                if start >= n_samples {
                    break;
                }
                let end = (start + chunk_sz).min(n_samples);
                handles.push(s.spawn(move || {
                    let mut lxtx = vec![0.0_f64; in_dim * in_dim];
                    let mut lxty = vec![0.0_f64; in_dim * out_dim];
                    for (x, y) in feats[start..end].iter().zip(targets[start..end].iter()) {
                        for i in 0..in_dim {
                            let xi = x[i];
                            if xi == 0.0 {
                                continue;
                            }
                            let y_row = &mut lxty[i * out_dim..(i + 1) * out_dim];
                            for (dst, &yv) in y_row.iter_mut().zip(y.iter()) {
                                *dst += xi * yv;
                            }
                            for j in i..in_dim {
                                lxtx[i * in_dim + j] += xi * x[j];
                            }
                        }
                    }
                    (lxtx, lxty)
                }));
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut xtx = vec![0.0_f64; in_dim * in_dim];
        let mut xty = vec![0.0_f64; in_dim * out_dim];
        for (lxtx, lxty) in partials {
            for (dst, src) in xtx.iter_mut().zip(lxtx.iter()) {
                *dst += *src;
            }
            for (dst, src) in xty.iter_mut().zip(lxty.iter()) {
                *dst += *src;
            }
        }
        (xtx, xty)
    } else {
        let mut xtx = vec![0.0_f64; in_dim * in_dim];
        let mut xty = vec![0.0_f64; in_dim * out_dim];
        for (x, y) in feats.iter().zip(targets.iter()) {
            for i in 0..in_dim {
                let xi = x[i];
                if xi == 0.0 {
                    continue;
                }
                let y_row = &mut xty[i * out_dim..(i + 1) * out_dim];
                for (dst, &yv) in y_row.iter_mut().zip(y.iter()) {
                    *dst += xi * yv;
                }
                for j in i..in_dim {
                    xtx[i * in_dim + j] += xi * x[j];
                }
            }
        }
        (xtx, xty)
    };

    let avg_diag = (0..in_dim).map(|i| xtx[i * in_dim + i]).sum::<f64>() / (in_dim as f64);
    let eff_lambda = if n_samples >= 256 {
        lambda.max(0.025 * avg_diag)
    } else {
        lambda
    };

    for i in 0..in_dim {
        for j in 0..i {
            xtx[i * in_dim + j] = xtx[j * in_dim + i];
        }
        xtx[i * in_dim + i] += eff_lambda;
    }

    let inv = invert_sym_pd(in_dim, &mut xtx)?;
    let mut w = vec![0.0_f64; in_dim * out_dim];
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(in_dim.max(1));
    let chunk_rows = in_dim.div_ceil(n_threads);
    std::thread::scope(|s| {
        for (t_idx, dst_chunk) in w.chunks_mut(chunk_rows * out_dim).enumerate() {
            let row_start = t_idx * chunk_rows;
            let inv_ref = &inv;
            let xty_ref = &xty;
            s.spawn(move || {
                let rows = dst_chunk.len() / out_dim;
                for local_r in 0..rows {
                    let i = row_start + local_r;
                    let dst = &mut dst_chunk[local_r * out_dim..(local_r + 1) * out_dim];
                    for k in 0..in_dim {
                        let a_ik = inv_ref[i * in_dim + k];
                        if a_ik == 0.0 {
                            continue;
                        }
                        let src = &xty_ref[k * out_dim..(k + 1) * out_dim];
                        for o in 0..out_dim {
                            dst[o] += a_ik * src[o];
                        }
                    }
                }
            });
        }
    });
    Ok(w)
}

fn build_rademacher_expansion(ws_dim: usize, exp_dim: usize) -> Vec<f64> {
    let mut w = vec![0.0_f64; ws_dim * exp_dim];
    let scale = 1.0 / (ws_dim as f64).sqrt();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for val in &mut w {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let bit = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 63) & 1;
        *val = if bit == 0 { scale } else { -scale };
    }
    w
}

fn expand_relu_features(
    ws_state: &[f64],
    domain_id: Option<usize>,
    w_expand: &[f64],
    ws_dim: usize,
    exp_dim: usize,
) -> Vec<f64> {
    let mut h = vec![0.0_f64; exp_dim];
    for (i, &xi) in ws_state[..ws_dim].iter().enumerate() {
        if xi == 0.0 {
            continue;
        }
        let row = &w_expand[i * exp_dim..(i + 1) * exp_dim];
        for (dst, &wv) in h.iter_mut().zip(row.iter()) {
            *dst += xi * wv;
        }
    }
    if let Some(d) = domain_id {
        let dom_off = (ws_dim + d) * exp_dim;
        if dom_off + exp_dim <= w_expand.len() {
            let row = &w_expand[dom_off..dom_off + exp_dim];
            for (dst, &wv) in h.iter_mut().zip(row.iter()) {
                *dst += 0.5 * wv;
            }
        }
    }
    for v in &mut h {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    h
}

/// WSP-synthesize `W_ncm_chunk: [ws_dim, seq_len * dim]` mapping verified Workspace states
/// directly to parallel `seq_len x dim` latent concept chunks via analytical Ridge.
pub fn compile_parallel_ncm_wsp(
    ws_states: &[Vec<f64>],
    target_chunks: &[Vec<f64>],
    ws_dim: usize,
    seq_len: usize,
    dim: usize,
    lambda: f64,
) -> Result<ParallelNcmGenerator, ErrorCode> {
    let out_dim = seq_len * dim;
    let w_ncm_chunk = solve_ridge_matrix(ws_states, target_chunks, ws_dim, out_dim, lambda)?;

    Ok(ParallelNcmGenerator {
        ws_dim,
        exp_dim: 0,
        dom_exp_dim: 0,
        num_domains: 0,
        seq_len,
        dim,
        w_expand: Vec::new(),
        w_dom_expand: Vec::new(),
        domain_means: Vec::new(),
        w_ncm_chunk,
        w_ncm_domain: Vec::new(),
        a_copy: Vec::new(),
        w_val: Vec::new(),
        coarse_len: 0,
        w_coarse: Vec::new(),
        w_q_attn: Vec::new(),
        w_k_heads: Vec::new(),
        rel_pos_bias: Vec::new(),
        w_v_attn: Vec::new(),
    })
}

/// WSP-synthesize domain-centered + non-linear ReLU-ETF-expanded `ParallelNcmGenerator`
/// with per-domain isolated residual WSP heads:
/// `z_hat = domain_means[d] + ReLU(x @ W_expand) @ W_ncm_chunk + ReLU(x @ W_dom_expand) @ W_ncm_domain[d]`.
#[allow(clippy::too_many_arguments)]
pub fn compile_parallel_ncm_wsp_expanded(
    ws_states: &[Vec<f64>],
    target_chunks: &[Vec<f64>],
    domain_ids: &[usize],
    num_domains: usize,
    ws_dim: usize,
    exp_dim: usize,
    dom_exp_dim: usize,
    seq_len: usize,
    dim: usize,
    lambda: f64,
) -> Result<ParallelNcmGenerator, ErrorCode> {
    if ws_states.is_empty()
        || ws_states.len() != target_chunks.len()
        || ws_states.len() != domain_ids.len()
        || num_domains == 0
        || exp_dim == 0
    {
        return Err(ErrorCode::InvalidInput);
    }
    let out_dim = seq_len * dim;
    let mut domain_means = vec![0.0_f64; num_domains * out_dim];
    let mut domain_counts = vec![0usize; num_domains];

    for (y, &d) in target_chunks.iter().zip(domain_ids.iter()) {
        if y.len() != out_dim || d >= num_domains {
            return Err(ErrorCode::ShapeMismatch);
        }
        domain_counts[d] += 1;
        let dst = &mut domain_means[d * out_dim..(d + 1) * out_dim];
        for (m, &yv) in dst.iter_mut().zip(y.iter()) {
            *m += yv;
        }
    }
    for (d, &cnt) in domain_counts.iter().enumerate() {
        if cnt > 0 {
            let inv = 1.0 / (cnt as f64);
            let dst = &mut domain_means[d * out_dim..(d + 1) * out_dim];
            for m in dst.iter_mut() {
                *m *= inv;
            }
        }
    }

    let w_expand = build_rademacher_expansion(ws_dim + num_domains, exp_dim);
    let mut expanded_feats = Vec::with_capacity(ws_states.len());
    let mut centered_targets = Vec::with_capacity(target_chunks.len());

    for ((x, y), &d) in ws_states
        .iter()
        .zip(target_chunks.iter())
        .zip(domain_ids.iter())
    {
        if x.len() != ws_dim {
            return Err(ErrorCode::ShapeMismatch);
        }
        expanded_feats.push(expand_relu_features(x, Some(d), &w_expand, ws_dim, exp_dim));
        let mean_d = &domain_means[d * out_dim..(d + 1) * out_dim];
        let res: Vec<f64> = y
            .iter()
            .zip(mean_d.iter())
            .map(|(&yv, &mv)| yv - mv)
            .collect();
        centered_targets.push(res);
    }

    let w_ncm_chunk =
        solve_ridge_matrix(&expanded_feats, &centered_targets, exp_dim, out_dim, lambda)?;

    let (w_dom_expand, w_ncm_domain) = if dom_exp_dim > 0 {
        let w_de = build_rademacher_expansion(ws_dim, dom_exp_dim);
        let mut by_dom_feats: Vec<Vec<Vec<f64>>> = vec![Vec::new(); num_domains];
        let mut by_dom_res: Vec<Vec<Vec<f64>>> = vec![Vec::new(); num_domains];

        for ((x, h_shared), (y_centered, &d)) in ws_states
            .iter()
            .zip(expanded_feats.iter())
            .zip(centered_targets.iter().zip(domain_ids.iter()))
        {
            let mut pred_shared = vec![0.0_f64; out_dim];
            for (i, &hi) in h_shared.iter().enumerate() {
                if hi == 0.0 {
                    continue;
                }
                let row = &w_ncm_chunk[i * out_dim..(i + 1) * out_dim];
                for (dst, &wv) in pred_shared.iter_mut().zip(row.iter()) {
                    *dst += hi * wv;
                }
            }
            let res2: Vec<f64> = y_centered
                .iter()
                .zip(pred_shared.iter())
                .map(|(&yc, &ps)| yc - ps)
                .collect();
            by_dom_feats[d].push(expand_relu_features(x, None, &w_de, ws_dim, dom_exp_dim));
            by_dom_res[d].push(res2);
        }

        let mut w_dom_all = vec![0.0_f64; num_domains * dom_exp_dim * out_dim];
        let dom_stride = dom_exp_dim * out_dim;
        std::thread::scope(|s| {
            for (d, dst_dom) in w_dom_all.chunks_mut(dom_stride).enumerate() {
                let f_ref = &by_dom_feats[d];
                let r_ref = &by_dom_res[d];
                if f_ref.is_empty() {
                    continue;
                }
                let adaptive_lambda = lambda * ((f_ref.len() as f64) / 250.0).sqrt().max(1.0);
                s.spawn(move || {
                    if let Ok(w_d) =
                        solve_ridge_matrix(f_ref, r_ref, dom_exp_dim, out_dim, adaptive_lambda)
                    {
                        dst_dom.copy_from_slice(&w_d);
                    }
                });
            }
        });
        (w_de, w_dom_all)
    } else {
        (Vec::new(), Vec::new())
    };

    Ok(ParallelNcmGenerator {
        ws_dim,
        exp_dim,
        dom_exp_dim,
        num_domains,
        seq_len,
        dim,
        w_expand,
        w_dom_expand,
        domain_means,
        w_ncm_chunk,
        w_ncm_domain,
        a_copy: Vec::new(),
        w_val: Vec::new(),
        coarse_len: 0,
        w_coarse: Vec::new(),
        w_q_attn: Vec::new(),
        w_k_heads: Vec::new(),
        rel_pos_bias: Vec::new(),
        w_v_attn: Vec::new(),
    })
}

fn downsample_chunk(z: &[f64], seq_len: usize, coarse_len: usize, dim: usize) -> Vec<f64> {
    let block = (seq_len / coarse_len).max(1);
    let inv = 1.0 / (block as f64);
    let mut out = vec![0.0_f64; coarse_len * dim];
    for c in 0..coarse_len {
        let dst = &mut out[c * dim..(c + 1) * dim];
        for b in 0..block {
            let t = (c * block + b).min(seq_len - 1);
            let src = &z[t * dim..(t + 1) * dim];
            for (dv, &sv) in dst.iter_mut().zip(src.iter()) {
                *dv += sv * inv;
            }
        }
    }
    out
}

fn upsample_chunk(z_coarse: &[f64], coarse_len: usize, seq_len: usize, dim: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; seq_len * dim];
    let block = (seq_len / coarse_len).max(1);
    for t in 0..seq_len {
        let c = (t / block).min(coarse_len - 1);
        out[t * dim..(t + 1) * dim].copy_from_slice(&z_coarse[c * dim..(c + 1) * dim]);
    }
    out
}

fn apply_multihead_rel_cross_attn(
    queries_in: &[f64],
    prompt_mat: &[f64],
    w_q_heads: &[f64],
    w_k_heads: &[f64],
    rel_pos_bias: &[f64],
    seq_len: usize,
    dim: usize,
) -> Vec<f64> {
    const HEADS: usize = 4;
    const REL_WIN: usize = 127;
    let dh = dim / HEADS;
    let inv_sqrt_dh = 1.0 / (dh as f64).sqrt();
    let mut active_pos = [false; 64];
    let mut active_cnt = 0usize;
    for s in 0..seq_len.min(64) {
        let xp = &prompt_mat[s * dim..(s + 1) * dim];
        if xp.iter().any(|&v| v != 0.0) {
            active_pos[s] = true;
            active_cnt += 1;
        }
    }
    let mut out = vec![0.0_f64; seq_len * dim];
    if active_cnt == 0 {
        return out;
    }

    let mut q_h = vec![0.0_f64; seq_len * dh];
    let mut k_h = vec![0.0_f64; seq_len * dh];
    let mut scores = [0.0_f64; 64];

    for h in 0..HEADS {
        let wq = &w_q_heads[h * dim * dh..(h + 1) * dim * dh];
        let wk = &w_k_heads[h * dim * dh..(h + 1) * dim * dh];
        let rpb = &rel_pos_bias[h * REL_WIN..(h + 1) * REL_WIN];

        q_h.fill(0.0);
        k_h.fill(0.0);
        for t in 0..seq_len {
            let qt = &queries_in[t * dim..(t + 1) * dim];
            let dst_q = &mut q_h[t * dh..(t + 1) * dh];
            for (i, &qv) in qt.iter().enumerate() {
                if qv != 0.0 {
                    let row = &wq[i * dh..(i + 1) * dh];
                    for (dv, &wv) in dst_q.iter_mut().zip(row.iter()) {
                        *dv += qv * wv;
                    }
                }
            }
            if t < 64 && active_pos[t] {
                let pt = &prompt_mat[t * dim..(t + 1) * dim];
                let dst_k = &mut k_h[t * dh..(t + 1) * dh];
                for (i, &pv) in pt.iter().enumerate() {
                    if pv != 0.0 {
                        let row = &wk[i * dh..(i + 1) * dh];
                        for (dv, &wv) in dst_k.iter_mut().zip(row.iter()) {
                            *dv += pv * wv;
                        }
                    }
                }
            }
        }

        let hoff = h * dh;
        for t in 0..seq_len {
            let qt = &q_h[t * dh..(t + 1) * dh];
            let mut max_s = f64::NEG_INFINITY;
            for s in 0..seq_len.min(64) {
                if !active_pos[s] {
                    continue;
                }
                let ks = &k_h[s * dh..(s + 1) * dh];
                let dot: f64 = qt.iter().zip(ks.iter()).map(|(a, b)| a * b).sum();
                let rel_idx = ((t as isize - s as isize) + 63).clamp(0, 126) as usize;
                let sc = dot * inv_sqrt_dh + rpb[rel_idx];
                scores[s] = sc;
                if sc > max_s {
                    max_s = sc;
                }
            }
            let mut sum_exp = 0.0;
            for s in 0..seq_len.min(64) {
                if active_pos[s] {
                    let w = (scores[s] - max_s).exp();
                    scores[s] = w;
                    sum_exp += w;
                }
            }
            let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
            let dst = &mut out[t * dim + hoff..t * dim + hoff + dh];
            for s in 0..seq_len.min(64) {
                if !active_pos[s] {
                    continue;
                }
                let w = scores[s] * inv_sum;
                let vs = &prompt_mat[s * dim + hoff..s * dim + hoff + dh];
                for (dv, &vv) in dst.iter_mut().zip(vs.iter()) {
                    *dv += w * vv;
                }
            }
        }
    }
    out
}

/// Apply `[seq_len, dim] @ [dim, dim] -> [seq_len, dim]`.
fn apply_val_proj(u: &[f64], w_val: &[f64], seq_len: usize, dim: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; seq_len * dim];
    for t in 0..seq_len {
        let ut = &u[t * dim..(t + 1) * dim];
        let dst = &mut out[t * dim..(t + 1) * dim];
        for (i, &uv) in ut.iter().enumerate() {
            if uv == 0.0 {
                continue;
            }
            let row = &w_val[i * dim..(i + 1) * dim];
            for (d_val, &wv) in dst.iter_mut().zip(row.iter()) {
                *d_val += uv * wv;
            }
        }
    }
    out
}

/// WSP-synthesize fast, low-memory NCM generator combining:
/// 1) Per-domain unblurred cross-position copy/routing `a_copy [num_domains, seq_len, seq_len]`
/// 2) Shared embedding-to-VQ-latent alignment `w_val [dim, dim]`
/// 3) Compact shared ReLU-ETF workspace residual `w_ncm_chunk [exp_dim, seq_len * dim]`
#[allow(clippy::too_many_arguments)]
pub fn compile_parallel_ncm_wsp_copy_and_residual(
    ws_states: &[Vec<f64>],
    prompt_mats: &[Vec<f64>],
    target_chunks: &[Vec<f64>],
    domain_ids: &[usize],
    num_domains: usize,
    ws_dim: usize,
    exp_dim: usize,
    seq_len: usize,
    dim: usize,
    lambda: f64,
) -> Result<ParallelNcmGenerator, ErrorCode> {
    let n = ws_states.len();
    if n == 0
        || prompt_mats.len() != n
        || target_chunks.len() != n
        || domain_ids.len() != n
        || num_domains == 0
        || exp_dim == 0
        || lambda <= 0.0
    {
        return Err(ErrorCode::InvalidInput);
    }
    let out_dim = seq_len * dim;
    let mut domain_means = vec![0.0_f64; num_domains * out_dim];
    let mut domain_counts = vec![0usize; num_domains];

    for (y, &d) in target_chunks.iter().zip(domain_ids.iter()) {
        if y.len() != out_dim || d >= num_domains {
            return Err(ErrorCode::ShapeMismatch);
        }
        domain_counts[d] += 1;
        let dst = &mut domain_means[d * out_dim..(d + 1) * out_dim];
        for (m, &yv) in dst.iter_mut().zip(y.iter()) {
            *m += yv;
        }
    }
    for (d, &cnt) in domain_counts.iter().enumerate() {
        if cnt > 0 {
            let inv = 1.0 / (cnt as f64);
            let dst = &mut domain_means[d * out_dim..(d + 1) * out_dim];
            for m in dst.iter_mut() {
                *m *= inv;
            }
        }
    }

    let mut centered_targets = Vec::with_capacity(n);
    let mut dom_indices: Vec<Vec<usize>> = vec![Vec::new(); num_domains];
    for (idx, ((x, p), (y, &d))) in ws_states
        .iter()
        .zip(prompt_mats.iter())
        .zip(target_chunks.iter().zip(domain_ids.iter()))
        .enumerate()
    {
        if x.len() != ws_dim || p.len() != out_dim {
            return Err(ErrorCode::ShapeMismatch);
        }
        let mean_d = &domain_means[d * out_dim..(d + 1) * out_dim];
        let res: Vec<f64> = y
            .iter()
            .zip(mean_d.iter())
            .map(|(&yv, &mv)| yv - mv)
            .collect();
        centered_targets.push(res);
        dom_indices[d].push(idx);
    }

    // Stage 0 (Coarse): ReLU-ETF workspace -> coarse [16, dim] -> upsampled [64, dim]
    let coarse_len = 16usize;
    let w_expand = build_rademacher_expansion(ws_dim + num_domains, exp_dim);
    let mut expanded_feats = Vec::with_capacity(n);
    let mut coarse_targets = Vec::with_capacity(n);
    for idx in 0..n {
        expanded_feats.push(expand_relu_features(
            &ws_states[idx],
            Some(domain_ids[idx]),
            &w_expand,
            ws_dim,
            exp_dim,
        ));
        coarse_targets.push(downsample_chunk(
            &centered_targets[idx],
            seq_len,
            coarse_len,
            dim,
        ));
    }
    let w_coarse = solve_ridge_matrix(
        &expanded_feats,
        &coarse_targets,
        exp_dim,
        coarse_len * dim,
        lambda,
    )?;

    let mut coarse_preds_up = Vec::with_capacity(n);
    let mut res_after_coarse = Vec::with_capacity(n);
    for idx in 0..n {
        let mut z_c = vec![0.0_f64; coarse_len * dim];
        for (i, &hi) in expanded_feats[idx].iter().enumerate() {
            if hi == 0.0 {
                continue;
            }
            let row = &w_coarse[i * coarse_len * dim..(i + 1) * coarse_len * dim];
            for (dst, &wv) in z_c.iter_mut().zip(row.iter()) {
                *dst += hi * wv;
            }
        }
        let z_up = upsample_chunk(&z_c, coarse_len, seq_len, dim);
        let res1: Vec<f64> = centered_targets[idx]
            .iter()
            .zip(z_up.iter())
            .map(|(&yc, &zu)| yc - zu)
            .collect();
        coarse_preds_up.push(z_up);
        res_after_coarse.push(res1);
    }

    // Stage A: Per-Domain 4-Head Content-Addressed + Relative-Position Cross-Attention
    const HEADS: usize = 4;
    const REL_WIN: usize = 127;
    let dh = dim / HEADS;

    let mut q_inputs = Vec::with_capacity(n);
    for idx in 0..n {
        let d = domain_ids[idx];
        let mean_d = &domain_means[d * out_dim..(d + 1) * out_dim];
        let mut q_in = vec![0.0_f64; out_dim];
        for o in 0..out_dim {
            q_in[o] = mean_d[o] + coarse_preds_up[idx][o];
        }
        q_inputs.push(q_in);
    }

    let q_head_stride = dim * dh;
    let q_dom_stride = HEADS * q_head_stride;
    let r_dom_stride = HEADS * REL_WIN;
    let v_dom_stride = dim * dim;

    let mut w_k_heads = vec![0.0_f64; HEADS * q_head_stride];
    for h in 0..HEADS {
        let hoff = h * dh;
        let wk_dst = &mut w_k_heads[h * q_head_stride..(h + 1) * q_head_stride];
        for c in 0..dh {
            wk_dst[(hoff + c) * dh + c] = 1.0;
        }
    }

    let n_threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .min(n.max(1));
    let chunk_sz = n.div_ceil(n_threads);

    let partials = std::thread::scope(|s_scope| {
        let q_in_ref = &q_inputs;
        let r_ref = &res_after_coarse;
        let p_ref = prompt_mats;
        let mut handles = Vec::new();
        for t_idx in 0..n_threads {
            let start = t_idx * chunk_sz;
            if start >= n {
                break;
            }
            let end = (start + chunk_sz).min(n);
            handles.push(s_scope.spawn(move || {
                let mut l_qxtx = vec![0.0_f64; num_domains * dim * dim];
                let mut l_qxty = vec![0.0_f64; num_domains * q_dom_stride];
                let mut l_rel = vec![0.0_f64; num_domains * r_dom_stride];
                let mut l_dom_cnt = vec![0usize; num_domains];

                for idx in start..end {
                    let d = domain_ids[idx];
                    l_dom_cnt[d] += 1;
                    let xp = &p_ref[idx];
                    let rt_all = &r_ref[idx];
                    let qi_all = &q_in_ref[idx];

                    let mut active_s = [false; 64];
                    let mut any_active = false;
                    for s in 0..seq_len.min(64) {
                        let xs = &xp[s * dim..(s + 1) * dim];
                        if xs.iter().any(|&v| v != 0.0) {
                            active_s[s] = true;
                            any_active = true;
                        }
                    }
                    if !any_active {
                        continue;
                    }

                    let qxtx_d = &mut l_qxtx[d * dim * dim..(d + 1) * dim * dim];
                    let qxty_d = &mut l_qxty[d * q_dom_stride..(d + 1) * q_dom_stride];
                    let rel_d = &mut l_rel[d * r_dom_stride..(d + 1) * r_dom_stride];

                    for t in 0..seq_len {
                        let qt = &qi_all[t * dim..(t + 1) * dim];
                        for i in 0..dim {
                            let qi = qt[i];
                            if qi == 0.0 {
                                continue;
                            }
                            let row = &mut qxtx_d[i * dim..i * dim + dim];
                            for j in i..dim {
                                row[j] += qi * qt[j];
                            }
                        }

                        for h in 0..HEADS {
                            let hoff = h * dh;
                            let rt = &rt_all[t * dim + hoff..t * dim + hoff + dh];
                            let mut best_s = usize::MAX;
                            let mut best_dot = f64::NEG_INFINITY;
                            for s in 0..seq_len.min(64) {
                                if !active_s[s] {
                                    continue;
                                }
                                let xs = &xp[s * dim + hoff..s * dim + hoff + dh];
                                let dot: f64 = rt.iter().zip(xs.iter()).map(|(a, b)| a * b).sum();
                                if dot > best_dot {
                                    best_dot = dot;
                                    best_s = s;
                                }
                            }
                            if best_s == usize::MAX {
                                continue;
                            }
                            let rel_idx =
                                ((t as isize - best_s as isize) + 63).clamp(0, 126) as usize;
                            rel_d[h * REL_WIN + rel_idx] += 1.0;

                            let target_k = &xp[best_s * dim + hoff..best_s * dim + hoff + dh];
                            let qxty_dh = &mut qxty_d[h * q_head_stride..(h + 1) * q_head_stride];
                            for i in 0..dim {
                                let qi = qt[i];
                                if qi == 0.0 {
                                    continue;
                                }
                                let y_row = &mut qxty_dh[i * dh..(i + 1) * dh];
                                for (dst, &kv) in y_row.iter_mut().zip(target_k.iter()) {
                                    *dst += qi * kv;
                                }
                            }
                        }
                    }
                }
                (l_qxtx, l_qxty, l_rel, l_dom_cnt)
            }));
        }
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });

    let mut qxtx_all = vec![0.0_f64; num_domains * dim * dim];
    let mut qxty_all = vec![0.0_f64; num_domains * q_dom_stride];
    let mut rel_all = vec![0.0_f64; num_domains * r_dom_stride];
    let mut dom_cnt_all = vec![0usize; num_domains];
    for (lq, lqy, lr, lc) in partials {
        for (d, s) in qxtx_all.iter_mut().zip(lq.iter()) {
            *d += *s;
        }
        for (d, s) in qxty_all.iter_mut().zip(lqy.iter()) {
            *d += *s;
        }
        for (d, s) in rel_all.iter_mut().zip(lr.iter()) {
            *d += *s;
        }
        for (d, s) in dom_cnt_all.iter_mut().zip(lc.iter()) {
            *d += *s;
        }
    }

    // Global fallback for any domain with 0 training samples
    let mut g_qxtx = vec![0.0_f64; dim * dim];
    let mut g_qxty = vec![0.0_f64; q_dom_stride];
    let mut g_rel = vec![0.0_f64; r_dom_stride];
    for d in 0..num_domains {
        let qxtx_d = &qxtx_all[d * dim * dim..(d + 1) * dim * dim];
        for (dst, &src) in g_qxtx.iter_mut().zip(qxtx_d.iter()) {
            *dst += src;
        }
        let qxty_d = &qxty_all[d * q_dom_stride..(d + 1) * q_dom_stride];
        for (dst, &src) in g_qxty.iter_mut().zip(qxty_d.iter()) {
            *dst += src;
        }
        let rel_d = &rel_all[d * r_dom_stride..(d + 1) * r_dom_stride];
        for (dst, &src) in g_rel.iter_mut().zip(rel_d.iter()) {
            *dst += src;
        }
    }

    let mut w_q_heads = vec![0.0_f64; num_domains * q_dom_stride];
    let mut rel_pos_bias = vec![0.0_f64; num_domains * r_dom_stride];

    std::thread::scope(|s_scope| {
        let dom_threads = n_threads.min(num_domains.max(1));
        let dom_chunk = num_domains.div_ceil(dom_threads);
        for (c_idx, (wq_chunk, rpb_chunk)) in w_q_heads
            .chunks_mut(dom_chunk * q_dom_stride)
            .zip(rel_pos_bias.chunks_mut(dom_chunk * r_dom_stride))
            .enumerate()
        {
            let d_base = c_idx * dom_chunk;
            let qxtx_ref = &qxtx_all;
            let qxty_ref = &qxty_all;
            let rel_ref = &rel_all;
            let cnt_ref = &dom_cnt_all;
            let g_qxtx_ref = &g_qxtx;
            let g_qxty_ref = &g_qxty;
            let g_rel_ref = &g_rel;
            s_scope.spawn(move || {
                let local_doms = wq_chunk.len() / q_dom_stride;
                for ld in 0..local_doms {
                    let d = d_base + ld;
                    let use_global = cnt_ref[d] == 0;
                    let mut qxtx = if use_global {
                        g_qxtx_ref.to_vec()
                    } else {
                        qxtx_ref[d * dim * dim..(d + 1) * dim * dim].to_vec()
                    };
                    let qxty = if use_global {
                        g_qxty_ref
                    } else {
                        &qxty_ref[d * q_dom_stride..(d + 1) * q_dom_stride]
                    };
                    let rel_src = if use_global {
                        g_rel_ref
                    } else {
                        &rel_ref[d * r_dom_stride..(d + 1) * r_dom_stride]
                    };

                    for h in 0..HEADS {
                        let r_src = &rel_src[h * REL_WIN..(h + 1) * REL_WIN];
                        let r_dst = &mut rpb_chunk[ld * r_dom_stride + h * REL_WIN
                            ..ld * r_dom_stride + (h + 1) * REL_WIN];
                        let total_rel: f64 = r_src.iter().sum::<f64>() + (REL_WIN as f64);
                        for (ri, &cnt) in r_src.iter().enumerate() {
                            r_dst[ri] = 0.5 * ((cnt + 1.0) / total_rel).ln();
                        }
                    }

                    for i in 0..dim {
                        for j in 0..i {
                            qxtx[i * dim + j] = qxtx[j * dim + i];
                        }
                        qxtx[i * dim + i] += lambda * (seq_len as f64);
                    }
                    if let Ok(qinv) = invert_sym_pd(dim, &mut qxtx) {
                        for h in 0..HEADS {
                            let qxty_h = &qxty[h * q_head_stride..(h + 1) * q_head_stride];
                            let wq_dst = &mut wq_chunk[ld * q_dom_stride + h * q_head_stride
                                ..ld * q_dom_stride + (h + 1) * q_head_stride];
                            for i in 0..dim {
                                for k in 0..dim {
                                    let a_ik = qinv[i * dim + k];
                                    if a_ik == 0.0 {
                                        continue;
                                    }
                                    let src = &qxty_h[k * dh..(k + 1) * dh];
                                    let dst = &mut wq_dst[i * dh..(i + 1) * dh];
                                    for o in 0..dh {
                                        dst[o] += a_ik * src[o];
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
    });

    let mut attn_outs = vec![Vec::new(); n];
    std::thread::scope(|s| {
        for (c_idx, a_slice) in attn_outs.chunks_mut(chunk_sz).enumerate() {
            let base = c_idx * chunk_sz;
            let q_ref = &q_inputs;
            let p_ref = prompt_mats;
            let wq_ref = &w_q_heads;
            let wk_ref = &w_k_heads;
            let rpb_ref = &rel_pos_bias;
            s.spawn(move || {
                for (li, a_dst) in a_slice.iter_mut().enumerate() {
                    let idx = base + li;
                    let d = domain_ids[idx];
                    *a_dst = apply_multihead_rel_cross_attn(
                        &q_ref[idx],
                        &p_ref[idx],
                        &wq_ref[d * q_dom_stride..(d + 1) * q_dom_stride],
                        wk_ref,
                        &rpb_ref[d * r_dom_stride..(d + 1) * r_dom_stride],
                        seq_len,
                        dim,
                    );
                }
            });
        }
    });

    let v_partials = std::thread::scope(|s_scope| {
        let ao_ref = &attn_outs;
        let r_ref = &res_after_coarse;
        let mut handles = Vec::new();
        for t_idx in 0..n_threads {
            let start = t_idx * chunk_sz;
            if start >= n {
                break;
            }
            let end = (start + chunk_sz).min(n);
            handles.push(s_scope.spawn(move || {
                let mut l_axtx = vec![0.0_f64; num_domains * v_dom_stride];
                let mut l_axty = vec![0.0_f64; num_domains * v_dom_stride];
                for idx in start..end {
                    let d = domain_ids[idx];
                    let ao = &ao_ref[idx];
                    let r1 = &r_ref[idx];
                    let axtx_d = &mut l_axtx[d * v_dom_stride..(d + 1) * v_dom_stride];
                    let axty_d = &mut l_axty[d * v_dom_stride..(d + 1) * v_dom_stride];
                    for t in 0..seq_len {
                        let at = &ao[t * dim..(t + 1) * dim];
                        let rt = &r1[t * dim..(t + 1) * dim];
                        for i in 0..dim {
                            let ai = at[i];
                            if ai == 0.0 {
                                continue;
                            }
                            let row_x = &mut axtx_d[i * dim..(i + 1) * dim];
                            for j in i..dim {
                                row_x[j] += ai * at[j];
                            }
                            let y_row = &mut axty_d[i * dim..(i + 1) * dim];
                            for (dst, &rv) in y_row.iter_mut().zip(rt.iter()) {
                                *dst += ai * rv;
                            }
                        }
                    }
                }
                (l_axtx, l_axty)
            }));
        }
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });

    let mut axtx_all = vec![0.0_f64; num_domains * v_dom_stride];
    let mut axty_all = vec![0.0_f64; num_domains * v_dom_stride];
    for (lx, ly) in v_partials {
        for (d, s) in axtx_all.iter_mut().zip(lx.iter()) {
            *d += *s;
        }
        for (d, s) in axty_all.iter_mut().zip(ly.iter()) {
            *d += *s;
        }
    }

    let mut g_axtx = vec![0.0_f64; v_dom_stride];
    let mut g_axty = vec![0.0_f64; v_dom_stride];
    for d in 0..num_domains {
        for (dst, &src) in g_axtx
            .iter_mut()
            .zip(axtx_all[d * v_dom_stride..(d + 1) * v_dom_stride].iter())
        {
            *dst += src;
        }
        for (dst, &src) in g_axty
            .iter_mut()
            .zip(axty_all[d * v_dom_stride..(d + 1) * v_dom_stride].iter())
        {
            *dst += src;
        }
    }

    let mut w_o_attn = vec![0.0_f64; num_domains * v_dom_stride];
    for d in 0..num_domains {
        let use_global = dom_cnt_all[d] == 0;
        let mut axtx = if use_global {
            g_axtx.clone()
        } else {
            axtx_all[d * v_dom_stride..(d + 1) * v_dom_stride].to_vec()
        };
        let axty = if use_global {
            &g_axty
        } else {
            &axty_all[d * v_dom_stride..(d + 1) * v_dom_stride]
        };
        for i in 0..dim {
            for j in 0..i {
                axtx[i * dim + j] = axtx[j * dim + i];
            }
            axtx[i * dim + i] += lambda * (seq_len as f64);
        }
        let ainv = invert_sym_pd(dim, &mut axtx)?;
        let w_o_d = &mut w_o_attn[d * v_dom_stride..(d + 1) * v_dom_stride];
        for i in 0..dim {
            for k in 0..dim {
                let a_ik = ainv[i * dim + k];
                if a_ik == 0.0 {
                    continue;
                }
                let src = &axty[k * dim..(k + 1) * dim];
                let dst = &mut w_o_d[i * dim..(i + 1) * dim];
                for o in 0..dim {
                    dst[o] += a_ik * src[o];
                }
            }
        }
    }

    // Stage B: Fine Residual W_ncm_chunk [exp_dim, seq_len * dim]
    let mut residual_targets = Vec::with_capacity(n);
    for idx in 0..n {
        let d = domain_ids[idx];
        let w_o_d = &w_o_attn[d * v_dom_stride..(d + 1) * v_dom_stride];
        let attn_pred = apply_val_proj(&attn_outs[idx], w_o_d, seq_len, dim);
        let res2: Vec<f64> = res_after_coarse[idx]
            .iter()
            .zip(attn_pred.iter())
            .map(|(&r1, &ap)| r1 - ap)
            .collect();
        residual_targets.push(res2);
    }
    let w_ncm_chunk =
        solve_ridge_matrix(&expanded_feats, &residual_targets, exp_dim, out_dim, lambda)?;

    Ok(ParallelNcmGenerator {
        ws_dim,
        exp_dim,
        dom_exp_dim: 0,
        num_domains,
        seq_len,
        dim,
        w_expand,
        w_dom_expand: Vec::new(),
        domain_means,
        w_ncm_chunk,
        w_ncm_domain: Vec::new(),
        a_copy: Vec::new(),
        w_val: Vec::new(),
        coarse_len,
        w_coarse,
        w_q_attn: w_q_heads,
        w_k_heads,
        rel_pos_bias,
        w_v_attn: w_o_attn,
    })
}

impl ParallelNcmGenerator {
    /// Parallel non-autoregressive NCM forward pass: `Workspace State -> z_hat [seq_len x dim]`.
    pub fn generate_concept_chunk(&self, ws_state: &[f64]) -> Vec<f64> {
        self.generate_concept_chunk_for_domain(ws_state, 0)
    }

    /// Domain-routed parallel NCM forward pass:
    /// `z_hat = domain_means[d] + ReLU(x @ w_expand) @ w_ncm_chunk + ReLU(x @ w_dom_expand) @ w_ncm_domain[d]`.
    pub fn generate_concept_chunk_for_domain(
        &self,
        ws_state: &[f64],
        domain_id: usize,
    ) -> Vec<f64> {
        let out_dim = self.seq_len * self.dim;
        let mut z_hat = if !self.domain_means.is_empty() && domain_id < self.num_domains {
            self.domain_means[domain_id * out_dim..(domain_id + 1) * out_dim].to_vec()
        } else {
            vec![0.0_f64; out_dim]
        };

        if self.exp_dim > 0 && !self.w_expand.is_empty() {
            let h = expand_relu_features(
                ws_state,
                Some(domain_id),
                &self.w_expand,
                self.ws_dim,
                self.exp_dim,
            );
            for (i, &hi) in h.iter().enumerate() {
                if hi == 0.0 {
                    continue;
                }
                let row = &self.w_ncm_chunk[i * out_dim..(i + 1) * out_dim];
                for (dst, &wv) in z_hat.iter_mut().zip(row.iter()) {
                    *dst += hi * wv;
                }
            }
            if self.dom_exp_dim > 0
                && !self.w_dom_expand.is_empty()
                && !self.w_ncm_domain.is_empty()
                && domain_id < self.num_domains
            {
                let hd = expand_relu_features(
                    ws_state,
                    None,
                    &self.w_dom_expand,
                    self.ws_dim,
                    self.dom_exp_dim,
                );
                let dom_base = domain_id * self.dom_exp_dim * out_dim;
                for (i, &hi) in hd.iter().enumerate() {
                    if hi == 0.0 {
                        continue;
                    }
                    let row =
                        &self.w_ncm_domain[dom_base + i * out_dim..dom_base + (i + 1) * out_dim];
                    for (dst, &wv) in z_hat.iter_mut().zip(row.iter()) {
                        *dst += hi * wv;
                    }
                }
            }
        } else {
            for (i, &xi) in ws_state.iter().enumerate() {
                if xi == 0.0 {
                    continue;
                }
                let row = &self.w_ncm_chunk[i * out_dim..(i + 1) * out_dim];
                for (dst, &wv) in z_hat.iter_mut().zip(row.iter()) {
                    *dst += xi * wv;
                }
            }
        }
        z_hat
    }

    /// Coarse-to-fine parallel NCM forward pass:
    /// `domain_means[d] -> coarse[16] upsampled -> 4-head content + rel-pos cross-attn -> fine residual`.
    pub fn generate_concept_chunk_with_prompt(
        &self,
        ws_state: &[f64],
        prompt_mat: &[f64],
        domain_id: usize,
    ) -> Vec<f64> {
        let out_dim = self.seq_len * self.dim;
        let mut z_hat = self.generate_concept_chunk_for_domain(ws_state, domain_id);
        let mut q_in = if !self.domain_means.is_empty() && domain_id < self.num_domains {
            self.domain_means[domain_id * out_dim..(domain_id + 1) * out_dim].to_vec()
        } else {
            vec![0.0_f64; out_dim]
        };
        if self.coarse_len > 0 && !self.w_coarse.is_empty() && !self.w_expand.is_empty() {
            let h = expand_relu_features(
                ws_state,
                Some(domain_id),
                &self.w_expand,
                self.ws_dim,
                self.exp_dim,
            );
            let c_dim = self.coarse_len * self.dim;
            let mut z_c = vec![0.0_f64; c_dim];
            for (i, &hi) in h.iter().enumerate() {
                if hi == 0.0 {
                    continue;
                }
                let row = &self.w_coarse[i * c_dim..(i + 1) * c_dim];
                for (dst, &wv) in z_c.iter_mut().zip(row.iter()) {
                    *dst += hi * wv;
                }
            }
            let z_up = upsample_chunk(&z_c, self.coarse_len, self.seq_len, self.dim);
            for o in 0..out_dim {
                z_hat[o] += z_up[o];
                q_in[o] += z_up[o];
            }
        }
        if !self.w_q_attn.is_empty() && !self.w_k_heads.is_empty() && !self.w_v_attn.is_empty() {
            const HEADS: usize = 4;
            const REL_WIN: usize = 127;
            let dh = self.dim / HEADS;
            let q_dom_stride = HEADS * self.dim * dh;
            let r_dom_stride = HEADS * REL_WIN;
            let v_dom_stride = self.dim * self.dim;
            let d = domain_id.min(self.num_domains.saturating_sub(1));

            let w_q_slice = if self.w_q_attn.len() >= self.num_domains * q_dom_stride {
                &self.w_q_attn[d * q_dom_stride..(d + 1) * q_dom_stride]
            } else {
                &self.w_q_attn
            };
            let rpb_slice = if self.rel_pos_bias.len() >= self.num_domains * r_dom_stride {
                &self.rel_pos_bias[d * r_dom_stride..(d + 1) * r_dom_stride]
            } else {
                &self.rel_pos_bias
            };
            let w_v_slice = if self.w_v_attn.len() >= self.num_domains * v_dom_stride {
                &self.w_v_attn[d * v_dom_stride..(d + 1) * v_dom_stride]
            } else {
                &self.w_v_attn
            };

            let ao = apply_multihead_rel_cross_attn(
                &q_in,
                prompt_mat,
                w_q_slice,
                &self.w_k_heads,
                rpb_slice,
                self.seq_len,
                self.dim,
            );
            let attn_pred = apply_val_proj(&ao, w_v_slice, self.seq_len, self.dim);
            for (dst, &ap) in z_hat.iter_mut().zip(attn_pred.iter()) {
                *dst += ap;
            }
        }
        z_hat
    }
}
