//! WSP Token-Level Language Decoder (`W_lm_head: [CTX_DIM, VOCAB_SIZE]`).
//!
//! Uses analytical Ridge projection `(X^T X + lambda I)^{-1} (X^T Y)` over
//! multi-scale KDA-style exponentially decayed context states `h_t in R^128`
//! to synthesize a full 8,192-vocabulary token decoder without backpropagation.

use crate::sys::ErrorCode;

pub const VOCAB_SIZE: usize = 8192;
pub const EMB_DIM: usize = 32;
pub const CTX_DIM: usize = 128; // 4 x EMB_DIM: [curr_tok, prev_tok, fast_kda(0.6), slow_kda(0.92)]
pub const FFN_DIM: usize = 512; // Nonlinear ReLU hidden expansion before W_lm_head

#[inline]
fn mix_u64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

/// Deterministic unit-norm token embedding in `R^EMB_DIM` (32-dim).
#[inline]
pub fn token_embedding(tok: u16, out: &mut [f64; EMB_DIM]) {
    let mut norm_sq = 0.0_f64;
    for (d, slot) in out.iter_mut().enumerate() {
        let h = mix_u64(((tok as u64) << 16) | (d as u64));
        // Signed Rademacher/normal-like projection
        let v = ((h & 0xFFFF) as f64) / 32768.0 - 1.0;
        *slot = v;
        norm_sq += v * v;
    }
    let inv = if norm_sq > 1e-12 {
        1.0 / norm_sq.sqrt()
    } else {
        1.0
    };
    for slot in out.iter_mut() {
        *slot *= inv;
    }
}

/// Multi-scale KDA-style recurrent context state `h_t in R^CTX_DIM`.
#[derive(Clone, Debug)]
pub struct TokenContextState {
    pub prev_emb: [f64; EMB_DIM],
    pub fast_kda: [f64; EMB_DIM],
    pub slow_kda: [f64; EMB_DIM],
}

impl TokenContextState {
    pub fn new() -> Self {
        Self {
            prev_emb: [0.0; EMB_DIM],
            fast_kda: [0.0; EMB_DIM],
            slow_kda: [0.0; EMB_DIM],
        }
    }

    pub fn reset(&mut self) {
        self.prev_emb.fill(0.0);
        self.fast_kda.fill(0.0);
        self.slow_kda.fill(0.0);
    }

    /// Advance one token and write the 128-dim context vector `h_out`.
    #[inline]
    pub fn step(&mut self, tok: u16, h_out: &mut [f64; CTX_DIM]) {
        let mut curr = [0.0_f64; EMB_DIM];
        token_embedding(tok, &mut curr);

        for d in 0..EMB_DIM {
            self.fast_kda[d] = 0.60 * self.fast_kda[d] + 0.40 * curr[d];
            self.slow_kda[d] = 0.92 * self.slow_kda[d] + 0.08 * curr[d];

            h_out[d] = curr[d];
            h_out[EMB_DIM + d] = self.prev_emb[d];
            h_out[2 * EMB_DIM + d] = self.fast_kda[d];
            h_out[3 * EMB_DIM + d] = self.slow_kda[d];
        }
        self.prev_emb = curr;
    }
}

impl Default for TokenContextState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct TokenLanguageDecoder {
    /// Deterministic task-conditioned ReLU projection `[CTX_DIM, FFN_DIM]` + bias `[FFN_DIM]`
    pub w_ffn: Vec<f64>,
    pub b_ffn: Vec<f64>,
    /// Analytical WSP-written LM head `[FFN_DIM, VOCAB_SIZE]` (512 x 8192 = 4,194,304 weights)
    pub w_lm_head: Vec<f64>,
}

fn build_relu_ffn_weights() -> (Vec<f64>, Vec<f64>) {
    let mut w_ffn = vec![0.0_f64; CTX_DIM * FFN_DIM];
    let mut b_ffn = vec![0.0_f64; FFN_DIM];
    let scale = 1.0 / (CTX_DIM as f64).sqrt();
    for i in 0..CTX_DIM {
        for j in 0..FFN_DIM {
            let h = mix_u64(((i as u64) << 20) ^ ((j as u64) * 0x9e37_79b9));
            let s = ((h & 0xFFFF) as f64) / 32768.0 - 1.0;
            w_ffn[i * FFN_DIM + j] = s * scale;
        }
    }
    for (j, slot) in b_ffn.iter_mut().enumerate() {
        let h = mix_u64((j as u64) ^ 0x517c_c1b7_2722_0a95);
        *slot = (((h & 0xFFFF) as f64) / 65536.0 - 0.5) * 0.25;
    }
    (w_ffn, b_ffn)
}

#[inline]
fn relu_ffn_forward(h: &[f64; CTX_DIM], w_ffn: &[f64], b_ffn: &[f64], z_out: &mut [f64; FFN_DIM]) {
    z_out.copy_from_slice(b_ffn);
    for (i, &hi) in h.iter().enumerate() {
        if hi == 0.0 {
            continue;
        }
        let row = &w_ffn[i * FFN_DIM..(i + 1) * FFN_DIM];
        for j in 0..FFN_DIM {
            z_out[j] += hi * row[j];
        }
    }
    for z in z_out.iter_mut() {
        if *z < 0.0 {
            *z = 0.0;
        }
    }
}

/// Invert an `n x n` symmetric positive-definite matrix via Gauss-Jordan elimination.
fn invert_matrix(n: usize, a: &mut [f64]) -> Result<Vec<f64>, ErrorCode> {
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
            let factor = a[r * n + col];
            if factor != 0.0 {
                for c in 0..n {
                    a[r * n + c] -= factor * a[col * n + c];
                    inv[r * n + c] -= factor * inv[col * n + c];
                }
            }
        }
    }

    Ok(inv)
}

/// WSP-synthesize `w_lm_head` (`[FFN_DIM, VOCAB_SIZE]`) over task-conditioned ReLU
/// activations `z = ReLU(h @ w_ffn + b_ffn)` from a `u16` token stream.
pub fn compile_token_decoder_wsp(
    tokens: &[u16],
    max_steps: usize,
    lambda: f64,
) -> Result<TokenLanguageDecoder, ErrorCode> {
    if tokens.len() < 2 || lambda <= 0.0 {
        return Err(ErrorCode::InvalidInput);
    }

    let (w_ffn, b_ffn) = build_relu_ffn_weights();
    let mut ztz = vec![0.0_f64; FFN_DIM * FFN_DIM];
    let mut zty = vec![0.0_f64; FFN_DIM * VOCAB_SIZE];
    let mut ctx = TokenContextState::new();
    let mut h = [0.0_f64; CTX_DIM];
    let mut z = [0.0_f64; FFN_DIM];

    let limit = tokens.len().saturating_sub(1).min(max_steps);
    for t in 0..limit {
        let cur = tokens[t];
        let nxt = tokens[t + 1];
        if cur == 101 {
            ctx.reset();
        }
        ctx.step(cur, &mut h);
        relu_ffn_forward(&h, &w_ffn, &b_ffn, &mut z);

        if (nxt as usize) < VOCAB_SIZE {
            for i in 0..FFN_DIM {
                let zi = z[i];
                if zi == 0.0 {
                    continue;
                }
                zty[i * VOCAB_SIZE + (nxt as usize)] += zi;
                for j in i..FFN_DIM {
                    let zj = z[j];
                    if zj != 0.0 {
                        ztz[i * FFN_DIM + j] += zi * zj;
                    }
                }
            }
        }
    }

    // Symmetrize Z^T Z and add lambda * I
    for i in 0..FFN_DIM {
        for j in 0..i {
            ztz[i * FFN_DIM + j] = ztz[j * FFN_DIM + i];
        }
        ztz[i * FFN_DIM + i] += lambda;
    }

    let inv_ztz = invert_matrix(FFN_DIM, &mut ztz)?;

    // W_lm_head = (Z^T Z + lambda I)^{-1} (Z^T Y)  [FFN_DIM x VOCAB_SIZE]
    let mut w_lm_head = vec![0.0_f64; FFN_DIM * VOCAB_SIZE];
    for i in 0..FFN_DIM {
        for k in 0..FFN_DIM {
            let a_ik = inv_ztz[i * FFN_DIM + k];
            if a_ik == 0.0 {
                continue;
            }
            let src_row = &zty[k * VOCAB_SIZE..(k + 1) * VOCAB_SIZE];
            let dst_row = &mut w_lm_head[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE];
            for v in 0..VOCAB_SIZE {
                dst_row[v] += a_ik * src_row[v];
            }
        }
    }

    Ok(TokenLanguageDecoder {
        w_ffn,
        b_ffn,
        w_lm_head,
    })
}

/// Evaluate Top-1 and Top-5 next-token accuracy on a held-out token stream.
pub fn evaluate_token_decoder(
    dec: &TokenLanguageDecoder,
    tokens: &[u16],
    max_eval_steps: usize,
) -> (f64, f64, usize) {
    let mut ctx = TokenContextState::new();
    let mut h = [0.0_f64; CTX_DIM];
    let mut z = [0.0_f64; FFN_DIM];
    let mut logits = vec![0.0_f64; VOCAB_SIZE];

    let mut top1 = 0usize;
    let mut top5 = 0usize;
    let mut total = 0usize;

    let limit = tokens.len().saturating_sub(1).min(max_eval_steps);
    for t in 0..limit {
        let cur = tokens[t];
        let nxt = tokens[t + 1] as usize;
        if cur == 101 {
            ctx.reset();
        }
        ctx.step(cur, &mut h);
        if nxt >= VOCAB_SIZE {
            continue;
        }
        relu_ffn_forward(&h, &dec.w_ffn, &dec.b_ffn, &mut z);

        logits.fill(0.0);
        for i in 0..FFN_DIM {
            let zi = z[i];
            if zi == 0.0 {
                continue;
            }
            let row = &dec.w_lm_head[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE];
            for v in 0..VOCAB_SIZE {
                logits[v] += zi * row[v];
            }
        }

        let target_logit = logits[nxt];
        let mut rank = 0usize;
        for &l in &logits {
            if l > target_logit {
                rank += 1;
                if rank >= 5 {
                    break;
                }
            }
        }
        if rank == 0 {
            top1 += 1;
        }
        if rank < 5 {
            top5 += 1;
        }
        total += 1;
    }

    let denom = total.max(1) as f64;
    (
        100.0 * (top1 as f64) / denom,
        100.0 * (top5 as f64) / denom,
        total,
    )
}

/// Autoregressively decode up to `max_new_tokens` from `prompt_tokens` using `ReLU + W_lm_head`.
pub fn generate_tokens(
    dec: &TokenLanguageDecoder,
    prompt_tokens: &[u16],
    max_new_tokens: usize,
) -> Vec<u16> {
    let mut ctx = TokenContextState::new();
    let mut h = [0.0_f64; CTX_DIM];
    for &tok in prompt_tokens {
        if tok == 101 {
            ctx.reset();
        }
        ctx.step(tok, &mut h);
    }

    let mut out = Vec::with_capacity(max_new_tokens);
    let mut z = [0.0_f64; FFN_DIM];
    let mut logits = vec![0.0_f64; VOCAB_SIZE];
    for _ in 0..max_new_tokens {
        relu_ffn_forward(&h, &dec.w_ffn, &dec.b_ffn, &mut z);
        logits.fill(0.0);
        for i in 0..FFN_DIM {
            let zi = z[i];
            if zi == 0.0 {
                continue;
            }
            let row = &dec.w_lm_head[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE];
            for v in 0..VOCAB_SIZE {
                logits[v] += zi * row[v];
            }
        }
        let next_tok = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u16)
            .unwrap_or(102);
        out.push(next_tok);
        if next_tok == 102 {
            break;
        }
        ctx.step(next_tok, &mut h);
    }
    out
}
