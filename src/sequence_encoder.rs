//! Order-Aware KDA + MLA Sequence Encoder & WordPiece Tokenizer (`sequence_encoder.rs`).
//!
//! Fixes Gap 1 (order-blind bag-of-embeddings):
//! - `SequenceEncoder` encodes token sequences using sinusoidal positional modulation +
//!   KDA delta-rule recurrence (`S_t = alpha * (S_{t-1} - (S_{t-1} k) k^T) + v k^T`) +
//!   bigram transition binding (`x_{t-1} ⊗ x_t` circular shift) so word order, clause
//!   direction ("A causes B" vs "B causes A"), and negation ("not approved" vs "approved")
//!   produce distinct, well-separated representations in `R^dim`.
//! - Hot-path `encode_into` is strictly zero-allocation using preallocated `SequenceScratch`.
//! - Cold-path `VocabTokenizer` loads `vocab_ada_h1.txt` for English <-> `u16` token IDs.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::sys::ErrorCode;

/// Preallocated scratchpad for zero-allocation `SequenceEncoder::encode_into`.
#[derive(Clone, Debug)]
pub struct SequenceScratch {
    pub dim: usize,
    /// KDA recurrent associative memory `[dim * dim]`.
    pub kda_state: Vec<f64>,
    pub prev_emb: Vec<f64>,
    pub cur_emb: Vec<f64>,
    pub k_buf: Vec<f64>,
    pub v_buf: Vec<f64>,
    pub step_out: Vec<f64>,
    pub mean_acc: Vec<f64>,
}

impl SequenceScratch {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            kda_state: vec![0.0_f64; dim * dim],
            prev_emb: vec![0.0_f64; dim],
            cur_emb: vec![0.0_f64; dim],
            k_buf: vec![0.0_f64; dim],
            v_buf: vec![0.0_f64; dim],
            step_out: vec![0.0_f64; dim],
            mean_acc: vec![0.0_f64; dim],
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.kda_state.fill(0.0);
        self.prev_emb.fill(0.0);
        self.cur_emb.fill(0.0);
        self.k_buf.fill(0.0);
        self.v_buf.fill(0.0);
        self.step_out.fill(0.0);
        self.mean_acc.fill(0.0);
    }
}

/// Order-aware KDA delta-rule + directional bigram sequence encoder.
#[derive(Clone, Copy, Debug)]
pub struct SequenceEncoder {
    pub dim: usize,
    pub kda_decay: f64,
    pub pos_scale: f64,
    pub bigram_scale: f64,
}

impl SequenceEncoder {
    pub const fn new(dim: usize) -> Self {
        Self {
            dim,
            kda_decay: 0.92,
            pos_scale: 0.15,
            bigram_scale: 0.55,
        }
    }

    /// Encode a token sequence into a unit-normalized `R^dim` vector (zero heap allocation).
    ///
    /// Preserves:
    /// - Token identity (mean & final pooled embeddings)
    /// - Word order (sinusoidal positional modulation + non-commutative circular-shifted
    ///   bigram binding `prev[(d + 1) % dim] * cur[d]`)
    /// - Long-range associative context (KDA delta-rule recurrent state `S_t`)
    pub fn encode_into(
        &self,
        tokens: &[u16],
        tok_emb: &[f64],
        vocab_size: usize,
        scratch: &mut SequenceScratch,
        out: &mut [f64],
    ) -> Result<(), ErrorCode> {
        let dim = self.dim;
        if dim == 0 || scratch.dim < dim || out.len() < dim {
            return Err(ErrorCode::BufferTooSmall);
        }
        scratch.reset();
        out[..dim].fill(0.0);
        if tokens.is_empty() {
            return Ok(());
        }

        let sqrt_d = (dim as f64).sqrt();
        let inv_sqrt_d = 1.0 / sqrt_d;
        let bigram_gain = self.bigram_scale * sqrt_d;
        for (pos, &tok) in tokens.iter().enumerate() {
            let tid = tok as usize;
            if tid < vocab_size && (tid + 1) * dim <= tok_emb.len() {
                scratch.cur_emb[..dim].copy_from_slice(&tok_emb[tid * dim..(tid + 1) * dim]);
            } else {
                scratch.cur_emb[..dim].fill(0.0);
            }

            // Sinusoidal positional modulation
            let p = (pos + 1) as f64;
            for d in 0..dim {
                let freq = 1.0 / (10000.0_f64).powf((2 * (d / 2)) as f64 / (dim as f64));
                let pe = if d % 2 == 0 {
                    (p * freq).sin()
                } else {
                    (p * freq).cos()
                };
                scratch.cur_emb[d] += self.pos_scale * inv_sqrt_d * pe;
            }

            // Non-commutative directional bigram binding: sqrt(d) * prev[(d + 1) % dim] * cur[d]
            // Unit-scale O(1) binding ensures ("A before B") != ("B before A") and ("not X") != ("X")
            if pos > 0 {
                for d in 0..dim {
                    let prev_rot = scratch.prev_emb[(d + 1) % dim];
                    scratch.v_buf[d] =
                        scratch.cur_emb[d] + bigram_gain * prev_rot * scratch.cur_emb[d];
                }
            } else {
                scratch.v_buf[..dim].copy_from_slice(&scratch.cur_emb[..dim]);
            }

            // Normalize key k for stable KDA delta-rule recurrence
            let mut k_norm_sq = 0.0_f64;
            for d in 0..dim {
                let kv = scratch.cur_emb[d];
                scratch.k_buf[d] = kv;
                k_norm_sq += kv * kv;
            }
            let inv_k = if k_norm_sq > 1e-12 {
                1.0 / k_norm_sq.sqrt()
            } else {
                0.0
            };
            for d in 0..dim {
                scratch.k_buf[d] *= inv_k;
            }

            // KDA update: S_t = alpha * (S_{t-1} - (S_{t-1} k) k^T) + v k^T; y_t = S_t k
            let alpha = self.kda_decay;
            for r in 0..dim {
                let row_off = r * dim;
                let mut s_k = 0.0_f64;
                for c in 0..dim {
                    s_k += scratch.kda_state[row_off + c] * scratch.k_buf[c];
                }
                let vr = scratch.v_buf[r];
                let mut y_r = 0.0_f64;
                for c in 0..dim {
                    let kc = scratch.k_buf[c];
                    let new_s = alpha * (scratch.kda_state[row_off + c] - s_k * kc) + vr * kc;
                    scratch.kda_state[row_off + c] = new_s;
                    y_r += new_s * kc;
                }
                scratch.step_out[r] = y_r;
                scratch.mean_acc[r] += y_r;
            }

            scratch.prev_emb[..dim].copy_from_slice(&scratch.cur_emb[..dim]);
        }

        // Fuse final KDA state readout (recency/order) + mean pooled readout (global semantics)
        let inv_len = 1.0 / (tokens.len() as f64);
        let mut norm_sq = 0.0_f64;
        for d in 0..dim {
            let v = 0.5 * scratch.step_out[d] + 0.5 * (scratch.mean_acc[d] * inv_len);
            out[d] = v;
            norm_sq += v * v;
        }
        if norm_sq > 1e-12 {
            let inv = 1.0 / norm_sq.sqrt();
            for v in out[..dim].iter_mut() {
                *v *= inv;
            }
        }
        Ok(())
    }
}

/// Cold-path WordPiece tokenizer & detokenizer backed by `vocab_ada_h1.txt`.
#[derive(Clone, Debug)]
pub struct VocabTokenizer {
    pub id_to_token: Vec<String>,
    pub token_to_id: HashMap<String, u16>,
    pub unk_id: u16,
    pub pad_id: u16,
    pub cls_id: u16,
    pub sep_id: u16,
}

impl VocabTokenizer {
    pub fn from_vocab_file(path: &Path) -> std::io::Result<Self> {
        let text = fs::read_to_string(path)?;
        let mut id_to_token = Vec::with_capacity(8192);
        let mut token_to_id = HashMap::with_capacity(8192);
        for (idx, line) in text.lines().enumerate() {
            let tok = line.trim().to_string();
            if idx < u16::MAX as usize {
                token_to_id.insert(tok.clone(), idx as u16);
                id_to_token.push(tok);
            }
        }
        let unk_id = token_to_id.get("[UNK]").copied().unwrap_or(100);
        let pad_id = token_to_id.get("[PAD]").copied().unwrap_or(0);
        let cls_id = token_to_id.get("[CLS]").copied().unwrap_or(101);
        let sep_id = token_to_id.get("[SEP]").copied().unwrap_or(102);
        Ok(Self {
            id_to_token,
            token_to_id,
            unk_id,
            pad_id,
            cls_id,
            sep_id,
        })
    }

    /// Encode raw text into `u16` WordPiece token IDs (preserves `[SPECIAL]` tokens).
    pub fn encode(&self, text: &str) -> Vec<u16> {
        let mut out = Vec::new();
        for raw_word in text.split_whitespace() {
            if let Some(&id) = self.token_to_id.get(raw_word) {
                out.push(id);
                continue;
            }
            // Split punctuation cleanly around words
            let lower = raw_word.to_ascii_lowercase();
            let mut buf = String::new();
            let flush = |word: &str, out: &mut Vec<u16>| {
                if word.is_empty() {
                    return;
                }
                if let Some(&id) = self.token_to_id.get(word) {
                    out.push(id);
                    return;
                }
                // Greedy longest-match WordPiece
                let chars: Vec<char> = word.chars().collect();
                let mut start = 0usize;
                while start < chars.len() {
                    let mut end = chars.len();
                    let mut matched = None;
                    while end > start {
                        let sub: String = if start == 0 {
                            chars[start..end].iter().collect()
                        } else {
                            let mut s = String::from("##");
                            for &c in &chars[start..end] {
                                s.push(c);
                            }
                            s
                        };
                        if let Some(&id) = self.token_to_id.get(&sub) {
                            matched = Some((id, end));
                            break;
                        }
                        end -= 1;
                    }
                    if let Some((id, next_start)) = matched {
                        out.push(id);
                        start = next_start;
                    } else {
                        out.push(self.unk_id);
                        start += 1;
                    }
                }
            };

            for ch in lower.chars() {
                if ch.is_ascii_alphanumeric() {
                    buf.push(ch);
                } else {
                    flush(&buf, &mut out);
                    buf.clear();
                    let mut p = [0u8; 4];
                    let ps = ch.encode_utf8(&mut p);
                    if let Some(&id) = self.token_to_id.get(ps) {
                        out.push(id);
                    }
                }
            }
            flush(&buf, &mut out);
        }
        out
    }

    /// Decode `u16` token IDs back into readable text (merges `##` subword pieces, skips `[PAD]`).
    pub fn decode(&self, tokens: &[u16]) -> String {
        let mut out = String::new();
        for &tid in tokens {
            if tid == self.pad_id {
                continue;
            }
            let piece = self
                .id_to_token
                .get(tid as usize)
                .map(|s| s.as_str())
                .unwrap_or("[UNK]");
            if piece == "[PAD]" {
                continue;
            }
            if let Some(suffix) = piece.strip_prefix("##") {
                out.push_str(suffix);
            } else {
                if !out.is_empty()
                    && !matches!(piece, "." | "," | ":" | ";" | ")" | "]" | "?" | "!")
                {
                    out.push(' ');
                }
                out.push_str(piece);
            }
        }
        out
    }
}
