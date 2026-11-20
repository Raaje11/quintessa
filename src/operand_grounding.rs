//! Semantic-to-Computational Operand Grounding & SACI Variable Binding.
//!
//! Zero-allocation, O(seq_len) numeric operand parser + semantic role binding:
//! `Prompt tokens -> NumericVocabTable -> BoundOperand[..16] -> SACI input buffer`.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub const MAX_BOUND_OPERANDS: usize = 16;
pub const GROUNDING_DIM: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum NumTokKind {
    None = 0,
    Digits = 1,
    Dot = 2,
    Minus = 3,
    Plus = 4,
    Exp = 5,
}

#[derive(Clone, Copy, Debug)]
pub struct NumTokEntry {
    pub kind: NumTokKind,
    pub is_continuation: bool,
    pub digit_count: u8,
    pub digits_val: u64,
}

impl NumTokEntry {
    pub const EMPTY: Self = Self {
        kind: NumTokKind::None,
        is_continuation: false,
        digit_count: 0,
        digits_val: 0,
    };
}

/// Direct-indexed `O(1)` lookup table for numeric/operator tokens across `vocab_size = 8192`.
#[derive(Clone, Debug)]
pub struct NumericVocabTable {
    pub entries: Vec<NumTokEntry>,
}

impl NumericVocabTable {
    /// Build with the standard ADA-O1 single-digit & operator token IDs (zero I/O fallback).
    pub fn default_ada_o1(vocab_size: usize) -> Self {
        let mut entries = vec![NumTokEntry::EMPTY; vocab_size];
        let mut set_entry = |id: usize, kind: NumTokKind, cont: bool, cnt: u8, val: u64| {
            if id < vocab_size {
                entries[id] = NumTokEntry {
                    kind,
                    is_continuation: cont,
                    digit_count: cnt,
                    digits_val: val,
                };
            }
        };
        set_entry(112, NumTokKind::Plus, false, 0, 0);
        set_entry(114, NumTokKind::Minus, false, 0, 0);
        set_entry(115, NumTokKind::Dot, false, 0, 0);
        set_entry(141, NumTokKind::Exp, false, 0, 0);
        for d in 0..10u64 {
            set_entry(117 + d as usize, NumTokKind::Digits, false, 1, d);
        }
        let cont_digits = [
            (318usize, 0u64),
            (329, 1),
            (344, 2),
            (343, 3),
            (340, 4),
            (341, 5),
            (338, 6),
            (330, 7),
            (342, 8),
            (348, 9),
        ];
        for (id, d) in cont_digits {
            set_entry(id, NumTokKind::Digits, true, 1, d);
        }
        Self { entries }
    }

    /// Cold-path initialization from `vocab.txt` (`one token per line`).
    pub fn from_vocab_file(path: &Path, vocab_size: usize) -> Self {
        let mut table = Self::default_ada_o1(vocab_size);
        let Ok(file) = File::open(path) else {
            return table;
        };
        let reader = BufReader::new(file);
        for (id, line_res) in reader.lines().enumerate() {
            if id >= vocab_size {
                break;
            }
            let Ok(raw) = line_res else {
                break;
            };
            let tok = raw.trim_end_matches(&['\r', '\n'][..]);
            let (cont, body) = if let Some(rest) = tok.strip_prefix("##") {
                (true, rest)
            } else {
                (false, tok)
            };
            if body.is_empty() {
                continue;
            }
            if body == "." {
                table.entries[id] = NumTokEntry {
                    kind: NumTokKind::Dot,
                    is_continuation: cont,
                    digit_count: 0,
                    digits_val: 0,
                };
            } else if body == "-" {
                table.entries[id] = NumTokEntry {
                    kind: NumTokKind::Minus,
                    is_continuation: cont,
                    digit_count: 0,
                    digits_val: 0,
                };
            } else if body == "+" {
                table.entries[id] = NumTokEntry {
                    kind: NumTokKind::Plus,
                    is_continuation: cont,
                    digit_count: 0,
                    digits_val: 0,
                };
            } else if body == "e" || body == "E" {
                table.entries[id] = NumTokEntry {
                    kind: NumTokKind::Exp,
                    is_continuation: cont,
                    digit_count: 0,
                    digits_val: 0,
                };
            } else if body.bytes().all(|b| b.is_ascii_digit()) && body.len() <= 18 {
                if let Ok(val) = body.parse::<u64>() {
                    table.entries[id] = NumTokEntry {
                        kind: NumTokKind::Digits,
                        is_continuation: cont,
                        digit_count: body.len() as u8,
                        digits_val: val,
                    };
                }
            }
        }
        table
    }

    #[inline]
    pub fn get(&self, tok: u16) -> NumTokEntry {
        let idx = tok as usize;
        if idx < self.entries.len() {
            self.entries[idx]
        } else {
            NumTokEntry::EMPTY
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BoundOperand {
    pub value: f64,
    pub arg_pos: u16,
    pub role_ctx: [f64; GROUNDING_DIM],
}

impl BoundOperand {
    pub const ZERO: Self = Self {
        value: 0.0,
        arg_pos: 0,
        role_ctx: [0.0; GROUNDING_DIM],
    };
}

/// Parse multi-token numbers from `tokens` and bind each to its local semantic role context
/// (preceding 3 variable/unit tokens + following 1 unit token) with zero heap allocation.
pub fn extract_bound_operands(
    tokens: &[u16],
    num_table: &NumericVocabTable,
    tok_emb: &[f64],
    vocab_size: usize,
    dim: usize,
    out: &mut [BoundOperand; MAX_BOUND_OPERANDS],
) -> usize {
    let mut count = 0usize;
    let n = tokens.len();
    let d_use = dim.min(GROUNDING_DIM);
    let mut i = 0usize;

    while i < n && count < MAX_BOUND_OPERANDS {
        let e = num_table.get(tokens[i]);
        let starts_num = match e.kind {
            NumTokKind::Digits => true,
            NumTokKind::Minus | NumTokKind::Plus => {
                i + 1 < n && num_table.get(tokens[i + 1]).kind == NumTokKind::Digits
            }
            _ => false,
        };
        if !starts_num {
            i += 1;
            continue;
        }

        let start_pos = i;
        let mut sign = 1.0_f64;
        if e.kind == NumTokKind::Minus {
            sign = -1.0;
            i += 1;
        } else if e.kind == NumTokKind::Plus {
            i += 1;
        }

        let mut int_part = 0.0_f64;
        let mut frac_part = 0.0_f64;
        let mut frac_scale = 1.0_f64;
        let mut in_frac = false;
        let mut exp_sign = 1.0_f64;
        let mut exp_val = 0.0_f64;
        let mut in_exp = false;
        let mut parsed_digits = false;

        while i < n {
            let cur = num_table.get(tokens[i]);
            match cur.kind {
                NumTokKind::Digits => {
                    if i > start_pos && !cur.is_continuation && !in_frac && !in_exp {
                        // Separate adjacent standalone numbers unless joined by `.` or `e`
                        let prev_kind = num_table.get(tokens[i - 1]).kind;
                        if prev_kind == NumTokKind::Digits {
                            break;
                        }
                    }
                    parsed_digits = true;
                    let pow10 = 10.0_f64.powi(cur.digit_count as i32);
                    let v = cur.digits_val as f64;
                    if in_exp {
                        exp_val = exp_val * pow10 + v;
                    } else if in_frac {
                        frac_scale *= pow10;
                        frac_part += v / frac_scale;
                    } else {
                        int_part = int_part * pow10 + v;
                    }
                    i += 1;
                }
                NumTokKind::Dot if !in_frac && !in_exp => {
                    if i + 1 < n && num_table.get(tokens[i + 1]).kind == NumTokKind::Digits {
                        in_frac = true;
                        i += 1;
                    } else {
                        break;
                    }
                }
                NumTokKind::Exp if parsed_digits && !in_exp => {
                    if i + 1 < n {
                        let nxt = num_table.get(tokens[i + 1]);
                        if nxt.kind == NumTokKind::Digits {
                            in_exp = true;
                            i += 1;
                            continue;
                        } else if (nxt.kind == NumTokKind::Minus || nxt.kind == NumTokKind::Plus)
                            && i + 2 < n
                            && num_table.get(tokens[i + 2]).kind == NumTokKind::Digits
                        {
                            exp_sign = if nxt.kind == NumTokKind::Minus {
                                -1.0
                            } else {
                                1.0
                            };
                            in_exp = true;
                            i += 2;
                            continue;
                        }
                    }
                    break;
                }
                _ => break,
            }
        }

        if parsed_digits {
            let mut val = sign * (int_part + frac_part);
            if in_exp {
                val *= 10.0_f64.powf(exp_sign * exp_val);
            }
            if val.is_finite() {
                let mut role_ctx = [0.0_f64; GROUNDING_DIM];
                let ctx_start = start_pos.saturating_sub(3);
                for &ctok in &tokens[ctx_start..start_pos] {
                    let tid = ctok as usize;
                    if tid < vocab_size {
                        let row = &tok_emb[tid * dim..tid * dim + d_use];
                        for (dst, &sv) in role_ctx[..d_use].iter_mut().zip(row.iter()) {
                            *dst += sv;
                        }
                    }
                }
                if i < n {
                    let tid = tokens[i] as usize;
                    if tid < vocab_size {
                        let row = &tok_emb[tid * dim..tid * dim + d_use];
                        for (dst, &sv) in role_ctx[..d_use].iter_mut().zip(row.iter()) {
                            *dst += 0.5 * sv;
                        }
                    }
                }
                let mut norm_sq = 0.0;
                for &rv in &role_ctx[..d_use] {
                    norm_sq += rv * rv;
                }
                if norm_sq > 1e-12 {
                    let inv = 1.0 / norm_sq.sqrt();
                    for rv in &mut role_ctx[..d_use] {
                        *rv *= inv;
                    }
                }
                out[count] = BoundOperand {
                    value: val,
                    arg_pos: start_pos as u16,
                    role_ctx,
                };
                count += 1;
            }
        } else {
            i += 1;
        }
    }

    count
}

/// Bind extracted operands into `saci_in_buf[..input_count]` using Reasoning `r_state`
/// semantic role affinity (falling back to positional order) and build a `128`-dim
/// grounded operand representation `grounded_out` for Workspace `s_state`.
pub fn bind_operands_to_saci(
    operands: &[BoundOperand],
    r_state: &[f64],
    saci_in_buf: &mut [f64],
    grounded_out: &mut [f64],
) {
    let in_n = saci_in_buf.len();
    let d_use = r_state.len().min(GROUNDING_DIM).min(grounded_out.len());
    grounded_out.fill(0.0);

    if operands.is_empty() {
        for (idx, dst) in saci_in_buf.iter_mut().enumerate() {
            *dst = r_state[idx % r_state.len().max(1)];
        }
        return;
    }

    // Score each extracted operand by semantic role affinity with `r_state` + argument order.
    let mut used = [false; MAX_BOUND_OPERANDS];
    for dst in saci_in_buf[..in_n].iter_mut() {
        let mut best_idx = None;
        let mut best_score = f64::NEG_INFINITY;
        for (oi, op) in operands.iter().enumerate().take(MAX_BOUND_OPERANDS) {
            if used[oi] {
                continue;
            }
            let dot: f64 = op.role_ctx[..d_use]
                .iter()
                .zip(r_state[..d_use].iter())
                .map(|(a, b)| a * b)
                .sum();
            let pos_prior = -0.05 * (oi as f64);
            let score = dot + pos_prior;
            if score > best_score {
                best_score = score;
                best_idx = Some(oi);
            }
        }
        if let Some(oi) = best_idx {
            used[oi] = true;
            // Bounded log-compressed numeric value for numerically stable SACI circuit execution
            let v = operands[oi].value;
            *dst = v.signum() * (1.0 + v.abs()).ln();
            let w = (*dst).tanh();
            for d in 0..d_use {
                grounded_out[d] += w * operands[oi].role_ctx[d];
            }
        } else {
            *dst = 0.0;
        }
    }

    let mut norm_sq = 0.0;
    for &gv in &grounded_out[..d_use] {
        norm_sq += gv * gv;
    }
    if norm_sq > 1e-12 {
        let inv = 1.0 / norm_sq.sqrt();
        for gv in &mut grounded_out[..d_use] {
            *gv *= inv;
        }
    }
}
