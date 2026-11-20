//! Conversational & Semantic Reasoning Dataset Synthesizer + 2-Layer Nonlinear ReLU-ETF
//! Knowledge & Reasoning Engine (`semantic_data.rs`).
//!
//! Fixes Gap 3 (conversational data, order/negation semantics, and nonlinear reasoning depth):
//! 1. `generate_semantic_corpus` creates deterministic, tokenized (`u16`) conversational,
//!    negation, word-order, and multi-hop relational reasoning episodes using `vocab_ada_h1.txt`.
//! 2. `DeepKnowledgeReasoner` upgrades single-layer linear `W_knowledge`/`W_reasoning` into a
//!    2-layer nonlinear ReLU-ETF residual block (`in_dim -> hidden_dim (ReLU) -> out_dim`)
//!    synthesized analytically via WSP Ridge projection.

use crate::sequence_encoder::VocabTokenizer;
use crate::solvers::solve_ridge_projection;
use crate::superposition_wsp::SplitMix64;
use crate::sys::ErrorCode;

const TOK_SUPERPOS: u16 = 10;
const TOK_THOUGHT: u16 = 11;
const TOK_SOLVER_START: u16 = 31;
const TOK_ROUTE: u16 = 79;
const TOK_BUDGET: u16 = 80;
const TOK_CLS: u16 = 101;
const TOK_SEP: u16 = 102;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticCategory {
    Conversation = 0,
    OrderCausal = 1,
    NegationLogic = 2,
    MultiHopChain = 3,
}

#[derive(Clone, Debug)]
pub struct SemanticEpisode {
    pub category: SemanticCategory,
    pub prompt_text: String,
    pub response_text: String,
    pub prompt_tokens: Vec<u16>,
    pub response_tokens: Vec<u16>,
}

/// Generate a balanced conversational + order-sensitive + negation + multi-hop semantic corpus.
pub fn generate_semantic_corpus(
    tok: &VocabTokenizer,
    count_per_category: usize,
    seed: u64,
) -> Vec<SemanticEpisode> {
    let mut rng = SplitMix64::new(seed);
    let mut episodes = Vec::with_capacity(count_per_category * 4);

    let conv_pairs: [(&str, &str); 14] = [
        (
            "hi",
            "hello! i am ada compiled by quintessa. ask a domain question or give me a reasoning task.",
        ),
        (
            "hello",
            "hello! i am ada compiled by quintessa. how can i help you today?",
        ),
        (
            "hey",
            "hello! ready for your query, rule check, or calculation.",
        ),
        (
            "what",
            "please specify your question, domain problem, or numeric inputs and i will solve and verify it.",
        ),
        (
            "hello who are you",
            "i am ada compiled by quintessa with verified reasoning and exact computation.",
        ),
        (
            "what can you do",
            "i route domain problems, execute exact mathematical circuits, and reason over structured knowledge.",
        ),
        (
            "how does your architecture work",
            "i combine orthogonal feature superposition, kda recurrent memory, saci circuits, and parallel concept decoding.",
        ),
        (
            "explain why verification matters",
            "verification checks activation sparsity, routing exclusivity, accuracy, etf coherence, and inference noise before emission.",
        ),
        (
            "can you remember previous turns",
            "yes my kda recurrent state keeps bounded conversational memory across turns without unbounded cache growth.",
        ),
        (
            "summarize the current status",
            "all active solver domains are routed deterministically and verified against ground truth.",
        ),
        (
            "what happens when a rule is violated",
            "the workspace verifier detects the constraint violation and triggers replanning or saci escalation.",
        ),
        (
            "how do you handle exact math",
            "deterministic formulas are compiled directly into in-neuron saci circuits instead of approximate text prediction.",
        ),
        (
            "thank you for the explanation",
            "you are welcome. provide the next query or constraint whenever ready.",
        ),
        (
            "give me a brief safety check",
            "all protected invariants remain frozen and verified within declared numerical tolerances.",
        ),
    ];

    let entities = [
        "sensor", "valve", "router", "kernel", "cache", "turbine", "relay", "module", "node",
        "gate", "clock", "bus",
    ];

    // 1. Conversational episodes
    for i in 0..count_per_category {
        let (p, r) = conv_pairs[i % conv_pairs.len()];
        episodes.push(SemanticEpisode {
            category: SemanticCategory::Conversation,
            prompt_text: p.to_string(),
            response_text: r.to_string(),
            prompt_tokens: tok.encode(p),
            response_tokens: tok.encode(r),
        });
    }

    // 2. Order-sensitive causal direction ("A causes B" vs "B causes A")
    for _ in 0..count_per_category {
        let a_idx = rng.next_usize(entities.len());
        let mut b_idx = rng.next_usize(entities.len());
        if b_idx == a_idx {
            b_idx = (a_idx + 1) % entities.len();
        }
        let a = entities[a_idx];
        let b = entities[b_idx];
        let p = format!("if {a} triggers {b} which component fires first");
        let r = format!("{a} fires first and causally precedes {b} in the execution chain.");
        episodes.push(SemanticEpisode {
            category: SemanticCategory::OrderCausal,
            prompt_text: p.clone(),
            response_text: r.clone(),
            prompt_tokens: tok.encode(&p),
            response_tokens: tok.encode(&r),
        });
    }

    // 3. Negation & conditional logic ("approved" vs "not approved")
    for i in 0..count_per_category {
        let a = entities[rng.next_usize(entities.len())];
        let (p, r) = if i % 2 == 0 {
            (
                format!("check {a} status: score is high and risk flag is not set"),
                format!("approved: {a} satisfies threshold with no active risk flag."),
            )
        } else {
            (
                format!("check {a} status: score is high and risk flag is set"),
                format!("rejected: {a} is blocked because the risk flag is active."),
            )
        };
        episodes.push(SemanticEpisode {
            category: SemanticCategory::NegationLogic,
            prompt_text: p.clone(),
            response_text: r.clone(),
            prompt_tokens: tok.encode(&p),
            response_tokens: tok.encode(&r),
        });
    }

    // 4. Multi-hop transitive chains ("A > B and B > C -> A > C")
    for _ in 0..count_per_category {
        let a_idx = rng.next_usize(entities.len());
        let b_idx = (a_idx + 1 + rng.next_usize(3)) % entities.len();
        let c_idx = (b_idx + 1 + rng.next_usize(3)) % entities.len();
        let (a, b, c) = (entities[a_idx], entities[b_idx], entities[c_idx]);
        let p = format!("given {a} exceeds {b} and {b} exceeds {c} rank {a} against {c}");
        let r = format!("by transitivity {a} exceeds {c} through intermediate link {b}.");
        episodes.push(SemanticEpisode {
            category: SemanticCategory::MultiHopChain,
            prompt_text: p.clone(),
            response_text: r.clone(),
            prompt_tokens: tok.encode(&p),
            response_tokens: tok.encode(&r),
        });
    }

    episodes
}

/// Map each semantic category to one of the 4 unused `data_dal` solver slots (`0, 1, 2, 25`)
/// so each category gets its own dedicated NCM centroid and 4-head cross-attention parameters
/// with zero collision against the 37 active `data_dal` solver domains.
#[inline]
pub const fn category_domain_id(cat: SemanticCategory) -> u16 {
    match cat {
        SemanticCategory::Conversation => 0,   // [SOLVER_00]
        SemanticCategory::OrderCausal => 1,    // [SOLVER_01]
        SemanticCategory::NegationLogic => 2,  // [SOLVER_02]
        SemanticCategory::MultiHopChain => 25, // [SOLVER_25]
    }
}

/// Serialize semantic episodes into the standard `train.bin`/`valid.bin` `u16` little-endian stream,
/// routing each category to its dedicated unused domain slot (`0, 1, 2, 25`).
pub fn encode_episodes_to_u16_stream(episodes: &[SemanticEpisode]) -> Vec<u16> {
    let mut stream = Vec::with_capacity(episodes.len() * 64);
    for ep in episodes {
        let solver_tok = TOK_SOLVER_START + category_domain_id(ep.category);
        stream.push(TOK_CLS);
        stream.push(TOK_SUPERPOS);
        stream.push(TOK_ROUTE);
        stream.push(0); // '=' slot
        stream.push(solver_tok);
        stream.push(TOK_BUDGET);
        // 4-token header padding matching data_dal episode layout
        stream.extend_from_slice(&[0, 0, 0, 0]);
        stream.extend_from_slice(&ep.prompt_tokens);
        stream.push(TOK_THOUGHT);
        stream.extend_from_slice(&ep.response_tokens);
        stream.push(TOK_SEP);
    }
    stream
}

/// 2-Layer Nonlinear ReLU-ETF Knowledge & Reasoning MLP:
///   `h = ReLU(x @ W_1)` where `W_1 in R^{in_dim x hidden_dim}` is deterministic Rademacher-ETF
///   `y = h @ W_2` where `W_2 in R^{hidden_dim x out_dim}` is solved analytically via WSP Ridge.
#[derive(Clone, Debug)]
pub struct DeepKnowledgeReasoner {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub out_dim: usize,
    pub w1: Vec<f64>,
    pub w2: Vec<f64>,
}

impl DeepKnowledgeReasoner {
    /// Analytically synthesize a 2-layer nonlinear ReLU-ETF map from `(xs, ys)` pairs.
    pub fn compile_wsp(
        xs: &[Vec<f64>],
        ys: &[Vec<f64>],
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        lambda: f64,
        seed: u64,
    ) -> Result<Self, ErrorCode> {
        let n = xs.len();
        if n == 0 || ys.len() != n || in_dim == 0 || hidden_dim == 0 || out_dim == 0 {
            return Err(ErrorCode::InvalidInput);
        }
        // Deterministic Rademacher-ETF expansion W1 [in_dim, hidden_dim]
        let mut rng = SplitMix64::new(seed);
        let scale = 1.0 / (in_dim as f64).sqrt();
        let mut w1 = vec![0.0_f64; in_dim * hidden_dim];
        for v in &mut w1 {
            *v = if (rng.next_u64() & 1) == 0 {
                scale
            } else {
                -scale
            };
        }

        // Compute hidden activations H [n, hidden_dim] = ReLU(X @ W1)
        let mut h_flat = vec![0.0_f64; n * hidden_dim];
        let mut y_flat = vec![0.0_f64; n * out_dim];
        for (idx, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
            if x.len() != in_dim || y.len() != out_dim {
                return Err(ErrorCode::ShapeMismatch);
            }
            let h_row = &mut h_flat[idx * hidden_dim..(idx + 1) * hidden_dim];
            for (i, &xi) in x.iter().enumerate() {
                if xi == 0.0 {
                    continue;
                }
                let w_row = &w1[i * hidden_dim..(i + 1) * hidden_dim];
                for (dst, &wv) in h_row.iter_mut().zip(w_row.iter()) {
                    *dst += xi * wv;
                }
            }
            for hv in h_row.iter_mut() {
                if *hv < 0.0 {
                    *hv = 0.0;
                }
            }
            y_flat[idx * out_dim..(idx + 1) * out_dim].copy_from_slice(y);
        }

        let w0 = vec![0.0_f64; hidden_dim * out_dim];
        let w2 = solve_ridge_projection(n, hidden_dim, out_dim, &h_flat, &y_flat, &w0, lambda)?;

        Ok(Self {
            in_dim,
            hidden_dim,
            out_dim,
            w1,
            w2,
        })
    }

    /// Zero-allocation forward pass using caller-provided `hidden_scratch` (`>= hidden_dim`).
    #[inline]
    pub fn forward_into(
        &self,
        x: &[f64],
        hidden_scratch: &mut [f64],
        out: &mut [f64],
    ) -> Result<(), ErrorCode> {
        if x.len() < self.in_dim
            || hidden_scratch.len() < self.hidden_dim
            || out.len() < self.out_dim
        {
            return Err(ErrorCode::BufferTooSmall);
        }
        let h = &mut hidden_scratch[..self.hidden_dim];
        h.fill(0.0);
        for (i, &xi) in x[..self.in_dim].iter().enumerate() {
            if xi == 0.0 {
                continue;
            }
            let row = &self.w1[i * self.hidden_dim..(i + 1) * self.hidden_dim];
            for (dst, &wv) in h.iter_mut().zip(row.iter()) {
                *dst += xi * wv;
            }
        }
        for hv in h.iter_mut() {
            if *hv < 0.0 {
                *hv = 0.0;
            }
        }

        out[..self.out_dim].fill(0.0);
        for (j, &hj) in h.iter().enumerate() {
            if hj == 0.0 {
                continue;
            }
            let row = &self.w2[j * self.out_dim..(j + 1) * self.out_dim];
            for (dst, &wv) in out[..self.out_dim].iter_mut().zip(row.iter()) {
                *dst += hj * wv;
            }
        }
        Ok(())
    }
}
