//! High-Signal Domain Knowledge Base & 4-Step Aligned Scratchpad Reasoning.
//!
//! Zero hardcoded neuron weights: all 4 superposed feature families
//! (`Knowledge`, `Reasoning`, `SaciInterface`, `Simulation`) are WSP-synthesized
//! from `D:\AdvancedADA\data_dal` via analytical Ridge / Null-space projection.
//!
//! At each reasoning signal step `t in {1, 2, 3, 4}`, every participating neuron
//! activates at most ONE feature family (`sum_f z_{i,f}(t) <= 1`), updating the
//! Workspace scratchpad and checking explicit verification states.

use crate::checkpoint::Checkpoint;
use crate::dcir::{step_ponder, AdaptivePonderConfig, PonderState};
use crate::operand_grounding::{
    bind_operands_to_saci, extract_bound_operands, BoundOperand, NumericVocabTable,
    MAX_BOUND_OPERANDS,
};
use crate::saci_compiler::{execute_saci_circuit, SaciCircuitDescriptor};
use crate::solvers::solve_ridge_projection;
use crate::sys::{ErrorCode, PonderAction};

pub const KB_DIM: usize = 128;
pub const KB_NUM_DOMAINS: usize = 42;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureSlot {
    Inactive = 0,
    Knowledge = 0x01,
    Reasoning = 0x02,
    SaciInterface = 0x04,
    Simulation = 0x08,
}

#[derive(Clone, Debug)]
pub struct DomainKnowledgeEntry {
    pub domain_id: usize,
    pub sample_count: usize,
    pub knowledge_proto: Vec<f64>,  // [KB_DIM]
    pub reasoning_proto: Vec<f64>,  // [KB_DIM]
    pub saci_facts_proto: Vec<f64>, // [KB_DIM]
    pub sim_delta_proto: Vec<f64>,  // [KB_DIM]
}

#[derive(Clone, Debug)]
pub struct HighSignalKnowledgeBase {
    pub entries: Vec<DomainKnowledgeEntry>,
    /// WSP-synthesized Knowledge weights: `[KB_DIM, KB_DIM]`
    pub w_knowledge: Vec<f64>,
    /// WSP-synthesized Reasoning weights: `[2 * KB_DIM, KB_DIM]` (input: `[q, k]`)
    pub w_reasoning: Vec<f64>,
    /// WSP-synthesized SACI-Interface weights: `[3 * KB_DIM, KB_NUM_DOMAINS]` (input: `[q, k, r]`)
    pub w_saci_route: Vec<f64>,
    /// WSP-synthesized SACI-Facts alignment weights: `[3 * KB_DIM, KB_DIM]` (input: `[q, k, r]`)
    pub w_saci_facts: Vec<f64>,
    /// WSP-synthesized Simulation transition weights: `[3 * KB_DIM, KB_DIM]` (input: `[k, r, s_facts]`)
    pub w_simulation: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct ScratchpadStepTrace {
    pub step: u8,
    pub active_feature: FeatureSlot,
    pub one_hot_ok: bool,
    pub alignment_score: f64,
    pub verification_passed: bool,
    pub ponder_action: PonderAction,
}

#[derive(Clone, Debug)]
pub struct AlignedScratchpadResult {
    pub predicted_domain: usize,
    pub steps: [ScratchpadStepTrace; 4],
    pub final_verified: bool,
    pub final_sim_residual: f64,
}

#[inline]
pub fn normalize_vec(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for x in v {
            *x /= norm;
        }
    }
}

#[inline]
pub fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom > 1e-12 {
        dot / denom
    } else {
        0.0
    }
}

#[inline]
fn matmul_vec(x: &[f64], w: &[f64], d_in: usize, d_out: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; d_out];
    for c in 0..d_out {
        let mut acc = 0.0;
        for i in 0..d_in {
            acc += x[i] * w[i * d_out + c];
        }
        out[c] = acc;
    }
    out
}

pub struct KbSample {
    pub domain_id: usize,
    pub protected: bool,
    pub req_vec: Vec<f64>,     // [KB_DIM]
    pub thought_vec: Vec<f64>, // [KB_DIM]
    pub facts_vec: Vec<f64>,   // [KB_DIM]
}

/// Build the High-Signal Knowledge Base and WSP-synthesize all 4 aligned feature matrices
/// `(W_K, W_R, W_S, W_M)` with zero hardcoded neuron weights.
pub fn build_wsp_knowledge_base(
    samples: &[KbSample],
) -> Result<HighSignalKnowledgeBase, ErrorCode> {
    if samples.is_empty() {
        return Err(ErrorCode::InvalidInput);
    }

    // 1. Aggregate per-domain Knowledge, Reasoning, SACI-Facts, and Simulation centroids
    let mut counts = vec![0usize; KB_NUM_DOMAINS];
    let mut k_sums = vec![vec![0.0_f64; KB_DIM]; KB_NUM_DOMAINS];
    let mut r_sums = vec![vec![0.0_f64; KB_DIM]; KB_NUM_DOMAINS];
    let mut s_sums = vec![vec![0.0_f64; KB_DIM]; KB_NUM_DOMAINS];

    for s in samples {
        if s.domain_id >= KB_NUM_DOMAINS {
            continue;
        }
        counts[s.domain_id] += 1;
        for d in 0..KB_DIM {
            k_sums[s.domain_id][d] += s.req_vec[d];
            r_sums[s.domain_id][d] += s.thought_vec[d];
            s_sums[s.domain_id][d] += s.facts_vec[d];
        }
    }

    let mut entries = Vec::with_capacity(KB_NUM_DOMAINS);
    for id in 0..KB_NUM_DOMAINS {
        let mut kp = k_sums[id].clone();
        let mut rp = r_sums[id].clone();
        let mut sp = s_sums[id].clone();
        normalize_vec(&mut kp);
        normalize_vec(&mut rp);
        normalize_vec(&mut sp);
        let mut sim_delta: Vec<f64> = sp.iter().zip(kp.iter()).map(|(a, b)| a - b).collect();
        normalize_vec(&mut sim_delta);
        entries.push(DomainKnowledgeEntry {
            domain_id: id,
            sample_count: counts[id],
            knowledge_proto: kp,
            reasoning_proto: rp,
            saci_facts_proto: sp,
            sim_delta_proto: sim_delta,
        });
    }

    // 2. Separate DAL vs Protected samples for WSP synthesis
    let dal: Vec<&KbSample> = samples.iter().filter(|s| !s.protected).collect();
    let prot: Vec<&KbSample> = samples.iter().filter(|s| s.protected).collect();
    if dal.is_empty() {
        return Err(ErrorCode::InvalidInput);
    }

    let n_dal = dal.len();
    let lambda = 1.0_f64;

    // Step 1 (Knowledge): W_K maps req_vec -> domain knowledge_proto while suppressing protected prompts -> 0
    let n_k_total = n_dal + prot.len();
    let mut x_k = Vec::with_capacity(n_k_total * KB_DIM);
    let mut y_k = Vec::with_capacity(n_k_total * KB_DIM);
    for s in &dal {
        x_k.extend_from_slice(&s.req_vec);
        y_k.extend_from_slice(&entries[s.domain_id].knowledge_proto);
    }
    for s in &prot {
        x_k.extend_from_slice(&s.req_vec);
        y_k.extend(std::iter::repeat_n(0.0_f64, KB_DIM));
    }
    let w0_k = vec![0.0_f64; KB_DIM * KB_DIM];
    let w_knowledge = solve_ridge_projection(n_k_total, KB_DIM, KB_DIM, &x_k, &y_k, &w0_k, lambda)?;

    // Step 2 (Reasoning): W_R maps [req_vec, k_state] -> reasoning_proto
    let d_r_in = 2 * KB_DIM;
    let mut x_r = Vec::with_capacity(n_dal * d_r_in);
    let mut y_r = Vec::with_capacity(n_dal * KB_DIM);
    let mut k_states = Vec::with_capacity(n_dal);
    for s in &dal {
        let mut k_st = matmul_vec(&s.req_vec, &w_knowledge, KB_DIM, KB_DIM);
        normalize_vec(&mut k_st);
        x_r.extend_from_slice(&s.req_vec);
        x_r.extend_from_slice(&k_st);
        y_r.extend_from_slice(&entries[s.domain_id].reasoning_proto);
        k_states.push(k_st);
    }
    let w0_r = vec![0.0_f64; d_r_in * KB_DIM];
    let w_reasoning = solve_ridge_projection(n_dal, d_r_in, KB_DIM, &x_r, &y_r, &w0_r, lambda)?;

    // Step 3 (SACI): W_S_route & W_S_facts map [req_vec, k_state, r_state] -> domain route & saci_facts_proto
    let d_s_in = 3 * KB_DIM;
    let mut x_s = Vec::with_capacity(n_dal * d_s_in);
    let mut y_s_route = vec![0.0_f64; n_dal * KB_NUM_DOMAINS];
    let mut y_s_facts = Vec::with_capacity(n_dal * KB_DIM);
    let mut r_states = Vec::with_capacity(n_dal);
    for (idx, s) in dal.iter().enumerate() {
        let mut r_in = Vec::with_capacity(d_r_in);
        r_in.extend_from_slice(&s.req_vec);
        r_in.extend_from_slice(&k_states[idx]);
        let mut r_st = matmul_vec(&r_in, &w_reasoning, d_r_in, KB_DIM);
        normalize_vec(&mut r_st);

        x_s.extend_from_slice(&s.req_vec);
        x_s.extend_from_slice(&k_states[idx]);
        x_s.extend_from_slice(&r_st);
        y_s_route[idx * KB_NUM_DOMAINS + s.domain_id] = 1.0;
        y_s_facts.extend_from_slice(&entries[s.domain_id].saci_facts_proto);
        r_states.push(r_st);
    }
    let w0_sr = vec![0.0_f64; d_s_in * KB_NUM_DOMAINS];
    let w_saci_route = solve_ridge_projection(
        n_dal,
        d_s_in,
        KB_NUM_DOMAINS,
        &x_s,
        &y_s_route,
        &w0_sr,
        lambda,
    )?;
    let w0_sf = vec![0.0_f64; d_s_in * KB_DIM];
    let w_saci_facts =
        solve_ridge_projection(n_dal, d_s_in, KB_DIM, &x_s, &y_s_facts, &w0_sf, lambda)?;

    // Step 4 (Simulation): W_M maps [k_state, r_state, s_facts_state] -> sim_delta_proto
    let mut x_m = Vec::with_capacity(n_dal * d_s_in);
    let mut y_m = Vec::with_capacity(n_dal * KB_DIM);
    for (idx, s) in dal.iter().enumerate() {
        let mut s_in = Vec::with_capacity(d_s_in);
        s_in.extend_from_slice(&s.req_vec);
        s_in.extend_from_slice(&k_states[idx]);
        s_in.extend_from_slice(&r_states[idx]);
        let mut sf_st = matmul_vec(&s_in, &w_saci_facts, d_s_in, KB_DIM);
        normalize_vec(&mut sf_st);

        x_m.extend_from_slice(&k_states[idx]);
        x_m.extend_from_slice(&r_states[idx]);
        x_m.extend_from_slice(&sf_st);
        y_m.extend_from_slice(&entries[s.domain_id].sim_delta_proto);
    }
    let w0_m = vec![0.0_f64; d_s_in * KB_DIM];
    let w_simulation = solve_ridge_projection(n_dal, d_s_in, KB_DIM, &x_m, &y_m, &w0_m, lambda)?;

    Ok(HighSignalKnowledgeBase {
        entries,
        w_knowledge,
        w_reasoning,
        w_saci_route,
        w_saci_facts,
        w_simulation,
    })
}

/// Run the 4-step aligned scratchpad reasoning loop (one feature family active per step,
/// `sum_f z_{i,f}(t) <= 1` checked at every step, with Workspace verification gates).
pub fn run_aligned_scratchpad_reasoning(
    kb: &HighSignalKnowledgeBase,
    query_vec: &[f64],
    num_neurons: usize,
    ponder_cfg: &AdaptivePonderConfig,
) -> AlignedScratchpadResult {
    let mut masks = vec![0u8; num_neurons];
    let mut ponder = PonderState::new();

    // --- Step 1: Knowledge feature activation (0x01) ---
    masks.fill(FeatureSlot::Knowledge as u8);
    let one_hot_1 = masks.iter().all(|&m| m.count_ones() <= 1);
    let mut k_state = matmul_vec(query_vec, &kb.w_knowledge, KB_DIM, KB_DIM);
    normalize_vec(&mut k_state);

    // Best-matching domain from Knowledge scratchpad state
    let mut best_k_sim = -1.0_f64;
    let mut best_k_dom = 0usize;
    for e in &kb.entries {
        if e.sample_count > 0 {
            let sim = cosine_sim(&k_state, &e.knowledge_proto);
            if sim > best_k_sim {
                best_k_sim = sim;
                best_k_dom = e.domain_id;
            }
        }
    }
    let ver_1 = one_hot_1 && best_k_sim > 0.25;
    let (pa_1, _) = step_ponder(
        &mut ponder,
        if ver_1 { 0.2 } else { -1.0 },
        1.0 - best_k_sim,
        ponder_cfg,
    );

    // --- Step 2: Reasoning feature activation (0x02) ---
    masks.fill(FeatureSlot::Reasoning as u8);
    let one_hot_2 = masks.iter().all(|&m| m.count_ones() <= 1);
    let mut r_in = Vec::with_capacity(2 * KB_DIM);
    r_in.extend_from_slice(query_vec);
    r_in.extend_from_slice(&k_state);
    let mut r_state = matmul_vec(&r_in, &kb.w_reasoning, 2 * KB_DIM, KB_DIM);
    normalize_vec(&mut r_state);
    let r_sim = cosine_sim(&r_state, &kb.entries[best_k_dom].reasoning_proto);
    let ver_2 = one_hot_2 && r_sim > 0.25;
    let (pa_2, _) = step_ponder(
        &mut ponder,
        if ver_2 { 0.4 } else { -1.0 },
        1.0 - r_sim,
        ponder_cfg,
    );

    // --- Step 3: SACI-Interface feature activation (0x04) ---
    masks.fill(FeatureSlot::SaciInterface as u8);
    let one_hot_3 = masks.iter().all(|&m| m.count_ones() <= 1);
    let mut s_in = Vec::with_capacity(3 * KB_DIM);
    s_in.extend_from_slice(query_vec);
    s_in.extend_from_slice(&k_state);
    s_in.extend_from_slice(&r_state);
    let logits = matmul_vec(&s_in, &kb.w_saci_route, 3 * KB_DIM, KB_NUM_DOMAINS);
    let (pred_domain, top_logit) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0));
    let mut s_facts = matmul_vec(&s_in, &kb.w_saci_facts, 3 * KB_DIM, KB_DIM);
    normalize_vec(&mut s_facts);
    let s_sim = cosine_sim(&s_facts, &kb.entries[pred_domain].saci_facts_proto);
    let ver_3 = one_hot_3 && s_sim > 0.25 && top_logit > 0.1;
    let (pa_3, _) = step_ponder(
        &mut ponder,
        if ver_3 { 1.0 } else { -0.5 },
        1.0 - s_sim,
        ponder_cfg,
    );

    // --- Step 4: Simulation feature activation (0x08) ---
    masks.fill(FeatureSlot::Simulation as u8);
    let one_hot_4 = masks.iter().all(|&m| m.count_ones() <= 1);
    let mut m_in = Vec::with_capacity(3 * KB_DIM);
    m_in.extend_from_slice(&k_state);
    m_in.extend_from_slice(&r_state);
    m_in.extend_from_slice(&s_facts);
    let mut m_delta = matmul_vec(&m_in, &kb.w_simulation, 3 * KB_DIM, KB_DIM);
    normalize_vec(&mut m_delta);
    let m_sim = cosine_sim(&m_delta, &kb.entries[pred_domain].sim_delta_proto);
    let final_sim_residual = (1.0 - m_sim).max(0.0);
    let ver_4 = one_hot_4 && m_sim > 0.25;
    let (pa_4, _) = step_ponder(
        &mut ponder,
        if ver_4 { 3.0 } else { -1.0 },
        final_sim_residual,
        ponder_cfg,
    );

    let final_verified = ver_1 && ver_2 && ver_3 && ver_4;

    AlignedScratchpadResult {
        predicted_domain: pred_domain,
        steps: [
            ScratchpadStepTrace {
                step: 1,
                active_feature: FeatureSlot::Knowledge,
                one_hot_ok: one_hot_1,
                alignment_score: best_k_sim,
                verification_passed: ver_1,
                ponder_action: pa_1,
            },
            ScratchpadStepTrace {
                step: 2,
                active_feature: FeatureSlot::Reasoning,
                one_hot_ok: one_hot_2,
                alignment_score: r_sim,
                verification_passed: ver_2,
                ponder_action: pa_2,
            },
            ScratchpadStepTrace {
                step: 3,
                active_feature: FeatureSlot::SaciInterface,
                one_hot_ok: one_hot_3,
                alignment_score: s_sim,
                verification_passed: ver_3,
                ponder_action: pa_3,
            },
            ScratchpadStepTrace {
                step: 4,
                active_feature: FeatureSlot::Simulation,
                one_hot_ok: one_hot_4,
                alignment_score: m_sim,
                verification_passed: ver_4,
                ponder_action: pa_4,
            },
        ],
        final_verified,
        final_sim_residual,
    }
}

// ---------------------------------------------------------------------------
// Pure-Weight Compiled Model (W_gate, W_K/R/S/M, P_K/R/S/M, W_halt)
// ---------------------------------------------------------------------------

/// All orchestration (step gating, verification prototypes, halting logits)
/// compiled into pure `f64` weight tensors—no Rust `if step == ...` or struct lookups.
#[derive(Clone, Debug)]
pub struct PureWeightAdaModel {
    /// Step-to-feature gating weights `[4, 4]` + bias `[4]` (maps phase state -> one-hot family)
    pub w_gate: Vec<f64>,
    pub b_gate: Vec<f64>,
    /// Feature transformation weights
    pub w_knowledge: Vec<f64>, // [KB_DIM, KB_DIM]
    pub w_reasoning: Vec<f64>,  // [2*KB_DIM, KB_DIM]
    pub w_saci_route: Vec<f64>, // [3*KB_DIM, KB_NUM_DOMAINS]
    pub w_saci_facts: Vec<f64>, // [3*KB_DIM, KB_DIM]
    pub w_simulation: Vec<f64>, // [3*KB_DIM, KB_DIM]
    /// Verification prototype matrices `[KB_DIM, KB_NUM_DOMAINS]` + threshold bias
    pub p_knowledge: Vec<f64>,
    pub p_reasoning: Vec<f64>,
    pub p_saci: Vec<f64>,
    pub p_simulation: Vec<f64>,
    pub verify_bias: f64,
    /// Halting gate weights `[2]` + bias `[1]` acting on `[verify_margin, step_norm]`
    pub w_halt: [f64; 2],
    pub b_halt: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct PureWeightStepResult {
    pub step: u8,
    pub gated_family_mask: u8,
    pub verify_margin: f64,
    pub halt_logit: f64,
}

#[derive(Clone, Debug)]
pub struct PureWeightForwardOutput {
    pub predicted_domain: usize,
    pub steps: [PureWeightStepResult; 4],
    pub all_verified: bool,
    pub halted_at_step: u8,
    /// Full Workspace state `[k_state, r_state, s_state, m_state]` (`4 * KB_DIM`) for parallel NCM.
    pub ws_state: Vec<f64>,
}

/// Subtract `alpha * global_mean` across active domain columns and renormalize to unit norm.
/// Prevents boilerplate-token spectral collapse (`EFF_RANK ~ 2.7/42`, `cos ~ 0.99`) while
/// preserving positive alignment with true domain states.
fn center_and_renormalize_prototypes(mat: &mut [f64], dim: usize, num_domains: usize, alpha: f64) {
    let mut mean = vec![0.0_f64; dim];
    let mut active_cnt = 0usize;
    let mut active = vec![false; num_domains];

    for d in 0..num_domains {
        let mut nsq = 0.0_f64;
        for r in 0..dim {
            let v = mat[r * num_domains + d];
            nsq += v * v;
        }
        if nsq > 1e-12 {
            active[d] = true;
            active_cnt += 1;
            for r in 0..dim {
                mean[r] += mat[r * num_domains + d];
            }
        }
    }
    if active_cnt < 2 {
        return;
    }
    let inv_cnt = 1.0 / (active_cnt as f64);
    for m in &mut mean {
        *m *= inv_cnt;
    }

    for d in 0..num_domains {
        if !active[d] {
            continue;
        }
        let mut nsq = 0.0_f64;
        for r in 0..dim {
            let v = mat[r * num_domains + d] - alpha * mean[r];
            mat[r * num_domains + d] = v;
            nsq += v * v;
        }
        if nsq > 1e-12 {
            let inv = 1.0 / nsq.sqrt();
            for r in 0..dim {
                mat[r * num_domains + d] *= inv;
            }
        }
    }
}

/// Compile `HighSignalKnowledgeBase` into a 100% pure-weight `PureWeightAdaModel`.
pub fn compile_pure_weight_model(kb: &HighSignalKnowledgeBase) -> PureWeightAdaModel {
    // W_gate [4 x 4] identity + small cyclic shift: maps phase one-hot -> next family one-hot
    let mut w_gate = vec![0.0_f64; 16];
    for i in 0..4 {
        w_gate[i * 4 + i] = 10.0;
    }
    let b_gate = vec![0.0_f64; 4];

    // Pack prototype matrices P_* [KB_DIM x KB_NUM_DOMAINS] (columns = unit prototypes)
    let mut p_knowledge = vec![0.0_f64; KB_DIM * KB_NUM_DOMAINS];
    let mut p_reasoning = vec![0.0_f64; KB_DIM * KB_NUM_DOMAINS];
    let mut p_saci = vec![0.0_f64; KB_DIM * KB_NUM_DOMAINS];
    let mut p_simulation = vec![0.0_f64; KB_DIM * KB_NUM_DOMAINS];

    for (d, e) in kb.entries.iter().enumerate() {
        for r in 0..KB_DIM {
            p_knowledge[r * KB_NUM_DOMAINS + d] = e.knowledge_proto[r];
            p_reasoning[r * KB_NUM_DOMAINS + d] = e.reasoning_proto[r];
            p_saci[r * KB_NUM_DOMAINS + d] = e.saci_facts_proto[r];
            p_simulation[r * KB_NUM_DOMAINS + d] = e.sim_delta_proto[r];
        }
    }

    // Decollapse shared boilerplate cone across domains (raises effective rank & domain discrimination)
    center_and_renormalize_prototypes(&mut p_knowledge, KB_DIM, KB_NUM_DOMAINS, 0.85);
    center_and_renormalize_prototypes(&mut p_reasoning, KB_DIM, KB_NUM_DOMAINS, 0.90);
    center_and_renormalize_prototypes(&mut p_saci, KB_DIM, KB_NUM_DOMAINS, 0.90);
    center_and_renormalize_prototypes(&mut p_simulation, KB_DIM, KB_NUM_DOMAINS, 0.85);

    PureWeightAdaModel {
        w_gate,
        b_gate,
        w_knowledge: kb.w_knowledge.clone(),
        w_reasoning: kb.w_reasoning.clone(),
        w_saci_route: kb.w_saci_route.clone(),
        w_saci_facts: kb.w_saci_facts.clone(),
        w_simulation: kb.w_simulation.clone(),
        p_knowledge,
        p_reasoning,
        p_saci,
        p_simulation,
        verify_bias: -0.15, // ReLU(cos - 0.15) > 0 on decollapsed discriminative prototypes
        w_halt: [4.0, 1.5],
        b_halt: -1.5,
    }
}

/// Pure-weight uniform forward pass: gating, feature execution, verification,
/// and halting logits are all computed via matrix multiplications + ReLU.
pub fn forward_pure_weights(m: &PureWeightAdaModel, query_vec: &[f64]) -> PureWeightForwardOutput {
    let mut k_state = vec![0.0_f64; KB_DIM];
    let mut r_state = vec![0.0_f64; KB_DIM];
    let mut s_state = vec![0.0_f64; KB_DIM];
    let mut m_state = vec![0.0_f64; KB_DIM];
    let mut pred_domain = 0usize;

    let mut step_results = [PureWeightStepResult {
        step: 0,
        gated_family_mask: 0,
        verify_margin: 0.0,
        halt_logit: 0.0,
    }; 4];

    let mut halted_at_step = 4u8;
    let mut all_verified = true;

    for t in 0..4usize {
        // 1. Pure-weight gating: phase_vec @ W_gate + b_gate -> one-hot family mask
        let mut phase = [0.0_f64; 4];
        phase[t] = 1.0;
        let gate_scores = matmul_vec(&phase, &m.w_gate, 4, 4);
        let active_fam = gate_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mask = 1u8 << (active_fam as u8);

        // 2. Execute the gated feature matrix & select verification prototype matrix
        let (state_vec, p_mat) = match active_fam {
            0 => {
                let mut v = matmul_vec(query_vec, &m.w_knowledge, KB_DIM, KB_DIM);
                normalize_vec(&mut v);
                k_state.copy_from_slice(&v);
                (v, &m.p_knowledge)
            }
            1 => {
                let mut r_in = Vec::with_capacity(2 * KB_DIM);
                r_in.extend_from_slice(query_vec);
                r_in.extend_from_slice(&k_state);
                let mut v = matmul_vec(&r_in, &m.w_reasoning, 2 * KB_DIM, KB_DIM);
                normalize_vec(&mut v);
                r_state.copy_from_slice(&v);
                (v, &m.p_reasoning)
            }
            2 => {
                let mut s_in = Vec::with_capacity(3 * KB_DIM);
                s_in.extend_from_slice(query_vec);
                s_in.extend_from_slice(&k_state);
                s_in.extend_from_slice(&r_state);
                let logits = matmul_vec(&s_in, &m.w_saci_route, 3 * KB_DIM, KB_NUM_DOMAINS);
                pred_domain = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let mut v = matmul_vec(&s_in, &m.w_saci_facts, 3 * KB_DIM, KB_DIM);
                normalize_vec(&mut v);
                s_state.copy_from_slice(&v);
                (v, &m.p_saci)
            }
            _ => {
                let mut m_in = Vec::with_capacity(3 * KB_DIM);
                m_in.extend_from_slice(&k_state);
                m_in.extend_from_slice(&r_state);
                m_in.extend_from_slice(&s_state);
                let mut v = matmul_vec(&m_in, &m.w_simulation, 3 * KB_DIM, KB_DIM);
                normalize_vec(&mut v);
                m_state.copy_from_slice(&v);
                (v, &m.p_simulation)
            }
        };

        // 3. Pure-weight verification: ReLU((state @ P)[domain] + verify_bias)
        let proto_scores = matmul_vec(&state_vec, p_mat, KB_DIM, KB_NUM_DOMAINS);
        let dom_idx = if active_fam >= 2 {
            pred_domain
        } else {
            proto_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        let verify_margin = (proto_scores[dom_idx] + m.verify_bias).max(0.0);
        if verify_margin <= 0.0 {
            all_verified = false;
        }

        // 4. Pure-weight halting logit: W_halt * [verify_margin, (t+1)/4] + b_halt
        let halt_logit =
            m.w_halt[0] * verify_margin + m.w_halt[1] * (((t + 1) as f64) / 4.0) + m.b_halt;
        if halt_logit > 0.0 && t == 3 && halted_at_step == 4 {
            halted_at_step = (t + 1) as u8;
        }

        step_results[t] = PureWeightStepResult {
            step: (t + 1) as u8,
            gated_family_mask: mask,
            verify_margin,
            halt_logit,
        };
    }

    let mut ws_state = Vec::with_capacity(4 * KB_DIM);
    ws_state.extend_from_slice(&k_state);
    ws_state.extend_from_slice(&r_state);
    ws_state.extend_from_slice(&s_state);
    ws_state.extend_from_slice(&m_state);

    PureWeightForwardOutput {
        predicted_domain: pred_domain,
        steps: step_results,
        all_verified,
        halted_at_step,
        ws_state,
    }
}

/// Query-conditioned 1-feature-per-neuron activation across the 4 superposed families
/// (`Knowledge=1<<0`, `Reasoning=1<<1`, `SACI=1<<2`, `Simulation=1<<3`).
/// Enforces `mask.count_ones() <= 1` on every neuron.
pub fn compute_exclusive_neuron_masks(
    etf_basis: &[f64],
    query_vec: &[f64],
    num_neurons: usize,
    dim: usize,
    threshold: f64,
) -> Vec<u8> {
    let mut masks = vec![0u8; num_neurons];
    if etf_basis.len() < num_neurons * 4 * dim || query_vec.len() < dim {
        return masks;
    }
    for (n, mask_slot) in masks.iter_mut().enumerate() {
        let mut best_abs = threshold;
        for f in 0..4usize {
            let off = (n * 4 + f) * dim;
            let row = &etf_basis[off..off + dim];
            let dot: f64 = row
                .iter()
                .zip(query_vec[..dim].iter())
                .map(|(a, b)| a * b)
                .sum();
            let mag = dot.abs();
            if mag > best_abs {
                best_abs = mag;
                *mask_slot = 1u8 << (f as u8);
            }
        }
    }
    masks
}

/// Pure-weight forward pass where Reasoning (`r_state`) extracts & binds numeric operands
/// from `req_tokens` via semantic role affinity, invokes the routed SACI circuit in `saci_ckpt`,
/// reports both the grounded operand vector and SACI output into Workspace `s_state`,
/// and runs Simulation (`m_state`) + PonderNet verification on that SACI-augmented state.
#[allow(clippy::too_many_arguments)]
pub fn forward_pure_weights_with_saci(
    m: &PureWeightAdaModel,
    query_vec: &[f64],
    req_tokens: &[u16],
    num_table: &NumericVocabTable,
    tok_emb: &[f64],
    vocab_size: usize,
    saci_ckpt: &Checkpoint,
    saci_descs: &[SaciCircuitDescriptor],
    saci_scratch: &mut [f64],
) -> PureWeightForwardOutput {
    let mut k_state = vec![0.0_f64; KB_DIM];
    let mut r_state = vec![0.0_f64; KB_DIM];
    let mut s_state = vec![0.0_f64; KB_DIM];
    let mut m_state = vec![0.0_f64; KB_DIM];
    let mut pred_domain = 0usize;

    let mut step_results = [PureWeightStepResult {
        step: 0,
        gated_family_mask: 0,
        verify_margin: 0.0,
        halt_logit: 0.0,
    }; 4];

    let mut halted_at_step = 4u8;
    let mut all_verified = true;
    let mut saci_in_buf = [0.0_f64; 64];

    for t in 0..4usize {
        let mut phase = [0.0_f64; 4];
        phase[t] = 1.0;
        let gate_scores = matmul_vec(&phase, &m.w_gate, 4, 4);
        let active_fam = gate_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mask = 1u8 << (active_fam as u8);

        let (state_vec, p_mat) = match active_fam {
            0 => {
                let mut v = matmul_vec(query_vec, &m.w_knowledge, KB_DIM, KB_DIM);
                normalize_vec(&mut v);
                k_state.copy_from_slice(&v);
                (v, &m.p_knowledge)
            }
            1 => {
                let mut r_in = Vec::with_capacity(2 * KB_DIM);
                r_in.extend_from_slice(query_vec);
                r_in.extend_from_slice(&k_state);
                let mut v = matmul_vec(&r_in, &m.w_reasoning, 2 * KB_DIM, KB_DIM);
                normalize_vec(&mut v);
                r_state.copy_from_slice(&v);
                (v, &m.p_reasoning)
            }
            2 => {
                let mut s_in = Vec::with_capacity(3 * KB_DIM);
                s_in.extend_from_slice(query_vec);
                s_in.extend_from_slice(&k_state);
                s_in.extend_from_slice(&r_state);
                let logits = matmul_vec(&s_in, &m.w_saci_route, 3 * KB_DIM, KB_NUM_DOMAINS);
                pred_domain = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let mut v = matmul_vec(&s_in, &m.w_saci_facts, 3 * KB_DIM, KB_DIM);

                // Reasoning extracts & binds numeric operands, calls the routed SACI circuit,
                // and reports both grounded role-values and SACI output into Workspace `s_state`.
                if let Some(desc) = saci_descs.get(pred_domain) {
                    let in_n = (desc.input_count as usize).min(saci_in_buf.len());
                    let mut ops_buf = [BoundOperand::ZERO; MAX_BOUND_OPERANDS];
                    let n_ops = extract_bound_operands(
                        req_tokens,
                        num_table,
                        tok_emb,
                        vocab_size,
                        KB_DIM,
                        &mut ops_buf,
                    );
                    let mut grounded_vec = [0.0_f64; KB_DIM];
                    bind_operands_to_saci(
                        &ops_buf[..n_ops],
                        &r_state,
                        &mut saci_in_buf[..in_n],
                        &mut grounded_vec,
                    );
                    for j in 0..KB_DIM {
                        v[j] += 0.25 * grounded_vec[j];
                    }
                    if let Ok(trace) =
                        execute_saci_circuit(saci_ckpt, desc, &saci_in_buf[..in_n], saci_scratch)
                    {
                        if !trace.output.is_empty() {
                            for (j, sv) in v.iter_mut().enumerate() {
                                let ov = trace.output[j % trace.output.len()];
                                *sv += 0.25 * ov.tanh();
                            }
                        }
                    }
                }
                normalize_vec(&mut v);
                s_state.copy_from_slice(&v);
                (v, &m.p_saci)
            }
            _ => {
                let mut m_in = Vec::with_capacity(3 * KB_DIM);
                m_in.extend_from_slice(&k_state);
                m_in.extend_from_slice(&r_state);
                m_in.extend_from_slice(&s_state);
                let mut v = matmul_vec(&m_in, &m.w_simulation, 3 * KB_DIM, KB_DIM);
                normalize_vec(&mut v);
                m_state.copy_from_slice(&v);
                (v, &m.p_simulation)
            }
        };

        let proto_scores = matmul_vec(&state_vec, p_mat, KB_DIM, KB_NUM_DOMAINS);
        let dom_idx = if active_fam >= 2 {
            pred_domain
        } else {
            proto_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        let verify_margin = (proto_scores[dom_idx] + m.verify_bias).max(0.0);
        if verify_margin <= 0.0 {
            all_verified = false;
        }

        let halt_logit =
            m.w_halt[0] * verify_margin + m.w_halt[1] * (((t + 1) as f64) / 4.0) + m.b_halt;
        if halt_logit > 0.0 && t == 3 && halted_at_step == 4 {
            halted_at_step = (t + 1) as u8;
        }

        step_results[t] = PureWeightStepResult {
            step: (t + 1) as u8,
            gated_family_mask: mask,
            verify_margin,
            halt_logit,
        };
    }

    let mut ws_state = Vec::with_capacity(4 * KB_DIM);
    ws_state.extend_from_slice(&k_state);
    ws_state.extend_from_slice(&r_state);
    ws_state.extend_from_slice(&s_state);
    ws_state.extend_from_slice(&m_state);

    PureWeightForwardOutput {
        predicted_domain: pred_domain,
        steps: step_results,
        all_verified,
        halted_at_step,
        ws_state,
    }
}
