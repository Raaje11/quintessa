//! Quintessa 5-Axis Verifier (`verifier.rs`).
//!
//! Independently verifies compiled superposition WSP networks against their symbolic ground-truth
//! engine across all five required axes:
//! 1. **Activation** — firing sparsity, forced-fire trigger rate, non-firing leakage, dead rules
//! 2. **Routing** — WSP exclusivity (`sum_f z_{i,f} <= 1`), active neuron ratio, route agreement
//! 3. **Accuracy** — random sparse, forced-fire (60%), and boundary (missing pos / present neg)
//! 4. **ETF Accuracy** — pairwise coherence vs Welch bound, feature recall/FPR, safety factor `z`
//! 5. **Inference Noise** — empirical vs theoretical `sqrt(k/d)` interference std, SNR (dB), max noise
//!
//! Zero string formatting inside this module (`GEMINI.md` §10-12); human-readable rendering
//! belongs in `observatory.rs`.

use crate::superposition_wsp::{
    FeatureSpace, RuleLayer, RuleSpec, SplitMix64, SuperpositionScratch,
};
use crate::sys::ErrorCode;
use crate::wsp::verify_exclusivity;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyStatus {
    Pass = 0,
    Fail = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ActivationMetrics {
    pub mean_active_rules: f64,
    pub activation_sparsity: f64,
    pub forced_fire_trigger_rate: f64,
    pub non_firing_leakage_rate: f64,
    pub dead_rule_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RoutingMetrics {
    pub exclusivity_violations: u32,
    pub active_neuron_ratio: f64,
    pub routing_accuracy: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AccuracyMetrics {
    pub random_sparse_error: f64,
    pub forced_fire_error: f64,
    pub boundary_missing_pos_error: f64,
    pub boundary_present_neg_error: f64,
    pub overall_agreement: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EtfAccuracyMetrics {
    pub max_pairwise_coherence: f64,
    pub rms_pairwise_coherence: f64,
    pub welch_bound: f64,
    pub present_feature_recall: f64,
    pub absent_feature_fpr: f64,
    pub safety_factor_z: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InferenceNoiseMetrics {
    pub measured_interference_std: f64,
    pub theoretical_interference_std: f64,
    pub snr_db: f64,
    pub max_abs_interference: f64,
    pub recommended_min_width: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VerifyThresholds {
    pub max_forced_fire_error: f64,
    pub max_boundary_error: f64,
    pub min_forced_fire_trigger_rate: f64,
    pub min_feature_recall: f64,
    pub max_absent_fpr: f64,
    pub min_snr_db: f64,
}

impl Default for VerifyThresholds {
    fn default() -> Self {
        Self {
            max_forced_fire_error: 0.02,        // <= 2% per-output disagreement
            max_boundary_error: 0.05,           // <= 5% on adversarial boundary cases
            min_forced_fire_trigger_rate: 0.95, // >= 95% forced rules actually fire
            min_feature_recall: 0.98,           // >= 98% present feature recall
            max_absent_fpr: 0.02,               // <= 2% absent feature false positive rate
            min_snr_db: 12.0,                   // >= 12 dB SNR
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VerifierReport {
    pub status: VerifyStatus,
    pub hidden_dim: u32,
    pub num_features: u32,
    pub num_rules: u32,
    pub num_outputs: u32,
    pub active_k: u32,
    pub trials: u32,
    pub activation: ActivationMetrics,
    pub routing: RoutingMetrics,
    pub accuracy: AccuracyMetrics,
    pub etf: EtfAccuracyMetrics,
    pub noise: InferenceNoiseMetrics,
}

/// Configuration for running the 5-axis Quintessa verification protocol.
#[derive(Clone, Copy, Debug)]
pub struct VerifyConfig {
    pub active_k: usize,
    pub trials: usize,
    pub forced_fire_prob: f64,
    pub readout_threshold: f64,
    pub seed: u64,
    pub coherence_sample_limit: usize,
    pub thresholds: VerifyThresholds,
}

impl Default for VerifyConfig {
    fn default() -> Self {
        Self {
            active_k: 3,
            trials: 1000,
            forced_fire_prob: 0.6,
            readout_threshold: 0.5,
            seed: 1,
            coherence_sample_limit: 256,
            thresholds: VerifyThresholds::default(),
        }
    }
}

/// Run the full 5-axis verification protocol on a compiled `RuleLayer` against its symbolic
/// `rules` ground truth. Optional `wsp_masks` are checked for strict WSP exclusivity.
pub fn verify_superposition_layer(
    in_space: &FeatureSpace,
    out_space: &FeatureSpace,
    layer: &RuleLayer,
    rules: &[RuleSpec],
    one_sentinel_feature: Option<u32>,
    wsp_masks: Option<&[u8]>,
    cfg: &VerifyConfig,
) -> Result<VerifierReport, ErrorCode> {
    if rules.len() != layer.num_rules || cfg.trials == 0 || cfg.active_k == 0 {
        return Err(ErrorCode::InvalidInput);
    }
    let m = in_space.num_features;
    let num_outputs = out_space.num_features;
    let k = cfg.active_k.min(m);

    let mut rng = SplitMix64::new(cfg.seed);
    let mut scratch = SuperpositionScratch::new(in_space.dim.max(out_space.dim), layer.num_rules);
    let mut active_mask = vec![false; m];
    let mut active_pairs: Vec<(u32, f64)> = Vec::with_capacity(k + 8);
    let mut gt_outputs = vec![false; num_outputs];

    // -----------------------------------------------------------------------
    // 1. ETF Coherence + Feature Readout & Inference Noise Measurement
    // -----------------------------------------------------------------------
    let (max_coh, rms_coh) = in_space.pairwise_coherence(cfg.coherence_sample_limit);
    let welch = in_space.welch_bound();

    let noise_trials = cfg.trials.min(400).max(50);
    let absent_probes_per_trial = 16usize.min(m.saturating_sub(k).max(1));
    let mut present_hits = 0u64;
    let mut present_total = 0u64;
    let mut absent_fps = 0u64;
    let mut absent_total = 0u64;
    let mut noise_sq_sum = 0.0_f64;
    let mut max_abs_noise = 0.0_f64;

    for _ in 0..noise_trials {
        active_mask.fill(false);
        active_pairs.clear();
        while active_pairs.len() < k {
            let f = rng.next_usize(m);
            if !active_mask[f] {
                active_mask[f] = true;
                active_pairs.push((f as u32, 1.0));
            }
        }
        in_space.write_into(&active_pairs, &mut scratch.hidden_a)?;

        // Present feature recall
        for &(f_idx, _) in &active_pairs {
            let d = in_space.dot(&scratch.hidden_a, f_idx)?;
            if d > cfg.readout_threshold {
                present_hits += 1;
            }
            present_total += 1;
        }
        // Absent feature interference & FPR
        for _ in 0..absent_probes_per_trial {
            let mut f = rng.next_usize(m);
            if active_mask[f] {
                f = (f + 1) % m;
            }
            if !active_mask[f] {
                let d = in_space.dot(&scratch.hidden_a, f as u32)?;
                let ad = d.abs();
                if ad > max_abs_noise {
                    max_abs_noise = ad;
                }
                noise_sq_sum += d * d;
                if d > cfg.readout_threshold {
                    absent_fps += 1;
                }
                absent_total += 1;
            }
        }
    }

    let present_feature_recall = if present_total > 0 {
        (present_hits as f64) / (present_total as f64)
    } else {
        1.0
    };
    let absent_feature_fpr = if absent_total > 0 {
        (absent_fps as f64) / (absent_total as f64)
    } else {
        0.0
    };
    let measured_std = if absent_total > 0 {
        (noise_sq_sum / (absent_total as f64)).sqrt()
    } else {
        0.0
    };
    let theoretical_std = ((k as f64) / (in_space.dim as f64)).sqrt();
    let snr_db = if measured_std > 1e-12 {
        10.0 * (1.0 / (measured_std * measured_std)).log10()
    } else {
        120.0
    };

    let mean_lits = (rules.iter().map(|r| r.literals.len()).sum::<usize>() as f64)
        / (rules.len().max(1) as f64);
    let safety_factor_z = 0.5 * ((in_space.dim as f64) / (mean_lits * (k as f64))).sqrt();

    // Recommended minimum power-of-two width for z >= 3.0 (sub-0.3% interference tail)
    let min_d_needed = (36.0 * mean_lits * (k as f64)).ceil() as u32;
    let recommended_min_width = [256u32, 512, 1024, 2048, 4096]
        .iter()
        .copied()
        .find(|&d| d >= min_d_needed)
        .unwrap_or(4096);

    // -----------------------------------------------------------------------
    // 2. Random Sparse & Forced-Fire Trials (Accuracy + Activation + Routing)
    // -----------------------------------------------------------------------
    let mut rand_bad = 0u64;
    let mut rand_tot = 0u64;
    let mut forced_bad = 0u64;
    let mut forced_tot = 0u64;
    let mut exact_route_hits = 0u64;
    let mut total_active_rules = 0u64;
    let mut non_firing_leaks = 0u64;
    let mut non_firing_total = 0u64;

    for _ in 0..cfg.trials {
        active_mask.fill(false);
        active_pairs.clear();
        if let Some(s_idx) = one_sentinel_feature {
            active_mask[s_idx as usize] = true;
            active_pairs.push((s_idx, 1.0));
        }

        let target_base = k + usize::from(one_sentinel_feature.is_some());
        while active_pairs.len() < target_base.min(m) {
            let f = rng.next_usize(m);
            if !active_mask[f] {
                active_mask[f] = true;
                active_pairs.push((f as u32, 1.0));
            }
        }

        let is_forced = rng.next_f64_open() < cfg.forced_fire_prob;
        if is_forced {
            let r_idx = rng.next_usize(rules.len());
            for lit in &rules[r_idx].literals {
                let fi = lit.feature as usize;
                if lit.sign > 0 && !active_mask[fi] {
                    active_mask[fi] = true;
                    active_pairs.push((lit.feature, 1.0));
                } else if lit.sign < 0 && active_mask[fi] {
                    active_mask[fi] = false;
                }
            }
            active_pairs.retain(|&(f_idx, _)| active_mask[f_idx as usize]);
        }

        in_space.write_into(&active_pairs, &mut scratch.hidden_a)?;
        let active_r = layer.forward_into(
            &scratch.hidden_a,
            &mut scratch.rule_acts,
            &mut scratch.hidden_b,
        )?;
        total_active_rules += active_r as u64;

        gt_outputs.fill(false);
        for (r_idx, rule) in rules.iter().enumerate() {
            let gt_fired = rule.eval_ground_truth(&active_mask);
            if gt_fired {
                gt_outputs[rule.output as usize] = true;
            } else {
                non_firing_total += 1;
                if scratch.rule_acts[r_idx] > 0.0 {
                    non_firing_leaks += 1;
                }
            }
        }

        let mut all_match = true;
        for (c, &want) in gt_outputs.iter().enumerate() {
            let pred = out_space.read_bool(&scratch.hidden_b, c as u32, cfg.readout_threshold)?;
            let err = u64::from(pred != want);
            if err != 0 {
                all_match = false;
            }
            if is_forced {
                forced_bad += err;
                forced_tot += 1;
            } else {
                rand_bad += err;
                rand_tot += 1;
            }
        }
        if all_match {
            exact_route_hits += 1;
        }
    }

    // -----------------------------------------------------------------------
    // 3. Per-Rule Boundary & Forced-Fire Trigger Verification (§7.3)
    // -----------------------------------------------------------------------
    let mut forced_triggers = 0u64;
    let mut dead_rules = 0u32;
    let mut bnd_miss_pos_bad = 0u64;
    let mut bnd_miss_pos_tot = 0u64;
    let mut bnd_pres_neg_bad = 0u64;
    let mut bnd_pres_neg_tot = 0u64;

    for (r_idx, rule) in rules.iter().enumerate() {
        // 3a. Exact forced-fire check for rule r_idx
        active_mask.fill(false);
        active_pairs.clear();
        if let Some(s_idx) = one_sentinel_feature {
            active_mask[s_idx as usize] = true;
            active_pairs.push((s_idx, 1.0));
        }
        for lit in &rule.literals {
            if lit.sign > 0 && !active_mask[lit.feature as usize] {
                active_mask[lit.feature as usize] = true;
                active_pairs.push((lit.feature, 1.0));
            }
        }
        in_space.write_into(&active_pairs, &mut scratch.hidden_a)?;
        layer.forward_into(
            &scratch.hidden_a,
            &mut scratch.rule_acts,
            &mut scratch.hidden_b,
        )?;
        if scratch.rule_acts[r_idx] > 0.0 {
            forced_triggers += 1;
        } else {
            dead_rules += 1;
        }

        // 3b. Boundary: exactly one positive literal missing
        if let Some(pos_lit) = rule.literals.iter().find(|l| l.sign > 0) {
            active_mask[pos_lit.feature as usize] = false;
            active_pairs.retain(|&(f_idx, _)| active_mask[f_idx as usize]);
            in_space.write_into(&active_pairs, &mut scratch.hidden_a)?;
            layer.forward_into(
                &scratch.hidden_a,
                &mut scratch.rule_acts,
                &mut scratch.hidden_b,
            )?;
            bnd_miss_pos_tot += 1;
            if scratch.rule_acts[r_idx] > 0.0 {
                bnd_miss_pos_bad += 1;
            }
            // Restore
            active_mask[pos_lit.feature as usize] = true;
            active_pairs.push((pos_lit.feature, 1.0));
        }

        // 3c. Boundary: one negated literal present
        if let Some(neg_lit) = rule.literals.iter().find(|l| l.sign < 0) {
            active_mask[neg_lit.feature as usize] = true;
            active_pairs.push((neg_lit.feature, 1.0));
            in_space.write_into(&active_pairs, &mut scratch.hidden_a)?;
            layer.forward_into(
                &scratch.hidden_a,
                &mut scratch.rule_acts,
                &mut scratch.hidden_b,
            )?;
            bnd_pres_neg_tot += 1;
            if scratch.rule_acts[r_idx] > 0.0 {
                bnd_pres_neg_bad += 1;
            }
        }
    }

    // -----------------------------------------------------------------------
    // 4. Routing Exclusivity Check
    // -----------------------------------------------------------------------
    let (exclusivity_violations, active_neuron_ratio) = if let Some(masks) = wsp_masks {
        let viols = if verify_exclusivity(masks).is_ok() {
            0
        } else {
            masks.iter().filter(|&&m| m.count_ones() > 1).count() as u32
        };
        let active_n = masks.iter().filter(|&&m| m != 0).count();
        let ratio = if !masks.is_empty() {
            (active_n as f64) / (masks.len() as f64)
        } else {
            0.0
        };
        (viols, ratio)
    } else {
        (
            0,
            (total_active_rules as f64) / ((cfg.trials * layer.num_rules).max(1) as f64),
        )
    };

    let mean_active_rules = (total_active_rules as f64) / (cfg.trials as f64);
    let activation_sparsity = 1.0 - (mean_active_rules / (layer.num_rules as f64)).clamp(0.0, 1.0);
    let forced_fire_trigger_rate = (forced_triggers as f64) / (rules.len() as f64);
    let non_firing_leakage_rate = if non_firing_total > 0 {
        (non_firing_leaks as f64) / (non_firing_total as f64)
    } else {
        0.0
    };

    let random_sparse_error = if rand_tot > 0 {
        (rand_bad as f64) / (rand_tot as f64)
    } else {
        0.0
    };
    let forced_fire_error = if forced_tot > 0 {
        (forced_bad as f64) / (forced_tot as f64)
    } else {
        0.0
    };
    let boundary_missing_pos_error = if bnd_miss_pos_tot > 0 {
        (bnd_miss_pos_bad as f64) / (bnd_miss_pos_tot as f64)
    } else {
        0.0
    };
    let boundary_present_neg_error = if bnd_pres_neg_tot > 0 {
        (bnd_pres_neg_bad as f64) / (bnd_pres_neg_tot as f64)
    } else {
        0.0
    };
    let total_eval = rand_tot + forced_tot;
    let overall_agreement = if total_eval > 0 {
        1.0 - ((rand_bad + forced_bad) as f64) / (total_eval as f64)
    } else {
        1.0
    };
    let routing_accuracy = (exact_route_hits as f64) / (cfg.trials as f64);

    let thr = cfg.thresholds;
    let passed = exclusivity_violations == 0
        && forced_fire_error <= thr.max_forced_fire_error
        && boundary_missing_pos_error <= thr.max_boundary_error
        && boundary_present_neg_error <= thr.max_boundary_error
        && forced_fire_trigger_rate >= thr.min_forced_fire_trigger_rate
        && present_feature_recall >= thr.min_feature_recall
        && absent_feature_fpr <= thr.max_absent_fpr
        && snr_db >= thr.min_snr_db;

    Ok(VerifierReport {
        status: if passed {
            VerifyStatus::Pass
        } else {
            VerifyStatus::Fail
        },
        hidden_dim: in_space.dim as u32,
        num_features: m as u32,
        num_rules: layer.num_rules as u32,
        num_outputs: num_outputs as u32,
        active_k: k as u32,
        trials: cfg.trials as u32,
        activation: ActivationMetrics {
            mean_active_rules,
            activation_sparsity,
            forced_fire_trigger_rate,
            non_firing_leakage_rate,
            dead_rule_count: dead_rules,
        },
        routing: RoutingMetrics {
            exclusivity_violations,
            active_neuron_ratio,
            routing_accuracy,
        },
        accuracy: AccuracyMetrics {
            random_sparse_error,
            forced_fire_error,
            boundary_missing_pos_error,
            boundary_present_neg_error,
            overall_agreement,
        },
        etf: EtfAccuracyMetrics {
            max_pairwise_coherence: max_coh,
            rms_pairwise_coherence: rms_coh,
            welch_bound: welch,
            present_feature_recall,
            absent_feature_fpr,
            safety_factor_z,
        },
        noise: InferenceNoiseMetrics {
            measured_interference_std: measured_std,
            theoretical_interference_std: theoretical_std,
            snr_db,
            max_abs_interference: max_abs_noise,
            recommended_min_width,
        },
    })
}
