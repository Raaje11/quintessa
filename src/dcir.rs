//! Domain Compilation IR (DCIR) & Adaptive PonderNet Reasoning Controller (Milestone 2).

use crate::sys::{AdaComponent, DomainConceptId, ErrorCode, PonderAction};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DomainOpKind {
    ConceptLookup = 0,
    Division = 1,
    Multiplication = 2,
    LinearMap = 3,
    Comparison = 4,
    StateUpdate = 5,
    InvariantCheck = 6,
}

#[derive(Clone, Debug)]
pub struct DomainExample {
    pub concept_id: DomainConceptId,
    pub features: Vec<f64>,
    pub target: Vec<f64>,
    pub protected: bool,
}

#[derive(Clone, Debug)]
pub struct DomainDataset {
    pub domain_id: u32,
    pub feature_dim: usize,
    pub target_dim: usize,
    pub examples: Vec<DomainExample>,
}

#[derive(Clone, Copy, Debug)]
pub struct ReasoningStep {
    pub step_id: u16,
    pub op: DomainOpKind,
    pub target_component: AdaComponent,
    pub halting_prior: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ConstraintSpec {
    pub constraint_id: u16,
    pub max_allowed_violation: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptivePonderConfig {
    pub min_steps: u16,
    pub max_steps: u16,
    pub halt_threshold: f64,
    pub halt_tolerance: f64,
    pub lambda_prior: f64,
    pub kl_beta: f64,
}

impl Default for AdaptivePonderConfig {
    fn default() -> Self {
        Self {
            min_steps: 1,
            max_steps: 8,
            halt_threshold: 0.75,
            halt_tolerance: 1e-4,
            lambda_prior: 0.3,
            kl_beta: 0.01,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PonderState {
    pub step: u16,
    pub survival_prob: f64,
    pub cumulative_halt_prob: f64,
    pub kl_regularization: f64,
}

impl PonderState {
    pub fn new() -> Self {
        Self {
            step: 0,
            survival_prob: 1.0,
            cumulative_halt_prob: 0.0,
            kl_regularization: 0.0,
        }
    }
}

impl Default for PonderState {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Execute one Adaptive PonderNet reasoning step:
/// Computes `p_n = lambda_n * prod_{j<n}(1 - lambda_j)`, updates KL divergence
/// against the geometric prior `P_G(lambda_prior)`, and decides whether to
/// `Continue`, `Halt` early, or `EscalateToSaci`.
pub fn step_ponder(
    state: &mut PonderState,
    halting_logit: f64,
    residual: f64,
    cfg: &AdaptivePonderConfig,
) -> (PonderAction, f64) {
    state.step += 1;
    let lambda_n = sigmoid(halting_logit).clamp(1e-6, 1.0 - 1e-6);

    let p_n = if state.step >= cfg.max_steps {
        state.survival_prob
    } else {
        lambda_n * state.survival_prob
    };
    state.cumulative_halt_prob += p_n;
    state.survival_prob *= 1.0 - lambda_n;

    // Geometric prior: P_G(n) = (1 - lambda_p)^(n-1) * lambda_p
    let lp = cfg.lambda_prior.clamp(1e-6, 1.0 - 1e-6);
    let prior_n = (1.0 - lp).powi((state.step - 1) as i32) * lp;
    if p_n > 1e-12 && prior_n > 1e-12 {
        state.kl_regularization += p_n * (p_n / prior_n).ln();
    }

    if state.step < cfg.min_steps {
        return (PonderAction::Continue, p_n);
    }
    if residual <= cfg.halt_tolerance && state.cumulative_halt_prob >= cfg.halt_threshold {
        return (PonderAction::Halt, p_n);
    }
    if state.step >= cfg.max_steps {
        if residual > cfg.halt_tolerance {
            return (PonderAction::EscalateToSaci, p_n);
        }
        return (PonderAction::Halt, p_n);
    }

    (PonderAction::Continue, p_n)
}

#[derive(Clone, Debug)]
pub struct ReasoningSpecification {
    pub steps: Vec<ReasoningStep>,
    pub constraints: Vec<ConstraintSpec>,
    pub ponder: AdaptivePonderConfig,
}

#[derive(Clone, Debug)]
pub struct DomainCompilationIr {
    pub domain_id: u32,
    pub feature_dim: usize,
    pub target_dim: usize,
    pub ops: Vec<DomainOpKind>,
    pub reasoning_steps: Vec<ReasoningStep>,
    pub constraints: Vec<ConstraintSpec>,
    pub ponder: AdaptivePonderConfig,
    pub trainable_examples: Vec<DomainExample>,
    pub protected_examples: Vec<DomainExample>,
}

/// Compile raw `DomainDataset` + `ReasoningSpecification` into validated `DomainCompilationIr`.
pub fn compile_dataset(
    dataset: &DomainDataset,
    reasoning: &ReasoningSpecification,
) -> Result<DomainCompilationIr, ErrorCode> {
    if dataset.feature_dim == 0 || dataset.target_dim == 0 || dataset.examples.is_empty() {
        return Err(ErrorCode::InvalidInput);
    }
    if reasoning.steps.is_empty() || reasoning.ponder.max_steps == 0 {
        return Err(ErrorCode::InvalidInput);
    }

    let mut trainable_examples = Vec::new();
    let mut protected_examples = Vec::new();

    for ex in &dataset.examples {
        if ex.features.len() != dataset.feature_dim || ex.target.len() != dataset.target_dim {
            return Err(ErrorCode::ShapeMismatch);
        }
        if ex
            .features
            .iter()
            .chain(ex.target.iter())
            .any(|v| !v.is_finite())
        {
            return Err(ErrorCode::InvalidInput);
        }
        if ex.protected {
            protected_examples.push(ex.clone());
        } else {
            trainable_examples.push(ex.clone());
        }
    }

    let ops = reasoning.steps.iter().map(|s| s.op).collect();

    Ok(DomainCompilationIr {
        domain_id: dataset.domain_id,
        feature_dim: dataset.feature_dim,
        target_dim: dataset.target_dim,
        ops,
        reasoning_steps: reasoning.steps.clone(),
        constraints: reasoning.constraints.clone(),
        ponder: reasoning.ponder,
        trainable_examples,
        protected_examples,
    })
}
