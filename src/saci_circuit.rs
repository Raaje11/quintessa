//! Program-to-Weight Compiled In-Neuron SACI Circuit (`SOLVER_07`: Forensic Timeline Audit).
//!
//! All arithmetic (signed delta, temporal impossibility deficit, anomaly indicator,
//! and invalid-skew guard) is compiled directly into ReLU MLP weight matrices `(W1, b1, W2)`
//! stored in the checkpoint's `SaciCircuits` tensor (`TensorId(5)`).
//!
//! Removing or replacing these weights with placeholders causes the circuit to fail
//! against the independent reference implementation—proving the computation lives in
//! the neural weights rather than an external Rust solver.

use crate::checkpoint::Checkpoint;
use crate::sys::{ErrorCode, NeuronGroupId, TensorId};

pub const FORENSIC_IN_DIM: usize = 4; // [child_start, parent_start, max_skew, bias=1.0]
pub const FORENSIC_HIDDEN_DIM: usize = 7;
pub const FORENSIC_OUT_DIM: usize = 4; // [signed_delta, deficit, is_anomaly, is_invalid_skew]

/// Total weight elements stored in `SaciCircuits` for `SOLVER_07`:
/// `W1 (7*4=28) + b1 (7) + W2 (4*7=28) = 63` `f64` parameters.
pub const FORENSIC_CIRCUIT_PARAMS: usize = FORENSIC_HIDDEN_DIM * FORENSIC_IN_DIM
    + FORENSIC_HIDDEN_DIM
    + FORENSIC_OUT_DIM * FORENSIC_HIDDEN_DIM;

const STEEP_K: f64 = 1.0e6;

/// Compile the Forensic Timeline Consistency algorithm into exact ReLU MLP weights
/// `(W1, b1, W2)` using Program-to-Weight synthesis (zero gradient descent).
pub fn synthesize_forensic_timeline_weights() -> Vec<f64> {
    let mut w = vec![0.0_f64; FORENSIC_CIRCUIT_PARAMS];
    let (w1, rest) = w.split_at_mut(FORENSIC_HIDDEN_DIM * FORENSIC_IN_DIM);
    let (b1, w2) = rest.split_at_mut(FORENSIC_HIDDEN_DIM);

    let mut set_w1 = |h: usize, c: usize, val: f64| {
        w1[h * FORENSIC_IN_DIM + c] = val;
    };
    let mut set_b1 = |h: usize, val: f64| {
        b1[h] = val;
    };
    let mut set_w2 = |o: usize, h: usize, val: f64| {
        w2[o * FORENSIC_HIDDEN_DIM + h] = val;
    };

    // h0 = ReLU(child - parent)
    set_w1(0, 0, 1.0);
    set_w1(0, 1, -1.0);

    // h1 = ReLU(parent - child)
    set_w1(1, 0, -1.0);
    set_w1(1, 1, 1.0);

    // h2 = ReLU(parent - child - skew) = deficit
    set_w1(2, 0, -1.0);
    set_w1(2, 1, 1.0);
    set_w1(2, 2, -1.0);

    // h3 = ReLU(k * (parent - child - skew))
    set_w1(3, 0, -STEEP_K);
    set_w1(3, 1, STEEP_K);
    set_w1(3, 2, -STEEP_K);

    // h4 = ReLU(k * (parent - child - skew) - 1.0)
    set_w1(4, 0, -STEEP_K);
    set_w1(4, 1, STEEP_K);
    set_w1(4, 2, -STEEP_K);
    set_b1(4, -1.0);

    // h5 = ReLU(-k * skew)
    set_w1(5, 2, -STEEP_K);

    // h6 = ReLU(-k * skew - 1.0)
    set_w1(6, 2, -STEEP_K);
    set_b1(6, -1.0);

    // Output 0: signed_delta = h0 - h1
    set_w2(0, 0, 1.0);
    set_w2(0, 1, -1.0);

    // Output 1: deficit = h2
    set_w2(1, 2, 1.0);

    // Output 2: is_anomaly = h3 - h4  (1.0 if deficit >= 1/k else 0.0)
    set_w2(2, 3, 1.0);
    set_w2(2, 4, -1.0);

    // Output 3: is_invalid_skew = h5 - h6  (1.0 if skew <= -1/k else 0.0)
    set_w2(3, 5, 1.0);
    set_w2(3, 6, -1.0);

    w
}

#[derive(Clone, Debug)]
pub struct CircuitExecutionTrace {
    pub circuit_id: u16,
    pub neuron_group: NeuronGroupId,
    pub tensor_id: TensorId,
    pub weights_read: usize,
    pub hidden_activations: [f64; FORENSIC_HIDDEN_DIM],
    pub output: [f64; FORENSIC_OUT_DIM],
}

/// Execute `SOLVER_07` strictly by reading the compiled weights from `ckpt.tensor_slice(tensor_id)`
/// and running the 2-layer ReLU forward pass. No Rust conditional logic computes the result.
pub fn execute_forensic_saci_circuit(
    ckpt: &Checkpoint,
    tensor_id: TensorId,
    neuron_group: NeuronGroupId,
    child_start: f64,
    parent_start: f64,
    max_skew: f64,
) -> Result<CircuitExecutionTrace, ErrorCode> {
    if !child_start.is_finite() || !parent_start.is_finite() || !max_skew.is_finite() {
        return Err(ErrorCode::InvalidInput);
    }

    let w = ckpt
        .tensor_slice(tensor_id)
        .ok_or(ErrorCode::ShapeMismatch)?;
    if w.len() < FORENSIC_CIRCUIT_PARAMS {
        return Err(ErrorCode::ShapeMismatch);
    }

    let w1 = &w[0..FORENSIC_HIDDEN_DIM * FORENSIC_IN_DIM];
    let b1 = &w[FORENSIC_HIDDEN_DIM * FORENSIC_IN_DIM
        ..FORENSIC_HIDDEN_DIM * FORENSIC_IN_DIM + FORENSIC_HIDDEN_DIM];
    let w2 =
        &w[FORENSIC_HIDDEN_DIM * FORENSIC_IN_DIM + FORENSIC_HIDDEN_DIM..FORENSIC_CIRCUIT_PARAMS];

    let x = [child_start, parent_start, max_skew, 1.0];
    let mut hidden = [0.0_f64; FORENSIC_HIDDEN_DIM];
    for h in 0..FORENSIC_HIDDEN_DIM {
        let mut acc = b1[h];
        for c in 0..FORENSIC_IN_DIM {
            acc += w1[h * FORENSIC_IN_DIM + c] * x[c];
        }
        hidden[h] = acc.max(0.0); // ReLU
    }

    let mut output = [0.0_f64; FORENSIC_OUT_DIM];
    for o in 0..FORENSIC_OUT_DIM {
        let mut acc = 0.0;
        for h in 0..FORENSIC_HIDDEN_DIM {
            acc += w2[o * FORENSIC_HIDDEN_DIM + h] * hidden[h];
        }
        output[o] = acc;
    }

    Ok(CircuitExecutionTrace {
        circuit_id: 7,
        neuron_group,
        tensor_id,
        weights_read: FORENSIC_CIRCUIT_PARAMS,
        hidden_activations: hidden,
        output,
    })
}

/// Independent reference implementation (used strictly by the verifier, never by the circuit).
pub fn reference_forensic_timeline(
    child_start: f64,
    parent_start: f64,
    max_skew: f64,
) -> [f64; FORENSIC_OUT_DIM] {
    let signed_delta = child_start - parent_start;
    let deficit = (parent_start - child_start - max_skew).max(0.0);
    let is_anomaly = if deficit > 0.0 { 1.0 } else { 0.0 };
    let is_invalid_skew = if max_skew < 0.0 { 1.0 } else { 0.0 };
    [signed_delta, deficit, is_anomaly, is_invalid_skew]
}

// ---------------------------------------------------------------------------
// Generic SACI Circuit Compiler & Mandatory Ablation Acceptance Gate
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaciCircuitTopology {
    TwoLayerReluMlp = 0,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CircuitSlot {
    pub circuit_id: u16,
    pub tensor_id: TensorId,
    pub offset: usize,
    pub len: usize,
}

#[derive(Clone, Debug)]
pub struct SaciCircuitProgram {
    pub circuit_id: u16,
    pub topology: SaciCircuitTopology,
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub out_dim: usize,
    pub weights: Vec<f64>,
}

impl SaciCircuitProgram {
    pub fn forensic_timeline_solver07() -> Self {
        Self {
            circuit_id: 7,
            topology: SaciCircuitTopology::TwoLayerReluMlp,
            in_dim: FORENSIC_IN_DIM,
            hidden_dim: FORENSIC_HIDDEN_DIM,
            out_dim: FORENSIC_OUT_DIM,
            weights: synthesize_forensic_timeline_weights(),
        }
    }

    pub fn expected_param_count(&self) -> usize {
        match self.topology {
            SaciCircuitTopology::TwoLayerReluMlp => {
                self.hidden_dim * self.in_dim + self.hidden_dim + self.out_dim * self.hidden_dim
            }
        }
    }
}

/// Validate that a set of `CircuitSlot`s are pairwise disjoint within each `TensorId`.
pub fn validate_circuit_isolation(slots: &[CircuitSlot]) -> Result<(), ErrorCode> {
    for (i, a) in slots.iter().enumerate() {
        for b in &slots[i + 1..] {
            if a.tensor_id == b.tensor_id {
                let a_end = a.offset + a.len;
                let b_end = b.offset + b.len;
                if a.offset < b_end && b.offset < a_end {
                    return Err(ErrorCode::ProtectedInvariant);
                }
            }
        }
    }
    Ok(())
}

/// Execute a generic `SaciCircuitProgram` strictly from its isolated `CircuitSlot`
/// inside `ckpt`, verifying `routed_circuit_id == slot.circuit_id`.
pub fn execute_isolated_saci_slot(
    ckpt: &Checkpoint,
    slot: &CircuitSlot,
    routed_circuit_id: u16,
    program: &SaciCircuitProgram,
    input: &[f64],
) -> Result<Vec<f64>, ErrorCode> {
    let topology = program.topology;
    let in_dim = program.in_dim;
    let hidden_dim = program.hidden_dim;
    let out_dim = program.out_dim;
    if routed_circuit_id != slot.circuit_id {
        return Err(ErrorCode::UnauthorizedWrite);
    }
    if input.len() != in_dim || input.iter().any(|v| !v.is_finite()) {
        return Err(ErrorCode::InvalidInput);
    }
    let tensor = ckpt
        .tensor_slice(slot.tensor_id)
        .ok_or(ErrorCode::ShapeMismatch)?;
    if slot.offset + slot.len > tensor.len() {
        return Err(ErrorCode::ShapeMismatch);
    }
    let w = &tensor[slot.offset..slot.offset + slot.len];

    match topology {
        SaciCircuitTopology::TwoLayerReluMlp => {
            let expected = hidden_dim * in_dim + hidden_dim + out_dim * hidden_dim;
            if w.len() != expected {
                return Err(ErrorCode::ShapeMismatch);
            }
            let w1 = &w[0..hidden_dim * in_dim];
            let b1 = &w[hidden_dim * in_dim..hidden_dim * in_dim + hidden_dim];
            let w2 = &w[hidden_dim * in_dim + hidden_dim..expected];

            let mut hidden = vec![0.0_f64; hidden_dim];
            for h in 0..hidden_dim {
                let mut acc = b1[h];
                for c in 0..in_dim {
                    acc += w1[h * in_dim + c] * input[c];
                }
                if !acc.is_finite() {
                    return Err(ErrorCode::InvalidInput);
                }
                hidden[h] = acc.max(0.0);
            }

            let mut out = vec![0.0_f64; out_dim];
            for o in 0..out_dim {
                let mut acc = 0.0;
                for h in 0..hidden_dim {
                    acc += w2[o * hidden_dim + h] * hidden[h];
                }
                if !acc.is_finite() {
                    return Err(ErrorCode::InvalidInput);
                }
                out[o] = acc;
            }
            Ok(out)
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CircuitAcceptanceReport {
    pub circuit_id: u16,
    pub cases_verified: usize,
    pub max_abs_error: f64,
    pub ablation_max_error: f64,
    pub passed: bool,
}

/// Generic `SaciCircuitProgram -> Checkpoint -> NeuralExecution -> ReferenceVerification + Mandatory Ablation` gate.
pub fn verify_saci_circuit_with_ablation<F>(
    program: &SaciCircuitProgram,
    ckpt: &Checkpoint,
    slot: &CircuitSlot,
    routed_circuit_id: u16,
    test_inputs: &[Vec<f64>],
    reference_fn: F,
    tolerance: f64,
) -> Result<CircuitAcceptanceReport, ErrorCode>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if program.circuit_id != slot.circuit_id
        || program.weights.len() != program.expected_param_count()
        || slot.len != program.weights.len()
        || test_inputs.is_empty()
    {
        return Err(ErrorCode::InvalidInput);
    }

    let mut max_abs_error = 0.0_f64;
    for inp in test_inputs {
        let got = execute_isolated_saci_slot(ckpt, slot, routed_circuit_id, program, inp)?;
        let exp = reference_fn(inp);
        if got.len() != exp.len() {
            return Err(ErrorCode::ShapeMismatch);
        }
        for (g, e) in got.iter().zip(exp.iter()) {
            let err = (g - e).abs();
            max_abs_error = max_abs_error.max(err);
            if err > tolerance {
                return Err(ErrorCode::VerificationFailed);
            }
        }
    }

    // Mandatory per-circuit ablation gate: replace ONLY `slot` weights with `cos()` placeholders
    let mut ablated_ckpt = ckpt.clone();
    let desc = ablated_ckpt
        .descriptor(slot.tensor_id)
        .ok_or(ErrorCode::ShapeMismatch)?
        .to_owned();
    for i in 0..slot.len {
        ablated_ckpt.data[desc.offset + slot.offset + i] = ((i as f64) * 0.029).cos();
    }

    let mut ablation_max_error = 0.0_f64;
    for inp in test_inputs {
        let got = execute_isolated_saci_slot(&ablated_ckpt, slot, routed_circuit_id, program, inp)?;
        let exp = reference_fn(inp);
        for (g, e) in got.iter().zip(exp.iter()) {
            ablation_max_error = ablation_max_error.max((g - e).abs());
        }
    }

    if ablation_max_error <= 1.0 {
        return Err(ErrorCode::VerificationFailed);
    }

    Ok(CircuitAcceptanceReport {
        circuit_id: program.circuit_id,
        cases_verified: test_inputs.len(),
        max_abs_error,
        ablation_max_error,
        passed: true,
    })
}
