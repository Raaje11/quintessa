//! Variable SACI Program-to-Weight Compiler (`.saci` -> `.sacib` -> `Checkpoint` -> `execute_saci_circuit`).
//!
//! Zero solver-specific runtime code: all algorithms are compiled from typed SSA
//! graphs into variable-topology neural weights, serialized into portable `.sacib`
//! binaries, loaded atomically into `Checkpoint` (`TensorId(5)` arena), and executed
//! via generic neural primitives using borrowed weight slices and preallocated scratch.

use crate::checkpoint::{Checkpoint, TensorDescriptor};
use crate::sys::{AdaComponent, ErrorCode, TensorId};

pub const SACIB_MAGIC: &[u8; 8] = b"SACIB\0\0\0";
pub const SACIB_VERSION: u32 = 1;
pub const SACIB_COMPILER_VERSION: u32 = 1;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaciDataType {
    F64 = 0,
    F32 = 1,
    Bool = 2,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericPrecision {
    F64 = 0,
    F32 = 1,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyKind {
    Linear = 0,
    MultiLayerRelu = 1,
    Sparse = 2,
    MultiBranch = 3,
    Recurrent = 4,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaciOp {
    Input = 0,
    Const = 1,
    Add = 2,
    Sub = 3,
    MulConst = 4,
    Relu = 5,
    Abs = 6,
    CmpLt = 7,
    CmpLe = 8,
    Max = 9,
    Min = 10,
    ReduceSum = 11,
    StateTransition = 12,
}

#[derive(Clone, Debug)]
pub struct SaciNode {
    pub id: u32,
    pub op: SaciOp,
    pub inputs: Vec<u32>,
    pub const_val: f64,
    pub output_type: SaciDataType,
}

#[derive(Clone, Debug)]
pub struct SaciProgram {
    pub circuit_id: u32,
    pub input_count: usize,
    pub output_nodes: Vec<u32>,
    pub nodes: Vec<SaciNode>,
    pub max_neurons: usize,
    pub max_parameters: usize,
    pub max_activation_bytes: usize,
    pub absolute_tolerance: f64,
    pub steepness_k: f64,
}

#[derive(Clone, Debug)]
pub struct SaciCircuitDescriptor {
    pub circuit_id: u32,
    pub topology: TopologyKind,
    pub layer_dims: Vec<u32>,
    pub neuron_ids: Vec<u32>,
    pub weight_offset: u64,
    pub weight_count: u64,
    pub input_count: u32,
    pub output_count: u32,
    pub precision: NumericPrecision,
    pub absolute_tolerance: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct SaciResourceReport {
    pub logical_params: u64,
    pub materialized_params: u64,
    pub nonzero_params: u64,
    pub serialized_bytes: u64,
    pub activation_memory_bytes: u64,
    pub peak_execution_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct CompiledSaciArtifact {
    pub descriptor: SaciCircuitDescriptor,
    pub weights: Vec<f64>,
    pub resources: SaciResourceReport,
    pub sacib_bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct LoadedSaciCircuit {
    pub descriptor: SaciCircuitDescriptor,
    pub resources: SaciResourceReport,
}

#[derive(Clone, Debug)]
pub struct SaciExecutionTrace {
    pub circuit_id: u32,
    pub topology: TopologyKind,
    pub weights_read: u64,
    pub layers_executed: usize,
    pub output: Vec<f64>,
}

/// Tracks dynamic SACI neuron allocation across the shared substrate (no hardcoded 128..192).
#[derive(Clone, Debug)]
pub struct SacisubstrateAllocator {
    pub total_neurons: usize,
    pub next_free: usize,
}

impl SacisubstrateAllocator {
    pub fn new(total_neurons: usize) -> Self {
        Self {
            total_neurons,
            next_free: 0,
        }
    }

    pub fn allocate(&mut self, count: usize) -> Result<Vec<u32>, ErrorCode> {
        if count == 0 || self.next_free + count > self.total_neurons {
            return Err(ErrorCode::CapacityExceeded);
        }
        let ids: Vec<u32> = (self.next_free..self.next_free + count)
            .map(|i| i as u32)
            .collect();
        self.next_free += count;
        Ok(ids)
    }
}

#[inline]
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Stage A + B: Parse a `.saci` declarative source into a validated `SaciProgram`.
pub fn parse_saci_source(src: &str) -> Result<SaciProgram, ErrorCode> {
    let mut format_version = 0u32;
    let mut circuit_id: Option<u32> = None;
    let mut input_count = 0usize;
    let mut output_count = 0usize;
    let mut max_neurons = 256usize;
    let mut max_parameters = 65536usize;
    let mut max_activation_bytes = 65536usize;
    let mut absolute_tolerance = 1e-9_f64;
    let mut steepness_k = 1.0e6_f64;

    let mut nodes: Vec<SaciNode> = Vec::new();
    let mut output_nodes: Vec<u32> = Vec::new();
    let mut section = "";

    for raw_line in src.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = &line[1..line.len() - 1];
            continue;
        }

        if section == "graph" {
            // Format:
            // INPUT <id>
            // CONST <id> <f64>
            // NODE <id> <OP> <in0,in1,...> [const=<f64>]
            // OUTPUTS <id0,id1,...>
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            match parts[0] {
                "INPUT" => {
                    if parts.len() < 2 {
                        return Err(ErrorCode::InvalidInput);
                    }
                    let id = parts[1]
                        .parse::<u32>()
                        .map_err(|_| ErrorCode::InvalidInput)?;
                    nodes.push(SaciNode {
                        id,
                        op: SaciOp::Input,
                        inputs: Vec::new(),
                        const_val: 0.0,
                        output_type: SaciDataType::F64,
                    });
                }
                "CONST" => {
                    if parts.len() < 3 {
                        return Err(ErrorCode::InvalidInput);
                    }
                    let id = parts[1]
                        .parse::<u32>()
                        .map_err(|_| ErrorCode::InvalidInput)?;
                    let val = parts[2]
                        .parse::<f64>()
                        .map_err(|_| ErrorCode::InvalidInput)?;
                    nodes.push(SaciNode {
                        id,
                        op: SaciOp::Const,
                        inputs: Vec::new(),
                        const_val: val,
                        output_type: SaciDataType::F64,
                    });
                }
                "NODE" => {
                    if parts.len() < 4 {
                        return Err(ErrorCode::InvalidInput);
                    }
                    let id = parts[1]
                        .parse::<u32>()
                        .map_err(|_| ErrorCode::InvalidInput)?;
                    let op = match parts[2] {
                        "ADD" => SaciOp::Add,
                        "SUB" => SaciOp::Sub,
                        "MUL_CONST" => SaciOp::MulConst,
                        "RELU" => SaciOp::Relu,
                        "ABS" => SaciOp::Abs,
                        "CMP_LT" => SaciOp::CmpLt,
                        "CMP_LE" => SaciOp::CmpLe,
                        "MAX" => SaciOp::Max,
                        "MIN" => SaciOp::Min,
                        "REDUCE_SUM" => SaciOp::ReduceSum,
                        "STATE_TRANSITION" => SaciOp::StateTransition,
                        _ => return Err(ErrorCode::InvalidInput),
                    };
                    let mut inputs = Vec::new();
                    for tok in parts[3].split(',') {
                        if !tok.is_empty() {
                            inputs.push(tok.parse::<u32>().map_err(|_| ErrorCode::InvalidInput)?);
                        }
                    }
                    let mut const_val = 0.0;
                    for extra in &parts[4..] {
                        if let Some(v) = extra.strip_prefix("const=") {
                            const_val = v.parse::<f64>().map_err(|_| ErrorCode::InvalidInput)?;
                        }
                    }
                    nodes.push(SaciNode {
                        id,
                        op,
                        inputs,
                        const_val,
                        output_type: SaciDataType::F64,
                    });
                }
                "OUTPUTS" => {
                    if parts.len() < 2 {
                        return Err(ErrorCode::InvalidInput);
                    }
                    for tok in parts[1].split(',') {
                        if !tok.is_empty() {
                            output_nodes
                                .push(tok.parse::<u32>().map_err(|_| ErrorCode::InvalidInput)?);
                        }
                    }
                }
                _ => return Err(ErrorCode::InvalidInput),
            }
            continue;
        }

        if let Some((k, v)) = line.split_once('=') {
            let key = k.trim();
            let val = v.trim().trim_matches('"');
            match (section, key) {
                ("saci", "format_version") => {
                    format_version = val.parse::<u32>().map_err(|_| ErrorCode::InvalidInput)?;
                }
                ("saci", "circuit_id") => {
                    circuit_id = Some(val.parse::<u32>().map_err(|_| ErrorCode::InvalidInput)?);
                }
                ("saci", "id") => {
                    if let Some(num) = val.strip_prefix("SOLVER_") {
                        circuit_id = Some(num.parse::<u32>().map_err(|_| ErrorCode::InvalidInput)?);
                    }
                }
                ("interface", "input_count") => {
                    input_count = val.parse::<usize>().map_err(|_| ErrorCode::InvalidInput)?;
                }
                ("interface", "output_count") => {
                    output_count = val.parse::<usize>().map_err(|_| ErrorCode::InvalidInput)?;
                }
                ("resources", "max_neurons") => {
                    max_neurons = val.parse::<usize>().map_err(|_| ErrorCode::InvalidInput)?;
                }
                ("resources", "max_parameters") => {
                    max_parameters = val.parse::<usize>().map_err(|_| ErrorCode::InvalidInput)?;
                }
                ("resources", "max_activation_bytes") => {
                    max_activation_bytes =
                        val.parse::<usize>().map_err(|_| ErrorCode::InvalidInput)?;
                }
                ("numerical", "absolute_tolerance") => {
                    absolute_tolerance = val.parse::<f64>().map_err(|_| ErrorCode::InvalidInput)?;
                }
                ("numerical", "steepness_k") => {
                    steepness_k = val.parse::<f64>().map_err(|_| ErrorCode::InvalidInput)?;
                }
                _ => {}
            }
        }
    }

    if format_version != 1 {
        return Err(ErrorCode::InvalidInput);
    }
    let cid = circuit_id.ok_or(ErrorCode::InvalidInput)?;
    if input_count == 0 || output_count == 0 || output_nodes.len() != output_count {
        return Err(ErrorCode::InvalidInput);
    }

    // Validate unique node IDs and valid input references (DAG order)
    for (idx, n) in nodes.iter().enumerate() {
        if nodes[..idx].iter().any(|prev| prev.id == n.id) {
            return Err(ErrorCode::InvalidInput);
        }
        for &inp_id in &n.inputs {
            if !nodes[..idx].iter().any(|prev| prev.id == inp_id) {
                return Err(ErrorCode::InvalidInput);
            }
        }
    }
    for &oid in &output_nodes {
        if !nodes.iter().any(|n| n.id == oid) {
            return Err(ErrorCode::InvalidInput);
        }
    }

    Ok(SaciProgram {
        circuit_id: cid,
        input_count,
        output_nodes,
        nodes,
        max_neurons,
        max_parameters,
        max_activation_bytes,
        absolute_tolerance,
        steepness_k,
    })
}

/// Stage C + D: Analyze SSA operations and select the appropriate circuit topology.
pub fn select_topology(program: &SaciProgram) -> TopologyKind {
    let mut has_nonlinear = false;
    let mut has_recurrent = false;
    for n in &program.nodes {
        match n.op {
            SaciOp::StateTransition => has_recurrent = true,
            SaciOp::Relu
            | SaciOp::Abs
            | SaciOp::CmpLt
            | SaciOp::CmpLe
            | SaciOp::Max
            | SaciOp::Min => {
                has_nonlinear = true;
            }
            _ => {}
        }
    }
    if has_recurrent {
        TopologyKind::Recurrent
    } else if has_nonlinear {
        TopologyKind::MultiLayerRelu
    } else {
        TopologyKind::Linear
    }
}

/// Stage E + F + G: Generic SSA Program-to-Weight synthesis, dynamic neuron placement,
/// validation, and `.sacib` serialization.
pub fn compile_saci_program(
    program: &SaciProgram,
    allocator: &mut SacisubstrateAllocator,
) -> Result<CompiledSaciArtifact, ErrorCode> {
    if program.input_count == 0 || program.output_nodes.is_empty() {
        return Err(ErrorCode::InvalidInput);
    }

    let topology = select_topology(program);
    let in_dim = program.input_count;
    let out_dim = program.output_nodes.len();

    // Map node_id -> linear representation over [inputs (0..in_dim), bias (in_dim)] OR hidden units
    // For Linear topology: every node is an exact affine vector of length `in_dim + 1`!
    // For MultiLayerRelu topology:
    //   Layer 1 projects `[inputs, 1.0]` -> hidden ReLU units (each nonlinear node emits its exact ReLU gadget).
    //   Layer 2 projects hidden ReLU units -> `[outputs]` (exact linear combinations).
    let mut id_to_idx = vec![0usize; program.nodes.len()];
    let find_node = |id: u32| -> Result<usize, ErrorCode> {
        program
            .nodes
            .iter()
            .position(|n| n.id == id)
            .ok_or(ErrorCode::InvalidInput)
    };
    for (i, n) in program.nodes.iter().enumerate() {
        id_to_idx[i] = n.id as usize;
    }

    let (layer_dims, weights) = match topology {
        TopologyKind::Linear => {
            // Each node's affine form over [x_0..x_{in_dim-1}, bias]: Vec<f64> of len `in_dim + 1`
            let stride = in_dim + 1;
            let mut forms = vec![vec![0.0_f64; stride]; program.nodes.len()];
            let mut next_input = 0usize;

            for (i, n) in program.nodes.iter().enumerate() {
                match n.op {
                    SaciOp::Input => {
                        if next_input >= in_dim {
                            return Err(ErrorCode::ShapeMismatch);
                        }
                        forms[i][next_input] = 1.0;
                        next_input += 1;
                    }
                    SaciOp::Const => {
                        forms[i][in_dim] = n.const_val;
                    }
                    SaciOp::Add => {
                        let a = find_node(n.inputs[0])?;
                        let b = find_node(n.inputs[1])?;
                        for k in 0..stride {
                            forms[i][k] = forms[a][k] + forms[b][k];
                        }
                    }
                    SaciOp::Sub => {
                        let a = find_node(n.inputs[0])?;
                        let b = find_node(n.inputs[1])?;
                        for k in 0..stride {
                            forms[i][k] = forms[a][k] - forms[b][k];
                        }
                    }
                    SaciOp::MulConst => {
                        let a = find_node(n.inputs[0])?;
                        for k in 0..stride {
                            forms[i][k] = forms[a][k] * n.const_val;
                        }
                    }
                    SaciOp::ReduceSum => {
                        for &inp_id in &n.inputs {
                            let a = find_node(inp_id)?;
                            for k in 0..stride {
                                forms[i][k] += forms[a][k];
                            }
                        }
                    }
                    _ => return Err(ErrorCode::InvalidInput),
                }
            }

            // Pack W1 [out_dim x in_dim] + b1 [out_dim]
            let mut w = vec![0.0_f64; out_dim * in_dim + out_dim];
            for (o, &oid) in program.output_nodes.iter().enumerate() {
                let idx = find_node(oid)?;
                for c in 0..in_dim {
                    w[o * in_dim + c] = forms[idx][c];
                }
                w[out_dim * in_dim + o] = forms[idx][in_dim];
            }
            (vec![in_dim as u32, out_dim as u32], w)
        }
        TopologyKind::MultiLayerRelu
        | TopologyKind::Sparse
        | TopologyKind::MultiBranch
        | TopologyKind::Recurrent => {
            // Lower affine subexpressions + nonlinear ReLU/Abs/Cmp/Max/Min gadgets into 2-layer ReLU MLP
            let aff_stride = in_dim + 1;
            let mut aff_forms = vec![vec![0.0_f64; aff_stride]; program.nodes.len()];
            let mut is_aff = vec![false; program.nodes.len()];

            // Hidden unit definitions: each hidden unit has affine coefficients over [x_0..x_{in_dim-1}, bias]
            let mut hidden_w1: Vec<Vec<f64>> = Vec::new();
            // Each node's representation over hidden units: Vec<(hidden_idx, coeff)>
            let mut h_forms: Vec<Vec<(usize, f64)>> = vec![Vec::new(); program.nodes.len()];

            let ensure_hidden = |idx: usize,
                                 is_aff: &[bool],
                                 aff_forms: &[Vec<f64>],
                                 h_forms: &[Vec<(usize, f64)>],
                                 hidden_w1: &mut Vec<Vec<f64>>|
             -> Vec<(usize, f64)> {
                if !is_aff[idx] {
                    return h_forms[idx].clone();
                }
                // Represent arbitrary affine form `a` as ReLU(a) - ReLU(-a)
                let pos_h = hidden_w1.len();
                hidden_w1.push(aff_forms[idx].clone());
                let neg_h = hidden_w1.len();
                hidden_w1.push(aff_forms[idx].iter().map(|v| -v).collect());
                vec![(pos_h, 1.0), (neg_h, -1.0)]
            };

            let mut next_input = 0usize;
            for (i, n) in program.nodes.iter().enumerate() {
                match n.op {
                    SaciOp::Input => {
                        if next_input >= in_dim {
                            return Err(ErrorCode::ShapeMismatch);
                        }
                        aff_forms[i][next_input] = 1.0;
                        is_aff[i] = true;
                        next_input += 1;
                    }
                    SaciOp::Const => {
                        aff_forms[i][in_dim] = n.const_val;
                        is_aff[i] = true;
                    }
                    SaciOp::Add => {
                        let a = find_node(n.inputs[0])?;
                        let b = find_node(n.inputs[1])?;
                        if is_aff[a] && is_aff[b] {
                            for k in 0..aff_stride {
                                aff_forms[i][k] = aff_forms[a][k] + aff_forms[b][k];
                            }
                            is_aff[i] = true;
                        } else {
                            let mut comb =
                                ensure_hidden(a, &is_aff, &aff_forms, &h_forms, &mut hidden_w1);
                            comb.extend(ensure_hidden(
                                b,
                                &is_aff,
                                &aff_forms,
                                &h_forms,
                                &mut hidden_w1,
                            ));
                            h_forms[i] = comb;
                        }
                    }
                    SaciOp::Sub => {
                        let a = find_node(n.inputs[0])?;
                        let b = find_node(n.inputs[1])?;
                        if is_aff[a] && is_aff[b] {
                            for k in 0..aff_stride {
                                aff_forms[i][k] = aff_forms[a][k] - aff_forms[b][k];
                            }
                            is_aff[i] = true;
                        } else {
                            let mut comb =
                                ensure_hidden(a, &is_aff, &aff_forms, &h_forms, &mut hidden_w1);
                            for (h, c) in
                                ensure_hidden(b, &is_aff, &aff_forms, &h_forms, &mut hidden_w1)
                            {
                                comb.push((h, -c));
                            }
                            h_forms[i] = comb;
                        }
                    }
                    SaciOp::MulConst => {
                        let a = find_node(n.inputs[0])?;
                        if is_aff[a] {
                            for k in 0..aff_stride {
                                aff_forms[i][k] = aff_forms[a][k] * n.const_val;
                            }
                            is_aff[i] = true;
                        } else {
                            h_forms[i] = h_forms[a]
                                .iter()
                                .map(|&(h, c)| (h, c * n.const_val))
                                .collect();
                        }
                    }
                    SaciOp::Relu => {
                        let a = find_node(n.inputs[0])?;
                        if !is_aff[a] {
                            return Err(ErrorCode::InvalidInput);
                        }
                        let hid = hidden_w1.len();
                        hidden_w1.push(aff_forms[a].clone());
                        h_forms[i] = vec![(hid, 1.0)];
                    }
                    SaciOp::Abs => {
                        let a = find_node(n.inputs[0])?;
                        if !is_aff[a] {
                            return Err(ErrorCode::InvalidInput);
                        }
                        let h_pos = hidden_w1.len();
                        hidden_w1.push(aff_forms[a].clone());
                        let h_neg = hidden_w1.len();
                        hidden_w1.push(aff_forms[a].iter().map(|v| -v).collect());
                        h_forms[i] = vec![(h_pos, 1.0), (h_neg, 1.0)];
                    }
                    SaciOp::CmpLt => {
                        // a < b  <=>  b - a > 0  =>  ReLU(k*(b - a)) - ReLU(k*(b - a) - 1)
                        let a = find_node(n.inputs[0])?;
                        let b = find_node(n.inputs[1])?;
                        if !is_aff[a] || !is_aff[b] {
                            return Err(ErrorCode::InvalidInput);
                        }
                        let k = program.steepness_k;
                        let mut diff_lo = vec![0.0; aff_stride];
                        for c in 0..aff_stride {
                            diff_lo[c] = k * (aff_forms[b][c] - aff_forms[a][c]);
                        }
                        let mut diff_hi = diff_lo.clone();
                        diff_hi[in_dim] -= 1.0;

                        let h_lo = hidden_w1.len();
                        hidden_w1.push(diff_lo);
                        let h_hi = hidden_w1.len();
                        hidden_w1.push(diff_hi);
                        h_forms[i] = vec![(h_lo, 1.0), (h_hi, -1.0)];
                    }
                    SaciOp::CmpLe => {
                        // a <= b  <=>  ReLU(k*(b - a) + 1) - ReLU(k*(b - a))
                        let a = find_node(n.inputs[0])?;
                        let b = find_node(n.inputs[1])?;
                        if !is_aff[a] || !is_aff[b] {
                            return Err(ErrorCode::InvalidInput);
                        }
                        let k = program.steepness_k;
                        let mut diff_hi = vec![0.0; aff_stride];
                        for c in 0..aff_stride {
                            diff_hi[c] = k * (aff_forms[b][c] - aff_forms[a][c]);
                        }
                        let diff_lo = diff_hi.clone();
                        diff_hi[in_dim] += 1.0;

                        let h_hi = hidden_w1.len();
                        hidden_w1.push(diff_hi);
                        let h_lo = hidden_w1.len();
                        hidden_w1.push(diff_lo);
                        h_forms[i] = vec![(h_hi, 1.0), (h_lo, -1.0)];
                    }
                    SaciOp::Max => {
                        let a = find_node(n.inputs[0])?;
                        let b = find_node(n.inputs[1])?;
                        if !is_aff[a] || !is_aff[b] {
                            return Err(ErrorCode::InvalidInput);
                        }
                        let mut comb =
                            ensure_hidden(a, &is_aff, &aff_forms, &h_forms, &mut hidden_w1);
                        let mut diff = vec![0.0; aff_stride];
                        for c in 0..aff_stride {
                            diff[c] = aff_forms[b][c] - aff_forms[a][c];
                        }
                        let h_d = hidden_w1.len();
                        hidden_w1.push(diff);
                        comb.push((h_d, 1.0));
                        h_forms[i] = comb;
                    }
                    SaciOp::Min => {
                        let a = find_node(n.inputs[0])?;
                        let b = find_node(n.inputs[1])?;
                        if !is_aff[a] || !is_aff[b] {
                            return Err(ErrorCode::InvalidInput);
                        }
                        let mut comb =
                            ensure_hidden(a, &is_aff, &aff_forms, &h_forms, &mut hidden_w1);
                        let mut diff = vec![0.0; aff_stride];
                        for c in 0..aff_stride {
                            diff[c] = aff_forms[a][c] - aff_forms[b][c];
                        }
                        let h_d = hidden_w1.len();
                        hidden_w1.push(diff);
                        comb.push((h_d, -1.0));
                        h_forms[i] = comb;
                    }
                    SaciOp::ReduceSum | SaciOp::StateTransition => {
                        let mut comb = Vec::new();
                        for &inp_id in &n.inputs {
                            let a = find_node(inp_id)?;
                            comb.extend(ensure_hidden(
                                a,
                                &is_aff,
                                &aff_forms,
                                &h_forms,
                                &mut hidden_w1,
                            ));
                        }
                        h_forms[i] = comb;
                    }
                }
            }

            // Resolve all output nodes first so `hidden_w1` reaches its final size before allocating `w`
            let mut out_terms = Vec::with_capacity(out_dim);
            for &oid in &program.output_nodes {
                let idx = find_node(oid)?;
                out_terms.push(ensure_hidden(
                    idx,
                    &is_aff,
                    &aff_forms,
                    &h_forms,
                    &mut hidden_w1,
                ));
            }

            let h_dim = hidden_w1.len().max(1);
            let total_w = h_dim * in_dim + h_dim + out_dim * h_dim;
            let mut w = vec![0.0_f64; total_w];
            let (w1, rest) = w.split_at_mut(h_dim * in_dim);
            let (b1, w2) = rest.split_at_mut(h_dim);

            for (h, row) in hidden_w1.iter().enumerate() {
                for c in 0..in_dim {
                    w1[h * in_dim + c] = row[c];
                }
                b1[h] = row[in_dim];
            }

            for (o, terms) in out_terms.into_iter().enumerate() {
                for (h, coeff) in terms {
                    w2[o * h_dim + h] += coeff;
                }
            }

            (vec![in_dim as u32, h_dim as u32, out_dim as u32], w)
        }
    };

    let neurons_needed = if layer_dims.len() == 2 {
        out_dim
    } else {
        layer_dims[1] as usize
    };
    let max_layer = *layer_dims.iter().max().unwrap_or(&0) as usize;
    let activation_memory_bytes = (max_layer * std::mem::size_of::<f64>()) as u64;

    if neurons_needed > program.max_neurons
        || weights.len() > program.max_parameters
        || activation_memory_bytes as usize > program.max_activation_bytes
    {
        return Err(ErrorCode::CapacityExceeded);
    }

    let neuron_ids = allocator.allocate(neurons_needed)?;
    let nonzero_params = weights.iter().filter(|&&v| v != 0.0).count() as u64;

    let descriptor = SaciCircuitDescriptor {
        circuit_id: program.circuit_id,
        topology,
        layer_dims,
        neuron_ids,
        weight_offset: 0,
        weight_count: weights.len() as u64,
        input_count: in_dim as u32,
        output_count: out_dim as u32,
        precision: NumericPrecision::F64,
        absolute_tolerance: program.absolute_tolerance,
    };

    let mut resources = SaciResourceReport {
        logical_params: weights.len() as u64,
        materialized_params: weights.len() as u64,
        nonzero_params,
        serialized_bytes: 0,
        activation_memory_bytes,
        peak_execution_bytes: (weights.len() * 8) as u64 + activation_memory_bytes,
    };

    let sacib_bytes = serialize_sacib(&descriptor, &weights, &resources);
    resources.serialized_bytes = sacib_bytes.len() as u64;

    Ok(CompiledSaciArtifact {
        descriptor,
        weights,
        resources,
        sacib_bytes,
    })
}

/// Serialize a compiled SACI circuit into portable little-endian `.sacib` format.
pub fn serialize_sacib(
    desc: &SaciCircuitDescriptor,
    weights: &[f64],
    resources: &SaciResourceReport,
) -> Vec<u8> {
    let mut payload = Vec::new();
    // Manifest
    payload.extend_from_slice(&desc.circuit_id.to_le_bytes());
    payload.extend_from_slice(&SACIB_COMPILER_VERSION.to_le_bytes());
    // Interface
    payload.extend_from_slice(&desc.input_count.to_le_bytes());
    payload.extend_from_slice(&desc.output_count.to_le_bytes());
    // Topology
    payload.push(desc.topology as u8);
    payload.push(desc.precision as u8);
    payload.extend_from_slice(&(desc.layer_dims.len() as u32).to_le_bytes());
    for &d in &desc.layer_dims {
        payload.extend_from_slice(&d.to_le_bytes());
    }
    payload.extend_from_slice(&(desc.neuron_ids.len() as u32).to_le_bytes());
    for &nid in &desc.neuron_ids {
        payload.extend_from_slice(&nid.to_le_bytes());
    }
    // Numerical & Resources
    payload.extend_from_slice(&desc.absolute_tolerance.to_le_bytes());
    payload.extend_from_slice(&resources.logical_params.to_le_bytes());
    payload.extend_from_slice(&resources.materialized_params.to_le_bytes());
    payload.extend_from_slice(&resources.nonzero_params.to_le_bytes());
    payload.extend_from_slice(&resources.activation_memory_bytes.to_le_bytes());
    payload.extend_from_slice(&resources.peak_execution_bytes.to_le_bytes());
    // Weight payload
    payload.extend_from_slice(&(weights.len() as u64).to_le_bytes());
    for &w in weights {
        payload.extend_from_slice(&w.to_le_bytes());
    }

    let hash = fnv1a_64(&payload);
    let total_size = (32 + payload.len()) as u64;

    let mut out = Vec::with_capacity(total_size as usize);
    out.extend_from_slice(SACIB_MAGIC);
    out.extend_from_slice(&SACIB_VERSION.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes()); // little-endian
    out.extend_from_slice(&total_size.to_le_bytes());
    out.extend_from_slice(&hash.to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

fn take_u32(slice: &[u8], off: &mut usize) -> Result<u32, ErrorCode> {
    let end = off.checked_add(4).ok_or(ErrorCode::ShapeMismatch)?;
    if end > slice.len() {
        return Err(ErrorCode::ShapeMismatch);
    }
    let v = u32::from_le_bytes(slice[*off..end].try_into().unwrap());
    *off = end;
    Ok(v)
}

fn take_u64(slice: &[u8], off: &mut usize) -> Result<u64, ErrorCode> {
    let end = off.checked_add(8).ok_or(ErrorCode::ShapeMismatch)?;
    if end > slice.len() {
        return Err(ErrorCode::ShapeMismatch);
    }
    let v = u64::from_le_bytes(slice[*off..end].try_into().unwrap());
    *off = end;
    Ok(v)
}

fn take_f64(slice: &[u8], off: &mut usize) -> Result<f64, ErrorCode> {
    let end = off.checked_add(8).ok_or(ErrorCode::ShapeMismatch)?;
    if end > slice.len() {
        return Err(ErrorCode::ShapeMismatch);
    }
    let v = f64::from_le_bytes(slice[*off..end].try_into().unwrap());
    *off = end;
    Ok(v)
}

/// Atomically load a `.sacib` binary into `checkpoint`'s `TensorId(5)` (`SaciCircuits`) arena.
/// If any check fails, `checkpoint` remains completely unmodified.
pub fn load_saci_binary(
    checkpoint: &mut Checkpoint,
    artifact: &[u8],
) -> Result<LoadedSaciCircuit, ErrorCode> {
    if artifact.len() < 32 || &artifact[0..8] != SACIB_MAGIC {
        return Err(ErrorCode::InvalidInput);
    }
    let mut off = 8usize;
    let version = take_u32(artifact, &mut off)?;
    let endian = take_u32(artifact, &mut off)?;
    let total_size = take_u64(artifact, &mut off)?;
    let expected_hash = take_u64(artifact, &mut off)?;

    if version != SACIB_VERSION || endian != 1 || total_size as usize != artifact.len() {
        return Err(ErrorCode::InvalidInput);
    }
    let payload = &artifact[32..];
    if fnv1a_64(payload) != expected_hash {
        return Err(ErrorCode::InvalidInput);
    }

    let mut p_off = 0usize;
    let circuit_id = take_u32(payload, &mut p_off)?;
    let _compiler_ver = take_u32(payload, &mut p_off)?;
    let input_count = take_u32(payload, &mut p_off)?;
    let output_count = take_u32(payload, &mut p_off)?;

    if p_off + 2 > payload.len() {
        return Err(ErrorCode::ShapeMismatch);
    }
    let topology = match payload[p_off] {
        0 => TopologyKind::Linear,
        1 => TopologyKind::MultiLayerRelu,
        2 => TopologyKind::Sparse,
        3 => TopologyKind::MultiBranch,
        4 => TopologyKind::Recurrent,
        _ => return Err(ErrorCode::InvalidInput),
    };
    let precision = match payload[p_off + 1] {
        0 => NumericPrecision::F64,
        1 => NumericPrecision::F32,
        _ => return Err(ErrorCode::InvalidInput),
    };
    p_off += 2;

    let n_dims = take_u32(payload, &mut p_off)? as usize;
    let mut layer_dims = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        layer_dims.push(take_u32(payload, &mut p_off)?);
    }
    let n_neurons = take_u32(payload, &mut p_off)? as usize;
    let mut neuron_ids = Vec::with_capacity(n_neurons);
    for _ in 0..n_neurons {
        neuron_ids.push(take_u32(payload, &mut p_off)?);
    }

    let absolute_tolerance = take_f64(payload, &mut p_off)?;
    let logical_params = take_u64(payload, &mut p_off)?;
    let materialized_params = take_u64(payload, &mut p_off)?;
    let nonzero_params = take_u64(payload, &mut p_off)?;
    let activation_memory_bytes = take_u64(payload, &mut p_off)?;
    let peak_execution_bytes = take_u64(payload, &mut p_off)?;

    let weight_count = take_u64(payload, &mut p_off)?;
    let mut weights = Vec::with_capacity(weight_count as usize);
    for _ in 0..weight_count {
        let val = take_f64(payload, &mut p_off)?;
        if !val.is_finite() {
            return Err(ErrorCode::InvalidInput);
        }
        weights.push(val);
    }

    // Atomic append to TensorId(5) arena (create descriptor if not present, or append if at end)
    let weight_offset = if let Some(desc_idx) = checkpoint
        .descriptors
        .iter()
        .position(|d| d.id == TensorId(5))
    {
        let desc = checkpoint.descriptors[desc_idx];
        if desc.offset + desc.len != checkpoint.data.len() {
            // Relocate TensorId(5) cleanly to end of data buffer so existing offsets remain valid
            let new_off = checkpoint.data.len();
            let existing = checkpoint.data[desc.offset..desc.offset + desc.len].to_vec();
            checkpoint.data.extend_from_slice(&existing);
            let arena_off = existing.len() as u64;
            checkpoint.data.extend_from_slice(&weights);
            checkpoint.descriptors[desc_idx].offset = new_off;
            checkpoint.descriptors[desc_idx].len = existing.len() + weights.len();
            checkpoint.descriptors[desc_idx].rows = checkpoint.descriptors[desc_idx].len;
            arena_off
        } else {
            let arena_off = desc.len as u64;
            checkpoint.data.extend_from_slice(&weights);
            checkpoint.descriptors[desc_idx].len += weights.len();
            checkpoint.descriptors[desc_idx].rows = checkpoint.descriptors[desc_idx].len;
            arena_off
        }
    } else {
        let new_off = checkpoint.data.len();
        checkpoint.data.extend_from_slice(&weights);
        checkpoint.descriptors.push(TensorDescriptor {
            id: TensorId(5),
            component: AdaComponent::SaciCircuits,
            rows: weights.len(),
            cols: 1,
            offset: new_off,
            len: weights.len(),
            frozen: true,
        });
        0
    };

    Ok(LoadedSaciCircuit {
        descriptor: SaciCircuitDescriptor {
            circuit_id,
            topology,
            layer_dims,
            neuron_ids,
            weight_offset,
            weight_count,
            input_count,
            output_count,
            precision,
            absolute_tolerance,
        },
        resources: SaciResourceReport {
            logical_params,
            materialized_params,
            nonzero_params,
            serialized_bytes: artifact.len() as u64,
            activation_memory_bytes,
            peak_execution_bytes,
        },
    })
}

/// Generic, zero-solver-specific SACI neural execution using borrowed checkpoint weights
/// and caller-provided `scratch` buffer.
pub fn execute_saci_circuit(
    checkpoint: &Checkpoint,
    descriptor: &SaciCircuitDescriptor,
    input: &[f64],
    scratch: &mut [f64],
) -> Result<SaciExecutionTrace, ErrorCode> {
    if input.len() != descriptor.input_count as usize || input.iter().any(|v| !v.is_finite()) {
        return Err(ErrorCode::InvalidInput);
    }
    let arena = checkpoint
        .tensor_slice(TensorId(5))
        .ok_or(ErrorCode::ShapeMismatch)?;
    let start = descriptor.weight_offset as usize;
    let end = start + descriptor.weight_count as usize;
    if end > arena.len() {
        return Err(ErrorCode::ShapeMismatch);
    }
    let w = &arena[start..end];

    let num_layers = descriptor.layer_dims.len().saturating_sub(1);
    if num_layers == 0 {
        return Err(ErrorCode::ShapeMismatch);
    }

    let max_dim = descriptor.layer_dims.iter().copied().max().unwrap_or(0) as usize;
    if scratch.len() < 2 * max_dim {
        return Err(ErrorCode::CapacityExceeded);
    }

    let (buf_a, buf_b) = scratch.split_at_mut(max_dim);
    buf_a[..input.len()].copy_from_slice(input);

    let mut w_off = 0usize;
    let mut cur_in_is_a = true;

    for l in 0..num_layers {
        let d_in = descriptor.layer_dims[l] as usize;
        let d_out = descriptor.layer_dims[l + 1] as usize;
        let is_last = l + 1 == num_layers;

        let mat_len = d_out * d_in;
        let bias_len = if descriptor.topology == TopologyKind::Linear || !is_last {
            d_out
        } else {
            0
        };
        if w_off + mat_len + bias_len > w.len() {
            return Err(ErrorCode::ShapeMismatch);
        }

        let mat = &w[w_off..w_off + mat_len];
        let bias = &w[w_off + mat_len..w_off + mat_len + bias_len];
        w_off += mat_len + bias_len;

        let (src, dst) = if cur_in_is_a {
            (&buf_a[..d_in], &mut buf_b[..d_out])
        } else {
            (&buf_b[..d_in], &mut buf_a[..d_out])
        };

        for o in 0..d_out {
            let mut acc = if bias_len > 0 { bias[o] } else { 0.0 };
            for c in 0..d_in {
                acc += mat[o * d_in + c] * src[c];
            }
            if !acc.is_finite() {
                return Err(ErrorCode::InvalidInput);
            }
            dst[o] = if !is_last && descriptor.topology != TopologyKind::Linear {
                acc.max(0.0)
            } else {
                acc
            };
        }
        cur_in_is_a = !cur_in_is_a;
    }

    let out_dim = descriptor.output_count as usize;
    let out_slice = if cur_in_is_a {
        &buf_a[..out_dim]
    } else {
        &buf_b[..out_dim]
    };

    Ok(SaciExecutionTrace {
        circuit_id: descriptor.circuit_id,
        topology: descriptor.topology,
        weights_read: descriptor.weight_count,
        layers_executed: num_layers,
        output: out_slice.to_vec(),
    })
}
