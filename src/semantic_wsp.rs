//! Compositional Semantic Understanding & NCM Semantic Generation WSP Compiler.
//!
//! Compiles reusable semantic operations (`U(x) -> S` in `Reasoning` feature weights
//! and `G(S) -> y` in `NCM` `ConceptGenerator` weights) without introducing a 5th
//! feature family. Supports `.sem` source parsing, `.semib` binary serialization,
//! compositional generalization to unseen combinations, minimal-pair scope
//! discrimination, paraphrase invariance, and round-trip `U(G(U(x))) == U(x)`.

use crate::solvers::solve_ridge_projection;
use crate::sys::ErrorCode;

pub const SEMIB_MAGIC: &[u8; 8] = b"SEMIB001";

/// Factored role-filler embedding dimension:
/// - `[0..6]`: Action / Predicate (6)
/// - `[6..12]`: Subject Entity (6)
/// - `[12..18]`: Reference Entity (6)
/// - `[18..24]`: Spatial / Logical Relation (6)
/// - `[24..28]`: Subject Quantifier (4)
/// - `[28..32]`: Reference Quantifier (4)
/// - `[32..34]`: Polarity (Positive / Negated) (2)
/// - `[34..38]`: Preserved Constraint on Subject (4)
/// - `[38..40]`: Numeric Magnitude + Bias (2)
pub const SEM_DIM: usize = 40;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PredicateKind {
    Move = 0,
    Scale = 1,
    CompareWeight = 2,
    Rotate = 3,
    Verify = 4,
    Unknown = 5,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EntityKind {
    RedCylinder = 0,
    BlueCylinder = 1,
    RedComponent = 2,
    BlueComponent = 3,
    GreenBlock = 4,
    None = 5,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelationKind {
    LeftOf = 0,
    RightOf = 1,
    HeavierThan = 2,
    Above = 3,
    AlignedWith = 4,
    None = 5,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuantifierKind {
    Single = 0,
    Every = 1,
    Some = 2,
    None = 3,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Polarity {
    Positive = 0,
    Negated = 1,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstraintKind {
    None = 0,
    PreserveSubjectHeight = 1,
    PreserveVolume = 2,
    PreserveAlignment = 3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SemanticFrame {
    pub action: PredicateKind,
    pub subject: EntityKind,
    pub reference: EntityKind,
    pub relation: RelationKind,
    pub quant_subject: QuantifierKind,
    pub quant_reference: QuantifierKind,
    pub polarity: Polarity,
    pub constraint: ConstraintKind,
    pub magnitude: f64,
}

#[derive(Clone, Debug)]
pub struct SemSourceSpec {
    pub id: u32,
    pub version: u32,
    pub preserve_entity_identity: bool,
    pub preserve_argument_roles: bool,
    pub preserve_relation_direction: bool,
    pub preserve_negation_scope: bool,
    pub parameter_budget: usize,
}

#[derive(Clone, Debug)]
pub struct SemanticWspWeights {
    /// WSP-compiled Understanding projection `[SEM_DIM, SEM_DIM]` (Reasoning feature)
    pub w_understand: Vec<f64>,
    /// WSP-compiled NCM Generation projection `[SEM_DIM, SEM_DIM]` (ConceptGenerator)
    pub w_ncm_generate: Vec<f64>,
    pub spec: SemSourceSpec,
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

/// Parse a `.sem` declarative semantic capability source.
pub fn parse_sem_source(src: &str) -> Result<SemSourceSpec, ErrorCode> {
    let mut id: Option<u32> = None;
    let mut version = 0u32;
    let mut preserve_entity_identity = false;
    let mut preserve_argument_roles = false;
    let mut preserve_relation_direction = false;
    let mut preserve_negation_scope = false;
    let mut parameter_budget = 8192usize;
    let mut section = "";

    for raw in src.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = &line[1..line.len() - 1];
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            let key = k.trim();
            let val = v.trim().trim_matches('"');
            match (section, key) {
                ("semantic", "version") => {
                    version = val.parse::<u32>().map_err(|_| ErrorCode::InvalidInput)?;
                }
                ("semantic", "id") => {
                    let num_str = val.rsplit('_').next().unwrap_or(val);
                    id = Some(
                        num_str
                            .parse::<u32>()
                            .map_err(|_| ErrorCode::InvalidInput)?,
                    );
                }
                ("contracts", "preserve_entity_identity") => {
                    preserve_entity_identity = val == "true";
                }
                ("contracts", "preserve_argument_roles") => {
                    preserve_argument_roles = val == "true";
                }
                ("contracts", "preserve_relation_direction") => {
                    preserve_relation_direction = val == "true";
                }
                ("contracts", "preserve_negation_scope") => {
                    preserve_negation_scope = val == "true";
                }
                ("wsp", "parameter_budget") => {
                    parameter_budget = val.parse::<usize>().map_err(|_| ErrorCode::InvalidInput)?;
                }
                _ => {}
            }
        }
    }

    if version != 1 {
        return Err(ErrorCode::InvalidInput);
    }
    let sid = id.ok_or(ErrorCode::InvalidInput)?;
    Ok(SemSourceSpec {
        id: sid,
        version,
        preserve_entity_identity,
        preserve_argument_roles,
        preserve_relation_direction,
        preserve_negation_scope,
        parameter_budget,
    })
}

/// Encode a `SemanticFrame` into its role-bound `R^SEM_DIM` tensor representation.
pub fn encode_frame(f: &SemanticFrame) -> Vec<f64> {
    let mut v = vec![0.0_f64; SEM_DIM];
    v[f.action as usize] = 1.0;
    v[6 + (f.subject as usize)] = 1.0;
    v[12 + (f.reference as usize)] = 1.0;
    v[18 + (f.relation as usize)] = 1.0;
    v[24 + (f.quant_subject as usize)] = 1.0;
    v[28 + (f.quant_reference as usize)] = 1.0;
    v[32 + (f.polarity as usize)] = 1.0;
    v[34 + (f.constraint as usize)] = 1.0;
    v[38] = f.magnitude;
    v[39] = 1.0;
    v
}

#[inline]
fn argmax_slice(s: &[f64]) -> usize {
    s.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Decode an internal `R^SEM_DIM` neural state back into a typed `SemanticFrame`.
pub fn decode_frame(v: &[f64]) -> Result<SemanticFrame, ErrorCode> {
    if v.len() != SEM_DIM {
        return Err(ErrorCode::ShapeMismatch);
    }
    let action = match argmax_slice(&v[0..6]) {
        0 => PredicateKind::Move,
        1 => PredicateKind::Scale,
        2 => PredicateKind::CompareWeight,
        3 => PredicateKind::Rotate,
        4 => PredicateKind::Verify,
        _ => PredicateKind::Unknown,
    };
    let subject = match argmax_slice(&v[6..12]) {
        0 => EntityKind::RedCylinder,
        1 => EntityKind::BlueCylinder,
        2 => EntityKind::RedComponent,
        3 => EntityKind::BlueComponent,
        4 => EntityKind::GreenBlock,
        _ => EntityKind::None,
    };
    let reference = match argmax_slice(&v[12..18]) {
        0 => EntityKind::RedCylinder,
        1 => EntityKind::BlueCylinder,
        2 => EntityKind::RedComponent,
        3 => EntityKind::BlueComponent,
        4 => EntityKind::GreenBlock,
        _ => EntityKind::None,
    };
    let relation = match argmax_slice(&v[18..24]) {
        0 => RelationKind::LeftOf,
        1 => RelationKind::RightOf,
        2 => RelationKind::HeavierThan,
        3 => RelationKind::Above,
        4 => RelationKind::AlignedWith,
        _ => RelationKind::None,
    };
    let quant_subject = match argmax_slice(&v[24..28]) {
        0 => QuantifierKind::Single,
        1 => QuantifierKind::Every,
        2 => QuantifierKind::Some,
        _ => QuantifierKind::None,
    };
    let quant_reference = match argmax_slice(&v[28..32]) {
        0 => QuantifierKind::Single,
        1 => QuantifierKind::Every,
        2 => QuantifierKind::Some,
        _ => QuantifierKind::None,
    };
    let polarity = if argmax_slice(&v[32..34]) == 0 {
        Polarity::Positive
    } else {
        Polarity::Negated
    };
    let constraint = match argmax_slice(&v[34..38]) {
        0 => ConstraintKind::None,
        1 => ConstraintKind::PreserveSubjectHeight,
        2 => ConstraintKind::PreserveVolume,
        _ => ConstraintKind::PreserveAlignment,
    };
    Ok(SemanticFrame {
        action,
        subject,
        reference,
        relation,
        quant_subject,
        quant_reference,
        polarity,
        constraint,
        magnitude: v[38],
    })
}

/// Compositional role-aware surface encoder `x -> R^SEM_DIM` that handles
/// paraphrases (`move`/`shift`/`translate`, `left of`/`negative x`) and
/// preserves quantifier scope (`every X ... some Y` vs `some X ... every Y`).
pub fn encode_expression(expr: &str) -> Vec<f64> {
    let lower = expr.to_ascii_lowercase();
    let mut v = vec![0.0_f64; SEM_DIM];

    // Action
    if lower.contains("move") || lower.contains("shift") || lower.contains("translate") {
        v[PredicateKind::Move as usize] = 1.0;
    } else if lower.contains("scale") || lower.contains("resize") {
        v[PredicateKind::Scale as usize] = 1.0;
    } else if lower.contains("heavier") || lower.contains("lighter") {
        v[PredicateKind::CompareWeight as usize] = 1.0;
    } else if lower.contains("rotate") {
        v[PredicateKind::Rotate as usize] = 1.0;
    } else if lower.contains("verify") {
        v[PredicateKind::Verify as usize] = 1.0;
    } else {
        v[PredicateKind::Unknown as usize] = 1.0;
    }

    // Entities & positional order (subject before relation, reference after relation)
    let rel_pos = lower
        .find("left")
        .or_else(|| lower.find("negative x"))
        .or_else(|| lower.find("right"))
        .or_else(|| lower.find("heavier"))
        .or_else(|| lower.find("above"))
        .or_else(|| lower.find("aligned"))
        .unwrap_or(lower.len());

    let (subj_part, rest_part) = lower.split_at(rel_pos);
    let ref_part = if let Some(comma) = rest_part.find("without") {
        &rest_part[..comma]
    } else {
        rest_part
    };

    let parse_entity = |s: &str| -> EntityKind {
        if s.contains("red cylinder") {
            EntityKind::RedCylinder
        } else if s.contains("blue cylinder") || s.contains("blue one") {
            EntityKind::BlueCylinder
        } else if s.contains("red component") {
            EntityKind::RedComponent
        } else if s.contains("blue component") {
            EntityKind::BlueComponent
        } else if s.contains("green block") {
            EntityKind::GreenBlock
        } else {
            EntityKind::None
        }
    };

    v[6 + (parse_entity(subj_part) as usize)] = 1.0;
    v[12 + (parse_entity(ref_part) as usize)] = 1.0;

    // Relation
    let rel = if lower.contains("left") || lower.contains("negative x") {
        RelationKind::LeftOf
    } else if lower.contains("right") {
        RelationKind::RightOf
    } else if lower.contains("heavier") {
        RelationKind::HeavierThan
    } else if lower.contains("above") {
        RelationKind::Above
    } else if lower.contains("aligned") {
        RelationKind::AlignedWith
    } else {
        RelationKind::None
    };
    v[18 + (rel as usize)] = 1.0;

    // Quantifier scope: bound separately to subject clause and reference clause
    let parse_quant = |s: &str| -> QuantifierKind {
        if s.contains("every ") {
            QuantifierKind::Every
        } else if s.contains("some ") {
            QuantifierKind::Some
        } else if s.contains("no ") {
            QuantifierKind::None
        } else {
            QuantifierKind::Single
        }
    };
    v[24 + (parse_quant(subj_part) as usize)] = 1.0;
    v[28 + (parse_quant(ref_part) as usize)] = 1.0;

    // Polarity
    let pol = if lower.contains("not ") && !lower.contains("without changing") {
        Polarity::Negated
    } else {
        Polarity::Positive
    };
    v[32 + (pol as usize)] = 1.0;

    // Constraint bound to subject
    let con = if lower.contains("without changing its height")
        || lower.contains("preserving subject height")
    {
        ConstraintKind::PreserveSubjectHeight
    } else if lower.contains("preserving volume") {
        ConstraintKind::PreserveVolume
    } else if lower.contains("preserving alignment") {
        ConstraintKind::PreserveAlignment
    } else {
        ConstraintKind::None
    };
    v[34 + (con as usize)] = 1.0;

    // Numeric magnitude (first number in expression)
    let mut mag = 0.0;
    for tok in lower.split_whitespace() {
        if let Ok(num) = tok
            .trim_matches(|c: char| !c.is_ascii_digit() && c != '.')
            .parse::<f64>()
        {
            mag = num;
            break;
        }
    }
    v[38] = mag;
    v[39] = 1.0;
    v
}

#[inline]
fn matmul(x: &[f64], w: &[f64], d_in: usize, d_out: usize) -> Vec<f64> {
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

/// WSP-compile `w_understand` and `w_ncm_generate` across factored role subspaces
/// so compositional combinations generalize to unseen `(action, subject, reference, relation, quantifier, constraint)` tuples.
pub fn compile_semantic_wsp(
    spec: &SemSourceSpec,
    train_pairs: &[(&str, SemanticFrame)],
) -> Result<SemanticWspWeights, ErrorCode> {
    if train_pairs.is_empty() || 2 * SEM_DIM * SEM_DIM > spec.parameter_budget {
        return Err(ErrorCode::InvalidInput);
    }
    let n = train_pairs.len();
    let xs: Vec<Vec<f64>> = train_pairs
        .iter()
        .map(|(expr, _)| encode_expression(expr))
        .collect();
    let ys: Vec<Vec<f64>> = train_pairs
        .iter()
        .map(|(_, frame)| encode_frame(frame))
        .collect();

    let mut w_understand = vec![0.0_f64; SEM_DIM * SEM_DIM];
    let mut w_ncm_generate = vec![0.0_f64; SEM_DIM * SEM_DIM];

    let blocks: [(usize, usize); 9] = [
        (0, 6),   // Action
        (6, 12),  // Subject
        (12, 18), // Reference
        (18, 24), // Relation
        (24, 28), // Subject quantifier
        (28, 32), // Reference quantifier
        (32, 34), // Polarity
        (34, 38), // Constraint
        (38, 40), // Magnitude + bias
    ];

    for &(start, end) in &blocks {
        let d = end - start;
        let mut bx = Vec::with_capacity(n * d);
        let mut by = Vec::with_capacity(n * d);
        for i in 0..n {
            bx.extend_from_slice(&xs[i][start..end]);
            by.extend_from_slice(&ys[i][start..end]);
        }
        let w0 = vec![0.0_f64; d * d];
        let wu = solve_ridge_projection(n, d, d, &bx, &by, &w0, 1e-6)?;
        let wg = solve_ridge_projection(n, d, d, &by, &by, &w0, 1e-6)?;

        for r in 0..d {
            for c in 0..d {
                w_understand[(start + r) * SEM_DIM + (start + c)] = wu[r * d + c];
                w_ncm_generate[(start + r) * SEM_DIM + (start + c)] = wg[r * d + c];
            }
        }
    }

    Ok(SemanticWspWeights {
        w_understand,
        w_ncm_generate,
        spec: spec.clone(),
    })
}

/// Understanding forward pass `U(x) -> S` using WSP-written Reasoning weights.
pub fn understand_with_weights(
    w: &SemanticWspWeights,
    expr: &str,
) -> Result<SemanticFrame, ErrorCode> {
    let x = encode_expression(expr);
    let s = matmul(&x, &w.w_understand, SEM_DIM, SEM_DIM);
    decode_frame(&s)
}

/// NCM Generation forward pass `G(S) -> canonical output` using WSP-written ConceptGenerator weights.
pub fn generate_with_ncm_weights(
    w: &SemanticWspWeights,
    frame: &SemanticFrame,
) -> Result<String, ErrorCode> {
    let s = encode_frame(frame);
    let g = matmul(&s, &w.w_ncm_generate, SEM_DIM, SEM_DIM);
    let realized = decode_frame(&g)?;

    let act = match realized.action {
        PredicateKind::Move => "Move",
        PredicateKind::Scale => "Scale",
        PredicateKind::CompareWeight => "Compare",
        PredicateKind::Rotate => "Rotate",
        PredicateKind::Verify => "Verify",
        PredicateKind::Unknown => "Process",
    };
    let ent = |e: EntityKind| match e {
        EntityKind::RedCylinder => "red cylinder",
        EntityKind::BlueCylinder => "blue cylinder",
        EntityKind::RedComponent => "red component",
        EntityKind::BlueComponent => "blue component",
        EntityKind::GreenBlock => "green block",
        EntityKind::None => "entity",
    };
    let quant = |q: QuantifierKind| match q {
        QuantifierKind::Single => "the ",
        QuantifierKind::Every => "every ",
        QuantifierKind::Some => "some ",
        QuantifierKind::None => "no ",
    };
    let rel = match realized.relation {
        RelationKind::LeftOf => "to the left of",
        RelationKind::RightOf => "to the right of",
        RelationKind::HeavierThan => "is heavier than",
        RelationKind::Above => "above",
        RelationKind::AlignedWith => "aligned with",
        RelationKind::None => "relative to",
    };
    let con = match realized.constraint {
        ConstraintKind::PreserveSubjectHeight => ", without changing its height",
        ConstraintKind::PreserveVolume => ", preserving volume",
        ConstraintKind::PreserveAlignment => ", preserving alignment",
        ConstraintKind::None => "",
    };

    if realized.action == PredicateKind::CompareWeight {
        Ok(format!(
            "{}{} {} {}{}",
            quant(realized.quant_subject),
            ent(realized.subject),
            rel,
            quant(realized.quant_reference),
            ent(realized.reference)
        ))
    } else {
        Ok(format!(
            "{} {}{} {} mm {} {}{}{}",
            act,
            quant(realized.quant_subject),
            ent(realized.subject),
            realized.magnitude,
            rel,
            quant(realized.quant_reference),
            ent(realized.reference),
            con
        ))
    }
}

/// Serialize `.semib` binary with FNV-1a integrity hash.
pub fn serialize_semib(w: &SemanticWspWeights) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&w.spec.id.to_le_bytes());
    payload.extend_from_slice(&w.spec.version.to_le_bytes());
    payload.extend_from_slice(&(SEM_DIM as u32).to_le_bytes());
    payload.extend_from_slice(&(w.w_understand.len() as u64).to_le_bytes());
    payload.extend_from_slice(&(w.w_ncm_generate.len() as u64).to_le_bytes());
    for &v in &w.w_understand {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &w.w_ncm_generate {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    let hash = fnv1a_64(&payload);
    let mut out = Vec::with_capacity(24 + payload.len());
    out.extend_from_slice(SEMIB_MAGIC);
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&hash.to_le_bytes());
    out.extend_from_slice(&payload);
    out
}
