//! FHP (FoundationHead Package) binary serializer and loader.
//!
//! Supports:
//! - `ADAFHP01`: Original static Pipeline A package (488-byte header for 256/96/3-SACI).
//! - `ADAFHP02`: Lossless static Pipeline A package including `WorkspaceIr` configuration.
//!
//! Dynamic WSP masks (`neuron_masks`, `layer_masks`, `compute_mask`) are per-query
//! runtime state (`WspScratch`) and are intentionally excluded from the static model
//! binary so candidate availability (`legal_masks`) is never conflated with live activation.

use std::fmt;
use std::io::{self, Read, Write};

use crate::etf::FeatureEtfIr;
use crate::ir::{
    ArchitectureIr, AttentionIr, KdaConfig, LayerSpec, MlaConfig, SaciIr, SaciOpSpec, WorkspaceIr,
};
use crate::runtime::CompiledAdaModel;
use crate::sys::{AttentionKind, Backend, LayerId, NumericalMode, SaciOpId};
use crate::wsp::StaticWspMetadata;

pub const FHP_MAGIC_V1: &[u8; 8] = b"ADAFHP01";
pub const FHP_MAGIC_V2: &[u8; 8] = b"ADAFHP02";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FhpVersion {
    V1,
    V2,
}

#[derive(Debug)]
pub enum FhpError {
    Io(io::Error),
    InvalidMagic([u8; 8]),
    InvalidHeader(&'static str),
    InvalidAttentionTopology,
    NonFiniteBasisValue(usize),
}

impl fmt::Display for FhpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FhpError::Io(e) => write!(f, "I/O error: {e}"),
            FhpError::InvalidMagic(m) => write!(f, "invalid FHP magic: {m:?}"),
            FhpError::InvalidHeader(msg) => write!(f, "invalid FHP header/payload: {msg}"),
            FhpError::InvalidAttentionTopology => {
                write!(f, "invalid attention topology (expected 6 KDA + 2 MLA)")
            }
            FhpError::NonFiniteBasisValue(idx) => {
                write!(f, "non-finite f64 basis value at index {idx}")
            }
        }
    }
}

impl std::error::Error for FhpError {}

impl From<io::Error> for FhpError {
    fn from(e: io::Error) -> Self {
        FhpError::Io(e)
    }
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8, FhpError> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16, FhpError> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, FhpError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, FhpError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_f64<R: Read>(r: &mut R) -> Result<f64, FhpError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

fn write_common_body<W: Write>(
    model: &CompiledAdaModel,
    include_workspace: bool,
    w: &mut W,
) -> Result<(), FhpError> {
    w.write_all(&model.ir.vq_codebook_size.to_le_bytes())?;
    w.write_all(&(model.ir.feature_etf.num_neurons as u32).to_le_bytes())?;
    w.write_all(&(model.ir.feature_etf.substrate_dim as u32).to_le_bytes())?;
    w.write_all(&(model.ir.tile_size as u32).to_le_bytes())?;
    w.write_all(&model.ir.feature_etf.welch_bound.to_le_bytes())?;
    w.write_all(&model.ir.feature_etf.max_coherence.to_le_bytes())?;

    // Attention IR (dynamic layer count & topology)
    w.write_all(&(model.ir.attention.layers.len() as u32).to_le_bytes())?;
    for layer in &model.ir.attention.layers {
        w.write_all(&[layer.id.0, layer.kind as u8])?;
        w.write_all(&layer.kda.chunk_size.to_le_bytes())?;
        w.write_all(&layer.kda.dplr_rank.to_le_bytes())?;
        w.write_all(&layer.kda.state_dim.to_le_bytes())?;
        w.write_all(&layer.mla.kv_latent_dim.to_le_bytes())?;
        w.write_all(&layer.mla.num_heads.to_le_bytes())?;
        w.write_all(&layer.mla.head_dim.to_le_bytes())?;
    }

    // SACI IR contracts
    w.write_all(&(model.ir.saci.ops.len() as u32).to_le_bytes())?;
    for op in &model.ir.saci.ops {
        w.write_all(&op.id.0.to_le_bytes())?;
        w.write_all(&[op.numerical_mode as u8, op.backend as u8])?;
        w.write_all(&op.tolerance.to_le_bytes())?;
        w.write_all(&op.precondition_flags.to_le_bytes())?;
        w.write_all(&op.postcondition_flags.to_le_bytes())?;
    }

    if include_workspace {
        w.write_all(&model.ir.workspace.scratchpad_slots.to_le_bytes())?;
        w.write_all(&model.ir.workspace.max_iters.to_le_bytes())?;
        w.write_all(&model.ir.workspace.halting_tolerance.to_le_bytes())?;
    }

    // Feature / ETF IR legal masks (candidate availability) & basis vectors
    w.write_all(&(model.ir.feature_etf.legal_masks.len() as u32).to_le_bytes())?;
    w.write_all(&model.ir.feature_etf.legal_masks)?;
    w.write_all(&(model.ir.feature_etf.basis.len() as u64).to_le_bytes())?;
    for &val in &model.ir.feature_etf.basis {
        w.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}

/// Serialize a `CompiledAdaModel` in `ADAFHP02` format (includes `WorkspaceIr`).
pub fn save_fhp<W: Write>(model: &CompiledAdaModel, w: &mut W) -> Result<(), FhpError> {
    w.write_all(FHP_MAGIC_V2)?;
    write_common_body(model, true, w)
}

/// Serialize a `CompiledAdaModel` in legacy `ADAFHP01` format.
pub fn save_fhp_v1<W: Write>(model: &CompiledAdaModel, w: &mut W) -> Result<(), FhpError> {
    w.write_all(FHP_MAGIC_V1)?;
    write_common_body(model, false, w)
}

/// Load and validate a `CompiledAdaModel` from `ADAFHP01` or `ADAFHP02` binary data.
pub fn load_fhp<R: Read>(r: &mut R) -> Result<(CompiledAdaModel, FhpVersion), FhpError> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    let version = if &magic == FHP_MAGIC_V1 {
        FhpVersion::V1
    } else if &magic == FHP_MAGIC_V2 {
        FhpVersion::V2
    } else {
        return Err(FhpError::InvalidMagic(magic));
    };

    let vq_codebook_size = read_u32(r)?;
    let num_neurons = read_u32(r)? as usize;
    let substrate_dim = read_u32(r)? as usize;
    let tile_size = read_u32(r)? as usize;
    let welch_bound = read_f64(r)?;
    let max_coherence = read_f64(r)?;

    if vq_codebook_size == 0 || num_neurons == 0 || substrate_dim < 3 || tile_size == 0 {
        return Err(FhpError::InvalidHeader(
            "zero or invalid architecture dimension",
        ));
    }

    let num_layers = read_u32(r)? as usize;
    if num_layers == 0 || num_layers > 256 {
        return Err(FhpError::InvalidAttentionTopology);
    }

    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let id = LayerId(read_u8(r)?);
        let kind = match read_u8(r)? {
            0 => AttentionKind::Kda,
            1 => AttentionKind::Mla,
            _ => return Err(FhpError::InvalidHeader("unknown AttentionKind")),
        };
        let kda = KdaConfig {
            chunk_size: read_u16(r)?,
            dplr_rank: read_u16(r)?,
            state_dim: read_u16(r)?,
        };
        let mla = MlaConfig {
            kv_latent_dim: read_u16(r)?,
            num_heads: read_u16(r)?,
            head_dim: read_u16(r)?,
        };
        layers.push(LayerSpec { id, kind, kda, mla });
    }

    let attention = AttentionIr { layers };

    let saci_len = read_u32(r)? as usize;
    let mut ops = Vec::with_capacity(saci_len);
    for _ in 0..saci_len {
        let id = SaciOpId(read_u16(r)?);
        let numerical_mode = match read_u8(r)? {
            0 => NumericalMode::Exact,
            1 => NumericalMode::CertifiedApproximate,
            _ => return Err(FhpError::InvalidHeader("unknown NumericalMode")),
        };
        let backend = match read_u8(r)? {
            0 => Backend::Cpu,
            1 => Backend::Gpu,
            _ => return Err(FhpError::InvalidHeader("unknown Backend")),
        };
        let tolerance = read_f64(r)?;
        let precondition_flags = read_u32(r)?;
        let postcondition_flags = read_u32(r)?;
        ops.push(SaciOpSpec {
            id,
            numerical_mode,
            tolerance,
            backend,
            precondition_flags,
            postcondition_flags,
        });
    }
    let saci = SaciIr::new(ops);

    let workspace = match version {
        FhpVersion::V1 => WorkspaceIr::default(),
        FhpVersion::V2 => WorkspaceIr {
            scratchpad_slots: read_u16(r)?,
            max_iters: read_u16(r)?,
            halting_tolerance: read_f64(r)?,
        },
    };

    let masks_len = read_u32(r)? as usize;
    if masks_len != num_neurons {
        return Err(FhpError::InvalidHeader("legal_masks length != num_neurons"));
    }
    let mut legal_masks = vec![0u8; masks_len];
    r.read_exact(&mut legal_masks)?;

    let basis_len = read_u64(r)? as usize;
    let expected_basis = num_neurons
        .checked_mul(4)
        .and_then(|v| v.checked_mul(substrate_dim))
        .ok_or(FhpError::InvalidHeader("basis size overflow"))?;
    if basis_len != expected_basis {
        return Err(FhpError::InvalidHeader(
            "basis length != num_neurons * 4 * substrate_dim",
        ));
    }

    let mut basis = Vec::with_capacity(basis_len);
    for idx in 0..basis_len {
        let val = read_f64(r)?;
        if !val.is_finite() {
            return Err(FhpError::NonFiniteBasisValue(idx));
        }
        basis.push(val);
    }

    let ir = ArchitectureIr {
        vq_codebook_size,
        tile_size,
        feature_etf: FeatureEtfIr {
            num_neurons,
            substrate_dim,
            basis,
            legal_masks,
            welch_bound,
            max_coherence,
        },
        attention,
        saci,
        workspace,
    };
    let static_wsp = StaticWspMetadata::from_arch_ir(&ir);

    Ok((CompiledAdaModel { ir, static_wsp }, version))
}
