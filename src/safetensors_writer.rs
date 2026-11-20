//! Zero-dependency SafeTensors pure-weights exporter.
//!
//! Writes standard `.safetensors` files (`u64` LE header length + UTF-8 JSON index +
//! contiguous raw little-endian weight buffers) with no external crates.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum WeightDtype {
    F32 = 0,
    F64 = 1,
    BF16 = 2,
    F16 = 3,
    I8 = 4,
}

impl WeightDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            WeightDtype::F32 => "F32",
            WeightDtype::F64 => "F64",
            WeightDtype::BF16 => "BF16",
            WeightDtype::F16 => "F16",
            WeightDtype::I8 => "I8",
        }
    }

    pub fn byte_size(self) -> usize {
        match self {
            WeightDtype::F64 => 8,
            WeightDtype::F32 => 4,
            WeightDtype::BF16 | WeightDtype::F16 => 2,
            WeightDtype::I8 => 1,
        }
    }
}

/// Explicit provenance classification stamped into `.safetensors` `__metadata__`
/// so any reader or inspector immediately knows whether weights are compiled or learned.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightOrigin {
    /// Analytically synthesized by Quintessa (WSP / SACI / ETF closed-form compiler).
    Compiled = 0,
    /// Trained empirically via gradient descent / backprop / fine-tuning (NOT compiled).
    Learned = 1,
    /// Mixed checkpoint containing both compiled circuits and learned weights.
    Hybrid = 2,
    /// Architectural skeleton / structured prior prior to training or WSP fitting.
    Initialized = 3,
    /// External `.safetensors` file with no Quintessa provenance header (treated as Learned).
    ExternalLearned = 4,
}

impl WeightOrigin {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Compiled => "compiled",
            Self::Learned => "learned",
            Self::Hybrid => "hybrid",
            Self::Initialized => "initialized",
            Self::ExternalLearned => "learned_external",
        }
    }

    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "compiled" => Some(Self::Compiled),
            "learned" | "trained" => Some(Self::Learned),
            "hybrid" | "mixed" => Some(Self::Hybrid),
            "initialized" | "init" | "prior" => Some(Self::Initialized),
            "learned_external" | "external" => Some(Self::ExternalLearned),
            _ => None,
        }
    }

    #[inline]
    pub const fn is_compiled_str(self) -> &'static str {
        match self {
            Self::Compiled => "true",
            Self::Hybrid => "partial",
            Self::Learned | Self::Initialized | Self::ExternalLearned => "false",
        }
    }
}

/// Mathematical rank classification of a tensor (Scalar, Vector, Matrix, 3D, 4D, ND).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MathRank {
    Scalar = 0,
    Vector = 1,
    Matrix = 2,
    Tensor3D = 3,
    Tensor4D = 4,
    TensorND = 5,
}

impl MathRank {
    #[inline]
    pub fn from_shape(shape: &[usize]) -> Self {
        match shape.len() {
            0 => Self::Scalar,
            1 => Self::Vector,
            2 => Self::Matrix,
            3 => Self::Tensor3D,
            4 => Self::Tensor4D,
            _ => Self::TensorND,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "Scalar",
            Self::Vector => "Vector",
            Self::Matrix => "Matrix",
            Self::Tensor3D => "Tensor3D",
            Self::Tensor4D => "Tensor4D",
            Self::TensorND => "TensorND",
        }
    }
}

pub struct NamedTensor<'a> {
    pub name: &'a str,
    pub shape: Vec<usize>,
    pub data: &'a [f64],
}

/// Per-tensor mixed-precision descriptor: each feature/block chooses its own storage `WeightDtype`.
pub struct TypedNamedTensor<'a> {
    pub name: &'a str,
    pub shape: Vec<usize>,
    pub dtype: WeightDtype,
    pub data: &'a [f64],
}

/// Pre-emission weight-space health summary across all tensors in a `.safetensors` artifact.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WeightHealthSummary {
    pub total_tensors: u32,
    pub nonfinite_count: u64,
    pub f16_overflow_count: u64,
    pub max_abs_weight: f64,
    pub outlier_tensor_count: u32,
    pub passed: bool,
}

/// Round-to-nearest-even `f64` -> `BF16` (`u16`).
#[inline]
pub fn f64_to_bf16(v: f64) -> u16 {
    let bits = (v as f32).to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

/// Convert `BF16` (`u16`) -> `f64`.
#[inline]
pub fn bf16_to_f64(bits: u16) -> f64 {
    f32::from_bits((bits as u32) << 16) as f64
}

/// Convert `f64` -> IEEE-754 half-precision `F16` (`u16`).
#[inline]
pub fn f64_to_f16(v: f64) -> u16 {
    let f = v as f32;
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    if exp == 255 {
        return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7C00; // infinity
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> (14 - new_exp);
        return sign | (m as u16);
    }
    sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

/// Convert IEEE-754 half-precision `F16` (`u16`) -> `f64`.
#[inline]
pub fn f16_to_f64(bits: u16) -> f64 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return if sign != 0 { -0.0 } else { 0.0 };
        }
        let val = (mant as f64) * 2.0_f64.powi(-24);
        return if sign != 0 { -val } else { val };
    }
    if exp == 31 {
        return if mant == 0 {
            if sign != 0 {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        } else {
            f64::NAN
        };
    }
    let val = (1.0 + (mant as f64) / 1024.0) * 2.0_f64.powi(exp - 15);
    if sign != 0 {
        -val
    } else {
        val
    }
}

/// Pre-emission weight-space health gate: checks for non-finite values, `F16` overflow
/// (`|w| > 65504.0`), and extreme outlier magnitudes (`|w| > 100.0`).
pub fn validate_mixed_tensor_health(tensors: &[TypedNamedTensor<'_>]) -> WeightHealthSummary {
    let mut nonfinite = 0u64;
    let mut f16_ovf = 0u64;
    let mut max_abs = 0.0_f64;
    let mut outlier_tensors = 0u32;

    for t in tensors {
        let mut t_max = 0.0_f64;
        for &v in t.data {
            if !v.is_finite() {
                nonfinite += 1;
                continue;
            }
            let av = v.abs();
            if av > t_max {
                t_max = av;
            }
            if t.dtype == WeightDtype::F16 && av > 65504.0 {
                f16_ovf += 1;
            }
        }
        if t_max > max_abs {
            max_abs = t_max;
        }
        if t_max > 100.0 && matches!(t.dtype, WeightDtype::F16 | WeightDtype::I8) {
            outlier_tensors += 1;
        }
    }

    WeightHealthSummary {
        total_tensors: tensors.len() as u32,
        nonfinite_count: nonfinite,
        f16_overflow_count: f16_ovf,
        max_abs_weight: max_abs,
        outlier_tensor_count: outlier_tensors,
        passed: nonfinite == 0 && f16_ovf == 0,
    }
}

fn push_json_escaped(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Write a pure-weights `.safetensors` file and return the total bytes written.
pub fn write_safetensors(
    path: &Path,
    dtype: WeightDtype,
    metadata: &[(&str, &str)],
    tensors: &[NamedTensor<'_>],
) -> io::Result<u64> {
    if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "SafeTensors output path must strictly have a .safetensors extension",
        ));
    }
    let elem_bytes = dtype.byte_size();
    let mut header = String::with_capacity(tensors.len() * 96 + 256);
    header.push('{');

    let mut first_entry = true;
    if !metadata.is_empty() {
        header.push_str("\"__metadata__\":{");
        for (i, &(k, v)) in metadata.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            push_json_escaped(&mut header, k);
            header.push(':');
            push_json_escaped(&mut header, v);
        }
        header.push('}');
        first_entry = false;
    }

    let mut offset: usize = 0;
    for t in tensors {
        let shape_elems = t
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "SafeTensors tensor shape element count overflow",
                )
            })?;
        if shape_elems != t.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "SafeTensors tensor shape product does not match data length",
            ));
        }
        if !first_entry {
            header.push(',');
        }
        first_entry = false;

        let nbytes = t.data.len() * elem_bytes;
        let end = offset + nbytes;

        push_json_escaped(&mut header, t.name);
        header.push_str(":{\"dtype\":\"");
        header.push_str(dtype.as_str());
        header.push_str("\",\"shape\":[");
        for (i, &dim) in t.shape.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            header.push_str(&dim.to_string());
        }
        header.push_str("],\"data_offsets\":[");
        header.push_str(&offset.to_string());
        header.push(',');
        header.push_str(&end.to_string());
        header.push_str("]}");

        offset = end;
    }
    header.push('}');

    // Pad JSON header with trailing spaces to 8-byte alignment per SafeTensors convention.
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }

    let header_bytes = header.as_bytes();
    let header_len = header_bytes.len() as u64;

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    w.write_all(&header_len.to_le_bytes())?;
    w.write_all(header_bytes)?;
    for t in tensors {
        write_typed_data(&mut w, dtype, t.data)?;
    }
    w.flush()?;

    Ok(8 + header_len + offset as u64)
}

/// Bulk-encode a tile of `f64` values into little-endian bytes for `dtype`,
/// appending into a preallocated reusable `dst` byte buffer (zero per-tile heap allocation
/// once `dst` has capacity `>= src.len() * dtype.byte_size()`).
///
/// Simultaneously updates in-flight health counters `(nonfinite, f16_overflow, max_abs)`
/// in a single cache-friendly pass.
pub fn encode_f64_tile_to_bytes(
    dtype: WeightDtype,
    src: &[f64],
    dst: &mut Vec<u8>,
) -> (u64, u64, f64) {
    dst.clear();
    let needed = src.len() * dtype.byte_size();
    if dst.capacity() < needed {
        dst.reserve(needed - dst.capacity());
    }
    let mut nonfinite = 0u64;
    let mut f16_ovf = 0u64;
    let mut max_abs = 0.0_f64;

    match dtype {
        WeightDtype::F64 => {
            for &val in src {
                if !val.is_finite() {
                    nonfinite += 1;
                } else {
                    let av = val.abs();
                    if av > max_abs {
                        max_abs = av;
                    }
                }
                dst.extend_from_slice(&val.to_le_bytes());
            }
        }
        WeightDtype::F32 => {
            for &val in src {
                if !val.is_finite() {
                    nonfinite += 1;
                } else {
                    let av = val.abs();
                    if av > max_abs {
                        max_abs = av;
                    }
                }
                dst.extend_from_slice(&(val as f32).to_le_bytes());
            }
        }
        WeightDtype::BF16 => {
            for &val in src {
                if !val.is_finite() {
                    nonfinite += 1;
                } else {
                    let av = val.abs();
                    if av > max_abs {
                        max_abs = av;
                    }
                }
                dst.extend_from_slice(&f64_to_bf16(val).to_le_bytes());
            }
        }
        WeightDtype::F16 => {
            for &val in src {
                if !val.is_finite() {
                    nonfinite += 1;
                } else {
                    let av = val.abs();
                    if av > max_abs {
                        max_abs = av;
                    }
                    if av > 65504.0 {
                        f16_ovf += 1;
                    }
                }
                dst.extend_from_slice(&f64_to_f16(val).to_le_bytes());
            }
        }
        WeightDtype::I8 => {
            for &val in src {
                if !val.is_finite() {
                    nonfinite += 1;
                } else {
                    let av = val.abs();
                    if av > max_abs {
                        max_abs = av;
                    }
                }
                let q = val.round().clamp(-127.0, 127.0) as i8;
                dst.push(q as u8);
            }
        }
    }
    (nonfinite, f16_ovf, max_abs)
}

fn write_typed_data<W: Write>(w: &mut W, dtype: WeightDtype, data: &[f64]) -> io::Result<()> {
    const CHUNK_ELEMS: usize = 16_384;
    let mut buf = Vec::with_capacity(CHUNK_ELEMS * dtype.byte_size());
    for chunk in data.chunks(CHUNK_ELEMS) {
        let _ = encode_f64_tile_to_bytes(dtype, chunk, &mut buf);
        w.write_all(&buf)?;
    }
    Ok(())
}

/// Write a per-tensor mixed-precision `.safetensors` file (strictly `.safetensors` extension)
/// after validating pre-emission weight-space health (zero NaNs/Infs, zero `F16` overflows).
pub fn write_mixed_safetensors(
    path: &Path,
    metadata: &[(&str, &str)],
    tensors: &[TypedNamedTensor<'_>],
) -> io::Result<(u64, WeightHealthSummary)> {
    if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "SafeTensors output path must strictly have a .safetensors extension",
        ));
    }
    let health = validate_mixed_tensor_health(tensors);
    if !health.passed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Pre-emission weight-space health gate failed (nonfinite={}, f16_overflow={})",
                health.nonfinite_count, health.f16_overflow_count
            ),
        ));
    }

    let mut header = String::with_capacity(tensors.len() * 96 + 256);
    header.push('{');

    let mut first_entry = true;
    if !metadata.is_empty() {
        header.push_str("\"__metadata__\":{");
        for (i, &(k, v)) in metadata.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            push_json_escaped(&mut header, k);
            header.push(':');
            push_json_escaped(&mut header, v);
        }
        header.push('}');
        first_entry = false;
    }

    let mut offset: usize = 0;
    for t in tensors {
        let elem_bytes = t.dtype.byte_size();
        let shape_elems = t
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "SafeTensors tensor shape element count overflow",
                )
            })?;
        if shape_elems != t.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "SafeTensors tensor shape product does not match data length",
            ));
        }
        if !first_entry {
            header.push(',');
        }
        first_entry = false;

        let byte_len = t.data.len().checked_mul(elem_bytes).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "SafeTensors byte size overflow",
            )
        })?;
        let end = offset.checked_add(byte_len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "SafeTensors offset overflow")
        })?;

        push_json_escaped(&mut header, t.name);
        header.push_str(":{\"dtype\":\"");
        header.push_str(t.dtype.as_str());
        header.push_str("\",\"shape\":[");
        for (i, &dim) in t.shape.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            header.push_str(&dim.to_string());
        }
        header.push_str("],\"data_offsets\":[");
        header.push_str(&offset.to_string());
        header.push(',');
        header.push_str(&end.to_string());
        header.push_str("]}");

        offset = end;
    }
    header.push('}');

    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }

    let header_bytes = header.as_bytes();
    let header_len = header_bytes.len() as u64;

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    w.write_all(&header_len.to_le_bytes())?;
    w.write_all(header_bytes)?;
    for t in tensors {
        write_typed_data(&mut w, t.dtype, t.data)?;
    }
    w.flush()?;

    Ok((8 + header_len + offset as u64, health))
}

#[derive(Clone, Debug)]
pub struct LoadedTensor {
    pub name: String,
    pub dtype: WeightDtype,
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

/// Zero-dependency SafeTensors reader (cold-path) supporting `F32` and `F64` tensors.
pub fn read_safetensors(path: &Path) -> io::Result<Vec<LoadedTensor>> {
    let raw = std::fs::read(path)?;
    if raw.len() < 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "SafeTensors file too short",
        ));
    }
    let hlen = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    if 8 + hlen > raw.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "SafeTensors header length exceeds file size",
        ));
    }
    let hdr = std::str::from_utf8(&raw[8..8 + hlen])
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid UTF-8 header"))?;
    let data_base = 8 + hlen;

    let mut out = Vec::new();
    let bytes = hdr.as_bytes();
    let mut pos = 0usize;

    fn skip_ws(b: &[u8], p: &mut usize) {
        while *p < b.len() && b[*p].is_ascii_whitespace() {
            *p += 1;
        }
    }
    fn parse_str(b: &[u8], p: &mut usize) -> Option<String> {
        skip_ws(b, p);
        if *p >= b.len() || b[*p] != b'"' {
            return None;
        }
        *p += 1;
        let mut s = String::new();
        while *p < b.len() {
            let c = b[*p];
            *p += 1;
            if c == b'"' {
                return Some(s);
            }
            if c == b'\\' && *p < b.len() {
                let esc = b[*p];
                *p += 1;
                s.push(esc as char);
            } else {
                s.push(c as char);
            }
        }
        None
    }
    fn skip_json_obj(b: &[u8], p: &mut usize) {
        skip_ws(b, p);
        if *p >= b.len() || b[*p] != b'{' {
            return;
        }
        let mut depth = 0i32;
        let mut in_str = false;
        while *p < b.len() {
            let c = b[*p];
            *p += 1;
            if in_str {
                if c == b'\\' {
                    *p += 1;
                } else if c == b'"' {
                    in_str = false;
                }
            } else if c == b'"' {
                in_str = true;
            } else if c == b'{' {
                depth += 1;
            } else if c == b'}' {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
        }
    }
    fn parse_usize_list(b: &[u8], p: &mut usize) -> Option<Vec<usize>> {
        skip_ws(b, p);
        if *p >= b.len() || b[*p] != b'[' {
            return None;
        }
        *p += 1;
        let mut vals = Vec::new();
        loop {
            skip_ws(b, p);
            if *p >= b.len() {
                return None;
            }
            if b[*p] == b']' {
                *p += 1;
                return Some(vals);
            }
            let start = *p;
            while *p < b.len() && b[*p].is_ascii_digit() {
                *p += 1;
            }
            let num: usize = std::str::from_utf8(&b[start..*p]).ok()?.parse().ok()?;
            vals.push(num);
            skip_ws(b, p);
            if *p < b.len() && b[*p] == b',' {
                *p += 1;
            }
        }
    }

    skip_ws(bytes, &mut pos);
    if pos < bytes.len() && bytes[pos] == b'{' {
        pos += 1;
    }
    loop {
        skip_ws(bytes, &mut pos);
        if pos >= bytes.len() || bytes[pos] == b'}' {
            break;
        }
        let key = match parse_str(bytes, &mut pos) {
            Some(k) => k,
            None => break,
        };
        skip_ws(bytes, &mut pos);
        if pos < bytes.len() && bytes[pos] == b':' {
            pos += 1;
        }
        if key == "__metadata__" {
            skip_json_obj(bytes, &mut pos);
        } else {
            skip_ws(bytes, &mut pos);
            if pos < bytes.len() && bytes[pos] == b'{' {
                pos += 1;
            }
            let mut dtype = WeightDtype::F32;
            let mut shape = Vec::new();
            let mut offsets = [0usize; 2];
            loop {
                skip_ws(bytes, &mut pos);
                if pos >= bytes.len() || bytes[pos] == b'}' {
                    if pos < bytes.len() {
                        pos += 1;
                    }
                    break;
                }
                let fk = match parse_str(bytes, &mut pos) {
                    Some(k) => k,
                    None => break,
                };
                skip_ws(bytes, &mut pos);
                if pos < bytes.len() && bytes[pos] == b':' {
                    pos += 1;
                }
                match fk.as_str() {
                    "dtype" => {
                        let ds = parse_str(bytes, &mut pos).unwrap_or_default();
                        dtype = match ds.as_str() {
                            "F64" => WeightDtype::F64,
                            "BF16" => WeightDtype::BF16,
                            "F16" => WeightDtype::F16,
                            "I8" => WeightDtype::I8,
                            _ => WeightDtype::F32,
                        };
                    }
                    "shape" => {
                        shape = parse_usize_list(bytes, &mut pos).unwrap_or_default();
                    }
                    "data_offsets" => {
                        let offs = parse_usize_list(bytes, &mut pos).unwrap_or_default();
                        if offs.len() == 2 {
                            offsets = [offs[0], offs[1]];
                        }
                    }
                    _ => {}
                }
                skip_ws(bytes, &mut pos);
                if pos < bytes.len() && bytes[pos] == b',' {
                    pos += 1;
                }
            }
            let start = data_base + offsets[0];
            let end = data_base + offsets[1];
            if end <= raw.len() && start <= end {
                let slice = &raw[start..end];
                let data: Vec<f64> = match dtype {
                    WeightDtype::F32 => slice
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                        .collect(),
                    WeightDtype::F64 => slice
                        .chunks_exact(8)
                        .map(|c| {
                            f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect(),
                    WeightDtype::BF16 => slice
                        .chunks_exact(2)
                        .map(|c| bf16_to_f64(u16::from_le_bytes([c[0], c[1]])))
                        .collect(),
                    WeightDtype::F16 => slice
                        .chunks_exact(2)
                        .map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]])))
                        .collect(),
                    WeightDtype::I8 => slice.iter().map(|&b| (b as i8) as f64).collect(),
                };
                out.push(LoadedTensor {
                    name: key,
                    dtype,
                    shape,
                    data,
                });
            }
        }
        skip_ws(bytes, &mut pos);
        if pos < bytes.len() && bytes[pos] == b',' {
            pos += 1;
        }
    }

    Ok(out)
}
