//! Bounded-memory streaming `.safetensors` inspector (`quintessa inspect`).
//!
//! Merged from `qresearch`: supports `catalogue`, `tensor-stats`, and `analyze`
//! in bounded chunks (`--chunk-mb`, `--max-memory-mb`) without loading full models.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use quintessa::safetensors_writer::{bf16_to_f64, f16_to_f64, MathRank, WeightOrigin};

#[derive(Clone, Copy, Debug)]
enum StreamDtype {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
    Unknown,
}

impl StreamDtype {
    fn from_str(s: &str) -> Self {
        match s {
            "F64" => Self::F64,
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "I64" => Self::I64,
            "I32" => Self::I32,
            "I16" => Self::I16,
            "I8" => Self::I8,
            "U8" => Self::U8,
            "BOOL" => Self::Bool,
            _ => Self::Unknown,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::F64 => "F64",
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::I64 => "I64",
            Self::I32 => "I32",
            Self::I16 => "I16",
            Self::I8 => "I8",
            Self::U8 => "U8",
            Self::Bool => "BOOL",
            Self::Unknown => "UNKNOWN",
        }
    }

    fn byte_size(self) -> usize {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 | Self::U8 | Self::Bool | Self::Unknown => 1,
        }
    }
}

struct HeaderEntry {
    name: String,
    dtype: StreamDtype,
    shape: Vec<usize>,
    off_start: u64,
    off_end: u64,
}

impl HeaderEntry {
    fn num_elems(&self) -> u64 {
        if self.shape.is_empty() {
            1
        } else {
            self.shape
                .iter()
                .fold(1u64, |acc, &d| acc.saturating_mul(d as u64))
        }
    }

    fn storage_bytes(&self) -> u64 {
        self.off_end.saturating_sub(self.off_start)
    }
}

type HeaderResult = (u64, Vec<(String, String)>, Vec<HeaderEntry>);

fn read_header(file: &mut File) -> io::Result<HeaderResult> {
    file.seek(SeekFrom::Start(0))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let hlen = u64::from_le_bytes(len_buf);
    if hlen > 128 * 1024 * 1024 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Header exceeds 128 MB bound",
        ));
    }
    let mut hdr_bytes = vec![0u8; hlen as usize];
    file.read_exact(&mut hdr_bytes)?;
    let hdr = std::str::from_utf8(&hdr_bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid UTF-8 header"))?;
    let bytes = hdr.as_bytes();
    let mut pos = 0usize;
    let mut meta_kv: Vec<(String, String)> = Vec::new();
    let mut out = Vec::new();

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
                s.push(b[*p] as char);
                *p += 1;
            } else {
                s.push(c as char);
            }
        }
        None
    }
    fn parse_meta_obj(b: &[u8], p: &mut usize, out: &mut Vec<(String, String)>) {
        skip_ws(b, p);
        if *p >= b.len() || b[*p] != b'{' {
            return;
        }
        *p += 1;
        loop {
            skip_ws(b, p);
            if *p >= b.len() || b[*p] == b'}' {
                if *p < b.len() {
                    *p += 1;
                }
                break;
            }
            let Some(k) = parse_str(b, p) else { break };
            skip_ws(b, p);
            if *p < b.len() && b[*p] == b':' {
                *p += 1;
            }
            if let Some(v) = parse_str(b, p) {
                out.push((k, v));
            } else {
                skip_obj(b, p);
            }
            skip_ws(b, p);
            if *p < b.len() && b[*p] == b',' {
                *p += 1;
            }
        }
    }
    fn skip_obj(b: &[u8], p: &mut usize) {
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
    fn parse_u64_list(b: &[u8], p: &mut usize) -> Option<Vec<u64>> {
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
            let num: u64 = std::str::from_utf8(&b[start..*p]).ok()?.parse().ok()?;
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
            parse_meta_obj(bytes, &mut pos, &mut meta_kv);
        } else {
            skip_ws(bytes, &mut pos);
            if pos < bytes.len() && bytes[pos] == b'{' {
                pos += 1;
            }
            let mut dtype = StreamDtype::Unknown;
            let mut shape = Vec::new();
            let mut off_start = 0u64;
            let mut off_end = 0u64;
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
                        dtype =
                            StreamDtype::from_str(&parse_str(bytes, &mut pos).unwrap_or_default())
                    }
                    "shape" => {
                        if let Some(s) = parse_u64_list(bytes, &mut pos) {
                            shape = s.into_iter().map(|v| v as usize).collect();
                        }
                    }
                    "data_offsets" => {
                        if let Some(o) = parse_u64_list(bytes, &mut pos) {
                            if o.len() == 2 {
                                off_start = o[0];
                                off_end = o[1];
                            }
                        }
                    }
                    _ => {}
                }
                skip_ws(bytes, &mut pos);
                if pos < bytes.len() && bytes[pos] == b',' {
                    pos += 1;
                }
            }
            out.push(HeaderEntry {
                name: key,
                dtype,
                shape,
                off_start,
                off_end,
            });
        }
        skip_ws(bytes, &mut pos);
        if pos < bytes.len() && bytes[pos] == b',' {
            pos += 1;
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok((8 + hlen, meta_kv, out))
}

fn detect_provenance(meta: &[(String, String)]) -> (WeightOrigin, &'static str) {
    let find = |k: &str| meta.iter().find(|(mk, _)| mk == k).map(|(_, v)| v.as_str());
    if let Some(orig) = find("weight_origin").and_then(WeightOrigin::from_str_opt) {
        let label = match orig {
            WeightOrigin::Compiled => "COMPILED (Quintessa analytical WSP/SACI)",
            WeightOrigin::Learned => "LEARNED (gradient-trained weights, NOT compiled)",
            WeightOrigin::Hybrid => "HYBRID (compiled circuits + learned weights)",
            WeightOrigin::Initialized => {
                "INITIALIZED (untrained architectural prior, NOT compiled)"
            }
            WeightOrigin::ExternalLearned => "LEARNED (external / non-compiled weights)",
        };
        return (orig, label);
    }
    if let Some(fmt) = find("format") {
        if fmt.starts_with("quintessa") {
            return (WeightOrigin::Compiled, "COMPILED (Quintessa legacy header)");
        }
    }
    (
        WeightOrigin::ExternalLearned,
        "LEARNED (external / non-compiled weights)",
    )
}

fn stream_stats(
    file: &mut File,
    data_base: u64,
    e: &HeaderEntry,
    chunk_bytes: usize,
) -> io::Result<(u64, f64, f64, f64, f64, f64, u64, u64)> {
    let esz = e.dtype.byte_size();
    let aligned = ((chunk_bytes / esz).max(1)) * esz;
    let mut buf = vec![0u8; aligned.min(e.storage_bytes() as usize).max(esz)];
    file.seek(SeekFrom::Start(data_base + e.off_start))?;
    let mut rem = e.storage_bytes() as usize;

    let (mut cnt, mut sum, mut sum_sq, mut mn, mut mx, mut zeros, mut nf) = (
        0u64,
        0.0_f64,
        0.0_f64,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0u64,
        0u64,
    );
    while rem > 0 {
        let n = (rem.min(buf.len()) / esz) * esz;
        if n == 0 {
            break;
        }
        file.read_exact(&mut buf[..n])?;
        rem -= n;
        for c in buf[..n].chunks_exact(esz) {
            let v = match e.dtype {
                StreamDtype::F64 => f64::from_le_bytes(c.try_into().unwrap()),
                StreamDtype::F32 => f32::from_le_bytes(c.try_into().unwrap()) as f64,
                StreamDtype::F16 => f16_to_f64(u16::from_le_bytes(c.try_into().unwrap())),
                StreamDtype::BF16 => bf16_to_f64(u16::from_le_bytes(c.try_into().unwrap())),
                StreamDtype::I64 => i64::from_le_bytes(c.try_into().unwrap()) as f64,
                StreamDtype::I32 => i32::from_le_bytes(c.try_into().unwrap()) as f64,
                StreamDtype::I16 => i16::from_le_bytes(c.try_into().unwrap()) as f64,
                StreamDtype::I8 => (c[0] as i8) as f64,
                StreamDtype::U8 | StreamDtype::Bool | StreamDtype::Unknown => c[0] as f64,
            };
            if !v.is_finite() {
                nf += 1;
                continue;
            }
            cnt += 1;
            sum += v;
            sum_sq += v * v;
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
            if v.abs() < 1e-12 {
                zeros += 1;
            }
        }
    }
    let mean = if cnt > 0 { sum / (cnt as f64) } else { 0.0 };
    let std = if cnt > 0 {
        (sum_sq / (cnt as f64) - mean * mean).max(0.0).sqrt()
    } else {
        0.0
    };
    Ok((
        cnt,
        mean,
        std,
        if cnt > 0 { mn } else { 0.0 },
        if cnt > 0 { mx } else { 0.0 },
        sum_sq.sqrt(),
        zeros,
        nf,
    ))
}

fn decode_elem(dtype: StreamDtype, c: &[u8]) -> f64 {
    match dtype {
        StreamDtype::F64 => f64::from_le_bytes(c.try_into().unwrap()),
        StreamDtype::F32 => f32::from_le_bytes(c.try_into().unwrap()) as f64,
        StreamDtype::F16 => f16_to_f64(u16::from_le_bytes(c.try_into().unwrap())),
        StreamDtype::BF16 => bf16_to_f64(u16::from_le_bytes(c.try_into().unwrap())),
        StreamDtype::I64 => i64::from_le_bytes(c.try_into().unwrap()) as f64,
        StreamDtype::I32 => i32::from_le_bytes(c.try_into().unwrap()) as f64,
        StreamDtype::I16 => i16::from_le_bytes(c.try_into().unwrap()) as f64,
        StreamDtype::I8 => (c[0] as i8) as f64,
        StreamDtype::U8 | StreamDtype::Bool | StreamDtype::Unknown => c[0] as f64,
    }
}

/// Bounded-RAM streaming diff between two matching-shape tensors (`max_abs_delta`, `rms_delta`, `l2_delta`, `cosine`).
fn stream_diff_tensors(
    fa: &mut File,
    base_a: u64,
    ea: &HeaderEntry,
    fb: &mut File,
    base_b: u64,
    eb: &HeaderEntry,
    chunk_elems: usize,
) -> io::Result<(f64, f64, f64, f64)> {
    let esz_a = ea.dtype.byte_size();
    let esz_b = eb.dtype.byte_size();
    let total = ea.num_elems().min(eb.num_elems()) as usize;
    let step = chunk_elems.max(1).min(total.max(1));
    let mut buf_a = vec![0u8; step * esz_a];
    let mut buf_b = vec![0u8; step * esz_b];

    fa.seek(SeekFrom::Start(base_a + ea.off_start))?;
    fb.seek(SeekFrom::Start(base_b + eb.off_start))?;

    let mut rem = total;
    let (mut max_abs, mut diff_sq, mut dot, mut na_sq, mut nb_sq) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

    while rem > 0 {
        let n = rem.min(step);
        fa.read_exact(&mut buf_a[..n * esz_a])?;
        fb.read_exact(&mut buf_b[..n * esz_b])?;
        for i in 0..n {
            let va = decode_elem(ea.dtype, &buf_a[i * esz_a..(i + 1) * esz_a]);
            let vb = decode_elem(eb.dtype, &buf_b[i * esz_b..(i + 1) * esz_b]);
            let d = (va - vb).abs();
            if d > max_abs {
                max_abs = d;
            }
            diff_sq += d * d;
            dot += va * vb;
            na_sq += va * va;
            nb_sq += vb * vb;
        }
        rem -= n;
    }

    let rms = if total > 0 {
        (diff_sq / (total as f64)).sqrt()
    } else {
        0.0
    };
    let denom = (na_sq.sqrt() * nb_sq.sqrt()).max(1e-18);
    let cos = (dot / denom).clamp(-1.0, 1.0);
    Ok((max_abs, rms, diff_sq.sqrt(), cos))
}

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  quintessa inspect catalogue <file.safetensors>");
        eprintln!(
            "  quintessa inspect tensor-stats <file.safetensors> --tensor <name> [--chunk-mb 8]"
        );
        eprintln!("  quintessa inspect analyze <file.safetensors> [--chunk-mb 8] [--max-memory-mb 64] [--out <stats.json>]");
        eprintln!("  quintessa inspect diff <a.safetensors> <b.safetensors> [--chunk-mb 4]");
        std::process::exit(2);
    }
    let sub = args[0].as_str();
    let path = Path::new(&args[1]);
    let mut file = File::open(path)?;
    let (data_base, meta_kv, entries) = read_header(&mut file)?;
    let (origin, origin_label) = detect_provenance(&meta_kv);
    let f64_count = entries
        .iter()
        .filter(|e| matches!(e.dtype, StreamDtype::F64))
        .count();

    match sub {
        "catalogue" => {
            let total_p: u64 = entries.iter().map(|e| e.num_elems()).sum();
            let total_b: u64 = entries.iter().map(|e| e.storage_bytes()).sum();
            println!(
                "File: {} | Tensors: {} | Params: {} | Bytes: {:.2} MB",
                path.display(),
                entries.len(),
                total_p,
                (total_b as f64) / 1e6
            );
            println!(
                "Provenance: {} [weight_origin={}, is_compiled={}]",
                origin_label,
                origin.as_str(),
                origin.is_compiled_str()
            );
            if f64_count > 0 {
                println!(
                    "Precision Warning: {f64_count} F64 tensor(s) detected (FP64 can fail/downcast on GPUs or drift on ill-conditioned ops; fix by capping to F32/BF16 or isolating exact math in fixed-point SACI)"
                );
            }
            for e in &entries {
                println!(
                    "  {:<46} | {:<8} | {:<4} | {:<18?} | {:>10}",
                    e.name,
                    MathRank::from_shape(&e.shape).as_str(),
                    e.dtype.as_str(),
                    e.shape,
                    e.num_elems()
                );
            }
        }
        "tensor-stats" => {
            let name = parse_flag(args, "--tensor").ok_or("Missing --tensor <name>")?;
            let chunk_mb: usize = parse_flag(args, "--chunk-mb")
                .and_then(|s| s.parse().ok())
                .unwrap_or(8)
                .clamp(1, 128);
            let e = entries
                .iter()
                .find(|x| x.name == name)
                .ok_or_else(|| format!("Tensor not found: {name}"))?;
            let (cnt, mean, std, mn, mx, l2, zeros, nf) =
                stream_stats(&mut file, data_base, e, chunk_mb * 1024 * 1024)?;
            let zpct = if cnt > 0 {
                100.0 * (zeros as f64) / (cnt as f64)
            } else {
                0.0
            };
            println!(
                "{} | rank={} dtype={} shape={:?} mean={:.5} std={:.5} min={:.4} max={:.4} l2={:.2} zero={:.1}% nonfinite={}",
                e.name,
                MathRank::from_shape(&e.shape).as_str(),
                e.dtype.as_str(),
                e.shape,
                mean,
                std,
                mn,
                mx,
                l2,
                zpct,
                nf
            );
        }
        "analyze" => {
            let chunk_mb: usize = parse_flag(args, "--chunk-mb")
                .and_then(|s| s.parse().ok())
                .unwrap_or(8);
            let max_mb: usize = parse_flag(args, "--max-memory-mb")
                .and_then(|s| s.parse().ok())
                .unwrap_or(64);
            if chunk_mb > max_mb {
                return Err(
                    format!("chunk_mb ({chunk_mb}) exceeds max_memory_mb ({max_mb})").into(),
                );
            }
            let mut out_f = match parse_flag(args, "--out") {
                Some(p) => Some(File::create(p)?),
                None => None,
            };
            if let Some(ref mut f) = out_f {
                writeln!(f, "[")?;
            }
            for (i, e) in entries.iter().enumerate() {
                let (cnt, mean, std, mn, mx, l2, zeros, nf) =
                    stream_stats(&mut file, data_base, e, chunk_mb * 1024 * 1024)?;
                let zpct = if cnt > 0 {
                    100.0 * (zeros as f64) / (cnt as f64)
                } else {
                    0.0
                };
                if let Some(ref mut f) = out_f {
                    let comma = if i + 1 < entries.len() { "," } else { "" };
                    writeln!(
                        f,
                        "  {{\"name\":\"{}\",\"rank\":\"{}\",\"dtype\":\"{}\",\"shape\":{:?},\"mean\":{:.6},\"std\":{:.6},\"min\":{:.4},\"max\":{:.4},\"l2\":{:.2},\"zero_pct\":{:.2},\"nonfinite\":{}}}{}",
                        e.name,
                        MathRank::from_shape(&e.shape).as_str(),
                        e.dtype.as_str(),
                        e.shape,
                        mean,
                        std,
                        mn,
                        mx,
                        l2,
                        zpct,
                        nf,
                        comma
                    )?;
                } else {
                    println!(
                        "{:<44} | {:<8} | {:<4} | mean={:>8.4} std={:>7.4} min={:>8.3} max={:>8.3} nf={}",
                        e.name,
                        MathRank::from_shape(&e.shape).as_str(),
                        e.dtype.as_str(),
                        mean,
                        std,
                        mn,
                        mx,
                        nf
                    );
                }
            }
            if let Some(ref mut f) = out_f {
                writeln!(f, "]")?;
            }
            println!(
                "Stream-analyzed {} tensors ({} MB chunks, {} MB budget)",
                entries.len(),
                chunk_mb,
                max_mb
            );
        }
        "diff" => {
            if args.len() < 3 {
                return Err(
                    "Usage: quintessa inspect diff <a.safetensors> <b.safetensors> [--chunk-mb 4]"
                        .into(),
                );
            }
            let path_b = Path::new(&args[2]);
            let mut file_b = File::open(path_b)?;
            let (base_b, _meta_b, entries_b) = read_header(&mut file_b)?;
            let chunk_mb: usize = parse_flag(args, "--chunk-mb")
                .and_then(|s| s.parse().ok())
                .unwrap_or(4)
                .clamp(1, 64);
            let chunk_elems = (chunk_mb * 1024 * 1024) / 8;

            println!("Diff: {} vs {}", path.display(), path_b.display());
            let mut matched = 0usize;
            let mut missing_in_b = 0usize;
            let mut worst_max_abs = 0.0_f64;

            for ea in &entries {
                if let Some(eb) = entries_b.iter().find(|x| x.name == ea.name) {
                    matched += 1;
                    if ea.shape != eb.shape {
                        println!(
                            "  {:<44} | SHAPE MISMATCH {:?} vs {:?}",
                            ea.name, ea.shape, eb.shape
                        );
                        continue;
                    }
                    let (max_abs, rms, l2, cos) = stream_diff_tensors(
                        &mut file,
                        data_base,
                        ea,
                        &mut file_b,
                        base_b,
                        eb,
                        chunk_elems,
                    )?;
                    if max_abs > worst_max_abs {
                        worst_max_abs = max_abs;
                    }
                    println!(
                        "  {:<44} | {:>4}->{:<4} | max_abs={:>9.5} rms={:>9.5} l2={:>9.4} cos={:.6}",
                        ea.name,
                        ea.dtype.as_str(),
                        eb.dtype.as_str(),
                        max_abs,
                        rms,
                        l2,
                        cos
                    );
                } else {
                    missing_in_b += 1;
                    println!("  {:<44} | ONLY IN A", ea.name);
                }
            }
            let only_in_b = entries_b
                .iter()
                .filter(|eb| !entries.iter().any(|ea| ea.name == eb.name))
                .count();
            println!(
                "Summary: matched={matched}, only_in_a={missing_in_b}, only_in_b={only_in_b}, worst_max_abs={worst_max_abs:.6}"
            );
        }
        other => return Err(format!("Unknown inspect subcommand: {other}").into()),
    }
    Ok(())
}
