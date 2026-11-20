//! Quintessa WSP Studio — Local Cold-Path Presentation Server (`quintessa studio`).
//!
//! Pure-std bounded-memory HTTP server on `127.0.0.1:<port>` serving the embedded
//! `Quintessa WSP Studio` node-graph UI and exposing:
//! - `GET  /`            -> Embedded Studio UI (`studio_ui.html`)
//! - `POST /api/compile` -> Parses `.qwf`, compiles via `compile_workflow_to_safetensors`,
//!                          runs 5-axis verifier + Weight-Space Health Gate, emits `.safetensors`
//! - `POST /api/inspect` -> Reads `.safetensors` header catalogue (bounded memory)

use std::fs::File;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;

use quintessa::observatory::format_verifier_report;
use quintessa::safetensors_writer::MathRank;
use quintessa::verifier::{VerifyConfig, VerifyStatus};
use quintessa::workflow_arch::{
    compile_workflow_to_safetensors, parse_workflow_spec, CompileProfile,
};

const STUDIO_HTML: &str = include_str!("studio_ui.html");
const MAX_REQUEST_BYTES: usize = 2 * 1024 * 1024; // 2 MB strict bound

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut port: u16 = 7744;
    for w in args.windows(2) {
        if w[0] == "--port" {
            port = w[1].parse()?;
        }
    }

    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr)?;
    println!("Quintessa WSP Studio running at http://{addr}");
    println!("Press Ctrl+C to stop.");

    for stream in listener.incoming() {
        match stream {
            Ok(mut s) => {
                let _ = handle_connection(&mut s);
            }
            Err(e) => eprintln!("Studio accept error: {e}"),
        }
    }
    Ok(())
}

fn handle_connection(stream: &mut TcpStream) -> io::Result<()> {
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    let mut header_end: Option<usize> = None;
    let mut content_len: usize = 0;

    loop {
        let n = match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => n,
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => break,
            Err(e) => return Err(e),
        };
        if buf.len() + n > MAX_REQUEST_BYTES {
            return send_json(
                stream,
                "413 Payload Too Large",
                r#"{"ok":false,"error":"Request exceeds 2 MB bound"}"#,
            );
        }
        buf.extend_from_slice(&chunk[..n]);
        if header_end.is_none() {
            if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
                header_end = Some(pos + 4);
                let hdr_str = String::from_utf8_lossy(&buf[..pos]);
                for line in hdr_str.lines() {
                    if let Some((k, v)) = line.split_once(':') {
                        if k.trim().eq_ignore_ascii_case("content-length") {
                            content_len = v.trim().parse().unwrap_or(0);
                        }
                    }
                }
            }
        }
        if let Some(hend) = header_end {
            if buf.len() >= hend + content_len {
                break;
            }
        }
    }

    let Some(hend) = header_end else {
        return Ok(());
    };
    let head = String::from_utf8_lossy(&buf[..hend]);
    let first_line = head.lines().next().unwrap_or("");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("/");
    let body = &buf[hend..buf.len().min(hend + content_len)];
    let body_str = String::from_utf8_lossy(body);

    match (method, path) {
        ("GET", "/") | ("GET", "/index.html") => send_html(stream, STUDIO_HTML),
        ("POST", "/api/compile") => {
            let resp = handle_compile(&body_str);
            send_json(stream, "200 OK", &resp)
        }
        ("POST", "/api/inspect") => {
            let resp = handle_inspect(&body_str);
            send_json(stream, "200 OK", &resp)
        }
        _ => send_json(
            stream,
            "404 Not Found",
            r#"{"ok":false,"error":"Endpoint not found"}"#,
        ),
    }
}

fn handle_compile(body: &str) -> String {
    let qwf = match extract_json_string(body, "qwf") {
        Some(s) => s,
        None => return r#"{"ok":false,"error":"Missing 'qwf' field"}"#.to_string(),
    };
    let profile_str = extract_json_string(body, "profile").unwrap_or_else(|| "balanced".into());
    let out_str = extract_json_string(body, "out")
        .unwrap_or_else(|| "output/studio_model.safetensors".into());
    let out_path = PathBuf::from(&out_str);
    if out_path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
        return r#"{"ok":false,"error":"Output path must strictly end with .safetensors"}"#
            .to_string();
    }
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }

    let mut spec = match parse_workflow_spec(&qwf) {
        Ok(s) => s,
        Err(e) => {
            return format!(
                r#"{{"ok":false,"error":"Parse error: {}"}}"#,
                escape_json(&format!("{e:?}"))
            )
        }
    };
    if let Some(p) = CompileProfile::from_str_opt(&profile_str) {
        spec.profile = p;
    }

    let verify_cfg = VerifyConfig {
        active_k: 3,
        trials: 200,
        ..VerifyConfig::default()
    };

    let res = match compile_workflow_to_safetensors(&spec, &verify_cfg, &out_path) {
        Ok(r) => r,
        Err(e) => {
            return format!(
                r#"{{"ok":false,"error":"Compilation error: {}"}}"#,
                escape_json(&format!("{e:?}"))
            )
        }
    };

    let tensors_json = read_safetensors_catalogue_json(&out_path).unwrap_or_else(|_| "[]".into());
    let mut ver_json = String::from("[");
    for (i, rep) in res.rule_reports.iter().enumerate() {
        if i > 0 {
            ver_json.push(',');
        }
        let passed = rep.status == VerifyStatus::Pass;
        let status = if passed { "PASS" } else { "FAIL" };
        let txt = format_verifier_report(rep);
        ver_json.push_str(&format!(
            r#"{{"passed":{},"status":"{}","report":"{}"}}"#,
            passed,
            status,
            escape_json(&txt)
        ));
    }
    ver_json.push(']');

    format!(
        r#"{{"ok":true,"out_path":"{}","profile":"{}","bytes_written":{},"total_tensors":{},"total_weights":{},"health":{{"passed":{},"nonfinite":{},"f16_overflow":{},"max_abs":{:.6},"outliers":{}}},"verifier":{},"tensors":{}}}"#,
        escape_json(&out_str),
        spec.profile.as_str(),
        res.bytes_written,
        res.total_tensors,
        res.total_weights,
        res.weight_health.passed,
        res.weight_health.nonfinite_count,
        res.weight_health.f16_overflow_count,
        res.weight_health.max_abs_weight,
        res.weight_health.outlier_tensor_count,
        ver_json,
        tensors_json
    )
}

fn handle_inspect(body: &str) -> String {
    let path_str = match extract_json_string(body, "path") {
        Some(s) => s,
        None => return r#"{"ok":false,"error":"Missing 'path' field"}"#.to_string(),
    };
    let path = PathBuf::from(&path_str);
    match read_safetensors_catalogue_full(&path) {
        Ok((tensors_json, total_params, total_bytes)) => format!(
            r#"{{"ok":true,"path":"{}","total_params":{},"total_bytes":{},"tensors":{}}}"#,
            escape_json(&path_str),
            total_params,
            total_bytes,
            tensors_json
        ),
        Err(e) => format!(
            r#"{{"ok":false,"error":"{}"}}"#,
            escape_json(&e.to_string())
        ),
    }
}

fn read_safetensors_catalogue_json(path: &PathBuf) -> io::Result<String> {
    let (json, _, _) = read_safetensors_catalogue_full(path)?;
    Ok(json)
}

fn read_safetensors_catalogue_full(path: &PathBuf) -> io::Result<(String, u64, u64)> {
    let mut f = File::open(path)?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)?;
    let hlen = u64::from_le_bytes(len_buf) as usize;
    if hlen > 64 * 1024 * 1024 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Header exceeds 64 MB",
        ));
    }
    let mut hbuf = vec![0u8; hlen];
    f.read_exact(&mut hbuf)?;
    let hdr = String::from_utf8(hbuf)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid UTF-8 header"))?;

    let mut entries = parse_header_entries(&hdr);
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut total_params = 0u64;
    let mut total_bytes = 0u64;
    let mut out = String::from("[");
    for (i, (name, dtype, shape, nbytes)) in entries.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let params: u64 = shape.iter().map(|&d| d as u64).product::<u64>().max(1);
        total_params += params;
        total_bytes += nbytes;
        let rank = MathRank::from_shape(&shape).as_str();
        let shape_s: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        out.push_str(&format!(
            r#"{{"name":"{}","dtype":"{}","rank":"{}","shape":[{}],"params":{}}}"#,
            escape_json(&name),
            escape_json(&dtype),
            rank,
            shape_s.join(","),
            params
        ));
    }
    out.push(']');
    Ok((out, total_params, total_bytes))
}

fn parse_header_entries(hdr: &str) -> Vec<(String, String, Vec<usize>, u64)> {
    let mut out = Vec::new();
    let bytes = hdr.as_bytes();
    let mut pos = 0usize;
    while let Some(rel) = hdr[pos..].find("\"dtype\"") {
        let dtype_pos = pos + rel;
        let obj_start = match hdr[..dtype_pos].rfind('{') {
            Some(i) => i,
            None => break,
        };
        let obj_end = match hdr[dtype_pos..].find('}') {
            Some(i) => dtype_pos + i + 1,
            None => break,
        };
        let obj = &hdr[obj_start..obj_end];
        // Find key name right before `:{`
        let before = hdr[..obj_start].trim_end();
        let before = before.strip_suffix(':').unwrap_or(before).trim_end();
        let name = if let Some(stripped) = before.strip_suffix('"') {
            if let Some(q) = stripped.rfind('"') {
                stripped[q + 1..].to_string()
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        if !name.is_empty() && name != "__metadata__" {
            let dtype = extract_json_string(obj, "dtype").unwrap_or_else(|| "F32".into());
            let shape = extract_json_usize_array(obj, "shape");
            let offs = extract_json_u64_array(obj, "data_offsets");
            let nbytes = if offs.len() == 2 && offs[1] >= offs[0] {
                offs[1] - offs[0]
            } else {
                0
            };
            out.push((name, dtype, shape, nbytes));
        }
        pos = obj_end;
        if pos >= bytes.len() {
            break;
        }
    }
    out
}

fn extract_json_string(src: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let start = src.find(&needle)? + needle.len();
    let rest = &src[start..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    if !after.starts_with('"') {
        return None;
    }
    let mut out = String::new();
    let mut chars = after[1..].chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                other => out.push(other),
            },
            other => out.push(other),
        }
    }
    None
}

fn extract_json_usize_array(src: &str, key: &str) -> Vec<usize> {
    let needle = format!("\"{key}\"");
    let Some(start) = src.find(&needle) else {
        return Vec::new();
    };
    let rest = &src[start + needle.len()..];
    let Some(lb) = rest.find('[') else {
        return Vec::new();
    };
    let Some(rb) = rest[lb..].find(']') else {
        return Vec::new();
    };
    rest[lb + 1..lb + rb]
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect()
}

fn extract_json_u64_array(src: &str, key: &str) -> Vec<u64> {
    let needle = format!("\"{key}\"");
    let Some(start) = src.find(&needle) else {
        return Vec::new();
    };
    let rest = &src[start + needle.len()..];
    let Some(lb) = rest.find('[') else {
        return Vec::new();
    };
    let Some(rb) = rest[lb..].find(']') else {
        return Vec::new();
    };
    rest[lb + 1..lb + rb]
        .split(',')
        .filter_map(|s| s.trim().parse::<u64>().ok())
        .collect()
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {}
            c => out.push(c),
        }
    }
    out
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn send_html(stream: &mut TcpStream, html: &str) -> io::Result<()> {
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        html.len()
    );
    stream.write_all(resp.as_bytes())?;
    stream.write_all(html.as_bytes())?;
    stream.flush()
}

fn send_json(stream: &mut TcpStream, status: &str, json: &str) -> io::Result<()> {
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        json.len()
    );
    stream.write_all(resp.as_bytes())?;
    stream.write_all(json.as_bytes())?;
    stream.flush()
}
