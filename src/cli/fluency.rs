//! CLI subcommand: `quintessa fluency` — Compile & run pure-weight Hopfield-PKM
//! Language Fluency & Multimodal Generation layers (`.safetensors`, pure Burn format).

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use quintessa::ada_compiler::AdaCapabilityDomain;
use quintessa::fluency_multimodal_layer::{
    FluencyCompileExample, UnifiedFluencyMultimodalLayer, MODAL_AUDIO_TOKENS, MODAL_CANVAS_ELEMS,
};

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn extract_json_str<'a>(line: &'a str, key: &str) -> Option<String> {
    let pos = line.find(key)?;
    let rest = &line[pos + key.len()..];
    let q1 = rest.find('"')?;
    let after = &rest[q1 + 1..];
    let bytes = after.as_bytes();
    let mut out = String::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            match bytes[i + 1] {
                b'n' => out.push('\n'),
                b't' => out.push('\t'),
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                other => out.push(other as char),
            }
            i += 2;
            continue;
        }
        if bytes[i] == b'"' {
            return Some(out);
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    None
}

/// Synthesize a structured 8x8 visual/CAD canvas target and 15 Hz RVQ audio tokens
/// corresponding to each domain's multimodal output geometry.
fn synth_multimodal_targets(
    domain_id: u8,
    idx: usize,
) -> ([f32; MODAL_CANVAS_ELEMS], [u16; MODAL_AUDIO_TOKENS]) {
    let mut canvas = [0.0_f32; MODAL_CANVAS_ELEMS];
    match domain_id {
        8 => {
            // Parametric CAD: Aluminum plate with centered through-bore hole
            for y in 1..7 {
                for x in 1..7 {
                    let in_hole = (y == 3 || y == 4) && (x == 3 || x == 4);
                    canvas[y * 8 + x] = if in_hole { 0.0 } else { 1.0 };
                }
            }
        }
        5 => {
            // Office/Spreadsheet: upward financial bar chart on 8x8 canvas
            for x in 0..8 {
                let h = ((x + 2) * 6) / 8;
                for y in (8 - h)..8 {
                    canvas[y * 8 + x] = 0.85;
                }
            }
        }
        _ => {
            // Semantic/Reasoning/Knowledge: smooth diagonal latent wave pattern
            for y in 0..8 {
                for x in 0..8 {
                    let v = (((x + y + (domain_id as usize)) % 8) as f32) / 7.0;
                    canvas[y * 8 + x] = v;
                }
            }
        }
    }
    let mut audio = [0u16; MODAL_AUDIO_TOKENS];
    for a in 0..MODAL_AUDIO_TOKENS {
        audio[a] = (((domain_id as usize) * 197 + a * 73 + (idx % 31) * 11) % 2048) as u16;
    }
    (canvas, audio)
}

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        eprintln!("Usage:");
        eprintln!("  quintessa fluency compile --dataset <train.jsonl> --out <model.safetensors> [--dim 512] [--max-slots 512] [--max-len 192]");
        eprintln!("  quintessa fluency infer --model <model.safetensors> --prompt \"...\"");
        std::process::exit(2);
    }

    match args[0].as_str() {
        "compile" => {
            let t0 = Instant::now();
            let ds_path = PathBuf::from(
                parse_flag(&args[1..], "--dataset")
                    .unwrap_or_else(|| r"D:\TRACR\ada_datasets\train.jsonl".to_string()),
            );
            let out_path =
                PathBuf::from(parse_flag(&args[1..], "--out").unwrap_or_else(|| {
                    r"D:\TRACR\output\ada_omni_fluency.safetensors".to_string()
                }));
            let dim: usize = parse_flag(&args[1..], "--dim")
                .and_then(|s| s.parse().ok())
                .unwrap_or(512);
            let max_slots: usize = parse_flag(&args[1..], "--max-slots")
                .and_then(|s| s.parse().ok())
                .unwrap_or(512);
            let max_len: usize = parse_flag(&args[1..], "--max-len")
                .and_then(|s| s.parse().ok())
                .unwrap_or(192);
            let beta: f32 = parse_flag(&args[1..], "--beta")
                .and_then(|s| s.parse().ok())
                .unwrap_or(36.0);

            let text = fs::read_to_string(&ds_path)?;
            let per_dom_cap = (max_slots / 9).max(1);
            let mut dom_counts = [0usize; 9];
            let mut examples = Vec::with_capacity(max_slots);

            for (idx, raw) in text.lines().enumerate() {
                let line = raw.trim();
                if line.is_empty() {
                    continue;
                }
                let Some(dom_str) = extract_json_str(line, "\"domain\":") else {
                    continue;
                };
                let Some(dom) = AdaCapabilityDomain::from_str_opt(&dom_str) else {
                    continue;
                };
                let d_idx = dom as usize;
                if dom_counts[d_idx] >= per_dom_cap {
                    continue;
                }
                let Some(prompt) = extract_json_str(line, "\"prompt\":") else {
                    continue;
                };
                let Some(expected) = extract_json_str(line, "\"expected\":") else {
                    continue;
                };
                let (canvas_target, audio_tokens) = synth_multimodal_targets(d_idx as u8, idx);
                examples.push(FluencyCompileExample {
                    prompt,
                    response: expected,
                    domain_id: d_idx as u8,
                    canvas_target,
                    audio_tokens,
                });
                dom_counts[d_idx] += 1;
                if examples.len() >= max_slots {
                    break;
                }
            }

            let layer = UnifiedFluencyMultimodalLayer::compile(dim, max_len, beta, &examples)
                .map_err(|e| format!("Fluency layer compile failed: {e:?}"))?;
            let bytes = layer
                .save_burn_safetensors(&out_path)
                .map_err(|e| format!("Failed to save Burn safetensors: {e:?}"))?;

            println!("=== Quintessa Unified Fluency & Multimodal Layer Compiled ===");
            println!("  Dataset       : {}", ds_path.display());
            println!("  Output Model  : {}", out_path.display());
            println!(
                "  Hopfield Slots: {} (balanced across 9 domains)",
                layer.num_slots
            );
            println!(
                "  PKM Sub-Keys  : {} x {} (dim={})",
                layer.pkm_side, layer.pkm_side, layer.dim
            );
            println!(
                "  Size on Disk  : {:.2} MB (pure Burn .safetensors)",
                (bytes as f64) / 1e6
            );
            println!("  Elapsed       : {} ms", t0.elapsed().as_millis());
            Ok(())
        }
        "infer" => {
            let t0 = Instant::now();
            let model_path =
                PathBuf::from(parse_flag(&args[1..], "--model").unwrap_or_else(|| {
                    r"D:\TRACR\output\ada_omni_fluency.safetensors".to_string()
                }));
            let prompt = parse_flag(&args[1..], "--prompt").ok_or("Missing --prompt \"...\"")?;

            let layer = UnifiedFluencyMultimodalLayer::load_burn_safetensors(&model_path)
                .map_err(|e| format!("Failed to load model: {e:?}"))?;
            let load_ms = t0.elapsed().as_millis();

            let t1 = Instant::now();
            let out = layer.forward(&prompt);
            let infer_us = t1.elapsed().as_micros();

            let dom_name = AdaCapabilityDomain::ALL
                .get(out.domain_id as usize)
                .map(|d| d.as_str())
                .unwrap_or("unknown");

            println!(
                "[model={} | slots={} | domain={} | hopfield_conf={:.4} | load={}ms | infer={}us]",
                model_path.display(),
                layer.num_slots,
                dom_name,
                out.top_attention_weight,
                load_ms,
                infer_us
            );
            println!("Text Output  : {}", out.text);
            println!("Audio RVQ 15Hz: {:?}", out.audio_rvq_tokens);
            println!("Visual/CAD 8x8 Canvas:");
            for y in 0..8 {
                print!("  ");
                for x in 0..8 {
                    let v = out.canvas_8x8[y * 8 + x];
                    let ch = if v > 0.75 {
                        '#'
                    } else if v > 0.35 {
                        '+'
                    } else if v > 0.10 {
                        '.'
                    } else {
                        ' '
                    };
                    print!("{ch}{ch}");
                }
                println!();
            }
            Ok(())
        }
        other => {
            Err(format!("Unknown fluency subcommand: {other} (expected compile | infer)").into())
        }
    }
}
