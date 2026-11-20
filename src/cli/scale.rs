//! CLI subcommand: `quintessa scale` — Trillion-Parameter Streaming & Sharded Compiler.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use quintessa::ada_compiler::AdaScaleSpec;
use quintessa::safetensors_writer::WeightOrigin;
use quintessa::sharded_compiler::{
    compile_sharded_model, parse_scale_spec, plan_scale_model, ScaleModelSpec, ShardCompileOptions,
};
use quintessa::sys::TelemetryBus;
use quintessa::workflow_arch::CompileProfile;

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn format_params(n: u64) -> String {
    if n >= 1_000_000_000_000 {
        format!("{:.3}T ({n})", (n as f64) / 1e12)
    } else if n >= 1_000_000_000 {
        format!("{:.3}B ({n})", (n as f64) / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.3}M ({n})", (n as f64) / 1e6)
    } else {
        format!("{n}")
    }
}

fn format_bytes(b: u64) -> String {
    const TB: f64 = 1024.0 * 1024.0 * 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    let bf = b as f64;
    if bf >= TB {
        format!("{:.2} TiB ({b} B)", bf / TB)
    } else if bf >= GB {
        format!("{:.2} GiB ({b} B)", bf / GB)
    } else if bf >= MB {
        format!("{:.2} MiB ({b} B)", bf / MB)
    } else {
        format!("{b} B")
    }
}

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let mut spec = if let Some(spec_path) = parse_flag(args, "--spec") {
        let src = fs::read_to_string(&spec_path)?;
        parse_scale_spec(&src)
            .map_err(|e| format!("Failed to parse scale spec '{spec_path}': {e:?}"))?
    } else {
        let preset = parse_flag(args, "--preset").unwrap_or_else(|| "1t-moe".to_string());
        match preset.to_ascii_lowercase().as_str() {
            "1t-moe" | "1t" => ScaleModelSpec::preset_1t_moe(),
            "1t-dense" => ScaleModelSpec::preset_1t_dense(),
            "100b-moe" | "100b" => ScaleModelSpec::preset_100b_moe(),
            "ada-1t" => AdaScaleSpec::preset_ada_1t()
                .lower_to_scale_spec()
                .map_err(|e| format!("ADA preset lower failed: {e:?}"))?,
            "ada-1b" => AdaScaleSpec::preset_ada_1b()
                .lower_to_scale_spec()
                .map_err(|e| format!("ADA preset lower failed: {e:?}"))?,
            "omni-1b" | "omni" => ScaleModelSpec::preset_omni_multimodal_1b(),
            "ada-local-3b" | "ada-3b" => AdaScaleSpec::preset_ada_local_3b()
                .lower_to_scale_spec()
                .map_err(|e| format!("ADA preset lower failed: {e:?}"))?,
            "micro" | "test" => ScaleModelSpec::preset_micro_test(),
            other => {
                return Err(format!(
                    "Unknown --preset '{other}' (expected 1t-moe | 1t-dense | 100b-moe | ada-1t | ada-local-3b | micro)"
                )
                .into());
            }
        }
    };

    if let Some(p_str) = parse_flag(args, "--profile") {
        spec.profile = CompileProfile::from_str_opt(&p_str).ok_or_else(|| {
            format!("Unknown --profile '{p_str}' (expected memory|latency|accuracy|balanced)")
        })?;
    }
    if let Some(gb_str) = parse_flag(args, "--shard-gb") {
        let gb: f64 = gb_str.parse()?;
        if !gb.is_finite() || gb <= 0.0 {
            return Err("--shard-gb must be a positive finite number".into());
        }
        spec.max_shard_bytes = (gb * 1024.0 * 1024.0 * 1024.0) as u64;
    }
    if let Some(mb_str) = parse_flag(args, "--max-memory-mb") {
        spec.max_memory_mb = mb_str.parse()?;
    }
    if let Some(orig_str) = parse_flag(args, "--origin") {
        spec.weight_origin = WeightOrigin::from_str_opt(&orig_str).ok_or_else(|| {
            format!("Unknown --origin '{orig_str}' (expected compiled|learned|hybrid|initialized)")
        })?;
    }
    if has_flag(args, "--allow-f64") {
        spec.gpu_safe_no_f64 = false;
    }

    let out_dir = PathBuf::from(
        parse_flag(args, "--out-dir").unwrap_or_else(|| format!("output/{}", spec.name)),
    );

    let mut opts = ShardCompileOptions {
        dry_run: has_flag(args, "--dry-run"),
        resume: has_flag(args, "--resume"),
        ..ShardCompileOptions::default()
    };
    if let Some(t_str) = parse_flag(args, "--threads") {
        opts.num_threads = t_str.parse()?;
    }
    if let Some(range_str) = parse_flag(args, "--shard-range") {
        let (s_str, e_str) = range_str
            .split_once("..")
            .ok_or("--shard-range must have format <start>..<end> (e.g. 0..8)")?;
        opts.shard_start = s_str.trim().parse()?;
        opts.shard_end = e_str.trim().parse()?;
    }

    let mut telemetry: TelemetryBus<256> = TelemetryBus::new();
    let plan =
        plan_scale_model(&spec, &mut telemetry).map_err(|e| format!("Plan failed: {e:?}"))?;
    let plan_ms = t0.elapsed().as_millis();

    println!("=== Quintessa Trillion-Scale Sharded Compiler ===");
    println!("  Model            : {}", plan.model_name);
    println!("  Compile Profile  : {}", plan.profile.as_str());
    println!(
        "  Weight Origin    : {} (is_compiled={})",
        plan.weight_origin.as_str(),
        plan.weight_origin.is_compiled_str()
    );
    println!(
        "  Precision Guard  : {}",
        if plan.gpu_safe_no_f64 {
            "GPU-safe (F64 capped to F32)"
        } else {
            "F64 allowed (CPU/exact math only)"
        }
    );
    println!("  Total Parameters : {}", format_params(plan.total_params));
    println!(
        "  Active / Token   : {}",
        format_params(plan.active_params_per_token)
    );
    println!(
        "  Total Storage    : {}",
        format_bytes(plan.total_data_bytes)
    );
    println!("  Total Tensors    : {}", plan.tensors.len());
    println!(
        "  Shards Planned   : {} (cap = {})",
        plan.shards.len(),
        format_bytes(plan.max_shard_bytes)
    );
    println!(
        "  RAM Budget       : {} MiB (strictly bounded)",
        plan.max_memory_mb
    );
    println!("  Plan Elapsed     : {plan_ms} ms");

    let report = compile_sharded_model(&plan, &out_dir, opts, &mut telemetry)
        .map_err(|e| format!("Sharded compilation failed: {e:?}"))?;

    if let Some(ref idx_path) = report.index_path {
        println!("  Index Manifest   : {}", idx_path.display());
    }
    if opts.dry_run {
        println!(
            "  Mode             : DRY-RUN (manifest emitted; 0 weight shards written, elapsed={} ms)",
            t0.elapsed().as_millis()
        );
    } else {
        println!(
            "  Shards Compiled  : {} (resumed={}, written={}, health={}, max_abs={:.4}, elapsed={} ms)",
            report.shards_compiled,
            report.shards_resumed,
            format_bytes(report.bytes_written_this_run),
            if report.overall_health.passed {
                "PASS"
            } else {
                "FAIL"
            },
            report.overall_health.max_abs_weight,
            t0.elapsed().as_millis()
        );
    }

    Ok(())
}
