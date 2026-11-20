use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use quintessa::observatory::format_verifier_report;
use quintessa::verifier::{VerifyConfig, VerifyStatus};
use quintessa::workflow_arch::{
    compile_workflow_to_safetensors, parse_workflow_spec, CompileProfile,
};

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    let arch_path = PathBuf::from(
        parse_flag(args, "--arch").ok_or("Missing required --arch <workflow.qwf> argument")?,
    );
    let out_path = PathBuf::from(
        parse_flag(args, "--out")
            .unwrap_or_else(|| "output/custom_workflow.safetensors".to_string()),
    );
    if out_path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
        return Err(format!(
            "Output path must strictly end with .safetensors (got: {})",
            out_path.display()
        )
        .into());
    }

    let mut verify_cfg = VerifyConfig::default();
    if let Some(k_str) = parse_flag(args, "--k") {
        verify_cfg.active_k = k_str.parse()?;
    }
    if let Some(t_str) = parse_flag(args, "--trials") {
        verify_cfg.trials = t_str.parse()?;
    }

    let src = fs::read_to_string(&arch_path)?;
    let mut spec = parse_workflow_spec(&src).map_err(|e| {
        format!(
            "Failed to parse workflow spec {}: {e:?}",
            arch_path.display()
        )
    })?;
    if let Some(profile_str) = parse_flag(args, "--profile") {
        spec.profile = match profile_str.to_ascii_lowercase().as_str() {
            "memory" => CompileProfile::Memory,
            "latency" => CompileProfile::Latency,
            "accuracy" => CompileProfile::Accuracy,
            "balanced" | "generalized" => CompileProfile::Balanced,
            other => {
                return Err(format!(
                    "Unknown --profile '{other}' (expected memory|latency|accuracy|balanced)"
                )
                .into())
            }
        };
    }

    let result = compile_workflow_to_safetensors(&spec, &verify_cfg, &out_path)
        .map_err(|e| format!("Workflow compilation failed: {e:?}"))?;

    for (i, rep) in result.rule_reports.iter().enumerate() {
        println!("--- Block Report #{i} ---");
        println!("{}", format_verifier_report(rep));
    }

    let all_pass = result.weight_health.passed
        && result
            .rule_reports
            .iter()
            .all(|r| r.status == VerifyStatus::Pass);

    println!(
        "Compiled workflow '{}' [profile={}] -> {} ({} bytes, {} tensors, {} weights, max_abs={:.4}, health={}, verifier={}, elapsed={} ms)",
        spec.name,
        spec.profile.as_str(),
        out_path.display(),
        result.bytes_written,
        result.total_tensors,
        result.total_weights,
        result.weight_health.max_abs_weight,
        if result.weight_health.passed { "PASS" } else { "FAIL" },
        if all_pass { "PASS" } else { "FAIL" },
        t0.elapsed().as_millis()
    );

    if !all_pass {
        return Err("Quintessa 5-axis verifier failed thresholds".into());
    }
    Ok(())
}
