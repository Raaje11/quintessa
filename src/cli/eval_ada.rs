//! CLI subcommand: `quintessa eval-ada` — ADA-1B 8-Domain General Intelligence & CAD Scorecard.

use std::fs;
use std::path::PathBuf;

use quintessa::ada_compiler::{compile_shared_saci_primitive_bank, SaciPrimitiveKind};
use quintessa::ada_scorecard::evaluate_benchmark_jsonl;

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let bench_path = PathBuf::from(
        parse_flag(args, "--benchmark")
            .unwrap_or_else(|| r"D:\TRACR\ada_datasets\benchmark_8domain_cad.jsonl".to_string()),
    );
    let floor: f64 = parse_flag(args, "--floor")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.60);

    // Verify the 8 shared transferable SACI primitives compile into pure GPU-safe F32 weights
    let bank = compile_shared_saci_primitive_bank(16)
        .map_err(|e| format!("Shared SACI primitive bank compile failed: {e:?}"))?;
    let mut probe_out = [0.0_f32; 3];
    bank.quantitative_gate
        .forward(&[1.0, 0.5, 0.25, 0.0, 0.5, 0.75, 0.5], &mut probe_out)
        .map_err(|e| format!("SACI primitive forward failed: {e:?}"))?;

    let text = fs::read_to_string(&bench_path)?;
    let report = evaluate_benchmark_jsonl(&text, floor)
        .map_err(|e| format!("Scorecard evaluation failed: {e:?}"))?;

    println!("=== ADA-1B Eight-Domain General Intelligence + Parametric CAD Scorecard ===");
    println!("  Benchmark File : {}", bench_path.display());
    println!(
        "  Domain Floor   : {:.1}% (anti-compensation enforced)",
        floor * 100.0
    );
    println!();
    println!(
        "  {:<24} | {:>6} | {:>9} | {:>7} | {}",
        "Capability Domain", "Weight", "Passed", "Score", "Floor"
    );
    println!(
        "  {:-<24}-+-{:-<6}-+-{:-<9}-+-{:-<7}-+------",
        "", "", "", ""
    );
    for ds in &report.domain_scores {
        let w_str = if ds.weight > 0.0 {
            format!("{:>5.0}%", ds.weight * 100.0)
        } else {
            String::from("  CAD ")
        };
        println!(
            "  {:<24} | {:>6} | {:>4}/{:<4} | {:>6.2}% | {}",
            ds.domain.as_str(),
            w_str,
            ds.passed_tasks,
            ds.total_tasks,
            ds.accuracy * 100.0,
            if ds.meets_floor { "PASS" } else { "FAIL" }
        );
    }
    println!();
    println!(
        "  Weighted GI Score       : {:.2}%",
        report.weighted_general_score * 100.0
    );
    println!(
        "  Parametric CAD Score    : {:.2}% (Domain Supremacy)",
        report.parametric_cad_score * 100.0
    );
    println!(
        "  Shared SACI Primitives  : {}/{} active (avg cross-domain transfer = {:.2} domains/primitive)",
        report.shared_primitives_covered,
        SaciPrimitiveKind::ALL.len(),
        SaciPrimitiveKind::ALL
            .iter()
            .map(|p| p.transfer_degree() as f64)
            .sum::<f64>()
            / (SaciPrimitiveKind::ALL.len() as f64)
    );
    println!(
        "  GI Certification        : {}",
        if report.certified_general_intelligence {
            "PASS"
        } else {
            "FAIL"
        }
    );

    if !report.certified_general_intelligence {
        return Err("ADA-1B General Intelligence certification failed domain floor".into());
    }
    Ok(())
}
