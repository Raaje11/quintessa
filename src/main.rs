mod cli;

fn print_help() {
    println!("quintessa - Unified TRACR/WSP Superposition Compiler, Evaluator & Runtime");
    println!();
    println!("USAGE:");
    println!("  quintessa [build] [BUILD_OPTIONS]       Build & verify a pure-weight ADA .safetensors model");
    println!("  quintessa infer --model <m.safetensors> [--prompt \"...\"] [--chunks <n>] [--interactive]");
    println!("                                          Run multi-turn, multi-chunk inference directly from .safetensors");
    println!("  quintessa eval --model <m.safetensors> --valid <valid.bin> [--saci-dir <dir>] [--vq <vq.bpk>]");
    println!("                                          Evaluate per-domain accuracy of a .safetensors model");
    println!("  quintessa eval-ada [--benchmark <benchmark_8domain_cad.jsonl>] [--floor <0.60>]");
    println!("                                          Evaluate ADA-1B 8-domain general intelligence + Parametric CAD scorecard");
    println!("  quintessa fluency <compile|infer> [...] Compile & run pure-weight Hopfield-PKM Fluency + Multimodal layer");
    println!("  quintessa lm <compile|residual|infer>   Statistical WSP Language Compiler + Residual Attribution + Per-Neuron GD");
    println!("  quintessa ada-o1 [--out <path>]         Build & execute zero-dead-parameter flagship ADA-o1 unified model");
    println!(
        "  quintessa workflow --arch <spec.qwf> --out <model.safetensors> [--profile <memory|latency|accuracy|balanced>] [--k <n>] [--trials <n>]"
    );
    println!("                                          Compile a custom workflow architecture to .safetensors");
    println!(
        "  quintessa inspect <catalogue|tensor-stats|analyze|diff> <file.safetensors> [file_b.safetensors] [--tensor <name>] [--chunk-mb <n>] [--max-memory-mb <n>]"
    );
    println!("                                          Stream-inspect weights & health of any .safetensors model");
    println!(
        "  quintessa scale [--preset <1t-moe|1t-dense|100b-moe|micro> | --spec <model.qscale>] --out-dir <dir> [--dry-run] [--shard-range <s..e>] [--resume] [--shard-gb <5.0>] [--max-memory-mb <256>] [--threads <n>]"
    );
    println!("                                          Plan & stream-compile trillion-parameter sharded .safetensors models");
    println!("  quintessa studio [--port <7744>]        Launch Quintessa WSP Studio web UI (node-graph compiler)");
    println!("  quintessa help                          Show this help message");
    println!();
    println!("BUILD OPTIONS:");
    println!("  --spec <path>         Architecture spec sheet (e.g. specs/ada_o1.spec)");
    println!("  --train <path>        Training token binary");
    println!("  --valid <path>        Validation token binary");
    println!("  --saci-dir <dir>      Directory of .saci circuit specs (e.g. specs/saci)");
    println!("  --vq <path>           Optional frozen VQ codebook (.bpk)");
    println!("  --out <path>          Output model path (strictly .safetensors)");
    println!("  --dtype <f32|f64>     Output weight precision");
    println!("  --ncm-exp-dim <n>     NCM expansion width (default: 512)");
    println!("  --max-ncm-eps <n>     Bounded training episode cap for NCM (default: 8192)");
    println!("  --model-kind <kind>   Model profile (ada-o1, compact-edge)");
    println!("  --etf-features <n>    Override ETF feature count");
    println!("  --form                Launch interactive build wizard");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() <= 1 {
        return cli::build::run(&args);
    }

    match args[1].as_str() {
        "-h" | "--help" | "help" => {
            print_help();
            Ok(())
        }
        "ada-o1" => cli::ada_o1::run(&args[2..]),
        "build" => {
            let mut forwarded = Vec::with_capacity(args.len() - 1);
            forwarded.push(args[0].clone());
            forwarded.extend_from_slice(&args[2..]);
            cli::build::run(&forwarded)
        }
        "eval" => cli::eval::run(&args[2..]),
        "eval-ada" => cli::eval_ada::run(&args[2..]),
        "fluency" => cli::fluency::run(&args[2..]),
        "infer" => cli::infer::run(&args[2..]),
        "inspect" => cli::inspect::run(&args[2..]),
        "lm" => cli::lm::run(&args[2..]),
        "pure-ada" => cli::pure_ada::run(&args[2..]),
        "scale" => cli::scale::run(&args[2..]),
        "studio" => cli::studio::run(&args[2..]),
        "workflow" => cli::workflow::run(&args[2..]),
        first if first.starts_with('-') => cli::build::run(&args),
        other => {
            eprintln!("Unknown subcommand: {other}");
            eprintln!("Run `quintessa --help` for usage.");
            std::process::exit(2);
        }
    }
}
