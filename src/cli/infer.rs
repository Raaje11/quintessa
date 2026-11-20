use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use quintessa::conversation::{ConversationSession, LoadedAdaModel};
use quintessa::operand_grounding::NumericVocabTable;
use quintessa::sequence_encoder::VocabTokenizer;

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    let model_path = PathBuf::from(parse_flag(args, "--model").unwrap_or_else(|| {
        if Path::new("output/ada_o1_512.safetensors").exists() {
            "output/ada_o1_512.safetensors".to_string()
        } else if Path::new("output/ada_o1.safetensors").exists() {
            "output/ada_o1.safetensors".to_string()
        } else {
            "quintessa/output/ada_o1_512.safetensors".to_string()
        }
    }));
    let saci_dir = PathBuf::from(parse_flag(args, "--saci-dir").unwrap_or_else(|| {
        if Path::new("specs/saci").exists() {
            "specs/saci".to_string()
        } else {
            "quintessa/specs/saci".to_string()
        }
    }));
    let vocab_path = PathBuf::from(
        parse_flag(args, "--vocab")
            .unwrap_or_else(|| r"D:\AdvancedADA\data\vocab_ada_h1.txt".to_string()),
    );
    let max_chunks: usize = parse_flag(args, "--chunks")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let tok = VocabTokenizer::from_vocab_file(&vocab_path)?;
    let model = LoadedAdaModel::from_safetensors(&model_path, Some(&saci_dir))
        .map_err(|e| format!("Failed to load {}: {e:?}", model_path.display()))?;
    let num_table = NumericVocabTable::from_vocab_file(&vocab_path, model.vocab_size);
    let mut session = ConversationSession::new(model.dim);

    println!(
        "Loaded {} (dim={}, vocab={}, saci_circuits={}, load={} ms)",
        model_path.display(),
        model.dim,
        model.vocab_size,
        model.saci_descs.len(),
        t0.elapsed().as_millis()
    );

    if let Some(prompt) = parse_flag(args, "--prompt") {
        let req_toks = tok.encode(&prompt);
        let t1 = Instant::now();
        let out = session
            .respond_turn(&model, &req_toks, &num_table, max_chunks)
            .map_err(|e| format!("Inference failed: {e:?}"))?;
        let raw_text = tok.decode(&out.tokens);
        let text = raw_text
            .strip_prefix("[THOUGHT]")
            .map(|s| s.trim_start())
            .unwrap_or(&raw_text);
        println!(
            "[turn={} domain={} verify_4={} chunks={} toks={} elapsed={} ms]",
            out.turn_index,
            out.predicted_domain,
            out.all_verified,
            out.chunks_generated,
            out.tokens.len(),
            t1.elapsed().as_millis()
        );
        println!("{text}");
        return Ok(());
    }

    if !has_flag(args, "--interactive") && parse_flag(args, "--prompt").is_none() {
        println!("Entering interactive multi-turn mode (type 'exit' or 'reset'):");
    }

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut line = String::new();
    loop {
        print!("\nyou> ");
        io::stdout().flush()?;
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }
        if prompt.eq_ignore_ascii_case("exit") || prompt.eq_ignore_ascii_case("quit") {
            break;
        }
        if prompt.eq_ignore_ascii_case("reset") {
            session.reset();
            println!("[session memory reset]");
            continue;
        }
        let req_toks = tok.encode(prompt);
        let t1 = Instant::now();
        let out = session
            .respond_turn(&model, &req_toks, &num_table, max_chunks)
            .map_err(|e| format!("Inference failed: {e:?}"))?;
        let raw_text = tok.decode(&out.tokens);
        let text = raw_text
            .strip_prefix("[THOUGHT]")
            .map(|s| s.trim_start())
            .unwrap_or(&raw_text);
        println!(
            "ada [turn={} dom={} ver={} chunks={} {}ms]> {}",
            out.turn_index,
            out.predicted_domain,
            out.all_verified,
            out.chunks_generated,
            t1.elapsed().as_millis(),
            text
        );
    }

    Ok(())
}
