//! CLI subcommand: `quintessa pure-ada` — Build, masked-train, save, reload, and run the
//! **100% un-stitched, single-model, NON-AUTOREGRESSIVE Parallel 64-Slot `PureAdaModel`**
//! (`ada_pure_unified.safetensors`).
//!
//! 1 single `forward()` pass emits all 64 output token slots + 3D voxel SDF + 2D pixel RGB in parallel.

use std::path::PathBuf;
use std::time::Instant;

use quintessa::pure_ada_model::{
    write_neural_pixel_ppm, write_neural_voxel_stl, PureAdaModel, PURE_PIXEL_COUNT,
    PURE_VOXEL_COUNT, PURE_VOXEL_N,
};

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn make_target_cube_bore() -> (Vec<f32>, Vec<f32>) {
    let mut vox = vec![1.0_f32; PURE_VOXEL_COUNT];
    for iz in 1..7 {
        for iy in 1..7 {
            for ix in 1..7 {
                let bore = (ix == 3 || ix == 4) && (iy == 3 || iy == 4);
                vox[iz * 64 + iy * 8 + ix] = if bore { 0.85 } else { -0.85 };
            }
        }
    }
    let mut px = vec![0.08_f32; PURE_PIXEL_COUNT];
    for y in 2..14 {
        for x in 2..14 {
            let bore = (6..=9).contains(&x) && (6..=9).contains(&y);
            let i = (y * 16 + x) * 3;
            if bore {
                px[i] = 0.05;
                px[i + 1] = 0.06;
                px[i + 2] = 0.08;
            } else {
                px[i] = 0.25;
                px[i + 1] = 0.65;
                px[i + 2] = 0.90;
            }
        }
    }
    (vox, px)
}

fn make_target_sphere() -> (Vec<f32>, Vec<f32>) {
    let mut vox = vec![1.0_f32; PURE_VOXEL_COUNT];
    for iz in 0..8 {
        for iy in 0..8 {
            for ix in 0..8 {
                let dx = (ix as f32) - 3.5;
                let dy = (iy as f32) - 3.5;
                let dz = (iz as f32) - 3.5;
                let d = (dx * dx + dy * dy + dz * dz).sqrt() - 2.8;
                vox[iz * 64 + iy * 8 + ix] = d.clamp(-1.0, 1.0);
            }
        }
    }
    let mut px = vec![0.08_f32; PURE_PIXEL_COUNT];
    for y in 0..16 {
        for x in 0..16 {
            let dx = (x as f32) - 7.5;
            let dy = (y as f32) - 7.5;
            let i = (y * 16 + x) * 3;
            if (dx * dx + dy * dy).sqrt() <= 5.5 {
                px[i] = 0.92;
                px[i + 1] = 0.42;
                px[i + 2] = 0.20;
            }
        }
    }
    (vox, px)
}

type DesignEpisode<'a> = (&'a [u8], &'a [u8], &'a [f32], &'a [f32]);

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = PathBuf::from(
        parse_flag(args, "--model")
            .unwrap_or_else(|| r"D:\TRACR\output\ada_pure_unified.safetensors".to_string()),
    );
    let out_prefix = PathBuf::from(
        parse_flag(args, "--out-prefix")
            .unwrap_or_else(|| r"D:\TRACR\output\pure_ada_neural_gen".to_string()),
    );

    if let Some(prompt) = parse_flag(args, "--prompt") {
        let t0 = Instant::now();
        let model = PureAdaModel::load_burn_safetensors(&model_path)
            .map_err(|e| format!("Failed to load {}: {e:?}", model_path.display()))?;
        let load_ms = t0.elapsed().as_millis();

        // 1 SINGLE PARALLEL FORWARD PASS (zero autoregressive loop!)
        let t1 = Instant::now();
        let out = model.forward_prompt_parallel(prompt.as_bytes(), None);
        let gen_us = t1.elapsed().as_micros();
        let gen_bytes = out.decoded_bytes();
        let gen_text = String::from_utf8_lossy(&gen_bytes);
        let solid_vox = out.voxel_sdf_8x8x8.iter().filter(|&&v| v <= 0.0).count();
        let saci_mass: f32 = out.last_router_probs[..8].iter().sum();

        let stl_p = out_prefix.with_extension("stl");
        let ppm_p = out_prefix.with_extension("ppm");
        let (stl_b, stl_t) = write_neural_voxel_stl(&out.voxel_sdf_8x8x8, 40.0, &stl_p)
            .map_err(|e| format!("STL write failed: {e:?}"))?;
        let ppm_b = write_neural_pixel_ppm(&out.pixel_rgb_16x16, &ppm_p)
            .map_err(|e| format!("PPM write failed: {e:?}"))?;

        println!("=== PureAdaModel Parallel 64-Slot Inference (1 Forward Pass, Zero Autoregression, Zero Stitching) ===");
        println!(
            "  Model      : {} (loaded in {} ms)",
            model_path.display(),
            load_ms
        );
        println!("  Prompt     : {prompt}");
        println!(
            "  Parallel64 : \"{gen_text}\" ({} us [1 pass!] | Ponder steps={} p={:.3} | SACI Router Mass={:.1}%)",
            gen_us,
            out.ponder_steps,
            out.ponder_halt_prob,
            saci_mass * 100.0
        );
        println!(
            "  Neural 3D  : {solid_vox}/512 solid voxels -> {} ({stl_b} B, {stl_t} tris)",
            stl_p.display()
        );
        println!(
            "  Neural 2D  : 16x16 RGB -> {} ({ppm_b} B)",
            ppm_p.display()
        );
        println!(
            "  [8x8 Mid-Plane Z=4 Neural Voxel Slice (directly from model.sim.voxel_proj + conv3d)]:"
        );
        for iy in 0..PURE_VOXEL_N {
            let mut row = String::with_capacity(16);
            for ix in 0..PURE_VOXEL_N {
                let v = out.voxel_sdf_8x8x8[4 * 64 + iy * 8 + ix];
                row.push_str(if v <= 0.0 { "##" } else { "  " });
            }
            println!("    |{row}|");
        }
        return Ok(());
    }

    let t0 = Instant::now();
    let mut model = PureAdaModel::compile_new(2026);
    let (cube_vox, cube_px) = make_target_cube_bore();
    let (sph_vox, sph_px) = make_target_sphere();

    let episodes: [DesignEpisode<'_>; 2] = [
        (
            b"design cube with bore",
            b"plan:box_union+bore_sub->verified",
            &cube_vox,
            &cube_px,
        ),
        (
            b"design solid sphere",
            b"plan:sphere_sdf+conv3d->verified",
            &sph_vox,
            &sph_px,
        ),
    ];

    let steps = 90usize;
    let mut loss_start = 0.0_f32;
    let mut loss_end = 0.0_f32;
    let mut vox_mse_end = 0.0_f32;
    let mut max_saci_drift = 0.0_f64;

    for step in 0..steps {
        let mut step_loss = 0.0_f32;
        for &(prompt, target, tvox, tpx) in &episodes {
            let rep = model.train_masked_step_parallel(prompt, target, Some(tvox), Some(tpx), 0.40);
            max_saci_drift = max_saci_drift.max(rep.saci_frozen_drift);
            step_loss += rep.lm_loss;
            vox_mse_end = rep.voxel_mse;
        }
        step_loss /= episodes.len() as f32;
        if step == 0 {
            loss_start = step_loss;
        }
        loss_end = step_loss;
    }

    let (bytes, n_tensors, n_params) = model
        .save_burn_safetensors(&model_path)
        .map_err(|e| format!("Save failed: {e:?}"))?;

    println!("=== PureAdaModel Compiled & Parallel-Trained (1-Pass 64-Slot NCM, Zero Autoregression, Zero Stitching) ===");
    println!(
        "  Output      : {} ({:.2} MB, {} tensors, {} params, pure Burn {{\"format\":\"burn\"}})",
        model_path.display(),
        (bytes as f64) / (1024.0 * 1024.0),
        n_tensors,
        n_params
    );
    println!(
        "  Training    : {} parallel steps | LM CE Loss: {:.4} -> {:.4} | 3D Voxel MSE: {:.4} | Frozen SACI Drift: {:.2e}",
        steps, loss_start, loss_end, vox_mse_end, max_saci_drift
    );
    println!("  Elapsed     : {} ms", t0.elapsed().as_millis());
    Ok(())
}
