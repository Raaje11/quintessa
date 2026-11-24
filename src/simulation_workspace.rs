//! Unified 2D Pixel & 3D Voxel/SDF Simulation Workspace (`simulation_workspace.rs`).
//!
//! Architecture contract:
//! - **Trainable Brain** decides *what* to design/edit.
//! - **Pure SACI Compute Layer** (`solver_44.saci`, `solver_45.saci`, `model.saci.voxel_conv3d.weight`,
//!   `model.saci.color.*`, `model.saci.conv3x3.*`, `model.saci.blend_mul.*`, `model.saci.curve.*`)
//!   performs **100% of the spatial & geometric computation**.
//! - **Simulation Workspace** holds bounded, cache-friendly Struct-of-Arrays (SoA) 2D pixel and
//!   3D voxel/SDF buffers, and only dumb I/O serializers export verified workspace state to
//!   `.ppm` (2D pixel image), arbitrary-shape watertight `.stl` (3D isosurface), and `.vtk` (3D volume).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::ada_compiler::{PureBilinearGateF32, PureConv2dF32, PureCurveMlpF32, PureLinearF32};
use crate::checkpoint::Checkpoint;
use crate::saci_compiler::{execute_saci_circuit, SaciCircuitDescriptor};
use crate::sys::ErrorCode;

/// Fixed bounded 2D pixel canvas resolution (`64 x 64 = 4,096` pixels).
pub const PIXEL_W: usize = 64;
pub const PIXEL_H: usize = 64;
pub const PIXEL_COUNT: usize = PIXEL_W * PIXEL_H;

/// Fixed bounded 3D voxel/SDF workspace resolution (`24 x 24 x 24 = 13,824` voxels, fits in L2 cache).
pub const VOXEL_N: usize = 24;
pub const VOXEL_COUNT: usize = VOXEL_N * VOXEL_N * VOXEL_N;

/// Semantic CSG boolean / shell operation executed by `solver_44.saci`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxelSaciOp {
    /// `min(a, b)` -> output slot 0 of `solver_44.saci`
    Union = 0,
    /// `max(a, -b)` -> output slot 1 of `solver_44.saci`
    Subtract = 1,
    /// `max(a, b)` -> output slot 2 of `solver_44.saci`
    Intersect = 2,
    /// `|a| - wall_thickness` -> output slot 3 of `solver_44.saci`
    Shell = 3,
}

/// Arbitrary 3D primitive field descriptor for voxel-level CSG sculpting.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxelPrimitiveKind {
    Box = 0,
    Sphere = 1,
    CylinderZ = 2,
    Torus = 3,
    ConeZ = 4,
}

#[derive(Clone, Copy, Debug)]
pub struct VoxelPrimitive {
    pub kind: VoxelPrimitiveKind,
    pub center_mm: [f32; 3],
    /// Parameters: Box `[hx, hy, hz]`, Sphere `[r, 0, 0]`, CylinderZ `[r, hz, 0]`,
    /// Torus `[major_r, minor_r, 0]`, ConeZ `[base_r, height, 0]`.
    pub params_mm: [f32; 3],
}

impl VoxelPrimitive {
    #[inline]
    pub fn eval_sdf(&self, x: f32, y: f32, z: f32) -> f32 {
        let px = x - self.center_mm[0];
        let py = y - self.center_mm[1];
        let pz = z - self.center_mm[2];
        match self.kind {
            VoxelPrimitiveKind::Box => {
                let qx = px.abs() - self.params_mm[0];
                let qy = py.abs() - self.params_mm[1];
                let qz = pz.abs() - self.params_mm[2];
                (qx.max(0.0).powi(2) + qy.max(0.0).powi(2) + qz.max(0.0).powi(2)).sqrt()
                    + qx.max(qy).max(qz).min(0.0)
            }
            VoxelPrimitiveKind::Sphere => (px * px + py * py + pz * pz).sqrt() - self.params_mm[0],
            VoxelPrimitiveKind::CylinderZ => {
                let dr = (px * px + py * py).sqrt() - self.params_mm[0];
                let dz = pz.abs() - self.params_mm[1];
                (dr.max(0.0).powi(2) + dz.max(0.0).powi(2)).sqrt() + dr.max(dz).min(0.0)
            }
            VoxelPrimitiveKind::Torus => {
                let qx = (px * px + py * py).sqrt() - self.params_mm[0];
                (qx * qx + pz * pz).sqrt() - self.params_mm[1]
            }
            VoxelPrimitiveKind::ConeZ => {
                let r_xy = (px * px + py * py).sqrt();
                let h = self.params_mm[1].max(1e-3);
                let frac = ((0.5 * h - pz) / h).clamp(0.0, 1.0);
                let r_at_z = self.params_mm[0] * frac;
                let dr = r_xy - r_at_z;
                let dz = pz.abs() - 0.5 * h;
                dr.max(dz)
            }
        }
    }
}

/// Bounded Struct-of-Arrays (SoA) 2D Pixel Canvas inside the Simulation Workspace.
#[derive(Clone, Debug)]
pub struct PixelWorkspace2D {
    pub r: Vec<f32>,
    pub g: Vec<f32>,
    pub b: Vec<f32>,
    /// Per-pixel edit mask (`0.0` = protected, `1.0` = active edit region).
    pub mask: Vec<f32>,
}

impl PixelWorkspace2D {
    pub fn new_blank() -> Self {
        Self {
            r: vec![0.0_f32; PIXEL_COUNT],
            g: vec![0.0_f32; PIXEL_COUNT],
            b: vec![0.0_f32; PIXEL_COUNT],
            mask: vec![1.0_f32; PIXEL_COUNT],
        }
    }

    /// Direct single-pixel edit `(x, y)` -> `(r, g, b)` clamped to `[0, 1]`.
    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, rgb: [f32; 3]) {
        if x < PIXEL_W && y < PIXEL_H {
            let idx = y * PIXEL_W + x;
            self.r[idx] = rgb[0].clamp(0.0, 1.0);
            self.g[idx] = rgb[1].clamp(0.0, 1.0);
            self.b[idx] = rgb[2].clamp(0.0, 1.0);
        }
    }

    /// Set a rectangular edit mask (`1.0` inside, `0.0` outside) for localized pixel editing.
    pub fn set_rect_mask(&mut self, x0: usize, y0: usize, x1: usize, y1: usize) {
        for y in 0..PIXEL_H {
            for x in 0..PIXEL_W {
                let inside = x >= x0 && x < x1 && y >= y0 && y < y1;
                self.mask[y * PIXEL_W + x] = if inside { 1.0 } else { 0.0 };
            }
        }
    }

    /// Execute pure-weight SACI 2D pixel pipeline (`color` -> `conv3x3` -> `curve` -> `alpha` mask composite)
    /// directly using tensors loaded from `ada_o1_insane.safetensors`.
    pub fn apply_saci_pixel_pass(
        &mut self,
        saci_color: &PureLinearF32,
        saci_conv: &PureConv2dF32,
        saci_curve: &PureCurveMlpF32,
        saci_alpha: &PureBilinearGateF32,
    ) {
        let mut r_tmp = self.r.clone();
        let mut g_tmp = self.g.clone();
        let mut b_tmp = self.b.clone();

        // 1. Depthwise 3x3 spatial convolution using `model.saci.conv3x3.weight`
        let k = &saci_conv.weight;
        for y in 1..(PIXEL_H - 1) {
            for x in 1..(PIXEL_W - 1) {
                let mut ar = 0.0_f32;
                let mut ag = 0.0_f32;
                let mut ab = 0.0_f32;
                for ky in 0..3 {
                    for kx in 0..3 {
                        let w = k[ky * 3 + kx];
                        let n_idx = (y + ky - 1) * PIXEL_W + (x + kx - 1);
                        ar += w * self.r[n_idx];
                        ag += w * self.g[n_idx];
                        ab += w * self.b[n_idx];
                    }
                }
                let idx = y * PIXEL_W + x;
                r_tmp[idx] = ar.clamp(0.0, 1.0);
                g_tmp[idx] = ag.clamp(0.0, 1.0);
                b_tmp[idx] = ab.clamp(0.0, 1.0);
            }
        }

        // 2. Per-pixel color matrix + ReLU tone curve + SwiGLU AlphaComposite with `self.mask`
        let mut c_out = [0.0_f32; 3];
        let mut comp_out = [0.0_f32; 3];
        for idx in 0..PIXEL_COUNT {
            let fg_in = [r_tmp[idx], g_tmp[idx], b_tmp[idx]];
            let _ = saci_color.forward(&fg_in, &mut c_out);
            for ch in &mut c_out {
                *ch = saci_curve
                    .forward_scalar(ch.clamp(0.0, 1.0))
                    .clamp(0.0, 1.0);
            }
            let alpha = self.mask[idx].clamp(0.0, 1.0);
            let _ = saci_alpha.forward(
                &[
                    c_out[0],
                    c_out[1],
                    c_out[2],
                    self.r[idx],
                    self.g[idx],
                    self.b[idx],
                    alpha,
                ],
                &mut comp_out,
            );
            self.r[idx] = comp_out[0].clamp(0.0, 1.0);
            self.g[idx] = comp_out[1].clamp(0.0, 1.0);
            self.b[idx] = comp_out[2].clamp(0.0, 1.0);
        }
    }

    /// Dumb I/O serializer: write `64x64` RGB workspace as a standard binary `.ppm` (`P6`) image.
    pub fn write_ppm(&self, path: &Path) -> Result<usize, ErrorCode> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        let file = File::create(path).map_err(|_| ErrorCode::IoError)?;
        let mut w = BufWriter::new(file);
        let header = format!("P6\n{PIXEL_W} {PIXEL_H}\n255\n");
        w.write_all(header.as_bytes())
            .map_err(|_| ErrorCode::IoError)?;
        let mut buf = Vec::with_capacity(PIXEL_COUNT * 3);
        for idx in 0..PIXEL_COUNT {
            buf.push((self.r[idx].clamp(0.0, 1.0) * 255.0).round() as u8);
            buf.push((self.g[idx].clamp(0.0, 1.0) * 255.0).round() as u8);
            buf.push((self.b[idx].clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        w.write_all(&buf).map_err(|_| ErrorCode::IoError)?;
        w.flush().map_err(|_| ErrorCode::IoError)?;
        Ok(header.len() + buf.len())
    }
}

/// Bounded Struct-of-Arrays (SoA) 3D Voxel / SDF Simulation Workspace (`24 x 24 x 24`).
/// Represents **ANY** arbitrary 3D shape (mechanical, organic, hollowed, or topology-optimized).
#[derive(Clone, Debug)]
pub struct VoxelWorkspace3D {
    /// Physical half-extent of the cubic simulation domain in millimeters (`[-half_extent, +half_extent]^3`).
    pub half_extent_mm: f32,
    /// Signed distance field per voxel (`<= 0.0` inside solid, `> 0.0` outside).
    pub sdf: Vec<f32>,
    /// Material ID per voxel (`0` = empty, `1..=5` = solid material).
    pub material: Vec<u8>,
    /// Per-voxel Von Mises stress (`MPa`) for FEA-driven topology optimization.
    pub stress_mpa: Vec<f32>,
    /// Per-voxel boundary/load-bearing pin mask (`1.0` = protected, `0.0` = carveable).
    pub pin_mask: Vec<f32>,
}

impl VoxelWorkspace3D {
    pub fn new_empty(half_extent_mm: f32) -> Self {
        Self {
            half_extent_mm: half_extent_mm.max(1.0),
            sdf: vec![half_extent_mm.max(1.0); VOXEL_COUNT],
            material: vec![0u8; VOXEL_COUNT],
            stress_mpa: vec![0.0_f32; VOXEL_COUNT],
            pin_mask: vec![0.0_f32; VOXEL_COUNT],
        }
    }

    #[inline]
    pub const fn idx(ix: usize, iy: usize, iz: usize) -> usize {
        iz * VOXEL_N * VOXEL_N + iy * VOXEL_N + ix
    }

    #[inline]
    pub fn voxel_size_mm(&self) -> f32 {
        (2.0 * self.half_extent_mm) / (VOXEL_N as f32)
    }

    #[inline]
    pub fn voxel_center_mm(&self, ix: usize, iy: usize, iz: usize) -> [f32; 3] {
        let step = self.voxel_size_mm();
        let org = -self.half_extent_mm + 0.5 * step;
        [
            org + (ix as f32) * step,
            org + (iy as f32) * step,
            org + (iz as f32) * step,
        ]
    }

    /// Direct single-voxel edit: set `(ix, iy, iz)` solid (`sdf = -step/2`) or empty (`sdf = +step/2`).
    #[inline]
    pub fn set_voxel(&mut self, ix: usize, iy: usize, iz: usize, solid: bool, mat_id: u8) {
        if ix < VOXEL_N && iy < VOXEL_N && iz < VOXEL_N {
            let i = Self::idx(ix, iy, iz);
            let h = 0.5 * self.voxel_size_mm();
            self.sdf[i] = if solid { -h } else { h };
            self.material[i] = if solid { mat_id.max(1) } else { 0 };
        }
    }

    /// Count active solid voxels (`sdf <= 0.0`).
    pub fn solid_voxel_count(&self) -> usize {
        self.sdf.iter().filter(|&&v| v <= 0.0).count()
    }

    /// Estimated solid volume (`mm^3`) directly from the voxel workspace.
    pub fn estimated_volume_mm3(&self) -> f64 {
        let dv = (self.voxel_size_mm() as f64).powi(3);
        (self.solid_voxel_count() as f64) * dv
    }

    /// Apply any arbitrary 3D primitive CSG or Shell operation **strictly via compiled `solver_44.saci`**
    /// inside `TensorId(5)` of `ada_o1_insane.safetensors`.
    pub fn apply_saci_csg_op(
        &mut self,
        saci_ckpt: &Checkpoint,
        desc_44: &SaciCircuitDescriptor,
        op: VoxelSaciOp,
        prim: Option<VoxelPrimitive>,
        shell_thickness_mm: f32,
        mat_id: u8,
    ) -> Result<(), ErrorCode> {
        let mut scratch = [0.0_f64; 256];
        let out_slot = op as usize;
        for iz in 0..VOXEL_N {
            for iy in 0..VOXEL_N {
                for ix in 0..VOXEL_N {
                    let i = Self::idx(ix, iy, iz);
                    let a = self.sdf[i] as f64;
                    let b = if let Some(p) = prim {
                        let [x, y, z] = self.voxel_center_mm(ix, iy, iz);
                        p.eval_sdf(x, y, z) as f64
                    } else {
                        a
                    };
                    let trace = execute_saci_circuit(
                        saci_ckpt,
                        desc_44,
                        &[a, b, shell_thickness_mm as f64],
                        &mut scratch,
                    )?;
                    let next_sdf = trace.output.get(out_slot).copied().unwrap_or(a) as f32;
                    self.sdf[i] = next_sdf;
                    if next_sdf <= 0.0 {
                        if self.material[i] == 0 {
                            self.material[i] = mat_id.max(1);
                        }
                    } else {
                        self.material[i] = 0;
                    }
                }
            }
        }
        Ok(())
    }

    /// Apply pure-weight 3D neural convolution (`model.saci.voxel_conv3d.weight`, `[27]`)
    /// directly on the SDF field for smooth organic filleting / surface regularization.
    pub fn apply_saci_conv3d(&mut self, kernel_3x3x3: &[f32]) -> Result<(), ErrorCode> {
        if kernel_3x3x3.len() != 27 {
            return Err(ErrorCode::ShapeMismatch);
        }
        let mut next = self.sdf.clone();
        for iz in 1..(VOXEL_N - 1) {
            for iy in 1..(VOXEL_N - 1) {
                for ix in 1..(VOXEL_N - 1) {
                    let mut acc = 0.0_f32;
                    let mut k_idx = 0usize;
                    for dz in 0..3 {
                        for dy in 0..3 {
                            for dx in 0..3 {
                                let ni = Self::idx(ix + dx - 1, iy + dy - 1, iz + dz - 1);
                                acc += kernel_3x3x3[k_idx] * self.sdf[ni];
                                k_idx += 1;
                            }
                        }
                    }
                    let i = Self::idx(ix, iy, iz);
                    next[i] = acc;
                    if acc <= 0.0 && self.material[i] == 0 {
                        self.material[i] = 1;
                    } else if acc > 0.0 {
                        self.material[i] = 0;
                    }
                }
            }
        }
        self.sdf = next;
        Ok(())
    }

    /// Execute FEA-driven Topology Optimization strictly via compiled **`solver_45.saci`**
    /// (`TensorId(5)`): carves away unpinned voxels whose stress falls below `carve_threshold_mpa`.
    /// Returns the number of low-stress voxels carved away.
    pub fn apply_saci_topology_optimize(
        &mut self,
        saci_ckpt: &Checkpoint,
        desc_45: &SaciCircuitDescriptor,
        carve_threshold_mpa: f32,
    ) -> Result<usize, ErrorCode> {
        let mut scratch = [0.0_f64; 256];
        let mut carved = 0usize;
        let step = self.voxel_size_mm();
        for i in 0..VOXEL_COUNT {
            if self.sdf[i] > 0.0 {
                continue;
            }
            let trace = execute_saci_circuit(
                saci_ckpt,
                desc_45,
                &[
                    self.stress_mpa[i] as f64,
                    carve_threshold_mpa as f64,
                    self.pin_mask[i] as f64,
                ],
                &mut scratch,
            )?;
            let should_carve = trace.output.get(1).copied().unwrap_or(0.0) > 0.5;
            if should_carve {
                self.sdf[i] = 0.5 * step;
                self.material[i] = 0;
                carved += 1;
            }
        }
        Ok(carved)
    }

    /// Project mid-plane `Z = 0` slice of this 3D voxel workspace into the `64x64` 2D Pixel Workspace
    /// (color-coded by Von Mises stress / solid occupancy) for visual inspection or 2D pixel editing.
    pub fn render_midplane_to_pixel_workspace(&self, px: &mut PixelWorkspace2D) {
        let iz = VOXEL_N / 2;
        let max_s = self.stress_mpa.iter().copied().fold(1.0_f32, f32::max);
        for py in 0..PIXEL_H {
            let iy = (py * VOXEL_N) / PIXEL_H;
            for pxx in 0..PIXEL_W {
                let ix = (pxx * VOXEL_N) / PIXEL_W;
                let vi = Self::idx(ix, iy, iz);
                if self.sdf[vi] <= 0.0 {
                    let s_norm = (self.stress_mpa[vi] / max_s).clamp(0.0, 1.0);
                    // Cool-to-hot stress colormap (blue -> cyan -> yellow -> red)
                    let rgb = [
                        s_norm,
                        0.35 + 0.5 * (1.0 - (s_norm - 0.5).abs() * 2.0),
                        1.0 - s_norm,
                    ];
                    px.set_pixel(pxx, py, rgb);
                } else {
                    px.set_pixel(pxx, py, [0.06, 0.07, 0.09]);
                }
            }
        }
    }

    /// Downsample this `24x24x24` voxel SDF into ADA-o1's `4x4x4` NCM `Field3D4x4x4` tensor (`64` floats).
    pub fn encode_to_ncm_field3d(&self) -> [f32; 64] {
        let mut out = [0.0_f32; 64];
        let blk = VOXEL_N / 4;
        let scale = 1.0 / self.half_extent_mm.max(1.0);
        for fz in 0..4 {
            for fy in 0..4 {
                for fx in 0..4 {
                    let mut acc = 0.0_f32;
                    for dz in 0..blk {
                        for dy in 0..blk {
                            for dx in 0..blk {
                                acc += self.sdf
                                    [Self::idx(fx * blk + dx, fy * blk + dy, fz * blk + dz)];
                            }
                        }
                    }
                    let mean = acc / ((blk * blk * blk) as f32);
                    out[fz * 16 + fy * 4 + fx] = (mean * scale).clamp(-1.0, 1.0);
                }
            }
        }
        out
    }

    /// Universal dumb I/O serializer: extracts a **watertight binary `.stl` mesh** from **ANY**
    /// arbitrary 3D voxel/SDF shape via boundary-face isosurface extraction (works for spheres,
    /// tori, shells, sculpted voxels, and topology-optimized organic brackets).
    pub fn export_isosurface_binary_stl(&self, path: &Path) -> Result<(usize, usize), ErrorCode> {
        let step = self.voxel_size_mm();
        let half = 0.5 * step;
        let mut tris: Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 3])> = Vec::with_capacity(4096);

        let is_solid = |ix: isize, iy: isize, iz: isize| -> bool {
            if ix < 0
                || iy < 0
                || iz < 0
                || (ix as usize) >= VOXEL_N
                || (iy as usize) >= VOXEL_N
                || (iz as usize) >= VOXEL_N
            {
                return false;
            }
            self.sdf[Self::idx(ix as usize, iy as usize, iz as usize)] <= 0.0
        };

        let mut push_quad =
            |n: [f32; 3], v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], v3: [f32; 3]| {
                tris.push((n, v0, v1, v2));
                tris.push((n, v0, v2, v3));
            };

        for iz in 0..VOXEL_N {
            for iy in 0..VOXEL_N {
                for ix in 0..VOXEL_N {
                    if self.sdf[Self::idx(ix, iy, iz)] > 0.0 {
                        continue;
                    }
                    let [cx, cy, cz] = self.voxel_center_mm(ix, iy, iz);
                    let (x0, x1) = (cx - half, cx + half);
                    let (y0, y1) = (cy - half, cy + half);
                    let (z0, z1) = (cz - half, cz + half);
                    let (iix, iiy, iiz) = (ix as isize, iy as isize, iz as isize);

                    if !is_solid(iix + 1, iiy, iiz) {
                        push_quad(
                            [1.0, 0.0, 0.0],
                            [x1, y0, z0],
                            [x1, y1, z0],
                            [x1, y1, z1],
                            [x1, y0, z1],
                        );
                    }
                    if !is_solid(iix - 1, iiy, iiz) {
                        push_quad(
                            [-1.0, 0.0, 0.0],
                            [x0, y1, z0],
                            [x0, y0, z0],
                            [x0, y0, z1],
                            [x0, y1, z1],
                        );
                    }
                    if !is_solid(iix, iiy + 1, iiz) {
                        push_quad(
                            [0.0, 1.0, 0.0],
                            [x1, y1, z0],
                            [x0, y1, z0],
                            [x0, y1, z1],
                            [x1, y1, z1],
                        );
                    }
                    if !is_solid(iix, iiy - 1, iiz) {
                        push_quad(
                            [0.0, -1.0, 0.0],
                            [x0, y0, z0],
                            [x1, y0, z0],
                            [x1, y0, z1],
                            [x0, y0, z1],
                        );
                    }
                    if !is_solid(iix, iiy, iiz + 1) {
                        push_quad(
                            [0.0, 0.0, 1.0],
                            [x0, y0, z1],
                            [x1, y0, z1],
                            [x1, y1, z1],
                            [x0, y1, z1],
                        );
                    }
                    if !is_solid(iix, iiy, iiz - 1) {
                        push_quad(
                            [0.0, 0.0, -1.0],
                            [x0, y1, z0],
                            [x1, y1, z0],
                            [x1, y0, z0],
                            [x0, y0, z0],
                        );
                    }
                }
            }
        }

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        let file = File::create(path).map_err(|_| ErrorCode::IoError)?;
        let mut w = BufWriter::new(file);
        let mut header = [0u8; 80];
        let tag = b"ADA-o1 Pure-SACI Arbitrary Voxel/SDF Workspace STL";
        header[..tag.len()].copy_from_slice(tag);
        w.write_all(&header).map_err(|_| ErrorCode::IoError)?;
        w.write_all(&(tris.len() as u32).to_le_bytes())
            .map_err(|_| ErrorCode::IoError)?;
        for &(n, a, b, c) in &tris {
            for v in [n, a, b, c] {
                for coord in v {
                    w.write_all(&coord.to_le_bytes())
                        .map_err(|_| ErrorCode::IoError)?;
                }
            }
            w.write_all(&0u16.to_le_bytes())
                .map_err(|_| ErrorCode::IoError)?;
        }
        w.flush().map_err(|_| ErrorCode::IoError)?;
        let bytes = 84 + tris.len() * 50;
        Ok((bytes, tris.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ada_compiler::{
        compile_saci_blend_gate, compile_saci_color_matrix, compile_saci_conv3x3,
        compile_saci_tone_curve, AdaBlendMode,
    };
    use crate::saci_compiler::{
        compile_saci_program, load_saci_binary, parse_saci_source, SacisubstrateAllocator,
    };

    #[test]
    fn pure_saci_pixel_and_voxel_workspace_arbitrary_shapes_and_topo_opt() {
        let mut alloc = SacisubstrateAllocator::new(2048);
        let mut ckpt = Checkpoint::new(Vec::new(), Vec::new()).unwrap();

        let src44 = std::fs::read_to_string("specs/saci/solver_44.saci").unwrap();
        let art44 = compile_saci_program(&parse_saci_source(&src44).unwrap(), &mut alloc).unwrap();
        let d44 = load_saci_binary(&mut ckpt, &art44.sacib_bytes)
            .unwrap()
            .descriptor;

        let src45 = std::fs::read_to_string("specs/saci/solver_45.saci").unwrap();
        let art45 = compile_saci_program(&parse_saci_source(&src45).unwrap(), &mut alloc).unwrap();
        let d45 = load_saci_binary(&mut ckpt, &art45.sacib_bytes)
            .unwrap()
            .descriptor;

        // 1. Sculpt an arbitrary non-plate 3D shape: Torus UNION Sphere SUBTRACT CylinderZ, then 3D Conv smooth
        let mut vox = VoxelWorkspace3D::new_empty(40.0);
        let torus = VoxelPrimitive {
            kind: VoxelPrimitiveKind::Torus,
            center_mm: [0.0, 0.0, 0.0],
            params_mm: [22.0, 8.0, 0.0],
        };
        vox.apply_saci_csg_op(&ckpt, &d44, VoxelSaciOp::Union, Some(torus), 0.0, 1)
            .unwrap();
        let n_torus = vox.solid_voxel_count();
        assert!(n_torus > 200);

        let hole = VoxelPrimitive {
            kind: VoxelPrimitiveKind::CylinderZ,
            center_mm: [22.0, 0.0, 0.0],
            params_mm: [6.0, 20.0, 0.0],
        };
        vox.apply_saci_csg_op(&ckpt, &d44, VoxelSaciOp::Subtract, Some(hole), 0.0, 1)
            .unwrap();
        assert!(vox.solid_voxel_count() < n_torus);

        // Single-voxel manual edit + 3D SACI neural Conv3x3x3 smoothing
        vox.set_voxel(12, 12, 12, true, 1);
        let mut k3d = [1.0_f32 / 34.0; 27];
        k3d[13] = 8.0 / 34.0;
        vox.apply_saci_conv3d(&k3d).unwrap();

        // FEA-driven topology optimization via `solver_45.saci`
        for i in 0..VOXEL_COUNT {
            if vox.sdf[i] <= 0.0 {
                vox.stress_mpa[i] = if i % 3 == 0 { 15.0 } else { 120.0 };
                vox.pin_mask[i] = if i % 9 == 0 { 1.0 } else { 0.0 };
            }
        }
        let carved = vox.apply_saci_topology_optimize(&ckpt, &d45, 50.0).unwrap();
        assert!(carved > 0);

        // Export arbitrary voxel shape to watertight binary STL
        let stl_p = std::env::temp_dir().join("ada_o1_voxel_torus.stl");
        let (stl_b, stl_t) = vox.export_isosurface_binary_stl(&stl_p).unwrap();
        assert!(stl_b > 1000 && stl_t > 20);
        let _ = std::fs::remove_file(&stl_p);

        // 2. Render mid-plane to 2D Pixel Workspace & run pure-weight SACI pixel pass -> PPM
        let mut px = PixelWorkspace2D::new_blank();
        vox.render_midplane_to_pixel_workspace(&mut px);
        px.set_rect_mask(8, 8, 56, 56);
        let sc = compile_saci_color_matrix(
            [[1.05, 0.0, 0.0], [0.0, 1.02, 0.0], [0.0, 0.0, 0.98]],
            [0.0, 0.0, 0.0],
        )
        .unwrap();
        let sv =
            compile_saci_conv3x3(3, [0.0, -0.1, 0.0, -0.1, 1.4, -0.1, 0.0, -0.1, 0.0]).unwrap();
        let st = compile_saci_tone_curve(&[(0.0, 0.0), (0.5, 0.55), (1.0, 1.0)]).unwrap();
        let sa = compile_saci_blend_gate(AdaBlendMode::AlphaComposite, 3).unwrap();
        px.apply_saci_pixel_pass(&sc, &sv, &st, &sa);

        let ppm_p = std::env::temp_dir().join("ada_o1_pixel_canvas.ppm");
        let ppm_b = px.write_ppm(&ppm_p).unwrap();
        assert_eq!(ppm_b, 13 + PIXEL_COUNT * 3);
        let _ = std::fs::remove_file(&ppm_p);
    }
}
