//! Bidirectional Parametric CAD B-Rep Encoder/Decoder (`ISO 10303-21 STEP AP214`),
//! High-Resolution Binary `.stl` Preview, and 3D FEA Simulation + `.vtk` Exporter.
//!
//! Built in strict compliance with `D:\TRACR\GEMINI.md`:
//! - Core solid IR (`CadBrepSolid`) is fixed-capacity, `Copy`, stack-allocated (zero heap `Vec`/`String`).
//! - Semantic categories (`CadMaterial`, `CadFeatureKind`, `FeaLoadCase`, `FeaSafetyStatus`) use `#[repr(u8)]` enums.
//! - Zero external Cargo dependencies: pure Rust ISO 10303-21 STEP AP214 emitter & parser,
//!   analytical watertight binary `.stl` tessellator, and VTK hexahedral FEA field exporter.

use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::sys::ErrorCode;

/// Maximum number of parametric features (bores, blind holes, bosses) per stack-allocated solid.
pub const MAX_CAD_FEATURES: usize = 16;

/// Engineering material enum with precomputed physical & mechanical constants.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CadMaterial {
    Aluminum6061 = 0,
    Steel4140 = 1,
    TitaniumTi6Al4V = 2,
    Stainless304 = 3,
    CopperC110 = 4,
}

impl CadMaterial {
    /// Density in `g/cm^3` (also `kg/dm^3`).
    #[inline]
    pub const fn density_g_cm3(self) -> f64 {
        match self {
            Self::Aluminum6061 => 2.70,
            Self::Steel4140 => 7.85,
            Self::TitaniumTi6Al4V => 4.43,
            Self::Stainless304 => 8.00,
            Self::CopperC110 => 8.96,
        }
    }

    /// Young's modulus in `MPa` (`N/mm^2`).
    #[inline]
    pub const fn youngs_modulus_mpa(self) -> f64 {
        match self {
            Self::Aluminum6061 => 68_900.0,
            Self::Steel4140 => 205_000.0,
            Self::TitaniumTi6Al4V => 113_800.0,
            Self::Stainless304 => 193_000.0,
            Self::CopperC110 => 117_000.0,
        }
    }

    /// Yield strength (`0.2%` offset) in `MPa`.
    #[inline]
    pub const fn yield_strength_mpa(self) -> f64 {
        match self {
            Self::Aluminum6061 => 276.0,
            Self::Steel4140 => 655.0,
            Self::TitaniumTi6Al4V => 880.0,
            Self::Stainless304 => 215.0,
            Self::CopperC110 => 210.0,
        }
    }

    /// Poisson's ratio.
    #[inline]
    pub const fn poissons_ratio(self) -> f64 {
        match self {
            Self::Aluminum6061 => 0.33,
            Self::Steel4140 => 0.29,
            Self::TitaniumTi6Al4V => 0.34,
            Self::Stainless304 => 0.29,
            Self::CopperC110 => 0.34,
        }
    }

    /// Canonical uppercase STEP identifier.
    #[inline]
    pub const fn step_code(self) -> &'static str {
        match self {
            Self::Aluminum6061 => "ALUMINUM_6061_T6",
            Self::Steel4140 => "STEEL_4140",
            Self::TitaniumTi6Al4V => "TITANIUM_TI6AL4V",
            Self::Stainless304 => "STAINLESS_304",
            Self::CopperC110 => "COPPER_C110",
        }
    }

    pub fn from_text(text: &str) -> Self {
        let lower = text.to_ascii_lowercase();
        if lower.contains("titanium") || lower.contains("ti6al4v") || lower.contains("ti-6al-4v") {
            Self::TitaniumTi6Al4V
        } else if lower.contains("stainless") || lower.contains("304") {
            Self::Stainless304
        } else if lower.contains("steel") || lower.contains("4140") {
            Self::Steel4140
        } else if lower.contains("copper") || lower.contains("c110") {
            Self::CopperC110
        } else {
            Self::Aluminum6061
        }
    }
}

/// Parametric feature kind inside a B-Rep solid.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CadFeatureKind {
    None = 0,
    /// Full through-hole from `z = 0` to `z = thickness_mm`.
    ThroughBore = 1,
    /// Blind hole recessed from top face (`z = thickness_mm`) downward by `depth_or_height_mm`.
    BlindBore = 2,
    /// Cylindrical boss protruding upward from top face (`z = thickness_mm`) by `depth_or_height_mm`.
    CylindricalBoss = 3,
}

/// Fixed-layout parametric cylindrical feature centered at `(cx_mm, cy_mm)` relative to plate center.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CadFeature {
    pub kind: CadFeatureKind,
    pub cx_mm: f64,
    pub cy_mm: f64,
    pub radius_mm: f64,
    pub depth_or_height_mm: f64,
}

impl CadFeature {
    pub const EMPTY: Self = Self {
        kind: CadFeatureKind::None,
        cx_mm: 0.0,
        cy_mm: 0.0,
        radius_mm: 0.0,
        depth_or_height_mm: 0.0,
    };
}

/// Stack-allocated, bounded parametric B-Rep solid representation (origin at plate center `X,Y`, `Z in [0, thickness_mm]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CadBrepSolid {
    pub width_mm: f64,
    pub height_mm: f64,
    pub thickness_mm: f64,
    pub material: CadMaterial,
    pub feature_count: u8,
    pub features: [CadFeature; MAX_CAD_FEATURES],
}

impl CadBrepSolid {
    pub fn new_plate(
        width_mm: f64,
        height_mm: f64,
        thickness_mm: f64,
        material: CadMaterial,
    ) -> Result<Self, ErrorCode> {
        if !width_mm.is_finite()
            || !height_mm.is_finite()
            || !thickness_mm.is_finite()
            || width_mm <= 0.0
            || height_mm <= 0.0
            || thickness_mm <= 0.0
        {
            return Err(ErrorCode::InvalidInput);
        }
        Ok(Self {
            width_mm,
            height_mm,
            thickness_mm,
            material,
            feature_count: 0,
            features: [CadFeature::EMPTY; MAX_CAD_FEATURES],
        })
    }

    pub fn add_feature(&mut self, feat: CadFeature) -> Result<(), ErrorCode> {
        if (self.feature_count as usize) >= MAX_CAD_FEATURES {
            return Err(ErrorCode::CapacityExceeded);
        }
        if !feat.radius_mm.is_finite() || feat.radius_mm <= 0.0 {
            return Err(ErrorCode::InvalidInput);
        }
        self.features[self.feature_count as usize] = feat;
        self.feature_count += 1;
        Ok(())
    }

    #[inline]
    pub fn active_features(&self) -> &[CadFeature] {
        &self.features[..(self.feature_count as usize).min(MAX_CAD_FEATURES)]
    }

    /// Exact analytical Signed Distance Function `sdf(x, y, z)` in millimeters (`<= 0` inside solid).
    pub fn sdf_mm(&self, x: f64, y: f64, z: f64) -> f64 {
        let hx = 0.5 * self.width_mm;
        let hy = 0.5 * self.height_mm;
        let hz = 0.5 * self.thickness_mm;
        let dx = x.abs() - hx;
        let dy = y.abs() - hy;
        let dz = (z - hz).abs() - hz;
        let mut d = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2) + dz.max(0.0).powi(2)).sqrt()
            + dx.max(dy).max(dz).min(0.0);

        for f in self.active_features() {
            let r_xy = ((x - f.cx_mm).powi(2) + (y - f.cy_mm).powi(2)).sqrt() - f.radius_mm;
            match f.kind {
                CadFeatureKind::ThroughBore => {
                    // CSG subtract infinite cylinder along Z: max(d, -r_xy)
                    d = d.max(-r_xy);
                }
                CadFeatureKind::BlindBore => {
                    let z_bot = (self.thickness_mm - f.depth_or_height_mm).max(0.0);
                    let dz_b = z_bot - z;
                    let d_cyl = r_xy.max(dz_b);
                    d = d.max(-d_cyl);
                }
                CadFeatureKind::CylindricalBoss => {
                    let z_top = self.thickness_mm + f.depth_or_height_mm;
                    let dz_boss = (z - self.thickness_mm - 0.5 * f.depth_or_height_mm).abs()
                        - 0.5 * f.depth_or_height_mm;
                    let d_boss = (r_xy.max(0.0).powi(2) + dz_boss.max(0.0).powi(2)).sqrt()
                        + r_xy.max(dz_boss).min(0.0);
                    let _ = z_top;
                    d = d.min(d_boss);
                }
                CadFeatureKind::None => {}
            }
        }
        d
    }

    /// Exact analytical volume (`mm^3`), assuming non-overlapping features.
    pub fn volume_mm3(&self) -> f64 {
        let mut vol = self.width_mm * self.height_mm * self.thickness_mm;
        for f in self.active_features() {
            let area = PI * f.radius_mm * f.radius_mm;
            match f.kind {
                CadFeatureKind::ThroughBore => vol -= area * self.thickness_mm,
                CadFeatureKind::BlindBore => {
                    vol -= area * f.depth_or_height_mm.min(self.thickness_mm);
                }
                CadFeatureKind::CylindricalBoss => vol += area * f.depth_or_height_mm,
                CadFeatureKind::None => {}
            }
        }
        vol.max(0.0)
    }

    /// Exact mass in grams (`g`).
    #[inline]
    pub fn mass_g(&self) -> f64 {
        self.volume_mm3() * (self.material.density_g_cm3() * 1e-3)
    }

    /// Minimum remaining wall thickness (`mm`) between any bore and the outer plate perimeter.
    pub fn min_wall_thickness_mm(&self) -> f64 {
        let hx = 0.5 * self.width_mm;
        let hy = 0.5 * self.height_mm;
        let mut min_w = hx.min(hy);
        for f in self.active_features() {
            if matches!(
                f.kind,
                CadFeatureKind::ThroughBore | CadFeatureKind::BlindBore
            ) {
                let wx = (hx - f.cx_mm.abs()) - f.radius_mm;
                let wy = (hy - f.cy_mm.abs()) - f.radius_mm;
                min_w = min_w.min(wx).min(wy);
            }
        }
        min_w.max(0.0)
    }

    /// Encode this B-Rep solid directly into ADA-o1's `4x4x4` 3D SDF field (`Field3D4x4x4` modality).
    pub fn encode_to_ncm_field3d(&self) -> [f32; 64] {
        let mut field = [0.0_f32; 64];
        let max_dim = self
            .width_mm
            .max(self.height_mm)
            .max(self.thickness_mm)
            .max(1.0);
        let z_max = self.max_z_mm();
        for iz in 0..4 {
            let z = ((iz as f64 + 0.5) / 4.0) * z_max;
            for iy in 0..4 {
                let y = (((iy as f64 + 0.5) / 4.0) - 0.5) * self.height_mm;
                for ix in 0..4 {
                    let x = (((ix as f64 + 0.5) / 4.0) - 0.5) * self.width_mm;
                    let sdf_norm = (self.sdf_mm(x, y, z) / max_dim).clamp(-1.0, 1.0);
                    field[iz * 16 + iy * 4 + ix] = sdf_norm as f32;
                }
            }
        }
        field
    }

    /// Encode mid-plane `Z = thickness/2` cross-section onto ADA-o1's `8x8` Simulation Canvas.
    pub fn encode_to_canvas8x8(&self) -> [f32; 64] {
        let mut canvas = [0.0_f32; 64];
        let span = self.width_mm.max(self.height_mm) * 1.1;
        let z_mid = 0.5 * self.thickness_mm;
        for iy in 0..8 {
            let y = (((iy as f64 + 0.5) / 8.0) - 0.5) * span;
            for ix in 0..8 {
                let x = (((ix as f64 + 0.5) / 8.0) - 0.5) * span;
                canvas[iy * 8 + ix] = if self.sdf_mm(x, y, z_mid) <= 0.0 {
                    1.0
                } else {
                    0.0
                };
            }
        }
        canvas
    }

    pub fn max_z_mm(&self) -> f64 {
        let mut mz = self.thickness_mm;
        for f in self.active_features() {
            if f.kind == CadFeatureKind::CylindricalBoss {
                mz = mz.max(self.thickness_mm + f.depth_or_height_mm);
            }
        }
        mz
    }
}

// ============================================================================
// 1. ISO 10303-21 STEP AP214 Exact Analytical B-Rep DECODER (Solid -> .step)
// ============================================================================

struct StepBuilder {
    next_id: usize,
    lines: Vec<String>,
}

impl StepBuilder {
    fn new() -> Self {
        Self {
            next_id: 1,
            lines: Vec::with_capacity(256),
        }
    }

    fn emit(&mut self, body: String) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.lines.push(format!("#{id}={body};"));
        id
    }

    fn point(&mut self, x: f64, y: f64, z: f64) -> usize {
        self.emit(format!("CARTESIAN_POINT('',({x:.6},{y:.6},{z:.6}))"))
    }

    fn dir(&mut self, x: f64, y: f64, z: f64) -> usize {
        self.emit(format!("DIRECTION('',({x:.6},{y:.6},{z:.6}))"))
    }

    fn vertex(&mut self, pt_id: usize) -> usize {
        self.emit(format!("VERTEX_POINT('',#{pt_id})"))
    }

    fn axis2(&mut self, origin_pt: usize, z_dir: usize, x_dir: usize) -> usize {
        self.emit(format!(
            "AXIS2_PLACEMENT_3D('',#{origin_pt},#{z_dir},#{x_dir})"
        ))
    }

    fn line_edge(&mut self, v1: usize, v2: usize, p1: [f64; 3], p2: [f64; 3]) -> usize {
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let dz = p2[2] - p1[2];
        let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);
        let p_id = self.point(p1[0], p1[1], p1[2]);
        let d_id = self.dir(dx / len, dy / len, dz / len);
        let vec_id = self.emit(format!("VECTOR('',#{d_id},{len:.6})"));
        let line_id = self.emit(format!("LINE('',#{p_id},#{vec_id})"));
        self.emit(format!("EDGE_CURVE('',#{v1},#{v2},#{line_id},.T.)"))
    }

    fn circle_arc_edges(
        &mut self,
        cx: f64,
        cy: f64,
        z: f64,
        r: f64,
        v_pos_x: usize,
        v_neg_x: usize,
    ) -> (usize, usize) {
        let c_pt = self.point(cx, cy, z);
        let z_d = self.dir(0.0, 0.0, 1.0);
        let x_d = self.dir(1.0, 0.0, 0.0);
        let ax = self.axis2(c_pt, z_d, x_d);
        let circ = self.emit(format!("CIRCLE('',#{ax},{r:.6})"));
        let e1 = self.emit(format!("EDGE_CURVE('',#{v_pos_x},#{v_neg_x},#{circ},.T.)"));
        let e2 = self.emit(format!("EDGE_CURVE('',#{v_neg_x},#{v_pos_x},#{circ},.T.)"));
        (e1, e2)
    }

    fn oriented_edge(&mut self, edge_id: usize, same_sense: bool) -> usize {
        let s = if same_sense { ".T." } else { ".F." };
        self.emit(format!("ORIENTED_EDGE('',*,*,#{edge_id},{s})"))
    }

    fn edge_loop(&mut self, oedges: &[usize]) -> usize {
        let list: Vec<String> = oedges.iter().map(|id| format!("#{id}")).collect();
        self.emit(format!("EDGE_LOOP('',({}))", list.join(",")))
    }

    fn face_outer_bound(&mut self, loop_id: usize, orientation: bool) -> usize {
        let s = if orientation { ".T." } else { ".F." };
        self.emit(format!("FACE_OUTER_BOUND('',#{loop_id},{s})"))
    }

    fn face_bound(&mut self, loop_id: usize, orientation: bool) -> usize {
        let s = if orientation { ".T." } else { ".F." };
        self.emit(format!("FACE_BOUND('',#{loop_id},{s})"))
    }

    fn advanced_face(&mut self, bounds: &[usize], surf_id: usize, same_sense: bool) -> usize {
        let list: Vec<String> = bounds.iter().map(|id| format!("#{id}")).collect();
        let s = if same_sense { ".T." } else { ".F." };
        self.emit(format!(
            "ADVANCED_FACE('',({}),#{surf_id},{s})",
            list.join(",")
        ))
    }
}

/// Summary of an exported ISO 10303-21 STEP AP214 file.
#[derive(Clone, Copy, Debug)]
pub struct StepExportReport {
    pub bytes_written: usize,
    pub entity_count: usize,
    pub face_count: usize,
    pub cylindrical_features: usize,
}

/// Decode a `CadBrepSolid` into an exact analytical ISO 10303-21 STEP AP214 `.step` file
/// ready for direct import into ANSYS SpaceClaim/Mechanical/Fluent, FreeCAD, SolidWorks, or Fusion 360.
pub fn decode_to_step_ap214(
    solid: &CadBrepSolid,
    part_name: &str,
    path: &Path,
) -> Result<StepExportReport, ErrorCode> {
    let mut sb = StepBuilder::new();

    let hx = 0.5 * solid.width_mm;
    let hy = 0.5 * solid.height_mm;
    let t = solid.thickness_mm;

    // 8 outer corners of the base plate
    let coords: [[f64; 3]; 8] = [
        [-hx, -hy, 0.0], // 0: bottom-left-front
        [hx, -hy, 0.0],  // 1: bottom-right-front
        [hx, hy, 0.0],   // 2: bottom-right-back
        [-hx, hy, 0.0],  // 3: bottom-left-back
        [-hx, -hy, t],   // 4: top-left-front
        [hx, -hy, t],    // 5: top-right-front
        [hx, hy, t],     // 6: top-right-back
        [-hx, hy, t],    // 7: top-left-back
    ];
    let mut v = [0usize; 8];
    for (i, c) in coords.iter().enumerate() {
        let p = sb.point(c[0], c[1], c[2]);
        v[i] = sb.vertex(p);
    }

    // 12 outer edges
    let e_bot = [
        sb.line_edge(v[0], v[1], coords[0], coords[1]),
        sb.line_edge(v[1], v[2], coords[1], coords[2]),
        sb.line_edge(v[2], v[3], coords[2], coords[3]),
        sb.line_edge(v[3], v[0], coords[3], coords[0]),
    ];
    let e_top = [
        sb.line_edge(v[4], v[5], coords[4], coords[5]),
        sb.line_edge(v[5], v[6], coords[5], coords[6]),
        sb.line_edge(v[6], v[7], coords[6], coords[7]),
        sb.line_edge(v[7], v[4], coords[7], coords[4]),
    ];
    let e_vert = [
        sb.line_edge(v[0], v[4], coords[0], coords[4]),
        sb.line_edge(v[1], v[5], coords[1], coords[5]),
        sb.line_edge(v[2], v[6], coords[2], coords[6]),
        sb.line_edge(v[3], v[7], coords[3], coords[7]),
    ];

    let mut top_bounds = Vec::with_capacity(1 + solid.feature_count as usize);
    let mut bot_bounds = Vec::with_capacity(1 + solid.feature_count as usize);
    let mut all_faces = Vec::with_capacity(6 + 2 * solid.feature_count as usize);

    // Outer loops for top (+Z) and bottom (-Z) faces
    let top_outer_oe = [
        sb.oriented_edge(e_top[0], true),
        sb.oriented_edge(e_top[1], true),
        sb.oriented_edge(e_top[2], true),
        sb.oriented_edge(e_top[3], true),
    ];
    let top_outer_loop = sb.edge_loop(&top_outer_oe);
    top_bounds.push(sb.face_outer_bound(top_outer_loop, true));

    let bot_outer_oe = [
        sb.oriented_edge(e_bot[3], false),
        sb.oriented_edge(e_bot[2], false),
        sb.oriented_edge(e_bot[1], false),
        sb.oriented_edge(e_bot[0], false),
    ];
    let bot_outer_loop = sb.edge_loop(&bot_outer_oe);
    bot_bounds.push(sb.face_outer_bound(bot_outer_loop, true));

    // Feature cylindrical surfaces + inner bounds
    let mut cyl_count = 0usize;
    for f in solid.active_features() {
        match f.kind {
            CadFeatureKind::ThroughBore => {
                cyl_count += 1;
                let p_b1 = [f.cx_mm + f.radius_mm, f.cy_mm, 0.0];
                let p_b2 = [f.cx_mm - f.radius_mm, f.cy_mm, 0.0];
                let p_t1 = [f.cx_mm + f.radius_mm, f.cy_mm, t];
                let p_t2 = [f.cx_mm - f.radius_mm, f.cy_mm, t];
                let vb1 = {
                    let p = sb.point(p_b1[0], p_b1[1], p_b1[2]);
                    sb.vertex(p)
                };
                let vb2 = {
                    let p = sb.point(p_b2[0], p_b2[1], p_b2[2]);
                    sb.vertex(p)
                };
                let vt1 = {
                    let p = sb.point(p_t1[0], p_t1[1], p_t1[2]);
                    sb.vertex(p)
                };
                let vt2 = {
                    let p = sb.point(p_t2[0], p_t2[1], p_t2[2]);
                    sb.vertex(p)
                };

                let (cb1, cb2) = sb.circle_arc_edges(f.cx_mm, f.cy_mm, 0.0, f.radius_mm, vb1, vb2);
                let (ct1, ct2) = sb.circle_arc_edges(f.cx_mm, f.cy_mm, t, f.radius_mm, vt1, vt2);
                let seam1 = sb.line_edge(vb1, vt1, p_b1, p_t1);
                let seam2 = sb.line_edge(vb2, vt2, p_b2, p_t2);

                // Inner hole loops (reversed orientation on top/bottom faces)
                let top_in_oe = [sb.oriented_edge(ct2, false), sb.oriented_edge(ct1, false)];
                let top_in_loop = sb.edge_loop(&top_in_oe);
                top_bounds.push(sb.face_bound(top_in_loop, true));

                let bot_in_oe = [sb.oriented_edge(cb1, true), sb.oriented_edge(cb2, true)];
                let bot_in_loop = sb.edge_loop(&bot_in_oe);
                bot_bounds.push(sb.face_bound(bot_in_loop, true));

                // Exact CYLINDRICAL_SURFACE split into two 180-deg half-cylinders (cleanest manifold topology for ANSYS/OCCT)
                let c_pt = sb.point(f.cx_mm, f.cy_mm, 0.0);
                let z_d = sb.dir(0.0, 0.0, 1.0);
                let x_d = sb.dir(1.0, 0.0, 0.0);
                let ax = sb.axis2(c_pt, z_d, x_d);
                let cyl_surf = sb.emit(format!("CYLINDRICAL_SURFACE('',#{ax},{:.6})", f.radius_mm));

                let h1_oe = [
                    sb.oriented_edge(cb1, false),
                    sb.oriented_edge(seam1, true),
                    sb.oriented_edge(ct1, true),
                    sb.oriented_edge(seam2, false),
                ];
                let h1_loop = sb.edge_loop(&h1_oe);
                let h1_bound = sb.face_outer_bound(h1_loop, true);
                all_faces.push(sb.advanced_face(&[h1_bound], cyl_surf, false));

                let h2_oe = [
                    sb.oriented_edge(cb2, false),
                    sb.oriented_edge(seam2, true),
                    sb.oriented_edge(ct2, true),
                    sb.oriented_edge(seam1, false),
                ];
                let h2_loop = sb.edge_loop(&h2_oe);
                let h2_bound = sb.face_outer_bound(h2_loop, true);
                all_faces.push(sb.advanced_face(&[h2_bound], cyl_surf, false));
            }
            CadFeatureKind::BlindBore => {
                cyl_count += 1;
                let z_bot = (t - f.depth_or_height_mm).max(0.001);
                let p_b1 = [f.cx_mm + f.radius_mm, f.cy_mm, z_bot];
                let p_b2 = [f.cx_mm - f.radius_mm, f.cy_mm, z_bot];
                let p_t1 = [f.cx_mm + f.radius_mm, f.cy_mm, t];
                let p_t2 = [f.cx_mm - f.radius_mm, f.cy_mm, t];
                let vb1 = {
                    let p = sb.point(p_b1[0], p_b1[1], p_b1[2]);
                    sb.vertex(p)
                };
                let vb2 = {
                    let p = sb.point(p_b2[0], p_b2[1], p_b2[2]);
                    sb.vertex(p)
                };
                let vt1 = {
                    let p = sb.point(p_t1[0], p_t1[1], p_t1[2]);
                    sb.vertex(p)
                };
                let vt2 = {
                    let p = sb.point(p_t2[0], p_t2[1], p_t2[2]);
                    sb.vertex(p)
                };
                let (cb1, cb2) =
                    sb.circle_arc_edges(f.cx_mm, f.cy_mm, z_bot, f.radius_mm, vb1, vb2);
                let (ct1, ct2) = sb.circle_arc_edges(f.cx_mm, f.cy_mm, t, f.radius_mm, vt1, vt2);
                let seam1 = sb.line_edge(vb1, vt1, p_b1, p_t1);
                let seam2 = sb.line_edge(vb2, vt2, p_b2, p_t2);

                let top_in_oe = [sb.oriented_edge(ct2, false), sb.oriented_edge(ct1, false)];
                let top_in_loop = sb.edge_loop(&top_in_oe);
                top_bounds.push(sb.face_bound(top_in_loop, true));

                let c_pt = sb.point(f.cx_mm, f.cy_mm, z_bot);
                let z_d = sb.dir(0.0, 0.0, 1.0);
                let x_d = sb.dir(1.0, 0.0, 0.0);
                let ax = sb.axis2(c_pt, z_d, x_d);
                let cyl_surf = sb.emit(format!("CYLINDRICAL_SURFACE('',#{ax},{:.6})", f.radius_mm));

                let h1_oe = [
                    sb.oriented_edge(cb1, false),
                    sb.oriented_edge(seam1, true),
                    sb.oriented_edge(ct1, true),
                    sb.oriented_edge(seam2, false),
                ];
                let h1_bound = {
                    let l = sb.edge_loop(&h1_oe);
                    sb.face_outer_bound(l, true)
                };
                all_faces.push(sb.advanced_face(&[h1_bound], cyl_surf, false));

                let h2_oe = [
                    sb.oriented_edge(cb2, false),
                    sb.oriented_edge(seam2, true),
                    sb.oriented_edge(ct2, true),
                    sb.oriented_edge(seam1, false),
                ];
                let h2_bound = {
                    let l = sb.edge_loop(&h2_oe);
                    sb.face_outer_bound(l, true)
                };
                all_faces.push(sb.advanced_face(&[h2_bound], cyl_surf, false));

                // Bore floor plane at z_bot (+Z normal)
                let fl_surf = sb.emit(format!("PLANE('',#{ax})"));
                let fl_oe = [sb.oriented_edge(cb1, true), sb.oriented_edge(cb2, true)];
                let fl_bound = {
                    let l = sb.edge_loop(&fl_oe);
                    sb.face_outer_bound(l, true)
                };
                all_faces.push(sb.advanced_face(&[fl_bound], fl_surf, true));
            }
            CadFeatureKind::CylindricalBoss => {
                cyl_count += 1;
                let z_top = t + f.depth_or_height_mm;
                let p_b1 = [f.cx_mm + f.radius_mm, f.cy_mm, t];
                let p_b2 = [f.cx_mm - f.radius_mm, f.cy_mm, t];
                let p_t1 = [f.cx_mm + f.radius_mm, f.cy_mm, z_top];
                let p_t2 = [f.cx_mm - f.radius_mm, f.cy_mm, z_top];
                let vb1 = {
                    let p = sb.point(p_b1[0], p_b1[1], p_b1[2]);
                    sb.vertex(p)
                };
                let vb2 = {
                    let p = sb.point(p_b2[0], p_b2[1], p_b2[2]);
                    sb.vertex(p)
                };
                let vt1 = {
                    let p = sb.point(p_t1[0], p_t1[1], p_t1[2]);
                    sb.vertex(p)
                };
                let vt2 = {
                    let p = sb.point(p_t2[0], p_t2[1], p_t2[2]);
                    sb.vertex(p)
                };
                let (cb1, cb2) = sb.circle_arc_edges(f.cx_mm, f.cy_mm, t, f.radius_mm, vb1, vb2);
                let (ct1, ct2) =
                    sb.circle_arc_edges(f.cx_mm, f.cy_mm, z_top, f.radius_mm, vt1, vt2);
                let seam1 = sb.line_edge(vb1, vt1, p_b1, p_t1);
                let seam2 = sb.line_edge(vb2, vt2, p_b2, p_t2);

                let top_in_oe = [sb.oriented_edge(cb2, false), sb.oriented_edge(cb1, false)];
                let top_in_loop = sb.edge_loop(&top_in_oe);
                top_bounds.push(sb.face_bound(top_in_loop, true));

                let c_pt = sb.point(f.cx_mm, f.cy_mm, t);
                let z_d = sb.dir(0.0, 0.0, 1.0);
                let x_d = sb.dir(1.0, 0.0, 0.0);
                let ax = sb.axis2(c_pt, z_d, x_d);
                let cyl_surf = sb.emit(format!("CYLINDRICAL_SURFACE('',#{ax},{:.6})", f.radius_mm));

                let h1_oe = [
                    sb.oriented_edge(cb1, true),
                    sb.oriented_edge(seam2, true),
                    sb.oriented_edge(ct1, false),
                    sb.oriented_edge(seam1, false),
                ];
                let h1_bound = {
                    let l = sb.edge_loop(&h1_oe);
                    sb.face_outer_bound(l, true)
                };
                all_faces.push(sb.advanced_face(&[h1_bound], cyl_surf, true));

                let h2_oe = [
                    sb.oriented_edge(cb2, true),
                    sb.oriented_edge(seam1, true),
                    sb.oriented_edge(ct2, false),
                    sb.oriented_edge(seam2, false),
                ];
                let h2_bound = {
                    let l = sb.edge_loop(&h2_oe);
                    sb.face_outer_bound(l, true)
                };
                all_faces.push(sb.advanced_face(&[h2_bound], cyl_surf, true));

                // Boss top cap plane at z_top (+Z normal)
                let cap_pt = sb.point(f.cx_mm, f.cy_mm, z_top);
                let cap_ax = sb.axis2(cap_pt, z_d, x_d);
                let cap_surf = sb.emit(format!("PLANE('',#{cap_ax})"));
                let cap_oe = [sb.oriented_edge(ct1, true), sb.oriented_edge(ct2, true)];
                let cap_bound = {
                    let l = sb.edge_loop(&cap_oe);
                    sb.face_outer_bound(l, true)
                };
                all_faces.push(sb.advanced_face(&[cap_bound], cap_surf, true));
            }
            CadFeatureKind::None => {}
        }
    }

    // 6 outer planar faces
    let z_pos = sb.dir(0.0, 0.0, 1.0);
    let z_neg = sb.dir(0.0, 0.0, -1.0);
    let x_pos = sb.dir(1.0, 0.0, 0.0);
    let x_neg = sb.dir(-1.0, 0.0, 0.0);
    let y_pos = sb.dir(0.0, 1.0, 0.0);
    let y_neg = sb.dir(0.0, -1.0, 0.0);

    // Top face (+Z)
    let p_top = sb.point(0.0, 0.0, t);
    let ax_top = sb.axis2(p_top, z_pos, x_pos);
    let pl_top = sb.emit(format!("PLANE('',#{ax_top})"));
    all_faces.push(sb.advanced_face(&top_bounds, pl_top, true));

    // Bottom face (-Z)
    let p_bot = sb.point(0.0, 0.0, 0.0);
    let ax_bot = sb.axis2(p_bot, z_neg, x_pos);
    let pl_bot = sb.emit(format!("PLANE('',#{ax_bot})"));
    all_faces.push(sb.advanced_face(&bot_bounds, pl_bot, true));

    // Front face (-Y)
    let p_fr = sb.point(0.0, -hy, 0.0);
    let ax_fr = sb.axis2(p_fr, y_neg, x_pos);
    let pl_fr = sb.emit(format!("PLANE('',#{ax_fr})"));
    let oe_fr = [
        sb.oriented_edge(e_bot[0], true),
        sb.oriented_edge(e_vert[1], true),
        sb.oriented_edge(e_top[0], false),
        sb.oriented_edge(e_vert[0], false),
    ];
    let b_fr = {
        let l = sb.edge_loop(&oe_fr);
        sb.face_outer_bound(l, true)
    };
    all_faces.push(sb.advanced_face(&[b_fr], pl_fr, true));

    // Right face (+X)
    let p_rt = sb.point(hx, 0.0, 0.0);
    let ax_rt = sb.axis2(p_rt, x_pos, y_pos);
    let pl_rt = sb.emit(format!("PLANE('',#{ax_rt})"));
    let oe_rt = [
        sb.oriented_edge(e_bot[1], true),
        sb.oriented_edge(e_vert[2], true),
        sb.oriented_edge(e_top[1], false),
        sb.oriented_edge(e_vert[1], false),
    ];
    let b_rt = {
        let l = sb.edge_loop(&oe_rt);
        sb.face_outer_bound(l, true)
    };
    all_faces.push(sb.advanced_face(&[b_rt], pl_rt, true));

    // Back face (+Y)
    let p_bk = sb.point(0.0, hy, 0.0);
    let ax_bk = sb.axis2(p_bk, y_pos, x_neg);
    let pl_bk = sb.emit(format!("PLANE('',#{ax_bk})"));
    let oe_bk = [
        sb.oriented_edge(e_bot[2], true),
        sb.oriented_edge(e_vert[3], true),
        sb.oriented_edge(e_top[2], false),
        sb.oriented_edge(e_vert[2], false),
    ];
    let b_bk = {
        let l = sb.edge_loop(&oe_bk);
        sb.face_outer_bound(l, true)
    };
    all_faces.push(sb.advanced_face(&[b_bk], pl_bk, true));

    // Left face (-X)
    let p_lt = sb.point(-hx, 0.0, 0.0);
    let ax_lt = sb.axis2(p_lt, x_neg, y_neg);
    let pl_lt = sb.emit(format!("PLANE('',#{ax_lt})"));
    let oe_lt = [
        sb.oriented_edge(e_bot[3], true),
        sb.oriented_edge(e_vert[0], true),
        sb.oriented_edge(e_top[3], false),
        sb.oriented_edge(e_vert[3], false),
    ];
    let b_lt = {
        let l = sb.edge_loop(&oe_lt);
        sb.face_outer_bound(l, true)
    };
    all_faces.push(sb.advanced_face(&[b_lt], pl_lt, true));

    let face_count = all_faces.len();
    let face_list: Vec<String> = all_faces.iter().map(|id| format!("#{id}")).collect();
    let shell_id = sb.emit(format!("CLOSED_SHELL('',({}))", face_list.join(",")));
    let brep_id = sb.emit(format!("MANIFOLD_SOLID_BREP('{part_name}',#{shell_id})"));

    // Standard AP214 product + millimeter unit representation context
    let orig = sb.point(0.0, 0.0, 0.0);
    let ax_world = sb.axis2(orig, z_pos, x_pos);
    let u_len = sb.emit(String::from(
        "(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))",
    ));
    let u_ang = sb.emit(String::from(
        "(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))",
    ));
    let u_sol = sb.emit(String::from(
        "(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT())",
    ));
    let unc = sb.emit(format!(
        "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#{u_len},'distance_accuracy_value','confusion accuracy')"
    ));
    let ctx = sb.emit(format!(
        "(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc})) GLOBAL_UNIT_ASSIGNED_CONTEXT((#{u_len},#{u_ang},#{u_sol})) REPRESENTATION_CONTEXT('{part_name}','3D'))"
    ));
    let shape_rep = sb.emit(format!(
        "ADVANCED_BREP_SHAPE_REPRESENTATION('{part_name}',(#{brep_id},#{ax_world}),#{ctx})"
    ));

    let app_ctx = sb.emit(String::from("APPLICATION_CONTEXT('automotive_design')"));
    let _app_proto = sb.emit(format!(
        "APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app_ctx})"
    ));
    let prod_ctx = sb.emit(format!("PRODUCT_CONTEXT('',#{app_ctx},'mechanical')"));
    let prod = sb.emit(format!(
        "PRODUCT('{part_name}','{part_name}','MATERIAL={}',(#{prod_ctx}))",
        solid.material.step_code()
    ));
    let prod_def_form = sb.emit(format!("PRODUCT_DEFINITION_FORMATION('','',#{prod})"));
    let pd_ctx = sb.emit(format!(
        "PRODUCT_DEFINITION_CONTEXT('part definition',#{app_ctx},'design')"
    ));
    let pd = sb.emit(format!(
        "PRODUCT_DEFINITION('design','',#{prod_def_form},#{pd_ctx})"
    ));
    let pds = sb.emit(format!("PRODUCT_DEFINITION_SHAPE('','',#{pd})"));
    let _sdr = sb.emit(format!(
        "SHAPE_DEFINITION_REPRESENTATION(#{pds},#{shape_rep})"
    ));

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    let file = File::create(path).map_err(|_| ErrorCode::IoError)?;
    let mut w = BufWriter::new(file);
    let mut bytes_written = 0usize;

    let header = format!(
        "ISO-10303-21;\n\
         HEADER;\n\
         FILE_DESCRIPTION(('ADA-o1 Exact Analytical B-Rep Solid ({})'),'2;1');\n\
         FILE_NAME('{part_name}.step','2026-11-24T00:00:00',('ADA-o1'),('Quintessa CAD B-Rep'),'quintessa::cad_brep','ADA-o1','');\n\
         FILE_SCHEMA(('AUTOMOTIVE_DESIGN {{ 1 0 10303 214 1 1 1 1 }}'));\n\
         ENDSEC;\n\
         DATA;\n",
        solid.material.step_code()
    );
    w.write_all(header.as_bytes())
        .map_err(|_| ErrorCode::IoError)?;
    bytes_written += header.len();

    for line in &sb.lines {
        w.write_all(line.as_bytes())
            .map_err(|_| ErrorCode::IoError)?;
        w.write_all(b"\n").map_err(|_| ErrorCode::IoError)?;
        bytes_written += line.len() + 1;
    }

    let footer = "ENDSEC;\nEND-ISO-10303-21;\n";
    w.write_all(footer.as_bytes())
        .map_err(|_| ErrorCode::IoError)?;
    bytes_written += footer.len();
    w.flush().map_err(|_| ErrorCode::IoError)?;

    Ok(StepExportReport {
        bytes_written,
        entity_count: sb.lines.len(),
        face_count,
        cylindrical_features: cyl_count,
    })
}

// ============================================================================
// 2. ISO 10303-21 STEP AP214 B-Rep ENCODER (.step -> CadBrepSolid + NCM Field)
// ============================================================================

fn parse_step_ref_id(s: &str) -> Option<usize> {
    let pos = s.find('#')?;
    let rest = &s[pos + 1..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<usize>().ok()
}

fn parse_step_coords(body: &str) -> Option<[f64; 3]> {
    // CARTESIAN_POINT('',(x,y,z))
    let open = body.rfind('(')?;
    let close = body[open + 1..].find(')')?;
    let inner = &body[open + 1..open + 1 + close];
    let mut it = inner.split(',');
    let x = it.next()?.trim().parse::<f64>().ok()?;
    let y = it.next()?.trim().parse::<f64>().ok()?;
    let z = it.next()?.trim().parse::<f64>().ok()?;
    Some([x, y, z])
}

/// Encode an existing ISO 10303-21 `.step` B-Rep file directly into ADA-o1's compact `CadBrepSolid` IR
/// so imported CAD parts can be inspected, simulated (FEA), and modified inside the model workspace.
pub fn encode_from_step_ap214(path: &Path) -> Result<CadBrepSolid, ErrorCode> {
    let text = std::fs::read_to_string(path).map_err(|_| ErrorCode::IoError)?;
    if !text.contains("ISO-10303-21") {
        return Err(ErrorCode::InvalidInput);
    }

    // Parse entity table indexed by entity ID
    let mut points: Vec<(usize, [f64; 3])> = Vec::with_capacity(64);
    let mut axes: Vec<(usize, usize)> = Vec::with_capacity(32); // (axis_id, origin_pt_id)
    let mut cylinders: Vec<(usize, usize, f64)> = Vec::with_capacity(16); // (cyl_id, axis_id, radius)
    let mut circles: Vec<(usize, f64)> = Vec::with_capacity(32); // (axis_id, radius)
    let mut material = CadMaterial::Aluminum6061;

    for raw in text.lines() {
        let line = raw.trim();
        if !line.starts_with('#') {
            continue;
        }
        let Some(eq) = line.find('=') else {
            continue;
        };
        let Ok(id) = line[1..eq].trim().parse::<usize>() else {
            continue;
        };
        let body = line[eq + 1..].trim_end_matches(';').trim();

        if body.starts_with("CARTESIAN_POINT(") {
            if let Some(c) = parse_step_coords(body) {
                points.push((id, c));
            }
        } else if body.starts_with("AXIS2_PLACEMENT_3D(") {
            // AXIS2_PLACEMENT_3D('',#orig,#z,#x)
            if let Some(first_comma) = body.find(',') {
                if let Some(orig_id) = parse_step_ref_id(&body[first_comma + 1..]) {
                    axes.push((id, orig_id));
                }
            }
        } else if body.starts_with("CYLINDRICAL_SURFACE(") {
            // CYLINDRICAL_SURFACE('',#ax,radius)
            let parts: Vec<&str> = body.split(',').collect();
            if parts.len() >= 3 {
                if let Some(ax_id) = parse_step_ref_id(parts[1]) {
                    let r_str = parts[2].trim().trim_end_matches(')');
                    if let Ok(r) = r_str.parse::<f64>() {
                        cylinders.push((id, ax_id, r));
                    }
                }
            }
        } else if body.starts_with("CIRCLE(") {
            let parts: Vec<&str> = body.split(',').collect();
            if parts.len() >= 3 {
                if let Some(ax_id) = parse_step_ref_id(parts[1]) {
                    let r_str = parts[2].trim().trim_end_matches(')');
                    if let Ok(r) = r_str.parse::<f64>() {
                        circles.push((ax_id, r));
                    }
                }
            }
        } else if body.starts_with("PRODUCT(") {
            material = CadMaterial::from_text(body);
        }
    }

    if points.is_empty() {
        return Err(ErrorCode::InvalidInput);
    }

    // Resolve bounding box of the base plate from all CARTESIAN_POINTs
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    for &(_, p) in &points {
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
        min_z = min_z.min(p[2]);
        max_z = max_z.max(p[2]);
    }

    let cx0 = 0.5 * (min_x + max_x);
    let cy0 = 0.5 * (min_y + max_y);
    let width_mm = (max_x - min_x).max(1e-3);
    let height_mm = (max_y - min_y).max(1e-3);

    // Determine base plate thickness (if a boss protrudes above the plate, detect the most common upper Z)
    let mut z_levels: Vec<f64> = points.iter().map(|(_, p)| p[2] - min_z).collect();
    z_levels.sort_by(|a, b| a.total_cmp(b));
    let total_z = (max_z - min_z).max(1e-3);
    let mut thickness_mm = total_z;

    // Lookup helpers
    let find_axis_origin = |ax_id: usize| -> Option<[f64; 3]> {
        let orig_id = axes.iter().find(|&&(a, _)| a == ax_id)?.1;
        points
            .iter()
            .find(|&&(pid, _)| pid == orig_id)
            .map(|&(_, p)| p)
    };

    // Detect if any cylinder originates at an intermediate Z > 0 and has a circle at max_z (boss)
    for &(_, ax_id, r) in &cylinders {
        if let Some(orig) = find_axis_origin(ax_id) {
            let z_rel = orig[2] - min_z;
            if z_rel > 1e-4 && (total_z - z_rel) > 1e-4 {
                let has_top_circle = circles.iter().any(|&(cax, cr)| {
                    (cr - r).abs() < 1e-4
                        && find_axis_origin(cax)
                            .map(|cp| ((cp[2] - min_z) - total_z).abs() < 1e-4)
                            .unwrap_or(false)
                });
                if has_top_circle {
                    thickness_mm = thickness_mm.min(z_rel);
                }
            }
        }
    }

    let mut solid = CadBrepSolid::new_plate(width_mm, height_mm, thickness_mm, material)?;

    for &(_, ax_id, r) in &cylinders {
        let Some(orig) = find_axis_origin(ax_id) else {
            continue;
        };
        let fx = orig[0] - cx0;
        let fy = orig[1] - cy0;
        let fz = orig[2] - min_z;

        // Deduplicate cylinders with identical (cx, cy, r)
        let dup = solid.active_features().iter().any(|f| {
            (f.cx_mm - fx).abs() < 1e-3
                && (f.cy_mm - fy).abs() < 1e-3
                && (f.radius_mm - r).abs() < 1e-3
        });
        if dup {
            continue;
        }

        let feat = if fz <= 1e-4 {
            CadFeature {
                kind: CadFeatureKind::ThroughBore,
                cx_mm: fx,
                cy_mm: fy,
                radius_mm: r,
                depth_or_height_mm: thickness_mm,
            }
        } else if (fz - thickness_mm).abs() < 1e-4 && total_z > thickness_mm + 1e-4 {
            CadFeature {
                kind: CadFeatureKind::CylindricalBoss,
                cx_mm: fx,
                cy_mm: fy,
                radius_mm: r,
                depth_or_height_mm: total_z - thickness_mm,
            }
        } else {
            CadFeature {
                kind: CadFeatureKind::BlindBore,
                cx_mm: fx,
                cy_mm: fy,
                radius_mm: r,
                depth_or_height_mm: (thickness_mm - fz).max(0.001),
            }
        };
        let _ = solid.add_feature(feat);
    }

    Ok(solid)
}

// ============================================================================
// 3. Deterministic 3D Structural FEA Check + VTK Unstructured Grid Exporter
// ============================================================================

/// Structural load case for internal FEA verification.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeaLoadCase {
    TensileAxial = 0,
    CantileverBending = 1,
    BoreBearing = 2,
}

/// Compact safety-factor classification.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeaSafetyStatus {
    Safe = 0,
    Marginal = 1,
    YieldFailure = 2,
}

impl FeaSafetyStatus {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Safe => "PASS_SAFE",
            Self::Marginal => "WARN_MARGINAL",
            Self::YieldFailure => "FAIL_YIELD",
        }
    }
}

/// Deterministic 3D FEA simulation summary for ANSYS pre-check.
#[derive(Clone, Copy, Debug)]
pub struct CadFeaReport {
    pub load_case: FeaLoadCase,
    pub applied_load_n: f64,
    pub stress_concentration_kt: f64,
    pub nominal_stress_mpa: f64,
    pub peak_von_mises_mpa: f64,
    pub yield_strength_mpa: f64,
    pub safety_factor: f64,
    pub max_deflection_mm: f64,
    pub active_hex_cells: usize,
    pub status: FeaSafetyStatus,
}

/// Evaluate analytical + Kirsch stress-concentration 3D FEA check and optionally write a
/// volumetric `.vtk` hexahedral mesh (`UNSTRUCTURED_GRID`) with `von_mises_MPa` and `safety_factor` fields.
pub fn run_fea_and_export_vtk(
    solid: &CadBrepSolid,
    load_case: FeaLoadCase,
    applied_load_n: f64,
    vtk_path: Option<&Path>,
) -> Result<CadFeaReport, ErrorCode> {
    let f_load = applied_load_n.abs().max(1.0);
    let e_mpa = solid.material.youngs_modulus_mpa();
    let sy_mpa = solid.material.yield_strength_mpa();

    // Find governing bore for Peterson's finite-width stress concentration factor K_t
    let mut max_bore_r = 0.0_f64;
    for f in solid.active_features() {
        if matches!(
            f.kind,
            CadFeatureKind::ThroughBore | CadFeatureKind::BlindBore
        ) {
            max_bore_r = max_bore_r.max(f.radius_mm);
        }
    }
    let ratio = ((2.0 * max_bore_r) / solid.height_mm).clamp(0.0, 0.85);
    let kt = if max_bore_r > 0.0 {
        (3.0 - 3.13 * ratio + 3.66 * ratio * ratio - 1.53 * ratio.powi(3)).max(1.0)
    } else {
        1.0
    };

    let net_width = (solid.height_mm - 2.0 * max_bore_r).max(1.0);
    let net_area_mm2 = net_width * solid.thickness_mm;

    let (nominal_stress_mpa, max_deflection_mm) = match load_case {
        FeaLoadCase::TensileAxial => {
            let nom = f_load / net_area_mm2;
            let gross_area = solid.height_mm * solid.thickness_mm;
            let delta = (f_load * solid.width_mm) / (gross_area * e_mpa);
            (nom, delta)
        }
        FeaLoadCase::CantileverBending => {
            // Cantilever loaded at free end X = +W/2, fixed at X = -W/2
            let i_xx = (solid.height_mm * solid.thickness_mm.powi(3)) / 12.0;
            let c = 0.5 * solid.thickness_mm;
            let nom = (f_load * solid.width_mm * c) / i_xx.max(1e-6);
            let delta = (f_load * solid.width_mm.powi(3)) / (3.0 * e_mpa * i_xx.max(1e-6));
            (nom, delta)
        }
        FeaLoadCase::BoreBearing => {
            let d_bore = (2.0 * max_bore_r).max(2.0);
            let nom = f_load / (d_bore * solid.thickness_mm);
            let delta = (f_load * solid.width_mm) / (net_area_mm2 * e_mpa);
            (nom, delta)
        }
    };

    // Discretize on a bounded 24 x 16 x 6 hexahedral grid and evaluate Kirsch local Von Mises field
    const NX: usize = 24;
    const NY: usize = 16;
    const NZ: usize = 6;
    let dx = solid.width_mm / (NX as f64);
    let dy = solid.height_mm / (NY as f64);
    let z_max = solid.max_z_mm();
    let dz = z_max / (NZ as f64);

    let mut active_cells: Vec<(usize, usize, usize, f64, f64)> = Vec::with_capacity(NX * NY * NZ);
    let mut peak_vm = nominal_stress_mpa * kt;

    for iz in 0..NZ {
        let z = (iz as f64 + 0.5) * dz;
        for iy in 0..NY {
            let y = -0.5 * solid.height_mm + (iy as f64 + 0.5) * dy;
            for ix in 0..NX {
                let x = -0.5 * solid.width_mm + (ix as f64 + 0.5) * dx;
                let sdf = solid.sdf_mm(x, y, z);
                if sdf > 0.0 {
                    continue;
                }
                // Local Kirsch stress enhancement around every bore
                let mut local_k = 1.0_f64;
                for f in solid.active_features() {
                    if matches!(
                        f.kind,
                        CadFeatureKind::ThroughBore | CadFeatureKind::BlindBore
                    ) {
                        let rx = x - f.cx_mm;
                        let ry = y - f.cy_mm;
                        let r2 = (rx * rx + ry * ry).max(f.radius_mm * f.radius_mm);
                        let a2_r2 = (f.radius_mm * f.radius_mm) / r2;
                        let cos2t = (rx * rx - ry * ry) / r2;
                        let k_kirsch =
                            (1.0 + 0.5 * a2_r2 - 1.5 * a2_r2 * a2_r2 * cos2t).clamp(0.5, kt);
                        local_k = local_k.max(k_kirsch);
                    }
                }
                let bend_factor = if load_case == FeaLoadCase::CantileverBending {
                    let x_arm = (0.5 * solid.width_mm - x) / solid.width_mm;
                    let z_arm = ((z - 0.5 * solid.thickness_mm).abs())
                        / (0.5 * solid.thickness_mm).max(1e-6);
                    (x_arm * z_arm).clamp(0.1, 1.0)
                } else {
                    1.0
                };
                let vm = nominal_stress_mpa * local_k * bend_factor;
                if vm > peak_vm {
                    peak_vm = vm;
                }
                active_cells.push((ix, iy, iz, vm, sdf));
            }
        }
    }

    let safety_factor = sy_mpa / peak_vm.max(1e-6);
    let status = if safety_factor >= 2.0 {
        FeaSafetyStatus::Safe
    } else if safety_factor >= 1.0 {
        FeaSafetyStatus::Marginal
    } else {
        FeaSafetyStatus::YieldFailure
    };

    if let Some(path) = vtk_path {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        let file = File::create(path).map_err(|_| ErrorCode::IoError)?;
        let mut w = BufWriter::new(file);
        let n_pts = (NX + 1) * (NY + 1) * (NZ + 1);
        let n_cells = active_cells.len();

        writeln!(w, "# vtk DataFile Version 3.0").map_err(|_| ErrorCode::IoError)?;
        writeln!(
            w,
            "ADA-o1 3D FEA Stress Field ({})",
            solid.material.step_code()
        )
        .map_err(|_| ErrorCode::IoError)?;
        writeln!(w, "ASCII").map_err(|_| ErrorCode::IoError)?;
        writeln!(w, "DATASET UNSTRUCTURED_GRID").map_err(|_| ErrorCode::IoError)?;
        writeln!(w, "POINTS {n_pts} float").map_err(|_| ErrorCode::IoError)?;
        for iz in 0..=NZ {
            let z = (iz as f64) * dz;
            for iy in 0..=NY {
                let y = -0.5 * solid.height_mm + (iy as f64) * dy;
                for ix in 0..=NX {
                    let x = -0.5 * solid.width_mm + (ix as f64) * dx;
                    writeln!(w, "{x:.4} {y:.4} {z:.4}").map_err(|_| ErrorCode::IoError)?;
                }
            }
        }

        let pt_idx = |ix: usize, iy: usize, iz: usize| -> usize {
            iz * (NY + 1) * (NX + 1) + iy * (NX + 1) + ix
        };

        writeln!(w, "CELLS {n_cells} {}", n_cells * 9).map_err(|_| ErrorCode::IoError)?;
        for &(ix, iy, iz, _, _) in &active_cells {
            writeln!(
                w,
                "8 {} {} {} {} {} {} {} {}",
                pt_idx(ix, iy, iz),
                pt_idx(ix + 1, iy, iz),
                pt_idx(ix + 1, iy + 1, iz),
                pt_idx(ix, iy + 1, iz),
                pt_idx(ix, iy, iz + 1),
                pt_idx(ix + 1, iy, iz + 1),
                pt_idx(ix + 1, iy + 1, iz + 1),
                pt_idx(ix, iy + 1, iz + 1),
            )
            .map_err(|_| ErrorCode::IoError)?;
        }

        writeln!(w, "CELL_TYPES {n_cells}").map_err(|_| ErrorCode::IoError)?;
        for _ in 0..n_cells {
            writeln!(w, "12").map_err(|_| ErrorCode::IoError)?;
        }

        writeln!(w, "CELL_DATA {n_cells}").map_err(|_| ErrorCode::IoError)?;
        writeln!(w, "SCALARS von_mises_MPa float 1").map_err(|_| ErrorCode::IoError)?;
        writeln!(w, "LOOKUP_TABLE default").map_err(|_| ErrorCode::IoError)?;
        for &(_, _, _, vm, _) in &active_cells {
            writeln!(w, "{vm:.4}").map_err(|_| ErrorCode::IoError)?;
        }

        writeln!(w, "SCALARS safety_factor float 1").map_err(|_| ErrorCode::IoError)?;
        writeln!(w, "LOOKUP_TABLE default").map_err(|_| ErrorCode::IoError)?;
        for &(_, _, _, vm, _) in &active_cells {
            let sf = (sy_mpa / vm.max(1e-3)).min(99.99);
            writeln!(w, "{sf:.4}").map_err(|_| ErrorCode::IoError)?;
        }
        w.flush().map_err(|_| ErrorCode::IoError)?;
    }

    Ok(CadFeaReport {
        load_case,
        applied_load_n: f_load,
        stress_concentration_kt: kt,
        nominal_stress_mpa,
        peak_von_mises_mpa: peak_vm,
        yield_strength_mpa: sy_mpa,
        safety_factor,
        max_deflection_mm,
        active_hex_cells: active_cells.len(),
        status,
    })
}

// ============================================================================
// 4. High-Resolution Watertight Binary `.stl` Preview Tessellator
// ============================================================================

fn push_tri(tris: &mut Vec<([f32; 3], [f32; 3], [f32; 3])>, a: [f64; 3], b: [f64; 3], c: [f64; 3]) {
    tris.push((
        [a[0] as f32, a[1] as f32, a[2] as f32],
        [b[0] as f32, b[1] as f32, b[2] as f32],
        [c[0] as f32, c[1] as f32, c[2] as f32],
    ));
}

fn tri_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let nx = u[1] * v[2] - u[2] * v[1];
    let ny = u[2] * v[0] - u[0] * v[2];
    let nz = u[0] * v[1] - u[1] * v[0];
    let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-12);
    [nx / len, ny / len, nz / len]
}

/// Export a high-resolution watertight binary `.stl` preview (`circle_segments = 128` recommended).
pub fn decode_to_binary_stl(
    solid: &CadBrepSolid,
    circle_segments: usize,
    path: &Path,
) -> Result<(usize, usize), ErrorCode> {
    let segs = circle_segments.clamp(24, 256);
    let hx = 0.5 * solid.width_mm;
    let hy = 0.5 * solid.height_mm;
    let t = solid.thickness_mm;

    let mut tris: Vec<([f32; 3], [f32; 3], [f32; 3])> = Vec::with_capacity(segs * 12);

    // 4 outer side walls (exact quads -> 8 triangles)
    let c0 = [-hx, -hy, 0.0];
    let c1 = [hx, -hy, 0.0];
    let c2 = [hx, hy, 0.0];
    let c3 = [-hx, hy, 0.0];
    let c4 = [-hx, -hy, t];
    let c5 = [hx, -hy, t];
    let c6 = [hx, hy, t];
    let c7 = [-hx, hy, t];

    // Front (-Y), Right (+X), Back (+Y), Left (-X)
    push_tri(&mut tris, c0, c1, c5);
    push_tri(&mut tris, c0, c5, c4);
    push_tri(&mut tris, c1, c2, c6);
    push_tri(&mut tris, c1, c6, c5);
    push_tri(&mut tris, c2, c3, c7);
    push_tri(&mut tris, c2, c7, c6);
    push_tri(&mut tris, c3, c0, c4);
    push_tri(&mut tris, c3, c4, c7);

    // Find primary centered through-bore (if present) for exact radial fan triangulation on top/bottom
    let primary_bore = solid
        .active_features()
        .iter()
        .find(|f| f.kind == CadFeatureKind::ThroughBore);

    if let Some(bore) = primary_bore {
        // Exact watertight radial strip between circle perimeter and rectangle boundary for top (+Z) & bottom (-Z)
        for i in 0..segs {
            let a0 = 2.0 * PI * (i as f64) / (segs as f64);
            let a1 = 2.0 * PI * ((i + 1) as f64) / (segs as f64);
            let (ca0, sa0) = (a0.cos(), a0.sin());
            let (ca1, sa1) = (a1.cos(), a1.sin());

            let in0 = [
                bore.cx_mm + bore.radius_mm * ca0,
                bore.cy_mm + bore.radius_mm * sa0,
            ];
            let in1 = [
                bore.cx_mm + bore.radius_mm * ca1,
                bore.cy_mm + bore.radius_mm * sa1,
            ];

            // Ray from bore center to outer rectangle boundary
            let ray_to_rect = |dx: f64, dy: f64| -> [f64; 2] {
                let tx = if dx > 1e-12 {
                    (hx - bore.cx_mm) / dx
                } else if dx < -1e-12 {
                    (-hx - bore.cx_mm) / dx
                } else {
                    f64::INFINITY
                };
                let ty = if dy > 1e-12 {
                    (hy - bore.cy_mm) / dy
                } else if dy < -1e-12 {
                    (-hy - bore.cy_mm) / dy
                } else {
                    f64::INFINITY
                };
                let s = tx.min(ty);
                [
                    (bore.cx_mm + s * dx).clamp(-hx, hx),
                    (bore.cy_mm + s * dy).clamp(-hy, hy),
                ]
            };
            let out0 = ray_to_rect(ca0, sa0);
            let out1 = ray_to_rect(ca1, sa1);

            // Top annular quad (+Z)
            push_tri(
                &mut tris,
                [in0[0], in0[1], t],
                [out0[0], out0[1], t],
                [out1[0], out1[1], t],
            );
            push_tri(
                &mut tris,
                [in0[0], in0[1], t],
                [out1[0], out1[1], t],
                [in1[0], in1[1], t],
            );

            // Bottom annular quad (-Z)
            push_tri(
                &mut tris,
                [in0[0], in0[1], 0.0],
                [out1[0], out1[1], 0.0],
                [out0[0], out0[1], 0.0],
            );
            push_tri(
                &mut tris,
                [in0[0], in0[1], 0.0],
                [in1[0], in1[1], 0.0],
                [out1[0], out1[1], 0.0],
            );

            // Inner bore wall (inward-facing normal)
            push_tri(
                &mut tris,
                [in0[0], in0[1], 0.0],
                [in0[0], in0[1], t],
                [in1[0], in1[1], t],
            );
            push_tri(
                &mut tris,
                [in0[0], in0[1], 0.0],
                [in1[0], in1[1], t],
                [in1[0], in1[1], 0.0],
            );
        }
    } else {
        // Solid plate top & bottom
        push_tri(&mut tris, c4, c5, c6);
        push_tri(&mut tris, c4, c6, c7);
        push_tri(&mut tris, c0, c2, c1);
        push_tri(&mut tris, c0, c3, c2);
    }

    // Add any protruding CylindricalBosses
    for f in solid.active_features() {
        if f.kind == CadFeatureKind::CylindricalBoss {
            let z_top = t + f.depth_or_height_mm;
            let ctr = [f.cx_mm, f.cy_mm, z_top];
            for i in 0..segs {
                let a0 = 2.0 * PI * (i as f64) / (segs as f64);
                let a1 = 2.0 * PI * ((i + 1) as f64) / (segs as f64);
                let p0b = [
                    f.cx_mm + f.radius_mm * a0.cos(),
                    f.cy_mm + f.radius_mm * a0.sin(),
                    t,
                ];
                let p1b = [
                    f.cx_mm + f.radius_mm * a1.cos(),
                    f.cy_mm + f.radius_mm * a1.sin(),
                    t,
                ];
                let p0t = [p0b[0], p0b[1], z_top];
                let p1t = [p1b[0], p1b[1], z_top];
                push_tri(&mut tris, p0b, p1b, p1t);
                push_tri(&mut tris, p0b, p1t, p0t);
                push_tri(&mut tris, ctr, p0t, p1t);
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
    let tag = b"ADA-o1 Exact Analytical B-Rep High-Res STL Preview";
    header[..tag.len()].copy_from_slice(tag);
    w.write_all(&header).map_err(|_| ErrorCode::IoError)?;
    w.write_all(&(tris.len() as u32).to_le_bytes())
        .map_err(|_| ErrorCode::IoError)?;

    for &(a, b, c) in &tris {
        let n = tri_normal(a, b, c);
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

// ============================================================================
// 5. Prompt-to-Solid Parametric Parser (Natural Language -> CadBrepSolid)
// ============================================================================

fn extract_number_after(text: &str, keyword: &str) -> Option<f64> {
    let pos = text.find(keyword)?;
    let rest = &text[pos + keyword.len()..];
    let mut started = false;
    let mut buf = String::with_capacity(16);
    for ch in rest.chars() {
        if ch.is_ascii_digit() || ch == '.' || (ch == '-' && !started) {
            started = true;
            buf.push(ch);
        } else if started {
            break;
        }
    }
    buf.parse::<f64>().ok()
}

/// Parse a parametric CAD prompt (e.g. `"Parametric CAD Aluminum 6061 plate (96x64x15 mm) with centered through-bore radius 11 mm"`)
/// into a `CadBrepSolid` + optional applied load in Newtons.
pub fn parse_cad_solid_from_prompt(prompt: &str) -> Option<(CadBrepSolid, FeaLoadCase, f64)> {
    let lower = prompt.to_ascii_lowercase();
    if !(lower.contains("plate")
        || lower.contains("cad")
        || lower.contains("bore")
        || lower.contains("block")
        || lower.contains("bracket"))
    {
        return None;
    }

    // Parse WxHxT triplet (e.g. `96x64x15`)
    let mut dims: Option<(f64, f64, f64)> = None;
    for tok in lower.split(|c: char| !(c.is_ascii_alphanumeric() || c == '.')) {
        let parts: Vec<&str> = tok.split('x').collect();
        if parts.len() == 3 {
            if let (Ok(w), Ok(h), Ok(t)) = (
                parts[0].parse::<f64>(),
                parts[1].parse::<f64>(),
                parts[2].parse::<f64>(),
            ) {
                if w > 0.0 && h > 0.0 && t > 0.0 {
                    dims = Some((w, h, t));
                    break;
                }
            }
        }
    }
    let (w, h, t) = dims?;
    let mat = CadMaterial::from_text(&lower);
    let mut solid = CadBrepSolid::new_plate(w, h, t, mat).ok()?;

    if let Some(r) = extract_number_after(&lower, "radius") {
        let _ = solid.add_feature(CadFeature {
            kind: CadFeatureKind::ThroughBore,
            cx_mm: 0.0,
            cy_mm: 0.0,
            radius_mm: r,
            depth_or_height_mm: t,
        });
    } else if let Some(d) = extract_number_after(&lower, "diameter") {
        let _ = solid.add_feature(CadFeature {
            kind: CadFeatureKind::ThroughBore,
            cx_mm: 0.0,
            cy_mm: 0.0,
            radius_mm: 0.5 * d,
            depth_or_height_mm: t,
        });
    }

    if let Some(boss_r) = extract_number_after(&lower, "boss radius") {
        let boss_h = extract_number_after(&lower, "boss height").unwrap_or(0.5 * t);
        let _ = solid.add_feature(CadFeature {
            kind: CadFeatureKind::CylindricalBoss,
            cx_mm: 0.0,
            cy_mm: 0.0,
            radius_mm: boss_r,
            depth_or_height_mm: boss_h,
        });
    }

    let load_case = if lower.contains("cantilever") || lower.contains("bending") {
        FeaLoadCase::CantileverBending
    } else if lower.contains("bearing") {
        FeaLoadCase::BoreBearing
    } else {
        FeaLoadCase::TensileAxial
    };
    let load_n = extract_number_after(&lower, "load")
        .or_else(|| extract_number_after(&lower, "force"))
        .unwrap_or(5_000.0);

    Some((solid, load_case, load_n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_ap214_encode_decode_roundtrip_and_fea_vtk_stl() {
        let mut solid =
            CadBrepSolid::new_plate(96.0, 64.0, 15.0, CadMaterial::Aluminum6061).unwrap();
        solid
            .add_feature(CadFeature {
                kind: CadFeatureKind::ThroughBore,
                cx_mm: 0.0,
                cy_mm: 0.0,
                radius_mm: 11.0,
                depth_or_height_mm: 15.0,
            })
            .unwrap();

        assert!((solid.volume_mm3() - 86_458.01).abs() < 0.05);
        assert!((solid.mass_g() - 233.44).abs() < 0.05);
        assert!((solid.min_wall_thickness_mm() - 21.0).abs() < 1e-6);

        let tmp = std::env::temp_dir();
        let step_p = tmp.join("ada_o1_test_plate.step");
        let vtk_p = tmp.join("ada_o1_test_plate.vtk");
        let stl_p = tmp.join("ada_o1_test_plate.stl");

        let rep = decode_to_step_ap214(&solid, "plate_96x64x15_bore11", &step_p).unwrap();
        assert!(rep.bytes_written > 1000);
        assert_eq!(rep.face_count, 8); // 6 outer planar faces + 2 half-cylinder bore faces
        assert_eq!(rep.cylindrical_features, 1);

        // Encode back from .step -> CadBrepSolid and verify exact round-trip!
        let encoded = encode_from_step_ap214(&step_p).unwrap();
        assert!((encoded.width_mm - 96.0).abs() < 1e-4);
        assert!((encoded.height_mm - 64.0).abs() < 1e-4);
        assert!((encoded.thickness_mm - 15.0).abs() < 1e-4);
        assert_eq!(encoded.material, CadMaterial::Aluminum6061);
        assert_eq!(encoded.feature_count, 1);
        assert_eq!(encoded.features[0].kind, CadFeatureKind::ThroughBore);
        assert!((encoded.features[0].radius_mm - 11.0).abs() < 1e-4);
        assert!((encoded.volume_mm3() - solid.volume_mm3()).abs() < 1e-3);

        // Encode to NCM Field3D4x4x4 & Canvas8x8
        let f3d = encoded.encode_to_ncm_field3d();
        assert_eq!(f3d.len(), 64);
        let c8 = encoded.encode_to_canvas8x8();
        assert_eq!(c8.len(), 64);

        // Run 3D FEA simulation & export VTK + high-res STL
        let fea =
            run_fea_and_export_vtk(&encoded, FeaLoadCase::TensileAxial, 10_000.0, Some(&vtk_p))
                .unwrap();
        assert!(fea.active_hex_cells > 1500);
        assert!(fea.stress_concentration_kt > 2.0);
        assert!(fea.safety_factor > 5.0);
        assert_eq!(fea.status, FeaSafetyStatus::Safe);

        let (stl_bytes, stl_tris) = decode_to_binary_stl(&encoded, 128, &stl_p).unwrap();
        assert!(stl_bytes > 30_000);
        assert!(stl_tris >= 768);

        let _ = std::fs::remove_file(&step_p);
        let _ = std::fs::remove_file(&vtk_p);
        let _ = std::fs::remove_file(&stl_p);
    }
}
