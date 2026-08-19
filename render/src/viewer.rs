//! miniquad-based 3D point cloud viewer.
//!
//! Renders points as GPU-accelerated vertex buffers with an orbit camera.
//!
//! Uses miniquad 0.4.x low-level API (raw OpenGL wrapper).
//! Uses nalgebra for raycasting in point selection.

use nalgebra::{Matrix4, Vector3, Vector4};
use std::collections::HashMap;

use egui_miniquad::EguiMq;
use miniquad::{
    conf, window, BufferId, BufferLayout, BufferSource, BufferType, BufferUsage, EventHandler,
    KeyCode, KeyMods, Pipeline, PipelineParams, PrimitiveType, RenderingBackend,
    ShaderMeta, ShaderSource, TouchPhase, UniformBlockLayout, UniformDesc, UniformType,
    VertexAttribute, VertexFormat,
};
use miniquad::graphics::raw_gl::{glEnable, GL_PROGRAM_POINT_SIZE};

use crate::camera::Camera;
use reader::PointCloud;

/// Navigation mode for touch / tablet input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavMode {
    /// Orbit around a fixed center point (default).
    Orbit,
    /// Fly through the cloud: 1-finger look, 2-finger move.
    Fly,
}

/// Default point render color (light gray).
const DEFAULT_COLOR: [f32; 3] = [1.0, 1.0, 1.0];

/// Highlight color for selected point (bright red).
const SELECTED_COLOR: [f32; 3] = [1.0, 0.15, 0.15];

/// Highlight point size (larger than default).
const SELECTED_POINT_SIZE: f32 = 10000.0;

/// Vertex shader source (GLSL 100).
const VERTEX_SHADER: &str = include_str!("shaders/point.glsl.vert");

/// Fragment shader source (GLSL 100).
const FRAGMENT_SHADER: &str = include_str!("shaders/point.glsl.frag");

/// The main viewer application.
/// Axis frame shader (vertex) - renders lines with VP matrix uniform
const AXIS_VERTEX_SHADER: &str = r#"
#version 100
attribute vec4 position;
attribute vec3 color;
uniform mat4 vp;
varying vec3 v_color;
void main() {
    gl_Position = vp * position;
    v_color = color;
}
"#;

/// Axis frame shader (fragment) - passes through vertex color
const AXIS_FRAGMENT_SHADER: &str = r#"
#version 100
precision mediump float;
varying vec3 v_color;
void main() {
    gl_FragColor = vec4(v_color, 1.0);
}
"#;

pub struct Viewer {
    cloud: PointCloud,
    camera: Camera,
    ctx: Option<Box<dyn RenderingBackend>>,
    pipeline: Option<Pipeline>,
    vertex_buffer: Option<BufferId>,
    index_buffer: Option<BufferId>,
    point_count: u32,
    initialized: bool,
    mouse_down: bool,
    last_mouse: (f32, f32),
    /// Index of the currently selected point (via Ctrl+click).
    selected_point: Option<usize>,
    /// Cached info about selected point for display: (index, x, y, z).
    selected_point_info: Option<(usize, f32, f32, f32)>,
    /// Whether the Ctrl key is currently held.
    ctrl_held: bool,
    /// Render size of regular points (adjustable with +/- keys).
    point_size: f32,
    /// Whether to show the coordinate origin axis frame.
    show_origin: bool,
    /// Axis frame vertex buffer (6 vertices: 3 axes × 2 endpoints, each with xyz + rgb).
    axis_vertex_buffer: Option<BufferId>,
    /// Axis frame index buffer (6 indices for 3 line segments).
    axis_index_buffer: Option<BufferId>,
    /// Axis frame line pipeline.
    axis_pipeline: Option<Pipeline>,
    /// Active touch points: id → (x, y). Supports 1-finger orbit and 2-finger pinch/pan.
    touch_points: HashMap<u64, (f32, f32)>,
    /// Current touch/tablet navigation mode.
    nav_mode: NavMode,
    /// Which mouse button is currently held down
    mouse_button_down: Option<miniquad::MouseButton>,
    /// Whether shift key is held
    shift_held: bool,
    /// egui context + renderer (created lazily after the window is live).
    egui_mq: Option<EguiMq>,
}

impl Viewer {
    /// Create a new viewer for the given point cloud.
    pub fn new(cloud: PointCloud, show_origin: bool) -> Self {
        let bounds = cloud_bounds(&cloud);
        let camera = Camera::from_bounds(bounds.0, bounds.1);

        Self {
            cloud,
            camera,
            ctx: None,
            pipeline: None,
            vertex_buffer: None,
            index_buffer: None,
            point_count: 0,
            initialized: false,
            mouse_down: false,
            last_mouse: (0.0, 0.0),
            selected_point: None,
            selected_point_info: None,
            ctrl_held: false,
            point_size: 1.0,
            show_origin,
            axis_vertex_buffer: None,
            axis_index_buffer: None,
            axis_pipeline: None,
            touch_points: HashMap::new(),
            nav_mode: NavMode::Orbit,
            mouse_button_down: None,
            shift_held: false,
            egui_mq: None,
        }
    }

    /// Initialize GPU resources (called once on first frame, inside miniquad loop).
    fn init(&mut self) {
        self.point_count = self.cloud.len() as u32;

        // Compute bounds for axis scaling
        let (bounds_min, bounds_max) = cloud_bounds(&self.cloud);

        // Create a single rendering backend shared by both our 3D rendering
        // and egui. Two separate backends pointing at the same GL context
        // corrupt each other's state tracking and cause uniform layout panics.
        let mut mq_ctx = window::new_rendering_backend();
        self.egui_mq = Some(EguiMq::new(&mut *mq_ctx));
        self.ctx = Some(mq_ctx);
        let ctx = self.ctx.as_mut().unwrap().as_mut();

        // Enable GL_PROGRAM_POINT_SIZE so vertex shader can control gl_PointSize
        unsafe {
            glEnable(GL_PROGRAM_POINT_SIZE);
        }

        // Build vertex data: each point = 4 floats (x, y, z, point_size)
        let vertices: Vec<f32> = self
            .cloud
            .points()
            .iter()
            .flat_map(|p| [p.x(), p.y(), p.z(), self.point_size])
            .collect();

        // Create vertex buffer
        self.vertex_buffer = Some(ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&vertices),
        ));

        // Create index buffer (0..N for drawing all points)
        let indices: Vec<u32> = (0..self.point_count).collect();
        self.index_buffer = Some(ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&indices),
        ));

        // Create shader with uniform metadata
        let shader_meta = ShaderMeta {
            images: vec![],
            uniforms: UniformBlockLayout {
                uniforms: vec![
                    UniformDesc::new("vp", UniformType::Mat4),
                    UniformDesc::new("color", UniformType::Float3),
                ],
            },
        };

        let shader = ctx
            .new_shader(
                ShaderSource::Glsl {
                    vertex: VERTEX_SHADER,
                    fragment: FRAGMENT_SHADER,
                },
                shader_meta,
            )
            .expect("Failed to compile shader");

        // Create pipeline for point rendering
        let buffer_layout = [BufferLayout::default()];
        let attributes = [VertexAttribute::new("position", VertexFormat::Float4)];

        self.pipeline = Some(
            ctx.new_pipeline(
                &buffer_layout,
                &attributes,
                shader,
                PipelineParams {
                    primitive_type: PrimitiveType::Points,
                    ..Default::default()
                },
            ),
        );

        // Build axis frame if enabled
        if self.show_origin {
            // Axis length scales with cloud bounds
            let scale = (bounds_max[0] - bounds_min[0])
                .max(bounds_max[1] - bounds_min[1])
                .max(bounds_max[2] - bounds_min[2])
                * 0.1;

            // 6 vertices: X axis (red), Y axis (green), Z axis (blue)
            // Each vertex: x, y, z, r, g, b
            let axis_verts: [f32; 36] = [
                // X axis (red)
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                scale, 0.0, 0.0, 1.0, 0.0, 0.0,
                // Y axis (green)
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, scale, 0.0, 0.0, 1.0, 0.0,
                // Z axis (blue)
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, scale, 0.0, 0.0, 1.0,
            ];

            self.axis_vertex_buffer = Some(ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&axis_verts),
            ));

            // Index buffer for 3 line segments (6 vertices)
            let axis_indices: [u32; 6] = [0, 1, 2, 3, 4, 5];
            self.axis_index_buffer = Some(ctx.new_buffer(
                BufferType::IndexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&axis_indices),
            ));

            // Shader with VP uniform and per-vertex color
            let axis_shader_meta = ShaderMeta {
                images: vec![],
                uniforms: UniformBlockLayout {
                    uniforms: vec![UniformDesc::new("vp", UniformType::Mat4)],
                },
            };

            let axis_shader = ctx
                .new_shader(
                    ShaderSource::Glsl {
                        vertex: AXIS_VERTEX_SHADER,
                        fragment: AXIS_FRAGMENT_SHADER,
                    },
                    axis_shader_meta,
                )
                .expect("Failed to compile axis shader");

            // Layout: xyz (Float3) + rgb (Float3)
            let axis_buffer_layout = [
                BufferLayout {
                    stride: 6 * 4, // 6 floats × 4 bytes
                    ..Default::default()
                },
            ];
            let axis_attributes = [
                VertexAttribute::new("position", VertexFormat::Float3),
                VertexAttribute::new("color", VertexFormat::Float3),
            ];

            self.axis_pipeline = Some(
                ctx.new_pipeline(
                    &axis_buffer_layout,
                    &axis_attributes,
                    axis_shader,
                    PipelineParams {
                        primitive_type: PrimitiveType::Lines,
                        ..Default::default()
                    },
                ),
            );
        }

        self.initialized = true;
    }

    /// Apply uniforms (view-projection matrix + color).
    fn apply_uniforms(&mut self, color: [f32; 3], _point_size: f32) {
        let (width, height) = window::screen_size();
        // Far plane scales with camera distance so large clouds aren't clipped
        let far = self.camera.distance * 10.0;
        // Near plane must be small enough to not clip points when zooming in close
        let near = (self.camera.distance * 0.001).max(0.001);
        let proj = perspective(45.0, width / height, near, far);
        let view = self.camera.view_matrix();
        let vp = multiply(&proj, &view);

        // Pack uniforms: mat4 (16 f32) + vec3 (3 f32) = 19 f32
        let uniforms: [f32; 19] = {
            let mut u = [0.0; 19];
            u[0..16].copy_from_slice(&vp);
            u[16] = color[0];
            u[17] = color[1];
            u[18] = color[2];
            u
        };

        self.ctx.as_mut().unwrap()
            .apply_uniforms_from_bytes(
                uniforms.as_ptr() as *const u8,
                std::mem::size_of_val(&uniforms),
            );
    }

    /// Apply VP matrix uniform for the axis frame.
    fn apply_axis_uniforms(&mut self) {
        let (width, height) = window::screen_size();
        let far = self.camera.distance * 10.0;
        let near = (self.camera.distance * 0.001).max(0.001);
        let proj = perspective(45.0, width / height, near, far);
        let view = self.camera.view_matrix();
        let vp = multiply(&proj, &view);

        self.ctx.as_mut().unwrap()
            .apply_uniforms_from_bytes(
                vp.as_ptr() as *const u8,
                std::mem::size_of_val(&vp),
            );
    }

    /// Rebuild the vertex buffer with the current point size.
    /// Point size is embedded in each vertex as the 4th component (w).
    fn rebuild_vertex_buffer(&mut self) {
        let ctx = self.ctx.as_mut().expect("Context not initialized");
        let vertices: Vec<f32> = self
            .cloud
            .points()
            .iter()
            .flat_map(|p| [p.x(), p.y(), p.z(), self.point_size])
            .collect();

        self.vertex_buffer = Some(ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&vertices),
        ));
    }

    /// Compute the VP matrix as a nalgebra Matrix4 (for raycasting).
    fn compute_vp_matrix(&self) -> Matrix4<f32> {
        let (width, height) = window::screen_size();
        let far = self.camera.distance * 10.0;
        let proj = perspective(45.0, width / height, 0.1, far);
        let view = self.camera.view_matrix();
        let vp = multiply(&proj, &view);
        Matrix4::from_column_slice(&vp)
    }

    /// Pick the nearest point to a screen-space click via raycasting.
    fn pick_point(&self, screen_x: f32, screen_y: f32) -> Option<usize> {
        let (width, height) = window::screen_size();
        let vp = self.compute_vp_matrix();
        let inv_vp = vp.try_inverse().unwrap();

        // Convert to NDC
        let ndc_x = (screen_x / width) * 2.0 - 1.0;
        let ndc_y = -(screen_y / height) * 2.0 + 1.0;

        // Unproject far point on the far clipping plane
        let far_clip = inv_vp * Vector4::new(ndc_x, ndc_y, 1.0, 1.0);
        let far_point = Vector3::new(far_clip.x, far_clip.y, far_clip.z) / far_clip.w;

        let ray_origin = self.camera.position().coords;
        let ray_direction = (far_point - ray_origin).normalize();

        // Brute-force: find point with minimum distance to ray
        let mut best_dist = f32::INFINITY;
        let mut best_idx = None;

        for (i, p) in self.cloud.points().iter().enumerate() {
            let point = Vector3::new(p.x(), p.y(), p.z());
            let v = point - ray_origin;
            let t = v.dot(&ray_direction);
            let projection = ray_origin + ray_direction * t;
            let dist = (point - projection).norm();
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(i);
            }
        }

        best_idx
    }
}

impl EventHandler for Viewer {
    fn update(&mut self) {
        // No per-frame updates needed - all navigation is event-driven (mouse/scroll)
    }

    fn draw(&mut self) {
        if !self.initialized {
            self.init();
        }

        let pipeline = self.pipeline.as_ref().expect("Pipeline not initialized");

        // Begin default pass (clear to dark background)
        self.ctx.as_mut().unwrap().begin_default_pass(miniquad::PassAction::clear_color(
            0.08, 0.08, 0.1, 1.0,
        ));

        // Apply pipeline
        self.ctx.as_mut().unwrap().apply_pipeline(pipeline);

        // Apply bindings and draw
        let vertex_buffer = self.vertex_buffer.unwrap();
        let index_buffer = self.index_buffer.unwrap();

        self.ctx.as_mut().unwrap().apply_bindings_from_slice(
            &[vertex_buffer],
            index_buffer,
            &[],
        );

        // Draw all points with default color
        self.apply_uniforms(DEFAULT_COLOR, self.point_size);
        self.ctx.as_mut().unwrap().draw(0, self.point_count as i32, 1);

        // Draw selected point in red (highlight)
        if let Some(idx) = self.selected_point {
            self.apply_uniforms(SELECTED_COLOR, SELECTED_POINT_SIZE);
            self.ctx.as_mut().unwrap().draw(idx as i32, 1, 1);
        }

        // Draw axis frame at origin (if enabled)
        if self.show_origin {
            if let Some(axis_pipeline) = self.axis_pipeline.as_ref() {
                let axis_buf = self.axis_vertex_buffer.unwrap();
                let axis_idx_buf = self.axis_index_buffer.unwrap();
                self.ctx.as_mut().unwrap().apply_pipeline(axis_pipeline);
                self.ctx.as_mut().unwrap().apply_bindings_from_slice(
                    &[axis_buf],
                    axis_idx_buf,
                    &[],
                );
                self.apply_axis_uniforms();
                self.ctx.as_mut().unwrap().draw(0, 6, 1);
            }
        }

        // End render pass
        self.ctx.as_mut().unwrap().end_render_pass();

        // ── egui HUD overlay ─────────────────────────────────────────────────
        // We move egui_mq out of self so we can freely borrow `self` inside
        // the closure (e.g. to call rebuild_vertex_buffer).
        let mut egui_opt = self.egui_mq.take();
        if let Some(egui_mq) = egui_opt.as_mut() {
            let nav_mode   = self.nav_mode;
            let point_size = self.point_size;
            let mut new_nav_mode   = nav_mode;
            let mut new_point_size = point_size;
            
            // Pre-calculate cloud bounds for display
            let (cloud_min, cloud_max) = cloud_bounds(&self.cloud);
            let extent_x = cloud_max[0] - cloud_min[0];
            let extent_y = cloud_max[1] - cloud_min[1];
            let extent_z = cloud_max[2] - cloud_min[2];
            let camera_dist = self.camera.distance;

            egui_mq.run(self.ctx.as_mut().unwrap().as_mut(), |_ctx, egui_ctx| {
                egui_ctx.set_style({
                    let mut style = (*egui_ctx.style()).clone();
                    style.text_styles.insert(
                        egui::TextStyle::Button,
                        egui::FontId::proportional(14.0),  // Reduced from 22
                    );
                    style.text_styles.insert(
                        egui::TextStyle::Body,
                        egui::FontId::proportional(12.0),  // Smaller body text
                    );
                    style.spacing.button_padding = egui::vec2(8.0, 4.0);  // Reduced from 16, 10
                    style.spacing.item_spacing   = egui::vec2(4.0, 2.0);  // Reduced from 8, 6
                    style
                });

                let (_w, _h) = window::screen_size();
                let selected_info = self.selected_point_info;
                
                // ─── TOP-LEFT: Point Cloud Measurements ───
                egui::Window::new("measurements")
                    .title_bar(false)
                    .resizable(false)
                    .collapsible(false)
                    .anchor(egui::Align2::LEFT_TOP, egui::vec2(12.0, 12.0))
                    .show(egui_ctx, |ui| {
                        ui.style_mut().text_styles.insert(
                            egui::TextStyle::Body,
                            egui::FontId::monospace(11.0),
                        );
                        ui.label(format!("Points: {}", self.cloud.len()));
                        ui.label(format!("X: {:.2} to {:.2} (Δ{:.2})", cloud_min[0], cloud_max[0], extent_x));
                        ui.label(format!("Y: {:.2} to {:.2} (Δ{:.2})", cloud_min[1], cloud_max[1], extent_y));
                        ui.label(format!("Z: {:.2} to {:.2} (Δ{:.2})", cloud_min[2], cloud_max[2], extent_z));
                        ui.separator();
                        ui.label(format!("Camera dist: {:.1}", camera_dist));
                        ui.separator();
                        ui.colored_label(
                            egui::Color32::from_rgb(100, 200, 255),
                            "CloudCompare Controls:"
                        );
                        ui.label("L-drag = Rotate");
                        ui.label("M-drag = Pan");
                        ui.label("Shift+L = Pan");
                        ui.label("R-drag = Zoom");
                        ui.label("Scroll = Zoom");
                        
                        // Show selected point info if available
                        if let Some((idx, x, y, z)) = selected_info {
                            ui.separator();
                            ui.colored_label(
                                egui::Color32::from_rgb(255, 100, 100),
                                format!("Point [{}]:", idx)
                            );
                            ui.label(format!("  x: {:.4}", x));
                            ui.label(format!("  y: {:.4}", y));
                            ui.label(format!("  z: {:.4}", z));
                        }
                    });

                // ─── BOTTOM-CENTER: Point Size Control HUD ───
                egui::Window::new("hud")
                    .title_bar(false)
                    .resizable(false)
                    .collapsible(false)
                    .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -8.0))
                    .fixed_size(egui::vec2(400.0, if selected_info.is_some() { 80.0 } else { 40.0 }))
                    .show(egui_ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Point Size:");
                            if ui.button("+").clicked() {
                                new_point_size = (point_size * 1.25).min(5.0);
                            }
                            ui.label(format!("{:.1}", point_size));
                            if ui.button("-").clicked() {
                                new_point_size = (point_size / 1.25).max(1.0);
                            }
                        });
                        
                        if let Some((idx, x, y, z)) = selected_info {
                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.label(format!("Point [{}]: x={:.4}, y={:.4}, z={:.4}", idx, x, y, z));
                            });
                        }
                    });
            });

            egui_mq.draw(self.ctx.as_mut().unwrap().as_mut());

            // Apply HUD changes now that egui_mq is no longer borrowing self
            if new_nav_mode != nav_mode {
                self.nav_mode = new_nav_mode;
                println!("[pcdviewr] Nav mode: {:?}", self.nav_mode);
            }
            if (new_point_size - point_size).abs() > 0.001 {
                self.point_size = new_point_size;
                println!("Point size: {:.1}", self.point_size);
                self.rebuild_vertex_buffer();
            }
        }
        self.egui_mq = egui_opt; // put it back

        self.ctx.as_mut().unwrap().commit_frame();
    }

    fn mouse_button_down_event(
        &mut self,
        button: miniquad::MouseButton,
        x: f32,
        y: f32,
    ) {
        // Forward to egui first
        if let Some(e) = self.egui_mq.as_mut() { e.mouse_button_down_event(button, x, y); }

        // Block 3D interaction when egui is handling the pointer
        if self.egui_mq.as_ref().map(|e| e.egui_ctx().is_pointer_over_area()).unwrap_or(false) {
            return;
        }

        // Ctrl+click: pick nearest point
        if self.ctrl_held && button == miniquad::MouseButton::Left {
            if let Some(idx) = self.pick_point(x, y) {
                self.selected_point = Some(idx);
                let p = &self.cloud.points()[idx];
                let px = p.x();
                let py = p.y();
                let pz = p.z();
                self.selected_point_info = Some((idx, px, py, pz));
                println!("Selected point {}: ({:.4}, {:.4}, {:.4})", idx, px, py, pz);
            }
            return;
        }

        // CloudCompare navigation:
        // Left drag = Rotate, Middle drag OR Shift+Left = Pan, Right drag = Zoom
        self.mouse_button_down = Some(button);
        self.mouse_down = true;
        self.last_mouse = (x, y);
    }

    fn mouse_button_up_event(
        &mut self,
        button: miniquad::MouseButton,
        x: f32,
        y: f32,
    ) {
        if let Some(e) = self.egui_mq.as_mut() { e.mouse_button_up_event(button, x, y); }
        self.mouse_button_down = None;
        self.mouse_down = false;
    }

    fn mouse_motion_event(&mut self, x: f32, y: f32) {
        if let Some(e) = self.egui_mq.as_mut() { e.mouse_motion_event(x, y); }

        if self.egui_mq.as_ref().map(|e| e.egui_ctx().is_pointer_over_area()).unwrap_or(false) {
            return;
        }

        if self.mouse_down {
            let dx = x - self.last_mouse.0;
            let dy = y - self.last_mouse.1;
            
            // CloudCompare navigation model:
            // Left drag = Rotate around center (orbit)
            // Middle drag OR Shift+Left = Pan
            // Right drag = Zoom
            match self.mouse_button_down {
                Some(miniquad::MouseButton::Left) => {
                    if self.shift_held {
                        // Shift+Left drag = Pan
                        self.camera.pan_mouse(dx, dy);
                    } else {
                        // Left drag = Orbit
                        self.camera.orbit(dx, dy);
                    }
                }
                Some(miniquad::MouseButton::Middle) => {
                    // Middle drag = Pan
                    self.camera.pan_mouse(dx, dy);
                }
                Some(miniquad::MouseButton::Right) => {
                    // Right drag = Zoom (vertical movement = zoom direction)
                    self.camera.zoom(dy * 0.1);
                }
                _ => {}
            }
            self.last_mouse = (x, y);
        }
    }

    fn mouse_wheel_event(&mut self, _x: f32, y: f32) {
        self.camera.zoom(y);
    }

    fn key_down_event(&mut self, key: KeyCode, mods: KeyMods, _repeat: bool) {
        if let Some(e) = self.egui_mq.as_mut() { e.key_down_event(key, mods); }
        
        // Track Shift key for pan mode
        if mods.shift {
            self.shift_held = true;
        }
        
        // Handle keys
        match key {
            KeyCode::Escape => {
                window::quit();
            }
            KeyCode::LeftControl | KeyCode::RightControl => {
                self.ctrl_held = true;
            }
            KeyCode::Equal | KeyCode::KpAdd => {
                self.point_size = (self.point_size * 1.25).min(5.0);
                println!("Point size: {:.1}", self.point_size);
                self.rebuild_vertex_buffer();
            }
            KeyCode::Minus | KeyCode::KpSubtract => {
                self.point_size = (self.point_size / 1.25).max(1.0);
                println!("Point size: {:.1}", self.point_size);
                self.rebuild_vertex_buffer();
            }
            _ => {}
        }
    }

    fn key_up_event(&mut self, key: KeyCode, _mods: KeyMods) {
        // Track movement keys
        match key {
            KeyCode::LeftControl | KeyCode::RightControl => {
                self.ctrl_held = false;
            }
            KeyCode::LeftShift | KeyCode::RightShift => {
                self.shift_held = false;
            }
            _ => {}
        }
    }

    fn char_event(&mut self, character: char, _mods: KeyMods, _repeat: bool) {
        if let Some(e) = self.egui_mq.as_mut() { e.char_event(character); }
    }

    /// Touch/tablet gesture handler.
    ///
    /// **Orbit mode** (default):
    ///   - 1 finger drag       → orbit around center
    ///   - 2 finger pinch      → zoom (damped)
    ///   - 2 finger drag       → pan center
    ///   - 3 finger tap        → switch to Fly mode
    ///
    /// **Fly mode** (press `F` or 3-finger tap to enter):
    ///   - 1 finger drag       → look around (yaw / pitch in place)
    ///   - 2 finger drag up    → fly forward into cloud
    ///   - 2 finger drag down  → fly backward
    ///   - 2 finger drag left  → strafe left
    ///   - 2 finger drag right → strafe right
    ///   - 3 finger tap        → switch back to Orbit mode
    fn touch_event(&mut self, phase: TouchPhase, id: u64, x: f32, y: f32) {
        match phase {
            TouchPhase::Started => {
                self.touch_points.insert(id, (x, y));
                // 3-finger tap toggles navigation mode
                if self.touch_points.len() == 3 {
                    self.nav_mode = match self.nav_mode {
                        NavMode::Orbit => {
                            println!("[pcdviewr] Nav mode: Fly  (1-finger=look, 2-finger=move, 3-finger=back to Orbit)");
                            NavMode::Fly
                        }
                        NavMode::Fly => {
                            println!("[pcdviewr] Nav mode: Orbit  (1-finger=orbit, 2-finger=pinch/pan, 3-finger=Fly)");
                            NavMode::Orbit
                        }
                    };
                }
            }
            TouchPhase::Moved => {
                // Snapshot old positions before updating
                let old = self.touch_points.clone();
                self.touch_points.insert(id, (x, y));

                let count = self.touch_points.len();

                match self.nav_mode {
                    NavMode::Orbit => {
                        if count == 1 {
                            if let Some(&(old_x, old_y)) = old.get(&id) {
                                self.camera.orbit(x - old_x, y - old_y);
                            }
                        } else if count == 2 {
                            let ids: Vec<u64> = self.touch_points.keys().copied().collect();
                            let (id_a, id_b) = (ids[0], ids[1]);

                            let (ax, ay) = self.touch_points[&id_a];
                            let (bx, by) = self.touch_points[&id_b];
                            let (old_ax, old_ay) = old.get(&id_a).copied().unwrap_or((ax, ay));
                            let (old_bx, old_by) = old.get(&id_b).copied().unwrap_or((bx, by));

                            // Damped pinch zoom
                            let old_dist = ((old_ax - old_bx).powi(2) + (old_ay - old_by).powi(2)).sqrt();
                            let new_dist = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();
                            if old_dist > 1.0 {
                                self.camera.zoom_pinch(new_dist / old_dist);
                            }

                            // Pan via midpoint delta
                            let dmx = (ax + bx) / 2.0 - (old_ax + old_bx) / 2.0;
                            let dmy = (ay + by) / 2.0 - (old_ay + old_by) / 2.0;
                            if dmx.abs() > 0.5 || dmy.abs() > 0.5 {
                                self.camera.pan_mouse(dmx, dmy);
                            }
                        }
                    }
                    NavMode::Fly => {
                        if count == 1 {
                            // Look around: just change yaw/pitch without moving the center
                            if let Some(&(old_x, old_y)) = old.get(&id) {
                                self.camera.orbit(x - old_x, y - old_y);
                            }
                        } else if count == 2 {
                            // Use midpoint delta for fly movement
                            let ids: Vec<u64> = self.touch_points.keys().copied().collect();
                            let (id_a, id_b) = (ids[0], ids[1]);

                            let (ax, ay) = self.touch_points[&id_a];
                            let (bx, by) = self.touch_points[&id_b];
                            let (old_ax, old_ay) = old.get(&id_a).copied().unwrap_or((ax, ay));
                            let (old_bx, old_by) = old.get(&id_b).copied().unwrap_or((bx, by));

                            let dmx = (ax + bx) / 2.0 - (old_ax + old_bx) / 2.0;
                            let dmy = (ay + by) / 2.0 - (old_ay + old_by) / 2.0;

                            // Vertical midpoint delta → fly forward/back (screen Y down = forward)
                            if dmy.abs() >= dmx.abs() {
                                self.camera.fly_forward(dmy);
                            } else {
                                // Horizontal midpoint delta → strafe
                                self.camera.strafe(dmx);
                            }
                        }
                    }
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.touch_points.remove(&id);
            }
        }
    }
}

/// Run the viewer with the given point cloud.
pub fn run(cloud: PointCloud, show_origin: bool) {
    println!("=== pcdviewr (CloudCompare-style navigation) ===");
    println!("Initial point_size: 1.0 (use +/- to adjust)");
    if show_origin {
        println!("Origin axis frame: enabled (X=red, Y=green, Z=blue)");
    }
    println!("");
    println!("NAVIGATION (CloudCompare model):");
    println!("  • Left mouse drag: Rotate around center point (ORBIT)");
    println!("  • Middle mouse drag: Pan the view");
    println!("  • Shift + Left drag: Pan the view (alternative)");
    println!("  • Right mouse drag: Zoom in/out");
    println!("  • Scroll wheel: Zoom in/out");
    println!("");
    println!("OTHER:");
    println!("  • Ctrl+Click: Select point and display coordinates");
    println!("  • +/- keys: Adjust point size");
    println!("  • Esc: Quit");
    println!("");
    println!("Info panel on left shows cloud bounds and selected point coords.");
    let viewer = Viewer::new(cloud, show_origin);

    miniquad::start(
        conf::Conf {
            window_title: "pcdviewr v6 (CloudCompare Nav)".to_string(),
            window_width: 1280,
            window_height: 720,
            ..Default::default()
        },
        move || Box::new(viewer),
    );
}

/// Compute min/max bounding box of a point cloud.
fn cloud_bounds(cloud: &PointCloud) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    for p in cloud.points() {
        min[0] = min[0].min(p.x());
        min[1] = min[1].min(p.y());
        min[2] = min[2].min(p.z());
        max[0] = max[0].max(p.x());
        max[1] = max[1].max(p.y());
        max[2] = max[2].max(p.z());
    }

    (min, max)
}

/// Perspective projection matrix (column-major).
fn perspective(fov_degrees: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let fov = fov_degrees.to_radians();
    let f = 1.0 / (fov.sin() / (fov.cos()));
    let range = 1.0 / (near - far);

    [
        f / aspect, 0.0, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 0.0, (near + far) * range, -1.0, 0.0,
        0.0, 2.0 * near * far * range, 0.0,
    ]
}

/// Multiply two 4x4 matrices (column-major).
fn multiply(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut result = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            result[col * 4 + row] = sum;
        }
    }
    result
}