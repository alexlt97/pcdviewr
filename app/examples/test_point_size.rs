//! Test gl_PointSize with uniform-based approach.
//!
//! Draws 3 points using separate draw calls, each with a different `point_size` uniform.
//! This tests whether the GPU respects gl_PointSize at all.
//!
//! Run with: `cargo run --package app --example test_point_size`

use miniquad::{
    conf, BufferId, BufferLayout, BufferSource, BufferType, BufferUsage, EventHandler,
    GlContext, PassAction, Pipeline, PipelineParams, PrimitiveType, RenderingBackend,
    ShaderMeta, ShaderSource, UniformBlockLayout, UniformDesc, UniformType, VertexAttribute,
    VertexFormat,
};
use miniquad::graphics::raw_gl::{glEnable, GL_PROGRAM_POINT_SIZE};

// Shader with point_size uniform
const VERTEX_SHADER: &str = r#"
#version 100
attribute vec4 position;
uniform float point_size;
void main() {
    gl_Position = position;
    gl_PointSize = point_size;
}
"#;

const FRAGMENT_SHADER: &str = r#"
#version 100
precision mediump float;
void main() {
    gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"#;

struct TestApp {
    ctx: Option<GlContext>,
    pipeline: Option<Pipeline>,
    // 3 separate vertex buffers, one per point
    buffers: [Option<BufferId>; 3],
    // Dummy index buffer for binding
    dummy_index: Option<BufferId>,
}

impl TestApp {
    fn init(&mut self) {
        self.ctx = Some(GlContext::new());
        let ctx = self.ctx.as_mut().unwrap();

        // Enable GL_PROGRAM_POINT_SIZE so vertex shader can control gl_PointSize
        unsafe {
            glEnable(GL_PROGRAM_POINT_SIZE);
        }
        println!("Enabled GL_PROGRAM_POINT_SIZE");

        // Each point gets its own vertex buffer (single vertex, 4 floats)
        let points: [(f32, f32, f32); 3] = [
            (-0.5, 0.0, 0.0),  // left
            (0.0, 0.0, 0.0),   // center
            (0.5, 0.0, 0.0),   // right
        ];

        for (i, &(x, y, z)) in points.iter().enumerate() {
            let verts: [f32; 4] = [x, y, z, 1.0];
            self.buffers[i] = Some(ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&verts),
            ));
        }

        // Dummy index buffer (single index)
        let dummy_index: [u32; 1] = [0];
        self.dummy_index = Some(ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&dummy_index),
        ));

        // Shader with point_size uniform
        let shader_meta = ShaderMeta {
            images: vec![],
            uniforms: UniformBlockLayout {
                uniforms: vec![UniformDesc::new("point_size", UniformType::Float1)],
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
    }

    fn set_point_size(&mut self, size: f32) {
        let ctx = self.ctx.as_mut().unwrap();
        let size_bytes = size.to_ne_bytes();
        ctx.apply_uniforms_from_bytes(size_bytes.as_ptr() as *const u8, 4);
        println!("  Set point_size uniform = {}", size);
    }
}

impl EventHandler for TestApp {
    fn update(&mut self) {}

    fn draw(&mut self) {
        if self.ctx.is_none() {
            self.init();
            println!("\n=== Drawing 3 points with uniform-based sizes ===");
        }

        let pipeline = self.pipeline.as_ref().expect("Pipeline not initialized");
        let ctx = self.ctx.as_mut().unwrap();

        ctx.begin_default_pass(PassAction::clear_color(0.1, 0.1, 0.15, 1.0));
        ctx.apply_pipeline(pipeline);

        let dummy_index = self.dummy_index.unwrap();
        let b0 = self.buffers[0].unwrap();
        let b1 = self.buffers[1].unwrap();
        let b2 = self.buffers[2].unwrap();

        // Draw left point (5px)
        ctx.apply_bindings_from_slice(&[b0], dummy_index, &[]);
        ctx.apply_uniforms_from_bytes((5.0f32).to_ne_bytes().as_ptr() as *const u8, 4);
        ctx.draw(0, 1, 1);

        // Draw center point (20px)
        ctx.apply_bindings_from_slice(&[b1], dummy_index, &[]);
        ctx.apply_uniforms_from_bytes((20.0f32).to_ne_bytes().as_ptr() as *const u8, 4);
        ctx.draw(0, 1, 1);

        // Draw right point (50px)
        ctx.apply_bindings_from_slice(&[b2], dummy_index, &[]);
        ctx.apply_uniforms_from_bytes((50.0f32).to_ne_bytes().as_ptr() as *const u8, 4);
        ctx.draw(0, 1, 1);

        ctx.end_render_pass();
    }
}

fn main() {
    println!("=== Point Size Test (uniform-based) ===");
    println!("Expected:");
    println!("  - Left point:  5 pixels (small)");
    println!("  - Center point: 20 pixels (medium)");
    println!("  - Right point: 50 pixels (large)");
    println!("========================================\n");

    miniquad::start(
        conf::Conf {
            window_title: "Point Size Test".to_string(),
            window_width: 640,
            window_height: 480,
            ..Default::default()
        },
        || Box::new(TestApp {
            ctx: None,
            pipeline: None,
            buffers: [None, None, None],
            dummy_index: None,
        }),
    );
}
