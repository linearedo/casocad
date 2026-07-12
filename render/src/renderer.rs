//! The wgpu viewport renderer: offscreen color+depth target, three passes
//! (analytic grid/axes, surface chunks, screen-space thick lines), ported
//! from the QRhi surface renderer.

use caso_surfaces::types::{SurfaceStatus, ViewportSurfaceScene};

use crate::camera::OrbitCamera;

pub const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const LINE_HALF_PX: f32 = 3.0;
/// Screen-space radius of point markers (sphere impostors).
const POINT_RADIUS_PX: f32 = 4.0;
/// Default viewport background (#241f32).
pub const DEFAULT_BACKGROUND: [f32; 3] = [0.141, 0.122, 0.196];

pub struct RenderOptions {
    pub background: [f32; 3],
    pub show_grid: bool,
    pub grid_spacing: f32,
    /// 0 = XY, 1 = XZ, 2 = YZ (matches the grid shader).
    pub grid_plane: u32,
    pub opacity: f32,
    pub wireframe: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            background: DEFAULT_BACKGROUND,
            show_grid: true,
            grid_spacing: 1.0,
            grid_plane: 0,
            // Semi-transparent so interior features (e.g. the default
            // scene's cylinder obstacle) are visible on first launch.
            opacity: 0.35,
            wireframe: false,
        }
    }
}

struct SurfaceChunk {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    /// Surface alpha < 1 (ghost previews): drawn after the opaque chunks
    /// with the blend pipeline, regardless of the global opacity slider.
    blended: bool,
}

pub struct ViewportRenderer {
    grid_pipeline: wgpu::RenderPipeline,
    surface_pipeline: wgpu::RenderPipeline,
    surface_blend_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    grid_bind_group: wgpu::BindGroup,
    surface_bind_group: wgpu::BindGroup,
    line_bind_group: wgpu::BindGroup,
    grid_uniforms: wgpu::Buffer,
    surface_uniforms: wgpu::Buffer,
    line_uniforms: wgpu::Buffer,
    chunks: Vec<SurfaceChunk>,
    line_buffer: Option<wgpu::Buffer>,
    line_vertex_count: u32,
    point_buffer: Option<wgpu::Buffer>,
    point_count: u32,
    color_texture: Option<wgpu::Texture>,
    depth_texture: Option<wgpu::Texture>,
    size: (u32, u32),
}

/// Create a GPU buffer and upload via the queue (avoids mappedAtCreation,
/// which some WebGPU implementations cap at small sizes).
fn upload_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let size = (contents.len().max(4) as u64).div_ceil(4) * 4;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, contents);
    buffer
}

fn uniform_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("caso uniform layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

impl ViewportRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = uniform_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("caso pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grid_axes"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/grid_axes.wgsl").into()),
        });
        let surface_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("surface"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/surface.wgsl").into()),
        });
        let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/line.wgsl").into()),
        });
        let point_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point_marker"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/point_marker.wgsl").into()),
        });

        let color_target = [Some(wgpu::ColorTargetState {
            format: TARGET_FORMAT,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let depth_disabled = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let depth_enabled = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let depth_test_no_write = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        let grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("grid pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &grid_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &grid_shader,
                entry_point: Some("fs_main"),
                targets: &color_target,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(depth_disabled.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Interleaved surface vertex: pos(3) normal(3) color(4) = 40 bytes.
        let surface_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 40,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 24,
                    shader_location: 2,
                },
            ],
        };
        let make_surface_pipeline = |label: &str, depth: wgpu::DepthStencilState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &surface_shader,
                    entry_point: Some("vs_main"),
                    buffers: std::slice::from_ref(&surface_vertex_layout),
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &surface_shader,
                    entry_point: Some("fs_main"),
                    targets: &color_target,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(depth),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let surface_pipeline = make_surface_pipeline("surface opaque", depth_enabled);
        let surface_blend_pipeline =
            make_surface_pipeline("surface transparent", depth_test_no_write.clone());

        // Line vertex: a(3) b(3) color(3) param(2) = 44 bytes.
        let line_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 44,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 24,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 36,
                    shader_location: 3,
                },
            ],
        };
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: Some("vs_main"),
                buffers: std::slice::from_ref(&line_vertex_layout),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: Some("fs_main"),
                targets: &color_target,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(depth_disabled.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Point instance: pos(3) color(3) = 24 bytes, one quad per instance.
        let point_instance_layout = wgpu::VertexBufferLayout {
            array_stride: 24,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
            ],
        };
        let point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &point_shader,
                entry_point: Some("vs_main"),
                buffers: std::slice::from_ref(&point_instance_layout),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &point_shader,
                entry_point: Some("fs_main"),
                targets: &color_target,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: Some(depth_disabled),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let grid_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid ubo"),
            size: 96,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let surface_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface ubo"),
            size: 80,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let line_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("line ubo"),
            size: 80,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let make_bind_group = |buffer: &wgpu::Buffer, label: &str| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        };
        let grid_bind_group = make_bind_group(&grid_uniforms, "grid bg");
        let surface_bind_group = make_bind_group(&surface_uniforms, "surface bg");
        let line_bind_group = make_bind_group(&line_uniforms, "line bg");

        Self {
            grid_pipeline,
            surface_pipeline,
            surface_blend_pipeline,
            line_pipeline,
            point_pipeline,
            grid_bind_group,
            surface_bind_group,
            line_bind_group,
            grid_uniforms,
            surface_uniforms,
            line_uniforms,
            chunks: Vec::new(),
            line_buffer: None,
            line_vertex_count: 0,
            point_buffer: None,
            point_count: 0,
            color_texture: None,
            depth_texture: None,
            size: (0, 0),
        }
    }

    /// (Re)create the offscreen color/depth target; returns the color view.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let width = width.max(1);
        let height = height.max(1);
        if self.size != (width, height) || self.color_texture.is_none() {
            self.size = (width, height);
            self.color_texture = Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewport color"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TARGET_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }));
            self.depth_texture = Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewport depth"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }));
        }
        self.color_texture
            .as_ref()
            .expect("created above")
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Upload the display-surface scene into GPU chunk buffers.
    pub fn set_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &ViewportSurfaceScene,
    ) {
        self.chunks.clear();
        // Thick-line vertex data for wire-only surfaces (1D objects,
        // outlines), concatenated into one static buffer per scene.
        let mut line_vertices: Vec<f32> = Vec::new();
        for surface in &scene.surfaces {
            if surface.status == SurfaceStatus::Failed {
                continue;
            }
            if surface.indices.is_empty() {
                for pair in surface.wire_indices.chunks_exact(2) {
                    let a = surface.vertices[pair[0] as usize];
                    let b = surface.vertices[pair[1] as usize];
                    push_line_segment(&mut line_vertices, a, b, surface.color);
                }
                continue;
            }
            let alpha = surface.alpha.clamp(0.0, 1.0);
            let mut interleaved: Vec<f32> = Vec::with_capacity(surface.vertices.len() * 10);
            for (vertex, normal) in surface.vertices.iter().zip(surface.normals.iter()) {
                interleaved.extend_from_slice(vertex);
                interleaved.extend_from_slice(normal);
                interleaved.extend_from_slice(&surface.color);
                interleaved.push(alpha);
            }
            let vertex_buffer = upload_buffer(
                device,
                queue,
                "chunk vertices",
                bytemuck::cast_slice(&interleaved),
                wgpu::BufferUsages::VERTEX,
            );
            let index_buffer = upload_buffer(
                device,
                queue,
                "chunk indices",
                bytemuck::cast_slice(&surface.indices),
                wgpu::BufferUsages::INDEX,
            );
            self.chunks.push(SurfaceChunk {
                vertex_buffer,
                index_buffer,
                index_count: surface.indices.len() as u32,
                blended: alpha < 0.999,
            });
        }
        self.line_vertex_count = (line_vertices.len() / 11) as u32;
        self.line_buffer = if line_vertices.is_empty() {
            None
        } else {
            Some(upload_buffer(
                device,
                queue,
                "line vertices",
                bytemuck::cast_slice(&line_vertices),
                wgpu::BufferUsages::VERTEX,
            ))
        };
    }

    /// Replace the point-marker instances (6 floats per point: xyz + rgb),
    /// drawn as constant-pixel-size sphere impostors.
    pub fn set_points(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[f32]) {
        self.point_count = (data.len() / 6) as u32;
        self.point_buffer = if data.is_empty() {
            None
        } else {
            Some(upload_buffer(
                device,
                queue,
                "point instances",
                bytemuck::cast_slice(data),
                wgpu::BufferUsages::VERTEX,
            ))
        };
    }

    /// Render one frame into the offscreen target.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &OrbitCamera,
        options: &RenderOptions,
    ) {
        let (width, height) = self.size;
        if width == 0 || height == 0 {
            return;
        }
        let basis = camera.basis();
        let max_ray = (camera.distance * 100.0).max(100.0) as f32;

        // Grid uniforms (96 bytes; layout matches grid_axes.wgsl).
        let grid_data: [f32; 24] = [
            basis.position.x as f32,
            basis.position.y as f32,
            basis.position.z as f32,
            camera.focal as f32,
            camera.target.x as f32,
            camera.target.y as f32,
            camera.target.z as f32,
            max_ray,
            basis.right.x as f32,
            basis.right.y as f32,
            basis.right.z as f32,
            options.grid_spacing,
            basis.up.x as f32,
            basis.up.y as f32,
            basis.up.z as f32,
            if options.show_grid { 1.0 } else { 0.0 },
            options.background[0],
            options.background[1],
            options.background[2],
            options.grid_plane as f32,
            width as f32,
            height as f32,
            0.0,
            0.0,
        ];
        queue.write_buffer(&self.grid_uniforms, 0, bytemuck::cast_slice(&grid_data));

        // Surface uniforms: column-major mvp + opacity.
        let matrix = camera.matrix(width, height);
        let mut surface_data = [0.0f32; 20];
        surface_data[..16].copy_from_slice(&matrix);
        surface_data[16] = options.opacity;
        queue.write_buffer(&self.surface_uniforms, 0, bytemuck::cast_slice(&surface_data));

        // Line uniforms (80 bytes; layout matches line.wgsl; clip_y_sign -1).
        let line_data: [f32; 20] = [
            basis.position.x as f32,
            basis.position.y as f32,
            basis.position.z as f32,
            camera.focal as f32,
            basis.right.x as f32,
            basis.right.y as f32,
            basis.right.z as f32,
            height as f32 / (width as f32).max(1.0),
            basis.up.x as f32,
            basis.up.y as f32,
            basis.up.z as f32,
            LINE_HALF_PX,
            camera.target.x as f32,
            camera.target.y as f32,
            camera.target.z as f32,
            -1.0,
            width as f32,
            height as f32,
            POINT_RADIUS_PX,
            0.0,
        ];
        queue.write_buffer(&self.line_uniforms, 0, bytemuck::cast_slice(&line_data));
        // Line vertices are uploaded once in set_scene; if per-frame overlay
        // lines are ever needed, give them their own small buffer here.

        let color_view = self
            .color_texture
            .as_ref()
            .expect("resize before render")
            .create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = self
            .depth_texture
            .as_ref()
            .expect("resize before render")
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("viewport encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: options.background[0] as f64,
                            g: options.background[1] as f64,
                            b: options.background[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                multiview_mask: None,
                timestamp_writes: None,
            });

            // 1. Analytic grid + axes (fullscreen, no depth).
            pass.set_pipeline(&self.grid_pipeline);
            pass.set_bind_group(0, &self.grid_bind_group, &[]);
            pass.draw(0..3, 0..1);

            // 2. Surface chunks (opaque or blended per the opacity slider),
            // then per-surface translucent chunks (ghost previews) on top.
            let surface_pipeline = if options.opacity >= 0.999 {
                &self.surface_pipeline
            } else {
                &self.surface_blend_pipeline
            };
            pass.set_pipeline(surface_pipeline);
            pass.set_bind_group(0, &self.surface_bind_group, &[]);
            for chunk in self.chunks.iter().filter(|chunk| !chunk.blended) {
                pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                pass.set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..chunk.index_count, 0, 0..1);
            }
            if self.chunks.iter().any(|chunk| chunk.blended) {
                pass.set_pipeline(&self.surface_blend_pipeline);
                for chunk in self.chunks.iter().filter(|chunk| chunk.blended) {
                    pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                    pass.set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..chunk.index_count, 0, 0..1);
                }
            }

            // 3. Thick lines (wire chunks + overlays, no depth).
            if let Some(line_buffer) = &self.line_buffer {
                pass.set_pipeline(&self.line_pipeline);
                pass.set_bind_group(0, &self.line_bind_group, &[]);
                pass.set_vertex_buffer(0, line_buffer.slice(..));
                pass.draw(0..self.line_vertex_count, 0..1);
            }

            // 4. Point markers (instanced sphere impostors, no depth).
            if let Some(point_buffer) = &self.point_buffer {
                pass.set_pipeline(&self.point_pipeline);
                pass.set_bind_group(0, &self.line_bind_group, &[]);
                pass.set_vertex_buffer(0, point_buffer.slice(..));
                pass.draw(0..4, 0..self.point_count);
            }
        }
        queue.submit([encoder.finish()]);
    }
}

/// Append the six vertices of one thick-line segment (a -> b).
fn push_line_segment(out: &mut Vec<f32>, a: [f32; 3], b: [f32; 3], color: [f32; 3]) {
    // (endpoint_sel, side) for the two triangles of the segment quad.
    const PARAMS: [[f32; 2]; 6] = [
        [0.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ];
    for param in PARAMS {
        out.extend_from_slice(&a);
        out.extend_from_slice(&b);
        out.extend_from_slice(&color);
        out.extend_from_slice(&param);
    }
}
