//! The wgpu viewport renderer: offscreen color+depth target, three passes
//! (analytic grid/axes, surface chunks, screen-space thick lines), ported
//! from the QRhi surface renderer.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use caso_meshing::{MeshTileKey, RenderLine};
use caso_surfaces::types::{mesh_tag_color, SurfaceStatus, ViewportSurface, ViewportSurfaceScene};
use web_time::Instant;

use crate::camera::OrbitCamera;

pub const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const LINE_HALF_PX: f32 = 3.0;
const LINE_INSTANCE_BYTES: u64 = 9 * std::mem::size_of::<f32>() as u64;
const LINE_UPLOAD_CHUNK_BYTES: u64 = 16 * 1024 * 1024;
pub const MESH_TILE_UPLOAD_BUDGET_BYTES: u64 = 1024 * 1024;
/// Screen-space radius of point markers (sphere impostors).
const POINT_RADIUS_PX: f32 = 4.0;
/// Opacity at and above which geometry surfaces write depth and the mesh
/// preview overlay (lines/points) starts depth-testing against them, so
/// near-opaque geometry can occlude mesh elements behind it. Below this,
/// mesh preview stays opacity-independent (x-ray inspection mode); see
/// `design_docs/mesh_preview_opacity_independence.md`.
const MESH_PREVIEW_OCCLUDE_OPACITY: f32 = 0.9;
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

struct LineChunk {
    vertex_buffer: wgpu::Buffer,
    instance_count: u32,
}

struct MeshTileBuffer {
    chunks: Vec<LineChunk>,
    line_count: usize,
    bytes: u64,
}

struct PendingMeshTile {
    lines: Arc<[RenderLine]>,
    uploaded: usize,
    chunks: Vec<LineChunk>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MeshTileRenderStats {
    pub generation: u64,
    pub upload_bytes: u64,
    pub upload_ms: f32,
    pub gpu_bytes: u64,
    pub active_lines: usize,
    pub active_tiles: usize,
    pub resident_tiles: usize,
    pub pending_tiles: usize,
}

pub struct ViewportRenderer {
    grid_pipeline: wgpu::RenderPipeline,
    surface_pipeline: wgpu::RenderPipeline,
    surface_blend_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    line_pipeline_depth_test: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    point_pipeline_depth_test: wgpu::RenderPipeline,
    grid_bind_group: wgpu::BindGroup,
    surface_bind_group: wgpu::BindGroup,
    line_bind_group: wgpu::BindGroup,
    grid_uniforms: wgpu::Buffer,
    surface_uniforms: wgpu::Buffer,
    line_uniforms: wgpu::Buffer,
    /// Surfaces from the last `set_scene` (base mesh + selection highlight):
    /// rebuilt only when the document or selection actually changes.
    base_chunks: Vec<SurfaceChunk>,
    /// Surfaces from the last `set_overlays` (boundary highlight ribbons,
    /// create-tool ghost): rebuilt on zoom-bucket/overlay-revision changes
    /// without touching `base_chunks` or `mesh_chunks`, so per-zoom-step
    /// ribbon rebuilds never re-upload the base mesh or the meshing preview.
    overlay_chunks: Vec<SurfaceChunk>,
    /// Surfaces from the last `set_mesh_overlays` (meshing-workspace preview
    /// — every previewed mesh element): rebuilt only on an actual
    /// `mesh_preview_revision` change, never by zoom or boundary-overlay
    /// churn, since this can be the largest surface set in the scene.
    mesh_chunks: Vec<SurfaceChunk>,
    /// Independent thick-line buffers. Zoom-driven tool-overlay refreshes
    /// must not concatenate and re-upload the potentially large mesh preview.
    base_lines: Vec<LineChunk>,
    overlay_lines: Vec<LineChunk>,
    mesh_lines: Vec<LineChunk>,
    mesh_tile_generation: Option<u64>,
    mesh_tile_target: BTreeSet<MeshTileKey>,
    active_mesh_tiles: BTreeSet<MeshTileKey>,
    mesh_tile_buffers: BTreeMap<MeshTileKey, MeshTileBuffer>,
    pending_mesh_tiles: BTreeMap<MeshTileKey, PendingMeshTile>,
    mesh_tile_stats: MeshTileRenderStats,
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

        // One line instance: a(3) b(3) color(3) = 36 bytes. The shader
        // derives the six quad corners from vertex_index.
        let line_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: LINE_INSTANCE_BYTES,
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
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 24,
                    shader_location: 2,
                },
            ],
        };
        let make_line_pipeline = |label: &str, depth: wgpu::DepthStencilState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
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
                depth_stencil: Some(depth),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let line_pipeline = make_line_pipeline("line pipeline", depth_disabled.clone());
        let line_pipeline_depth_test =
            make_line_pipeline("line pipeline (depth test)", depth_test_no_write.clone());

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
        let make_point_pipeline = |label: &str, depth: wgpu::DepthStencilState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
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
                depth_stencil: Some(depth),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let point_pipeline = make_point_pipeline("point pipeline", depth_disabled);
        let point_pipeline_depth_test =
            make_point_pipeline("point pipeline (depth test)", depth_test_no_write);

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
            line_pipeline_depth_test,
            point_pipeline,
            point_pipeline_depth_test,
            grid_bind_group,
            surface_bind_group,
            line_bind_group,
            grid_uniforms,
            surface_uniforms,
            line_uniforms,
            base_chunks: Vec::new(),
            overlay_chunks: Vec::new(),
            mesh_chunks: Vec::new(),
            base_lines: Vec::new(),
            overlay_lines: Vec::new(),
            mesh_lines: Vec::new(),
            mesh_tile_generation: None,
            mesh_tile_target: BTreeSet::new(),
            active_mesh_tiles: BTreeSet::new(),
            mesh_tile_buffers: BTreeMap::new(),
            pending_mesh_tiles: BTreeMap::new(),
            mesh_tile_stats: MeshTileRenderStats::default(),
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
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
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

    /// Build GPU chunk buffers plus raw thick-line vertex floats for one set
    /// of surfaces. Shared by the base-scene and overlay upload paths so
    /// rebuilding one side never re-walks or re-uploads the other's.
    fn build_surfaces(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surfaces: &[ViewportSurface],
    ) -> (Vec<SurfaceChunk>, Vec<[f32; 9]>) {
        let mut chunks = Vec::new();
        // Thick-line instances for wire-only surfaces (1D objects, outlines).
        let mut line_instances = Vec::new();
        for surface in surfaces {
            if surface.status == SurfaceStatus::Failed {
                continue;
            }
            // Inspector surfaces intentionally combine colored face fills
            // with element outlines. Regular display surfaces keep their
            // historical fill-only behavior so CAD tessellation stays hidden.
            if surface.indices.is_empty() || surface.object_kind == "mesh_inspector" {
                for pair in surface.wire_indices.chunks_exact(2) {
                    let a = surface.vertices[pair[0] as usize];
                    let b = surface.vertices[pair[1] as usize];
                    push_line_segment(&mut line_instances, a, b, surface.color);
                }
            }
            if surface.indices.is_empty() {
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
            chunks.push(SurfaceChunk {
                vertex_buffer,
                index_buffer,
                index_count: surface.indices.len() as u32,
                blended: alpha < 0.999,
            });
        }
        (chunks, line_instances)
    }

    fn upload_lines(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        instances: &[[f32; 9]],
    ) -> Vec<LineChunk> {
        let instances_per_chunk = line_instances_per_chunk(device.limits().max_buffer_size);
        instances
            .chunks(instances_per_chunk)
            .map(|instances| LineChunk {
                vertex_buffer: upload_buffer(
                    device,
                    queue,
                    label,
                    bytemuck::cast_slice(instances),
                    wgpu::BufferUsages::VERTEX,
                ),
                instance_count: instances.len() as u32,
            })
            .collect()
    }

    /// Upload the base display-surface scene (the actual mesh + selection
    /// highlight) into GPU chunk buffers. Call only when the document or
    /// selection changes — never on zoom or overlay-only updates, see
    /// `set_overlays`.
    pub fn set_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &ViewportSurfaceScene,
    ) {
        let (chunks, line_instances) = Self::build_surfaces(device, queue, &scene.surfaces);
        self.base_chunks = chunks;
        self.base_lines = Self::upload_lines(device, queue, "base line instances", &line_instances);
    }

    /// Upload boundary-tool overlay surfaces (highlight ribbons, create-tool
    /// ghost) into their own GPU chunk buffers, independent of both the base
    /// mesh and the meshing preview. This is the path zoom-bucket changes
    /// should call, so their cost scales only with overlay geometry — never
    /// with the base mesh or the (potentially much larger) mesh preview.
    pub fn set_overlays(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surfaces: &[ViewportSurface],
    ) {
        let (chunks, line_instances) = Self::build_surfaces(device, queue, surfaces);
        self.overlay_chunks = chunks;
        self.overlay_lines =
            Self::upload_lines(device, queue, "overlay line instances", &line_instances);
    }

    /// Upload the meshing-workspace preview surfaces (every previewed mesh
    /// element) into their own GPU chunk buffers. Call only on an actual
    /// `mesh_preview_revision` change — never on zoom or boundary-overlay
    /// churn, since this is typically the largest surface set in the scene.
    pub fn set_mesh_overlays(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surfaces: &[ViewportSurface],
    ) {
        let (chunks, line_instances) = Self::build_surfaces(device, queue, surfaces);
        self.mesh_chunks = chunks;
        self.mesh_lines = Self::upload_lines(device, queue, "mesh line instances", &line_instances);
    }

    /// Set the newest camera-dependent tile target. Existing active buffers
    /// remain visible until every target tile is fully uploaded.
    pub fn set_mesh_tile_target(
        &mut self,
        generation: u64,
        keys: impl IntoIterator<Item = MeshTileKey>,
    ) {
        if self
            .mesh_tile_generation
            .is_some_and(|current| generation < current)
        {
            return;
        }
        let target = keys.into_iter().collect::<BTreeSet<_>>();
        if self.mesh_tile_generation == Some(generation) && self.mesh_tile_target == target {
            return;
        }
        self.mesh_tile_generation = Some(generation);
        self.mesh_tile_target = target;
        self.pending_mesh_tiles
            .retain(|key, _| self.mesh_tile_target.contains(key));
        self.try_activate_mesh_tiles();
        self.refresh_mesh_tile_stats(0, 0.0);
    }

    /// Queue one decoded tile for bounded GPU upload. Stale generations and
    /// tiles no longer required by the target are ignored.
    pub fn upsert_mesh_tile(
        &mut self,
        generation: u64,
        key: MeshTileKey,
        lines: Arc<[RenderLine]>,
    ) {
        if self.mesh_tile_generation != Some(generation)
            || !self.mesh_tile_target.contains(&key)
            || self.mesh_tile_buffers.contains_key(&key)
            || self.pending_mesh_tiles.contains_key(&key)
        {
            return;
        }
        if lines.is_empty() {
            self.mesh_tile_buffers.insert(
                key,
                MeshTileBuffer {
                    chunks: Vec::new(),
                    line_count: 0,
                    bytes: 0,
                },
            );
            self.try_activate_mesh_tiles();
        } else {
            self.pending_mesh_tiles.insert(
                key,
                PendingMeshTile {
                    lines,
                    uploaded: 0,
                    chunks: Vec::new(),
                },
            );
        }
        self.refresh_mesh_tile_stats(0, 0.0);
    }

    /// Upload no more than `budget_bytes` of complete 36-byte line instances.
    /// Returns true while either decode delivery or GPU upload is incomplete.
    pub fn upload_pending_mesh_tiles(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        budget_bytes: u64,
    ) -> bool {
        if budget_bytes < LINE_INSTANCE_BYTES {
            self.refresh_mesh_tile_stats(0, 0.0);
            return self.mesh_tile_stats.pending_tiles != 0;
        }
        let started = Instant::now();
        let mut budget = budget_bytes;
        let mut uploaded_bytes = 0;
        while budget >= LINE_INSTANCE_BYTES {
            let Some(key) = self.pending_mesh_tiles.keys().next().copied() else {
                break;
            };
            let pending = self
                .pending_mesh_tiles
                .get_mut(&key)
                .expect("key came from pending map");
            let remaining = pending.lines.len() - pending.uploaded;
            let count =
                mesh_upload_instance_count(budget, device.limits().max_buffer_size, remaining);
            if count == 0 {
                break;
            }
            let start = pending.uploaded;
            let instances = pending.lines[start..start + count]
                .iter()
                .map(line_instance)
                .collect::<Vec<_>>();
            pending.chunks.push(LineChunk {
                vertex_buffer: upload_buffer(
                    device,
                    queue,
                    "mesh tile line instances",
                    bytemuck::cast_slice(&instances),
                    wgpu::BufferUsages::VERTEX,
                ),
                instance_count: count as u32,
            });
            pending.uploaded += count;
            let bytes = count as u64 * LINE_INSTANCE_BYTES;
            budget -= bytes;
            uploaded_bytes += bytes;

            if pending.uploaded == pending.lines.len() {
                let pending = self
                    .pending_mesh_tiles
                    .remove(&key)
                    .expect("completed pending tile");
                self.mesh_tile_buffers.insert(
                    key,
                    MeshTileBuffer {
                        chunks: pending.chunks,
                        line_count: pending.lines.len(),
                        bytes: pending.lines.len() as u64 * LINE_INSTANCE_BYTES,
                    },
                );
                self.try_activate_mesh_tiles();
            }
        }
        let upload_ms = if uploaded_bytes == 0 {
            0.0
        } else {
            started.elapsed().as_secs_f32() * 1_000.0
        };
        self.refresh_mesh_tile_stats(uploaded_bytes, upload_ms);
        self.mesh_tile_stats.pending_tiles != 0
    }

    /// Cancel incomplete work for a moving camera without touching the
    /// complete tile set currently visible on the GPU.
    pub fn defer_mesh_tiles(&mut self) {
        self.mesh_tile_target = self.active_mesh_tiles.clone();
        self.pending_mesh_tiles.clear();
        self.refresh_mesh_tile_stats(0, 0.0);
    }

    /// Atomically activate a fully resident set. Incomplete requests leave
    /// the previous active set untouched.
    pub fn set_active_mesh_tiles(&mut self, keys: impl IntoIterator<Item = MeshTileKey>) -> bool {
        let keys = keys.into_iter().collect::<BTreeSet<_>>();
        if keys
            .iter()
            .any(|key| !self.mesh_tile_buffers.contains_key(key))
        {
            return false;
        }
        self.active_mesh_tiles = keys;
        self.mesh_tile_buffers
            .retain(|key, _| self.active_mesh_tiles.contains(key));
        self.refresh_mesh_tile_stats(
            self.mesh_tile_stats.upload_bytes,
            self.mesh_tile_stats.upload_ms,
        );
        true
    }

    /// Evict only unprotected tiles; active, target, and in-flight tiles
    /// cannot be removed.
    pub fn evict_mesh_tiles(&mut self, keys: impl IntoIterator<Item = MeshTileKey>) {
        for key in keys {
            if !self.active_mesh_tiles.contains(&key)
                && !self.mesh_tile_target.contains(&key)
                && !self.pending_mesh_tiles.contains_key(&key)
            {
                self.mesh_tile_buffers.remove(&key);
            }
        }
        self.refresh_mesh_tile_stats(
            self.mesh_tile_stats.upload_bytes,
            self.mesh_tile_stats.upload_ms,
        );
    }

    pub fn clear_mesh_tiles(&mut self) {
        self.mesh_tile_generation = None;
        self.mesh_tile_target.clear();
        self.active_mesh_tiles.clear();
        self.mesh_tile_buffers.clear();
        self.pending_mesh_tiles.clear();
        self.mesh_tile_stats = MeshTileRenderStats::default();
    }

    pub fn mesh_tile_stats(&self) -> MeshTileRenderStats {
        self.mesh_tile_stats
    }

    pub fn active_mesh_tiles(&self) -> &BTreeSet<MeshTileKey> {
        &self.active_mesh_tiles
    }

    fn try_activate_mesh_tiles(&mut self) {
        if self
            .mesh_tile_target
            .iter()
            .all(|key| self.mesh_tile_buffers.contains_key(key))
        {
            let target = self.mesh_tile_target.clone();
            self.set_active_mesh_tiles(target);
        }
    }

    fn refresh_mesh_tile_stats(&mut self, upload_bytes: u64, upload_ms: f32) {
        self.mesh_tile_stats = MeshTileRenderStats {
            generation: self.mesh_tile_generation.unwrap_or(0),
            upload_bytes,
            upload_ms,
            gpu_bytes: self.mesh_tile_buffers.values().map(|tile| tile.bytes).sum(),
            active_lines: self
                .active_mesh_tiles
                .iter()
                .filter_map(|key| self.mesh_tile_buffers.get(key))
                .map(|tile| tile.line_count)
                .sum(),
            active_tiles: self.active_mesh_tiles.len(),
            resident_tiles: self.mesh_tile_buffers.len(),
            pending_tiles: self
                .mesh_tile_target
                .iter()
                .filter(|key| !self.mesh_tile_buffers.contains_key(key))
                .count(),
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
        queue.write_buffer(
            &self.surface_uniforms,
            0,
            bytemuck::cast_slice(&surface_data),
        );

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
        // Line vertices are uploaded only when their respective scene,
        // tool-overlay, or mesh-preview set changes.

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
            // At/above MESH_PREVIEW_OCCLUDE_OPACITY, surfaces write depth so
            // the mesh preview overlay (below) can be occluded by geometry.
            let occlude_mesh_preview = options.opacity >= MESH_PREVIEW_OCCLUDE_OPACITY;
            let surface_pipeline = if occlude_mesh_preview {
                &self.surface_pipeline
            } else {
                &self.surface_blend_pipeline
            };
            pass.set_pipeline(surface_pipeline);
            pass.set_bind_group(0, &self.surface_bind_group, &[]);
            let all_chunks = || {
                self.base_chunks
                    .iter()
                    .chain(self.overlay_chunks.iter())
                    .chain(self.mesh_chunks.iter())
            };
            for chunk in all_chunks().filter(|chunk| !chunk.blended) {
                pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                pass.set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..chunk.index_count, 0, 0..1);
            }
            if all_chunks().any(|chunk| chunk.blended) {
                pass.set_pipeline(&self.surface_blend_pipeline);
                for chunk in all_chunks().filter(|chunk| chunk.blended) {
                    pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                    pass.set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..chunk.index_count, 0, 0..1);
                }
            }

            // 3. Thick lines (wire chunks + overlays). Depth-tested against
            // geometry once opacity crosses MESH_PREVIEW_OCCLUDE_OPACITY;
            // otherwise always on top (x-ray mesh inspection).
            if !self.base_lines.is_empty()
                || !self.overlay_lines.is_empty()
                || !self.mesh_lines.is_empty()
                || !self.active_mesh_tiles.is_empty()
            {
                let line_pipeline = if occlude_mesh_preview {
                    &self.line_pipeline_depth_test
                } else {
                    &self.line_pipeline
                };
                pass.set_pipeline(line_pipeline);
                pass.set_bind_group(0, &self.line_bind_group, &[]);
                for lines in self
                    .base_lines
                    .iter()
                    .chain(self.overlay_lines.iter())
                    .chain(self.mesh_lines.iter())
                {
                    pass.set_vertex_buffer(0, lines.vertex_buffer.slice(..));
                    pass.draw(0..6, 0..lines.instance_count);
                }
                for tile in self
                    .active_mesh_tiles
                    .iter()
                    .filter_map(|key| self.mesh_tile_buffers.get(key))
                {
                    for lines in &tile.chunks {
                        pass.set_vertex_buffer(0, lines.vertex_buffer.slice(..));
                        pass.draw(0..6, 0..lines.instance_count);
                    }
                }
            }

            // 4. Point markers (instanced sphere impostors). Same depth-test
            // cutover as thick lines above.
            if let Some(point_buffer) = &self.point_buffer {
                let point_pipeline = if occlude_mesh_preview {
                    &self.point_pipeline_depth_test
                } else {
                    &self.point_pipeline
                };
                pass.set_pipeline(point_pipeline);
                pass.set_bind_group(0, &self.line_bind_group, &[]);
                pass.set_vertex_buffer(0, point_buffer.slice(..));
                pass.draw(0..4, 0..self.point_count);
            }
        }
        queue.submit([encoder.finish()]);
    }
}

fn line_instances_per_chunk(max_buffer_size: u64) -> usize {
    let bytes = max_buffer_size.min(LINE_UPLOAD_CHUNK_BYTES);
    assert!(
        bytes >= LINE_INSTANCE_BYTES,
        "wgpu device cannot hold one line instance"
    );
    (bytes / LINE_INSTANCE_BYTES) as usize
}

fn mesh_upload_instance_count(budget: u64, max_buffer_size: u64, remaining: usize) -> usize {
    (budget.min(max_buffer_size) / LINE_INSTANCE_BYTES)
        .min(u64::from(u32::MAX))
        .min(remaining as u64) as usize
}

fn line_instance(line: &RenderLine) -> [f32; 9] {
    let color = mesh_tag_color((line.color_id % 60_000).max(1) as u32);
    [
        line.a[0], line.a[1], line.a[2], line.b[0], line.b[1], line.b[2], color[0], color[1],
        color[2],
    ]
}

/// Append one thick-line instance (a -> b); the shader expands its quad.
fn push_line_segment(out: &mut Vec<[f32; 9]>, a: [f32; 3], b: [f32; 3], color: [f32; 3]) {
    out.push([
        a[0], a[1], a[2], b[0], b[1], b[2], color[0], color[1], color[2],
    ]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use caso_meshing::MeshTileDetail;

    #[cfg(not(target_arch = "wasm32"))]
    static HEADLESS_GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn tile(node_id: u64) -> MeshTileKey {
        MeshTileKey {
            node_id,
            detail: MeshTileDetail::Preview,
        }
    }

    fn render_line(edge_id: u64) -> RenderLine {
        RenderLine {
            edge_id,
            a: [0.0, 0.0, 0.0],
            b: [1.0, 0.0, 0.0],
            color_id: 1,
            opacity: 1.0,
            highlighted: false,
            selected: false,
        }
    }

    #[test]
    fn line_chunks_are_whole_instances_within_the_upload_limit() {
        assert_eq!(MESH_TILE_UPLOAD_BUDGET_BYTES, 1024 * 1024);
        assert_eq!(line_instances_per_chunk(100), 2);
        let count = line_instances_per_chunk(u64::MAX);
        assert_eq!(count as u64 * LINE_INSTANCE_BYTES, 16_777_188);
        assert!((count as u64 + 1) * LINE_INSTANCE_BYTES > LINE_UPLOAD_CHUNK_BYTES);
        let mesh_count =
            mesh_upload_instance_count(MESH_TILE_UPLOAD_BUDGET_BYTES, u64::MAX, usize::MAX);
        assert!(mesh_count as u64 * LINE_INSTANCE_BYTES <= MESH_TILE_UPLOAD_BUDGET_BYTES);
        assert!((mesh_count as u64 + 1) * LINE_INSTANCE_BYTES > MESH_TILE_UPLOAD_BUDGET_BYTES);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn renderer_pipelines_compile_headlessly() {
        let _guard = HEADLESS_GPU.lock().expect("headless GPU test lock");
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            return;
        };
        let (device, _queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("headless wgpu device");
        let _renderer = ViewportRenderer::new(&device);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn mesh_tiles_upload_incrementally_and_activate_atomically() {
        let _guard = HEADLESS_GPU.lock().expect("headless GPU test lock");
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            return;
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("headless wgpu device");
        let mut renderer = ViewportRenderer::new(&device);
        let parent = tile(1);
        let children = [tile(2), tile(3)];

        renderer.set_mesh_tile_target(1, [parent]);
        renderer.upsert_mesh_tile(1, parent, Arc::from([render_line(1)]));
        assert!(renderer.upload_pending_mesh_tiles(&device, &queue, 0));
        assert_eq!(renderer.mesh_tile_stats().upload_bytes, 0);
        assert_eq!(renderer.mesh_tile_stats().upload_ms, 0.0);
        renderer.upload_pending_mesh_tiles(&device, &queue, MESH_TILE_UPLOAD_BUDGET_BYTES);
        assert_eq!(renderer.active_mesh_tiles(), &BTreeSet::from([parent]));

        let budget_lines = (MESH_TILE_UPLOAD_BUDGET_BYTES / LINE_INSTANCE_BYTES) as usize;
        renderer.set_mesh_tile_target(2, children);
        renderer.upsert_mesh_tile(2, children[0], Arc::from([render_line(2)]));
        renderer.upsert_mesh_tile(
            2,
            children[1],
            vec![render_line(3); budget_lines + 10].into(),
        );
        renderer.defer_mesh_tiles();
        assert_eq!(renderer.active_mesh_tiles(), &BTreeSet::from([parent]));
        assert_eq!(renderer.mesh_tile_stats().upload_bytes, 0);
        assert_eq!(renderer.mesh_tile_stats().pending_tiles, 0);

        renderer.set_mesh_tile_target(3, children);
        renderer.upsert_mesh_tile(3, children[0], Arc::from([render_line(2)]));
        renderer.upsert_mesh_tile(
            3,
            children[1],
            vec![render_line(3); budget_lines + 10].into(),
        );

        let mut upload_frames = 0;
        loop {
            upload_frames += 1;
            let pending =
                renderer.upload_pending_mesh_tiles(&device, &queue, MESH_TILE_UPLOAD_BUDGET_BYTES);
            assert!(renderer.mesh_tile_stats().upload_bytes <= MESH_TILE_UPLOAD_BUDGET_BYTES);
            if !pending {
                break;
            }
            assert_eq!(renderer.active_mesh_tiles(), &BTreeSet::from([parent]));
            if upload_frames == 1 {
                renderer.evict_mesh_tiles([parent, children[0]]);
                assert_eq!(renderer.mesh_tile_stats().resident_tiles, 2);
            }
        }
        assert!(upload_frames >= 2);
        assert_eq!(
            renderer.active_mesh_tiles(),
            &children.into_iter().collect()
        );
        assert_eq!(renderer.mesh_tile_stats().active_lines, budget_lines + 11);
        let retained_bytes = renderer.mesh_tile_stats().gpu_bytes;
        renderer.set_mesh_tile_target(4, children);
        renderer.upsert_mesh_tile(4, children[0], Arc::from([render_line(99)]));
        assert_eq!(renderer.mesh_tile_stats().gpu_bytes, retained_bytes);
        assert_eq!(renderer.mesh_tile_stats().resident_tiles, 2);

        // Reversal cannot activate stale children or leave a hole while the
        // evicted parent is uploaded again.
        renderer.set_mesh_tile_target(5, [parent]);
        renderer.upsert_mesh_tile(4, children[0], Arc::from([render_line(4)]));
        assert_eq!(
            renderer.active_mesh_tiles(),
            &children.into_iter().collect()
        );
        renderer.upsert_mesh_tile(5, parent, Arc::from([render_line(1)]));
        renderer.upload_pending_mesh_tiles(&device, &queue, MESH_TILE_UPLOAD_BUDGET_BYTES);
        assert_eq!(renderer.active_mesh_tiles(), &BTreeSet::from([parent]));
        renderer.evict_mesh_tiles([parent]);
        assert_eq!(renderer.mesh_tile_stats().resident_tiles, 1);
    }
}
