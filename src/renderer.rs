// --- File: renderer.rs ---
use crate::config::SimulationConfig;
// REMOVED unused import: RENDER_RESOLUTION_SCALE
use crate::constants::{BACKGROUND_COLOR, MAX_ORGANISMS};
// Import SimulationState and Organism (Organism needed for iteration type)
use crate::simulation::{Organism, OrganismGpuData, SimulationState};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

// --- GPU Data Structures ---

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GlobalUniforms {
    render_resolution: [f32; 2], // Now likely full window resolution
    iso_level: f32,
    smoothness: f32,
    background_color: [f32; 4],
    // NEW: Grid parameters added
    grid_dims: [i32; 2],
    grid_cell_size: f32,
    _padding_grid: f32, // Add padding for alignment (vec2<i32> = 8 bytes, f32=4, f32=4 -> 16 bytes total)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FullscreenVertex {
    position: [f32; 2],
}

impl FullscreenVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<FullscreenVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2],
        }
    }
}

// --- Renderer ---
pub struct Renderer<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>, // Window size

    // Metaball rendering resources
    render_pipeline: wgpu::RenderPipeline,
    organism_storage_buffer: wgpu::Buffer,
    // Removed: max_organisms_in_buffer: usize, // Capacity is now implicitly MAX_ORGANISMS (+ headroom)
    global_uniform_buffer: wgpu::Buffer,
    organism_count_buffer: wgpu::Buffer,
    bind_group_layout_globals: wgpu::BindGroupLayout,
    bind_group_layout_storage: wgpu::BindGroupLayout,
    bind_group_globals: wgpu::BindGroup,
    bind_group_storage: wgpu::BindGroup, // Needs recreation if grid_offsets_buffer resizes

    // Fullscreen quad resources
    fullscreen_vertex_buffer: wgpu::Buffer,

    // --- Grid rendering resources ---
    grid_offsets_buffer: wgpu::Buffer,
    grid_indices_buffer: wgpu::Buffer,
    // Removed: max_grid_indices_in_buffer: usize, // Capacity is now related to MAX_ORGANISMS
    // --- End Grid ---
}

// --- Constants for Metaballs (Tunable) ---
const METABALL_ISO_LEVEL: f32 = 0.95;
const METABALL_SMOOTHNESS: f32 = 0.95;
const VISUAL_RADIUS_MULTIPLIER: f32 = 1.2;

// --- NEW: Constants for Buffer Allocation ---
// Add some headroom beyond MAX_ORGANISMS for safety/flexibility
const ORGANISM_BUFFER_HEADROOM_FACTOR: f32 = 1.1;
// Estimate max indices needed per organism (e.g., if grid is dense). Adjust if needed.
const GRID_INDICES_HEADROOM_PER_ORGANISM: usize = 4;

impl<'a> Renderer<'a> {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let size = PhysicalSize::new(size.width.max(1), size.height.max(1));

        let render_width = size.width;
        let render_height = size.height;
        log::info!("Window/Render size: {}x{}", size.width, size.height);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find adapter");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // --- Create Shaders ---
        let shader_source = include_str!("shader.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Grid Metaball Shader Module"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // --- Create Buffers ---
        let fullscreen_vertices = [
            FullscreenVertex {
                position: [-1.0, -1.0],
            },
            FullscreenVertex {
                position: [1.0, -1.0],
            },
            FullscreenVertex {
                position: [-1.0, 1.0],
            },
            FullscreenVertex {
                position: [-1.0, 1.0],
            },
            FullscreenVertex {
                position: [1.0, -1.0],
            },
            FullscreenVertex {
                position: [1.0, 1.0],
            },
        ];
        let fullscreen_vertex_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Fullscreen Quad Vertex Buffer"),
                contents: bytemuck::cast_slice(&fullscreen_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        // Global Uniform Buffer
        let global_uniforms = GlobalUniforms {
            render_resolution: [render_width as f32, render_height as f32],
            iso_level: METABALL_ISO_LEVEL,
            smoothness: METABALL_SMOOTHNESS,
            background_color: [
                BACKGROUND_COLOR.r as f32,
                BACKGROUND_COLOR.g as f32,
                BACKGROUND_COLOR.b as f32,
                BACKGROUND_COLOR.a as f32,
            ],
            grid_dims: [0, 0],
            grid_cell_size: 0.0,
            _padding_grid: 0.0,
        };
        let global_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Uniform Buffer"),
            contents: bytemuck::cast_slice(&[global_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Organism Count Buffer
        let organism_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Count Buffer"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Pre-allocate Organism Storage Buffer based on MAX_ORGANISMS ---
        let max_organisms_capacity =
            ((MAX_ORGANISMS as f32 * ORGANISM_BUFFER_HEADROOM_FACTOR) as usize).max(16);
        let organism_storage_buffer_size = (max_organisms_capacity
            * std::mem::size_of::<OrganismGpuData>())
            as wgpu::BufferAddress;
        log::info!(
            "Pre-allocating Organism Storage Buffer for {} organisms ({} bytes)",
            max_organisms_capacity,
            organism_storage_buffer_size
        );
        let organism_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Storage Buffer (Pre-allocated)"),
            size: organism_storage_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Pre-allocate Grid Indices Buffer ---
        let max_grid_indices_capacity =
            (max_organisms_capacity * GRID_INDICES_HEADROOM_PER_ORGANISM).max(256);
        let grid_indices_buffer_size =
            (max_grid_indices_capacity * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        log::info!(
            "Pre-allocating Grid Indices Buffer for {} indices ({} bytes)",
            max_grid_indices_capacity,
            grid_indices_buffer_size
        );
        let grid_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid Indices Buffer (Pre-allocated)"),
            size: grid_indices_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Grid Offsets Buffer (Still dynamically sized) ---
        // Start with a minimal buffer, will be resized in render() if needed based on window size
        let initial_grid_cells = 1;
        let grid_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid Offsets Buffer (Initial)"),
            size: (initial_grid_cells * std::mem::size_of::<[u32; 2]>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Bind Group Layouts (Unchanged) ---
        let bind_group_layout_globals =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Globals Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    // GlobalUniforms
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<GlobalUniforms>() as _,
                        ),
                    },
                    count: None,
                }],
            });

        let bind_group_layout_storage = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Storage Bind Group Layout"),
            entries: &[
                // Organism Data
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<OrganismGpuData>() as _),
                    }, count: None,
                },
                // Organism Count
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<u32>() as _),
                    }, count: None,
                },
                // Grid Offsets
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<[u32; 2]>() as _),
                    }, count: None,
                },
                // Grid Indices
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<u32>() as _),
                    }, count: None,
                },
            ],
        });

        // --- Create Bind Groups (Using pre-allocated buffers) ---
        let bind_group_globals = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Globals Bind Group"),
            layout: &bind_group_layout_globals,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: global_uniform_buffer.as_entire_binding(),
            }],
        });

        // Create initial storage bind group - might be recreated later if grid_offsets resizes
        let bind_group_storage = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Storage Bind Group (Initial Pre-allocated)"),
            layout: &bind_group_layout_storage,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: organism_storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: organism_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grid_indices_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Create Render Pipeline (Unchanged) ---
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout_globals, &bind_group_layout_storage],
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            fullscreen_vertex_buffer,
            organism_storage_buffer,
            // Removed max_organisms_in_buffer field
            global_uniform_buffer,
            organism_count_buffer,
            bind_group_layout_globals,
            bind_group_layout_storage,
            bind_group_globals,
            bind_group_storage, // Store the initial one
            grid_offsets_buffer,
            grid_indices_buffer,
            // Removed max_grid_indices_in_buffer field
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        let win_new_size = PhysicalSize::new(new_size.width.max(1), new_size.height.max(1));
        if win_new_size.width > 0 && win_new_size.height > 0 && win_new_size != self.size {
            self.size = win_new_size;
            self.config.width = win_new_size.width;
            self.config.height = win_new_size.height;
            self.surface.configure(&self.device, &self.config);

            let screen_res_data = [self.size.width as f32, self.size.height as f32];
            self.queue.write_buffer(
                &self.global_uniform_buffer,
                0,
                bytemuck::cast_slice(&screen_res_data),
            );

            // NOTE: grid_offsets_buffer will be resized dynamically within render()
            // if the required size changes due to the new window dimensions affecting grid layout.
            // This is expected and less frequent than organism-count-based resizing.

            log::info!(
                "Renderer resized window to {}x{}",
                win_new_size.width,
                win_new_size.height
            );
        }
    }

    pub fn render(
        &mut self,
        simulation_state: &SimulationState,
        _config: &SimulationConfig,
    ) -> Result<(), wgpu::SurfaceError> {
        let output_surface_texture = self.surface.get_current_texture()?;
        let output_surface_view = output_surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let organisms: &[Organism] = &simulation_state.organisms;
        let current_organism_count = organisms.len();

        // --- Prepare Organism GPU Data ---
        let gpu_data_for_buffer: Vec<OrganismGpuData> = organisms
            .iter()
            .map(|org| {
                let render_radius = org.radius * VISUAL_RADIUS_MULTIPLIER.max(0.01);
                OrganismGpuData {
                    world_position: org.position.into(),
                    radius: render_radius,
                    _padding1: 0.0,
                    color: org.color.into(),
                }
            })
            .collect();

        // --- Assert that organism count doesn't exceed buffer capacity ---
        // This should ideally never fail if MAX_ORGANISMS is respected.
        debug_assert!(
            (current_organism_count as u64 * std::mem::size_of::<OrganismGpuData>() as u64)
                <= self.organism_storage_buffer.size(),
            "Organism count ({}) exceeds pre-allocated buffer size ({})!",
            current_organism_count,
            self.organism_storage_buffer.size()
        );

        // --- Prepare Grid GPU Data ---
        let gpu_grid_indices = simulation_state.get_gpu_grid_indices();
        let gpu_grid_offsets = simulation_state.get_gpu_grid_offsets();
        let (grid_width, grid_height) = simulation_state.get_grid_dimensions();
        let grid_cell_size = simulation_state.get_grid_cell_size();

        // --- Assert that grid index count doesn't exceed buffer capacity ---
        let required_indices_bytes = gpu_grid_indices.len() * std::mem::size_of::<u32>();
        debug_assert!(
            required_indices_bytes as u64 <= self.grid_indices_buffer.size(),
            "Required grid indices ({}) exceed pre-allocated buffer size ({})!",
            gpu_grid_indices.len(),
            self.grid_indices_buffer.size()
        );

        // --- Flag to track if *storage bind group* needs recreating (only for grid_offsets now) ---
        let mut needs_bind_group_recreation = false;

        // --- REMOVED: Resize Organism Storage Buffer ---
        // --- REMOVED: Resize Grid Indices Buffer ---

        // --- Resize Grid Offsets Buffer if Needed (Based on grid dimensions / window size) ---
        let required_offsets_bytes =
            (gpu_grid_offsets.len() * std::mem::size_of::<[u32; 2]>()) as wgpu::BufferAddress;

        if required_offsets_bytes > self.grid_offsets_buffer.size()
            || (required_offsets_bytes == 0 && self.grid_offsets_buffer.size() != 0)
        {
            // Recalculate required size (ensure non-zero)
            let new_buffer_size =
                required_offsets_bytes.max(std::mem::size_of::<[u32; 2]>() as u64);

            log::info!(
                "Resizing grid offsets buffer from {} to {} bytes (grid dims {}x{})",
                self.grid_offsets_buffer.size(),
                new_buffer_size,
                grid_width,
                grid_height
            );

            self.grid_offsets_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Grid Offsets Buffer (Resized)"),
                size: new_buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            needs_bind_group_recreation = true; // Recreate bind group if this buffer changes
        }

        // --- Recreate Storage Bind Group IF grid_offsets_buffer was resized ---
        if needs_bind_group_recreation {
            log::debug!("Recreating storage bind group due to grid offsets buffer resize.");
            self.bind_group_storage = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Storage Bind Group (Recreated)"),
                layout: &self.bind_group_layout_storage,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.organism_storage_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.organism_count_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.grid_offsets_buffer.as_entire_binding(),
                    }, // Use potentially new buffer
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.grid_indices_buffer.as_entire_binding(),
                    },
                ],
            });
        }

        // --- Update Buffers ---
        // Organism Data
        if current_organism_count > 0 {
            self.queue.write_buffer(
                &self.organism_storage_buffer,
                0,
                bytemuck::cast_slice(&gpu_data_for_buffer),
            );
        }
        // Organism Count
        self.queue.write_buffer(
            &self.organism_count_buffer,
            0,
            bytemuck::cast_slice(&[current_organism_count as u32]),
        );

        // Grid Data
        if !gpu_grid_offsets.is_empty() && self.grid_offsets_buffer.size() > 0 {
            self.queue.write_buffer(
                &self.grid_offsets_buffer,
                0,
                bytemuck::cast_slice(gpu_grid_offsets),
            );
        }
        // Only write indices if count > 0 and buffer has size
        if !gpu_grid_indices.is_empty() && self.grid_indices_buffer.size() > 0 {
            self.queue.write_buffer(
                &self.grid_indices_buffer,
                0,
                bytemuck::cast_slice(gpu_grid_indices),
            );
        }

        // Global Uniforms (Update grid params)
        let grid_uniform_data = GridUniformUpdate {
            grid_dims: [grid_width, grid_height],
            grid_cell_size,
        };
        let grid_dims_offset = std::mem::size_of::<[f32; 2]>() // render_resolution
            + std::mem::size_of::<f32>()     // iso_level
            + std::mem::size_of::<f32>()     // smoothness
            + std::mem::size_of::<[f32; 4]>(); // background_color
        self.queue.write_buffer(
            &self.global_uniform_buffer,
            grid_dims_offset as wgpu::BufferAddress,
            bytemuck::bytes_of(&grid_uniform_data),
        );

        // --- Create Command Encoder ---
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // --- Render Pass ---
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Metaball Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(BACKGROUND_COLOR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.fullscreen_vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.bind_group_globals, &[]);
            render_pass.set_bind_group(1, &self.bind_group_storage, &[]); // Use potentially recreated group
            render_pass.draw(0..6, 0..1);
        }

        // --- Submit and Present ---
        self.queue.submit(std::iter::once(encoder.finish()));
        output_surface_texture.present();

        Ok(())
    }
}

// Helper struct for partial update of global uniforms
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GridUniformUpdate {
    grid_dims: [i32; 2],
    grid_cell_size: f32,
}
// --- End of File: renderer.rs ---
