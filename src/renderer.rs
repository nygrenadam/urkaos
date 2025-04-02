// --- File: renderer.rs ---
use crate::config::SimulationConfig;
// Import the new constant
use crate::constants::{BACKGROUND_COLOR, MAX_ORGANISMS, RENDER_RESOLUTION_SCALE};
// Use the Organism struct for input type, and the GpuData struct for buffer mapping
use crate::simulation::{Organism, OrganismGpuData};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

// --- GPU Data Structures ---

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GlobalUniforms {
    // This resolution will now be the *render target* resolution
    render_resolution: [f32; 2],
    iso_level: f32,
    smoothness: f32,
    background_color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FullscreenVertex {
    position: [f32; 2], // Corresponds to @location(0) in shader
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
    metaball_pipeline: wgpu::RenderPipeline,
    organism_storage_buffer: wgpu::Buffer,
    max_organisms_in_buffer: usize,
    global_uniform_buffer: wgpu::Buffer,
    organism_count_buffer: wgpu::Buffer,
    bind_group_layout_globals: wgpu::BindGroupLayout,
    bind_group_layout_storage: wgpu::BindGroupLayout,
    bind_group_globals: wgpu::BindGroup,
    bind_group_storage: wgpu::BindGroup,

    // Fullscreen quad resources (used by both passes)
    fullscreen_vertex_buffer: wgpu::Buffer,

    // --- NEW: Render-to-texture resources ---
    render_texture: wgpu::Texture,
    render_texture_view: wgpu::TextureView,
    upscale_sampler: wgpu::Sampler,
    upscale_pipeline: wgpu::RenderPipeline,
    upscale_bind_group_layout: wgpu::BindGroupLayout,
    upscale_bind_group: wgpu::BindGroup,
    render_width: u32,
    render_height: u32,
    // --- End NEW ---
}

// --- Constants for Metaballs (Tunable) ---
const METABALL_ISO_LEVEL: f32 = 0.9;
const METABALL_SMOOTHNESS: f32 = 0.5;
const VISUAL_RADIUS_MULTIPLIER: f32 = 1.5;

impl<'a> Renderer<'a> {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let size = PhysicalSize::new(size.width.max(1), size.height.max(1));

        // Calculate initial render target size
        let render_width = ((size.width as f32 * RENDER_RESOLUTION_SCALE).round() as u32).max(1);
        let render_height = ((size.height as f32 * RENDER_RESOLUTION_SCALE).round() as u32).max(1);
        log::info!(
            "Window size: {}x{}, Render target size: {}x{}",
            size.width,
            size.height,
            render_width,
            render_height
        );

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
            .expect("Failed to find an appropriate adapter");

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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, // Surface is final target
            format: surface_format,
            width: size.width, // Config uses window size
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // --- Create Shaders ---
        let metaball_shader_source = include_str!("shader.wgsl");
        let metaball_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Metaball Shader Module"),
            source: wgpu::ShaderSource::Wgsl(metaball_shader_source.into()),
        });

        // --- NEW: Upscale Shader ---
        let upscale_shader_source = include_str!("upscale.wgsl");
        let upscale_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Upscale Shader Module"),
            source: wgpu::ShaderSource::Wgsl(upscale_shader_source.into()),
        });
        // --- End NEW ---

        // --- Create Buffers ---
        // FIX 2: Restore vertex data
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

        // Global Uniform Buffer (Metaballs) - Uses RENDER resolution
        let global_uniforms = GlobalUniforms {
            render_resolution: [render_width as f32, render_height as f32], // Use render target size
            iso_level: METABALL_ISO_LEVEL,
            smoothness: METABALL_SMOOTHNESS,
            background_color: [
                BACKGROUND_COLOR.r as f32,
                BACKGROUND_COLOR.g as f32,
                BACKGROUND_COLOR.b as f32,
                BACKGROUND_COLOR.a as f32,
            ],
        };
        let global_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Uniform Buffer"),
            contents: bytemuck::cast_slice(&[global_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Organism Count Buffer (Metaballs) - No changes needed here
        let organism_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Count Buffer"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Organism Storage Buffer (Metaballs) - No changes needed here
        let initial_max_organisms = MAX_ORGANISMS.max(1024);
        let storage_buffer_size =
            (initial_max_organisms * std::mem::size_of::<OrganismGpuData>()) as wgpu::BufferAddress;
        let organism_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Storage Buffer"),
            size: storage_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Bind Group Layouts ---
        // Metaball Layouts
        let bind_group_layout_globals =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Globals Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
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
        let bind_group_layout_storage =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Storage Bind Group Layout"),
                entries: &[
                    // Organism Data (Storage Buffer)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<OrganismGpuData>() as _,
                            ),
                        },
                        count: None,
                    },
                    // Organism Count (Uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<u32>() as _),
                        },
                        count: None,
                    },
                ],
            });

        // --- NEW: Upscale Bind Group Layout ---
        let upscale_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Upscale Bind Group Layout"),
                entries: &[
                    // Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Texture View
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
        // --- End NEW ---

        // --- Create Intermediate Texture & Sampler ---
        let (render_texture, render_texture_view, upscale_sampler) =
            Self::create_render_target(&device, render_width, render_height, config.format);

        // --- Create Bind Groups ---
        // Metaball Bind Groups
        let bind_group_globals = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Globals Bind Group"),
            layout: &bind_group_layout_globals,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: global_uniform_buffer.as_entire_binding(),
            }],
        });
        let bind_group_storage = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Storage Bind Group (Initial)"),
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
            ],
        });

        // --- NEW: Upscale Bind Group ---
        let upscale_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Upscale Bind Group"),
            layout: &upscale_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&upscale_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&render_texture_view),
                },
            ],
        });
        // --- End NEW ---

        // --- Create Pipelines ---
        // Metaball Pipeline (uses metaball shader)
        let metaball_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Metaball Render Pipeline Layout"),
                bind_group_layouts: &[
                    &bind_group_layout_globals, // Group 0
                    &bind_group_layout_storage, // Group 1
                ],
                push_constant_ranges: &[],
            });
        let metaball_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Metaball Render Pipeline"),
            layout: Some(&metaball_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &metaball_shader, // Use metaball shader
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &metaball_shader, // Use metaball shader
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    // Target is the intermediate texture format
                    format: config.format, // Assuming same format for simplicity
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(), // TriangleList is default
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // --- NEW: Upscale Pipeline ---
        let upscale_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Upscale Pipeline Layout"),
                bind_group_layouts: &[&upscale_bind_group_layout], // Group 0
                push_constant_ranges: &[],
            });
        let upscale_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Upscale Pipeline"),
            layout: Some(&upscale_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &upscale_shader, // Use upscale shader
                entry_point: Some("vs_main"),
                buffers: &[FullscreenVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &upscale_shader, // Use upscale shader
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    // Target is the final surface format
                    format: config.format,
                    blend: None, // Overwrite the final output
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
        // --- End NEW ---

        Self {
            surface,
            device,
            queue,
            config,
            size,
            metaball_pipeline, // Renamed from render_pipeline
            fullscreen_vertex_buffer,
            organism_storage_buffer,
            max_organisms_in_buffer: initial_max_organisms,
            global_uniform_buffer,
            organism_count_buffer,
            bind_group_layout_globals,
            bind_group_layout_storage,
            bind_group_globals,
            bind_group_storage,
            // NEW fields
            render_texture,
            render_texture_view,
            upscale_sampler,
            upscale_pipeline,
            upscale_bind_group_layout,
            upscale_bind_group,
            render_width,
            render_height,
            // END NEW
        }
    }

    // --- NEW: Helper to create render target resources ---
    fn create_render_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Target Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // Usage: Render Target + Texture Binding (to be read by upscale shader)
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[], // Required empty for now
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Upscale Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear, // Linear filtering for smoother upscale
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest, // No mipmaps
            ..Default::default()
        });
        (texture, view, sampler)
    }
    // --- End NEW ---

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        let win_new_size = PhysicalSize::new(new_size.width.max(1), new_size.height.max(1));
        if win_new_size.width > 0 && win_new_size.height > 0 && win_new_size != self.size {
            self.size = win_new_size; // Update window size tracker
            // Update surface config size
            self.config.width = win_new_size.width;
            self.config.height = win_new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Calculate new render target size
            self.render_width =
                ((win_new_size.width as f32 * RENDER_RESOLUTION_SCALE).round() as u32).max(1);
            self.render_height =
                ((win_new_size.height as f32 * RENDER_RESOLUTION_SCALE).round() as u32).max(1);
            log::info!(
                "Resized window to {}x{}, render target to {}x{}",
                win_new_size.width,
                win_new_size.height,
                self.render_width,
                self.render_height
            );

            // --- Recreate render target texture & view ---
            let (new_texture, new_view, _) = Self::create_render_target(
                &self.device,
                self.render_width,
                self.render_height,
                self.config.format, // Use surface format
            );
            self.render_texture = new_texture;
            self.render_texture_view = new_view;
            // Sampler doesn't need recreation unless filter modes change

            // --- Recreate upscale bind group ---
            self.upscale_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Upscale Bind Group (Resized)"),
                layout: &self.upscale_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(&self.upscale_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        // Use the NEW view
                        resource: wgpu::BindingResource::TextureView(&self.render_texture_view),
                    },
                ],
            });

            // Update global uniform buffer with the new RENDER resolution
            let screen_res_data = [self.render_width as f32, self.render_height as f32];
            self.queue.write_buffer(
                &self.global_uniform_buffer,
                0, // Offset matches GlobalUniforms.render_resolution
                bytemuck::cast_slice(&screen_res_data),
            );

            println!(
                "Renderer resized window to {}x{}, render target to {}x{}",
                win_new_size.width,
                win_new_size.height,
                self.render_width,
                self.render_height
            );
        }
    }

    /// Renders the current state of the organisms using two passes:
    /// 1. Metaballs rendered to an intermediate low-resolution texture.
    /// 2. Intermediate texture upscaled and rendered to the screen.
    pub fn render(
        &mut self,
        organisms: &[Organism],
        _config: &SimulationConfig,
    ) -> Result<(), wgpu::SurfaceError> {
        // --- Get final output texture ---
        let output_surface_texture = self.surface.get_current_texture()?;
        let output_surface_view = output_surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // --- Prepare GPU Data for Metaballs (Same as before) ---
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
        let current_organism_count = gpu_data_for_buffer.len();

        // --- Resize Storage Buffer if Needed ---
        if current_organism_count > self.max_organisms_in_buffer {
            // FIX 1 & 3: Restore actual calculation logic
            let new_max_organisms = (current_organism_count * 3 / 2)
                .max(self.max_organisms_in_buffer + 1)
                .max(16) // Ensure a minimum size
                .min(MAX_ORGANISMS * 2) // Add a reasonable upper limit if desired
                .next_power_of_two();
            let new_buffer_size =
                (new_max_organisms * std::mem::size_of::<OrganismGpuData>()) as wgpu::BufferAddress;

            println!(
                "Resizing organism storage buffer from {} to {} organisms ({} bytes)",
                self.max_organisms_in_buffer, new_max_organisms, new_buffer_size
            );

            self.organism_storage_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Organism Storage Buffer (Resized)"),
                size: new_buffer_size, // Use the calculated u64 size
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.max_organisms_in_buffer = new_max_organisms;

            // Recreate the storage bind group
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
                ],
            });
        }

        // --- Update Buffers (Same as before) ---
        if current_organism_count > 0 {
            self.queue.write_buffer(
                &self.organism_storage_buffer,
                0,
                bytemuck::cast_slice(&gpu_data_for_buffer),
            );
        }
        self.queue.write_buffer(
            &self.organism_count_buffer,
            0,
            bytemuck::cast_slice(&[current_organism_count as u32]),
        );

        // --- Create Command Encoder ---
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder (Metaball + Upscale)"),
            });

        // --- PASS 1: Render Metaballs to Intermediate Texture ---
        {
            let mut metaball_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Metaball Render Pass (to Texture)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    // Target the INTERMEDIATE texture view
                    view: &self.render_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(BACKGROUND_COLOR),
                        store: wgpu::StoreOp::Store, // Store result in the texture
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            metaball_pass.set_pipeline(&self.metaball_pipeline);
            metaball_pass.set_vertex_buffer(0, self.fullscreen_vertex_buffer.slice(..));
            metaball_pass.set_bind_group(0, &self.bind_group_globals, &[]);
            metaball_pass.set_bind_group(1, &self.bind_group_storage, &[]);
            metaball_pass.draw(0..6, 0..1);
        } // Drop metaball_pass

        // --- PASS 2: Render Intermediate Texture to Screen (Upscale) ---
        {
            let mut upscale_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Upscale Render Pass (to Screen)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    // Target the FINAL surface view
                    view: &output_surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Don't need to clear if the quad covers the whole screen
                        load: wgpu::LoadOp::Load, // Or Clear if desired
                        store: wgpu::StoreOp::Store, // Store result in the surface
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            upscale_pass.set_pipeline(&self.upscale_pipeline);
            upscale_pass.set_vertex_buffer(0, self.fullscreen_vertex_buffer.slice(..));
            upscale_pass.set_bind_group(0, &self.upscale_bind_group, &[]); // Bind texture+sampler
            upscale_pass.draw(0..6, 0..1);
        } // Drop upscale_pass

        // --- Submit and Present ---
        self.queue.submit(std::iter::once(encoder.finish()));
        output_surface_texture.present();

        Ok(())
    }
}
// --- End of File: renderer.rs ---