use crate::config::SimulationConfig;
use crate::constants::{BACKGROUND_COLOR, MAX_ORGANISMS};
// Use the Organism struct for input type, and the GpuData struct for buffer mapping
use crate::simulation::{Organism, OrganismGpuData};
// Remove the local definition of OrganismGpuData, as it's now imported from simulation.rs
// use bytemuck::{Pod, Zeroable}; // Already imported via simulation.rs potentially, but keep for clarity if needed
use bytemuck::{Pod, Zeroable};
// Keep Pod/Zeroable import here for other structs
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

// --- GPU Data Structures ---

// REMOVED: OrganismGpuData struct definition - now imported from simulation.rs

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GlobalUniforms {
    screen_resolution: [f32; 2],
    iso_level: f32,
    smoothness: f32,
    background_color: [f32; 4],
}

// NEW: Vertex struct for the fullscreen quad
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FullscreenVertex {
    position: [f32; 2], // Corresponds to @location(0) in shader
}

impl FullscreenVertex {
    // Vertex layout description
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<FullscreenVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2], // Matches @location(0) vec2<f32>
        }
    }
}

// --- Renderer ---
pub struct Renderer<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    fullscreen_vertex_buffer: wgpu::Buffer,
    organism_storage_buffer: wgpu::Buffer,
    max_organisms_in_buffer: usize,
    global_uniform_buffer: wgpu::Buffer,
    organism_count_buffer: wgpu::Buffer,
    bind_group_layout_globals: wgpu::BindGroupLayout,
    bind_group_layout_storage: wgpu::BindGroupLayout,
    bind_group_globals: wgpu::BindGroup,
    bind_group_storage: wgpu::BindGroup,
}

// --- Constants for Metaballs (Tunable) ---
const METABALL_ISO_LEVEL: f32 = 0.7;
const METABALL_SMOOTHNESS: f32 = 0.4;
// --- NEW: Visual Scaling Factor ---
// Increase this value (> 1.0) to make metaballs appear larger visually
// Decrease this value (< 1.0) to make metaballs appear smaller visually
// 1.0 means no visual scaling (matches simulation radius)
const VISUAL_RADIUS_MULTIPLIER: f32 = 2.5; // Example: 50% larger visual radius

impl<'a> Renderer<'a> {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let size = PhysicalSize::new(size.width.max(1), size.height.max(1));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
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
                    // Potentially require shader storage buffer support if not default? Check limits if issues arise.
                    required_limits: wgpu::Limits::default(),
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
            present_mode: wgpu::PresentMode::Fifo, // Consider Mailbox for lower latency if vsync is not strictly needed
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader_source = include_str!("shader.wgsl"); // Ensure this points to the correct shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Metaball Shader Module"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // --- Create Buffers ---

        // Fullscreen Quad Vertex Buffer
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
            screen_resolution: [size.width as f32, size.height as f32],
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

        // Organism Count Buffer
        let organism_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Count Buffer"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Organism Storage Buffer
        let initial_max_organisms = MAX_ORGANISMS.max(1024);
        // Use the imported OrganismGpuData struct size
        let storage_buffer_size =
            (initial_max_organisms * std::mem::size_of::<OrganismGpuData>()) as wgpu::BufferAddress;
        let organism_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Storage Buffer"),
            size: storage_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Bind Group Layouts ---
        let bind_group_layout_globals =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Globals Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // Visibility needs to include Fragment if used there
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
                        // Visibility needs to include Fragment if used there
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<OrganismGpuData>() as _ // Use imported struct size
                            ),
                        },
                        count: None,
                    },
                    // Organism Count (Uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        // Visibility needs to include Fragment if used there
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<u32>() as _
                            ),
                        },
                        count: None,
                    },
                ],
            });

        // --- Create Initial Bind Groups ---
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

        // --- Render Pipeline ---
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Metaball Render Pipeline Layout"),
                bind_group_layouts: &[
                    &bind_group_layout_globals, // Group 0
                    &bind_group_layout_storage, // Group 1
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Metaball Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[FullscreenVertex::desc()], // Use the fullscreen quad layout
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING), // Enable alpha blending
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // Draw triangles
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for a fullscreen quad
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None, // No depth buffer needed
            multisample: wgpu::MultisampleState::default(), // No MSAA
            multiview: None,
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
            // REMOVED: organism_gpu_data field
            max_organisms_in_buffer: initial_max_organisms,
            global_uniform_buffer,
            organism_count_buffer,
            bind_group_layout_globals,
            bind_group_layout_storage,
            bind_group_globals,
            bind_group_storage,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        let sim_new_size = PhysicalSize::new(new_size.width.max(1), new_size.height.max(1));
        if sim_new_size.width > 0 && sim_new_size.height > 0 && sim_new_size != self.size {
            self.size = sim_new_size;
            self.config.width = sim_new_size.width;
            self.config.height = sim_new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Update only the screen resolution part of the global uniform buffer
            let screen_res_data = [sim_new_size.width as f32, sim_new_size.height as f32];
            self.queue.write_buffer(
                &self.global_uniform_buffer,
                0, // Offset matches GlobalUniforms.screen_resolution
                bytemuck::cast_slice(&screen_res_data),
            );

            println!(
                "Renderer resized to {}x{}",
                sim_new_size.width, sim_new_size.height
            );
        }
    }

    /// Renders the current state of the organisms using the metaball shader.
    /// Takes a slice of `Organism` from the simulation.
    pub fn render(
        &mut self,
        organisms: &[Organism],     // Still take simulation organisms as input
        _config: &SimulationConfig, // Keep for potential future use
    ) -> Result<(), wgpu::SurfaceError> {
        let output_texture = self.surface.get_current_texture()?;
        let view = output_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Metaball Render Encoder"),
            });

        // --- Prepare GPU Data with Visual Scaling ---
        // Create the Vec<OrganismGpuData> here, applying the visual multiplier
        let gpu_data_for_buffer: Vec<OrganismGpuData> = organisms
            .iter()
            .map(|org| {
                // Apply visual scaling factor to the radius
                let render_radius = org.radius * VISUAL_RADIUS_MULTIPLIER.max(0.01); // Ensure positive radius

                OrganismGpuData {
                    world_position: org.position.into(), // Convert Vec2 to [f32; 2]
                    radius: render_radius,               // Use the visually scaled radius
                    _padding1: 0.0,                      // Explicitly set padding
                    color: org.color.into(),             // Convert Vec4 to [f32; 4]
                }
            })
            .collect();

        let current_organism_count = gpu_data_for_buffer.len();

        // --- Resize Storage Buffer if Needed ---
        if current_organism_count > self.max_organisms_in_buffer {
            // Calculate new size (same logic as before)
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

            // Create new buffer
            self.organism_storage_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Organism Storage Buffer (Resized)"),
                size: new_buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.max_organisms_in_buffer = new_max_organisms;

            // Recreate the bind group to point to the new buffer
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

        // --- Update Buffers ---
        // Write the prepared (and potentially scaled) data to the GPU buffer
        if current_organism_count > 0 {
            self.queue.write_buffer(
                &self.organism_storage_buffer,
                0,
                bytemuck::cast_slice(&gpu_data_for_buffer), // Use the prepared Vec
            );
        }
        // else: If count is 0, we don't need to write to the storage buffer,
        //       the shader will just loop 0 times based on the count uniform.

        // Update the organism count uniform buffer
        self.queue.write_buffer(
            &self.organism_count_buffer,
            0,
            bytemuck::cast_slice(&[current_organism_count as u32]),
        );

        // --- Render Pass ---
        {
            // Scoped borrow of encoder
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Metaball Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(BACKGROUND_COLOR), // Use the constant directly
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);

            // Set the vertex buffer for the fullscreen quad
            render_pass.set_vertex_buffer(0, self.fullscreen_vertex_buffer.slice(..));

            // Bind uniform/storage groups
            render_pass.set_bind_group(0, &self.bind_group_globals, &[]); // Globals like resolution, iso, etc.
            render_pass.set_bind_group(1, &self.bind_group_storage, &[]); // Organism data + count

            // Draw the fullscreen quad (6 vertices, 1 instance)
            render_pass.draw(0..6, 0..1);
        } // render_pass is dropped, releasing borrow of encoder

        // Submit the command encoder's commands to the queue
        self.queue.submit(std::iter::once(encoder.finish()));
        // Present the frame to the surface
        output_texture.present();

        Ok(())
    }
}
