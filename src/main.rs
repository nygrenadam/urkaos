use std::{sync::Arc, time::Instant};
use rand::Rng;
use winit::{
    event::{Event, WindowEvent, ElementState},
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
    dpi::PhysicalSize,
};
use wgpu::util::DeviceExt;
use glam::{Vec2, Vec4};
use bytemuck::{Pod, Zeroable};

// --- Constants ---
const BACKGROUND_COLOR: wgpu::Color = wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
const PLANT_COLOR: Vec4 = Vec4::new(0.1, 0.7, 0.1, 1.0);
const FISH_COLOR: Vec4 = Vec4::new(0.2, 0.3, 0.9, 1.0);

const ORGANISM_RADIUS: f32 = 5.0;
const INITIAL_SPEED_MULTIPLIER: f32 = 1.0;
const MIN_SPEED_MULTIPLIER: f32 = 0.0;
const MAX_SPEED_MULTIPLIER: f32 = 10.0;
const SPEED_ADJUST_FACTOR: f32 = 0.5;
const MOVEMENT_SPEED: f32 = 50.0;
const FIXED_TIMESTEP: f64 = 1.0 / 60.0;
const MIN_LIFETIME: f32 = 15.0;
const MAX_LIFETIME: f32 = 45.0;
const GROWTH_RATE: f32 = 1.0;
const SPAWN_OFFSET_RADIUS: f32 = (ORGANISM_RADIUS * 2.0) * 1.5;
const INITIAL_PLANT_COUNT: usize = 70;
const INITIAL_FISH_COUNT: usize = 30;
const WINDOW_WIDTH: u32 = 1280;
const WINDOW_HEIGHT: u32 = 720;
const MAX_ORGANISMS: usize = 50_000;

// --- Data Structures ---

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum OrganismType {
    Plant,
    Fish,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2],
        }
    }
}

const QUAD_VERTICES: &[Vertex] = &[
    Vertex { position: [-1.0, -1.0] }, Vertex { position: [ 1.0, -1.0] },
    Vertex { position: [ 1.0,  1.0] }, Vertex { position: [-1.0,  1.0] },
];
const QUAD_INDICES: &[u16] = &[0, 1, 2, 0, 2, 3];

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct InstanceData {
    world_position: [f32; 2],
    radius: f32,
    color: [f32; 4],
}

impl InstanceData {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceData>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: mem::size_of::<[f32; 2]>() as wgpu::BufferAddress, shader_location: 2, format: wgpu::VertexFormat::Float32 },
                wgpu::VertexAttribute { offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress, shader_location: 3, format: wgpu::VertexFormat::Float32x4 },
            ],
        }
    }
}

#[derive(Debug)]
struct Organism {
    kind: OrganismType,
    position: Vec2,
    velocity: Vec2,
    age: f32,
    lifetime: f32,
    growth_rate: f32,
}

struct SimulationState {
    organisms: Vec<Organism>,
    rng: rand::rngs::ThreadRng,
    window_size: PhysicalSize<u32>,
    speed_multiplier: f32,
    is_paused: bool, // Added pause flag
}

impl SimulationState {
    fn new(window_size: PhysicalSize<u32>) -> Self {
        let mut state = Self {
            organisms: Vec::new(), // Initialize empty first
            rng: rand::thread_rng(),
            window_size,
            speed_multiplier: INITIAL_SPEED_MULTIPLIER,
            is_paused: false,
        };
        state.initialize_organisms(); // Populate organisms
        state
    }

    // Helper to populate organisms, used by new() and restart()
    fn initialize_organisms(&mut self) {
        self.organisms.clear(); // Clear existing organisms first
        let total_initial_count = INITIAL_PLANT_COUNT + INITIAL_FISH_COUNT;
        self.organisms.reserve(total_initial_count); // Reserve capacity

        // Create initial plants
        for _ in 0..INITIAL_PLANT_COUNT {
            self.organisms.push(Self::create_random_organism(&mut self.rng, self.window_size, OrganismType::Plant));
        }
        // Create initial fish
        for _ in 0..INITIAL_FISH_COUNT {
            self.organisms.push(Self::create_random_organism(&mut self.rng, self.window_size, OrganismType::Fish));
        }
    }


    fn create_random_organism(rng: &mut rand::rngs::ThreadRng, window_size: PhysicalSize<u32>, kind: OrganismType) -> Organism {
        let position = Vec2::new(rng.gen_range(0.0..window_size.width as f32), rng.gen_range(0.0..window_size.height as f32));
        let lifetime = rng.gen_range(MIN_LIFETIME..MAX_LIFETIME);
        let velocity = match kind {
            OrganismType::Plant => Vec2::ZERO,
            OrganismType::Fish => {
                let angle = rng.gen_range(0.0..std::f32::consts::TAU);
                Vec2::new(angle.cos(), angle.sin()) * MOVEMENT_SPEED
            }
        };
        Organism { kind, position, velocity, age: 0.0, lifetime, growth_rate: GROWTH_RATE }
    }

    fn create_offspring(parent: &Organism, window_size: PhysicalSize<u32>, rng: &mut rand::rngs::ThreadRng) -> Organism {
        let angle_offset = rng.gen_range(0.0..std::f32::consts::TAU);
        let offset = Vec2::new(angle_offset.cos(), angle_offset.sin()) * SPAWN_OFFSET_RADIUS;
        let mut position = parent.position + offset;
        position.x = position.x.clamp(0.0, window_size.width as f32);
        position.y = position.y.clamp(0.0, window_size.height as f32);
        let lifetime = rng.gen_range(MIN_LIFETIME..MAX_LIFETIME);
        let kind = parent.kind;
        let velocity = match kind {
            OrganismType::Plant => Vec2::ZERO,
            OrganismType::Fish => {
                let angle = rng.gen_range(0.0..std::f32::consts::TAU);
                Vec2::new(angle.cos(), angle.sin()) * MOVEMENT_SPEED
            }
        };
        Organism { kind, position, velocity, age: 0.0, lifetime, growth_rate: GROWTH_RATE }
    }

    fn update(&mut self, delta_time: f32) {
        // --- Pause Check ---
        if self.is_paused {
            return; // Do nothing if paused
        }
        // --- End Pause Check ---

        let dt = delta_time * self.speed_multiplier;
        let mut new_organisms = Vec::new();
        let mut organisms_to_remove = Vec::new();
        let window_size = self.window_size;
        let current_organism_count = self.organisms.len();

        for (i, organism) in self.organisms.iter_mut().enumerate() {
            organism.age += dt;
            if organism.age >= organism.lifetime {
                organisms_to_remove.push(i); continue;
            }
            organism.position += organism.velocity * dt;
            if organism.position.x < 0.0 { organism.position.x += window_size.width as f32; }
            else if organism.position.x >= window_size.width as f32 { organism.position.x -= window_size.width as f32; }
            if organism.position.y < 0.0 { organism.position.y += window_size.height as f32; }
            else if organism.position.y >= window_size.height as f32 { organism.position.y -= window_size.height as f32; }

            let spawn_probability_this_tick = (organism.growth_rate / organism.lifetime) * dt;
            if spawn_probability_this_tick > 0.0 && self.rng.gen_bool(spawn_probability_this_tick.clamp(0.0, 1.0) as f64) {
                if current_organism_count + new_organisms.len() < MAX_ORGANISMS {
                    new_organisms.push(Self::create_offspring(organism, window_size, &mut self.rng));
                }
            }
            if organism.kind == OrganismType::Fish && self.rng.gen_bool(0.01 * self.speed_multiplier as f64) {
                let angle = self.rng.gen_range(0.0..std::f32::consts::TAU);
                organism.velocity = Vec2::new(angle.cos(), angle.sin()) * MOVEMENT_SPEED;
            }
        }
        organisms_to_remove.sort_unstable_by(|a, b| b.cmp(a));
        for i in organisms_to_remove { self.organisms.swap_remove(i); }
        self.organisms.extend(new_organisms);
    }

    fn adjust_speed(&mut self, increase: bool) {
        // Prevent speed change if paused? Optional.
        // if self.is_paused { return; }
        self.speed_multiplier = if increase {
            (self.speed_multiplier + SPEED_ADJUST_FACTOR).min(MAX_SPEED_MULTIPLIER)
        } else {
            (self.speed_multiplier - SPEED_ADJUST_FACTOR).max(MIN_SPEED_MULTIPLIER)
        };
        println!("Speed Multiplier: {:.2}", self.speed_multiplier);
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 { self.window_size = new_size; }
    }

    // --- New Methods ---
    fn toggle_pause(&mut self) {
        self.is_paused = !self.is_paused;
        println!("Simulation {}", if self.is_paused { "Paused" } else { "Resumed" });
    }

    fn restart(&mut self) {
        println!("Restarting simulation...");
        self.initialize_organisms(); // Re-create organisms
        self.speed_multiplier = INITIAL_SPEED_MULTIPLIER; // Reset speed
        self.is_paused = false; // Ensure not paused
        // self.rng = rand::thread_rng(); // Get a new RNG state implicitly on next use
    }
    // --- End New Methods ---
}

struct Renderer<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    instance_data: Vec<InstanceData>,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    max_instances: usize,
    window: Arc<Window>,
}

impl<'a> Renderer<'a> {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let size = PhysicalSize::new(size.width.max(1), size.height.max(1));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::PRIMARY, ..Default::default() });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false,
        }).await.unwrap();
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor { label: Some("Device"), required_features: wgpu::Features::empty(), required_limits: wgpu::Limits::default() }, None,
        ).await.unwrap();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: surface_format,
            width: size.width, height: size.height, present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0], view_formats: vec![], desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let shader_source = include_str!("shader.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Shader Module"), source: wgpu::ShaderSource::Wgsl(shader_source.into()) });
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"), contents: bytemuck::cast_slice(&[size.width as f32, size.height as f32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None,
                }, count: None,
            }], label: Some("Uniform Bind Group Layout"),
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() }],
            label: Some("Uniform Bind Group"),
        });
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"), bind_group_layouts: &[&uniform_bind_group_layout], push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"), layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader, entry_point: "vs_main",
                buffers: &[Vertex::desc(), InstanceData::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: "fs_main", targets: &[Some(wgpu::ColorTargetState {
                    format: config.format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, strip_index_format: None, front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, polygon_mode: wgpu::PolygonMode::Fill, unclipped_depth: false, conservative: false,
            },
            depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview: None,
        });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"), contents: bytemuck::cast_slice(QUAD_VERTICES), usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"), contents: bytemuck::cast_slice(QUAD_INDICES), usage: wgpu::BufferUsages::INDEX,
        });
        let max_instances = (INITIAL_PLANT_COUNT + INITIAL_FISH_COUNT) * 10;
        let instance_data = Vec::with_capacity(max_instances);
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"), size: (max_instances * std::mem::size_of::<InstanceData>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        Self {
            surface, device, queue, config, size, render_pipeline, vertex_buffer, index_buffer, instance_buffer,
            instance_data, uniform_buffer, uniform_bind_group, max_instances, window,
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        let new_size = PhysicalSize::new(new_size.width.max(1), new_size.height.max(1));
        if new_size.width > 0 && new_size.height > 0 && new_size != self.size {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            let resolution = [new_size.width as f32, new_size.height as f32];
            self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&resolution));
        }
    }

    fn render(&mut self, organisms: &[Organism]) -> Result<(), wgpu::SurfaceError> {
        let output_texture = self.surface.get_current_texture()?;
        let view = output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });

        self.instance_data.clear();
        self.instance_data.extend(organisms.iter().map(|org| {
            let color = match org.kind {
                OrganismType::Plant => PLANT_COLOR,
                OrganismType::Fish => FISH_COLOR,
            };
            InstanceData { world_position: org.position.to_array(), radius: ORGANISM_RADIUS, color: color.to_array() }
        }));

        let current_instance_count = self.instance_data.len();
        let required_buffer_size = (current_instance_count * std::mem::size_of::<InstanceData>()) as wgpu::BufferAddress;

        if current_instance_count > self.max_instances {
            let new_max_instances = (current_instance_count * 2).next_power_of_two();
            let new_buffer_size = (new_max_instances * std::mem::size_of::<InstanceData>()) as wgpu::BufferAddress;
            println!("Resizing instance buffer from {} to {} instances ({} bytes)", self.max_instances, new_max_instances, new_buffer_size);
            self.instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Instance Buffer (Resized)"), size: new_buffer_size, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
            });
            self.max_instances = new_max_instances;
        }
        if !self.instance_data.is_empty() {
            self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&self.instance_data));
        }
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"), color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(BACKGROUND_COLOR), store: wgpu::StoreOp::Store },
                })], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            if current_instance_count > 0 {
                render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..required_buffer_size));
                render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..QUAD_INDICES.len() as u32, 0, 0..current_instance_count as u32);
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output_texture.present();
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    let window = Arc::new(WindowBuilder::new()
        .with_title("Urkaos Life Simulation")
        .with_inner_size(PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .build(&event_loop)?);

    let mut renderer = pollster::block_on(Renderer::new(window.clone()));
    // SimulationState::new now handles initialization itself
    let mut simulation_state = SimulationState::new(renderer.size);
    let mut last_frame_time = Instant::now();
    let mut time_accumulator = 0.0;

    event_loop.run(move |event, elwt: &EventLoopWindowTarget<()>| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            Event::AboutToWait => {
                // Only run simulation updates if not paused
                if !simulation_state.is_paused {
                    let now = Instant::now();
                    let delta_time = now.duration_since(last_frame_time).as_secs_f64();
                    last_frame_time = now;
                    time_accumulator += delta_time;
                    while time_accumulator >= FIXED_TIMESTEP {
                        simulation_state.update(FIXED_TIMESTEP as f32);
                        time_accumulator -= FIXED_TIMESTEP;
                    }
                } else {
                    // If paused, reset last_frame_time to avoid a large jump when resuming
                    last_frame_time = Instant::now();
                    time_accumulator = 0.0; // Reset accumulator too
                }
                window.request_redraw(); // Always request redraw to show current state (even if paused)
            }
            Event::WindowEvent { window_id, event } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(physical_size) => {
                    renderer.resize(physical_size);
                    simulation_state.resize(physical_size);
                }
                WindowEvent::ScaleFactorChanged { .. } => {
                    let new_inner_size = window.inner_size();
                    renderer.resize(new_inner_size);
                    simulation_state.resize(new_inner_size);
                }
                WindowEvent::KeyboardInput { event: key_event, .. } => {
                    if key_event.state == ElementState::Pressed && !key_event.repeat {
                        // Handle new keys
                        match key_event.physical_key {
                            PhysicalKey::Code(KeyCode::ArrowUp) => simulation_state.adjust_speed(true),
                            PhysicalKey::Code(KeyCode::ArrowDown) => simulation_state.adjust_speed(false),
                            PhysicalKey::Code(KeyCode::Space) => simulation_state.toggle_pause(), // Toggle pause
                            PhysicalKey::Code(KeyCode::KeyR) => simulation_state.restart(), // Restart simulation
                            PhysicalKey::Code(KeyCode::Escape) => elwt.exit(),
                            _ => {}
                        }
                    }
                }
                WindowEvent::RedrawRequested => {
                    match renderer.render(&simulation_state.organisms) {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => { eprintln!("WGPU Error: OutOfMemory"); elwt.exit(); }
                        Err(e) => eprintln!("WGPU Error: {:?}", e),
                    }
                    // Update title with pause state
                    let plant_count = simulation_state.organisms.iter().filter(|o| o.kind == OrganismType::Plant).count();
                    let fish_count = simulation_state.organisms.len() - plant_count;
                    let paused_text = if simulation_state.is_paused { " [PAUSED]" } else { "" };
                    window.set_title(&format!(
                        "Urkaos - Plants: {}, Fish: {} - Speed: {:.1}x{}",
                        plant_count, fish_count,
                        simulation_state.speed_multiplier,
                        paused_text // Add paused indicator
                    ));
                }
                _ => {}
            },
            _ => {}
        }
    })?;
    Ok(())
}