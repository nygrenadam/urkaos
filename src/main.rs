// use wgpu::util::DeviceExt; // Removed unused import
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec4};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::{collections::HashMap, sync::Arc, time::Instant};
use wgpu::util::DeviceExt;
// Keep this if create_buffer_init is used (it is!)
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

// --- Global Constants ---
const BACKGROUND_COLOR: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};
const ORGANISM_RADIUS: f32 = 5.0;
const INITIAL_SPEED_MULTIPLIER: f32 = 1.0;
const MIN_SPEED_MULTIPLIER: f32 = 0.0;
const MAX_SPEED_MULTIPLIER: f32 = 10.0;
const SPEED_ADJUST_FACTOR: f32 = 0.5;
const FIXED_TIMESTEP: f64 = 1.0 / 60.0;
const SPAWN_OFFSET_RADIUS: f32 = (ORGANISM_RADIUS * 2.0) * 1.5;
const INITIAL_PLANT_COUNT: usize = 70;
const INITIAL_FISH_COUNT: usize = 30;
const INITIAL_BUG_COUNT: usize = 50;
const WINDOW_WIDTH: u32 = 1280;
const WINDOW_HEIGHT: u32 = 720;
const MAX_ORGANISMS: usize = 50_000;
const PLANT_BASE_COLOR: Vec4 = Vec4::new(0.1, 0.7, 0.1, 1.0);
const FISH_BASE_COLOR: Vec4 = Vec4::new(0.2, 0.3, 0.9, 1.0);
const BUG_BASE_COLOR: Vec4 = Vec4::new(0.9, 0.6, 0.1, 1.0);
const FPS_UPDATE_INTERVAL_SECS: f64 = 0.5;
const GRID_CELL_SIZE: f32 = 100.0;

const INITIAL_COLOR_MUTATION_MAX_DELTA: f32 = 0.15;
const OFFSPRING_COLOR_MUTATION_MAX_DELTA: f32 = 0.05;

// --- Configuration Structs ---

#[derive(Debug, Clone)]
struct PlantConfig {
    /* fields */ min_lifetime: f32,
    max_lifetime: f32,
    base_growth_rate: f32,
    aging_rate_when_eaten: f32,
    clustering_radius: f32,
    clustering_radius_sq: f32,
    clustering_spawn_boost_factor: f32,
}
impl Default for PlantConfig {
    fn default() -> Self {
        let cr = ORGANISM_RADIUS * 5.0;
        Self {
            min_lifetime: 15.0,
            max_lifetime: 45.0,
            base_growth_rate: 1.0,
            aging_rate_when_eaten: 5.0,
            clustering_radius: cr,
            clustering_radius_sq: cr * cr,
            clustering_spawn_boost_factor: 1.5,
        }
    }
}

#[derive(Debug, Clone)]
struct FishConfig {
    /* fields */ min_lifetime: f32,
    max_lifetime: f32,
    base_growth_rate: f32,
    movement_speed: f32,
    perception_radius: f32,
    perception_radius_sq: f32,
    eating_radius: f32,
    eating_radius_sq: f32,
    influence_plant: f32,
    influence_fish: f32,
    influence_bug: f32,
    eating_spawn_boost_factor: f32,
}
impl Default for FishConfig {
    fn default() -> Self {
        let pr = 100.0;
        let er = ORGANISM_RADIUS * 2.5;
        Self {
            min_lifetime: 15.0,
            max_lifetime: 40.0,
            base_growth_rate: 1.0,
            movement_speed: 50.0,
            perception_radius: pr,
            perception_radius_sq: pr * pr,
            eating_radius: er,
            eating_radius_sq: er * er,
            influence_plant: 0.5,
            influence_fish: -0.2,
            influence_bug: -0.1,
            eating_spawn_boost_factor: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
struct BugConfig {
    /* fields */ min_lifetime: f32,
    max_lifetime: f32,
    base_growth_rate: f32,
    movement_speed: f32,
    max_turn_angle_per_sec: f32,
    eating_radius: f32,
    eating_radius_sq: f32,
    eating_spawn_boost_factor: f32,
}
impl Default for BugConfig {
    fn default() -> Self {
        let er = ORGANISM_RADIUS * 2.0;
        Self {
            min_lifetime: 10.0,
            max_lifetime: 30.0,
            base_growth_rate: 0.8,
            movement_speed: 40.0,
            max_turn_angle_per_sec: std::f32::consts::PI / 4.0,
            eating_radius: er,
            eating_radius_sq: er * er,
            eating_spawn_boost_factor: 1.8,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct SimulationConfig {
    plant: PlantConfig,
    fish: FishConfig,
    bug: BugConfig,
}

// --- Core Data Structures ---

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum OrganismType {
    Plant,
    Fish,
    Bug,
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
    Vertex {
        position: [-1.0, -1.0],
    },
    Vertex {
        position: [1.0, -1.0],
    },
    Vertex {
        position: [1.0, 1.0],
    },
    Vertex {
        position: [-1.0, 1.0],
    },
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
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
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
    color: Vec4,
}

type GridKey = (i32, i32);
type SimRng = StdRng;

struct SimulationState {
    organisms: Vec<Organism>,
    rng: SimRng,
    window_size: PhysicalSize<u32>,
    speed_multiplier: f32,
    is_paused: bool,
    config: SimulationConfig,
    grid: HashMap<GridKey, Vec<usize>>,
    grid_width: i32,
    grid_height: i32,
}

impl SimulationState {
    fn new(window_size: PhysicalSize<u32>, config: SimulationConfig) -> Self {
        let grid_width = (window_size.width as f32 / GRID_CELL_SIZE).ceil() as i32;
        let grid_height = (window_size.height as f32 / GRID_CELL_SIZE).ceil() as i32;
        let mut state = Self {
            organisms: Vec::new(),
            rng: SimRng::from_entropy(),
            window_size,
            speed_multiplier: INITIAL_SPEED_MULTIPLIER,
            is_paused: false,
            config,
            grid: HashMap::new(),
            grid_width,
            grid_height,
        };
        state.initialize_organisms();
        state
    }
    fn initialize_organisms(&mut self) {
        self.organisms.clear();
        let total_initial_count = INITIAL_PLANT_COUNT + INITIAL_FISH_COUNT + INITIAL_BUG_COUNT;
        self.organisms.reserve(total_initial_count);
        for _ in 0..INITIAL_PLANT_COUNT {
            self.organisms.push(Self::create_organism(
                &mut self.rng,
                self.window_size,
                OrganismType::Plant,
                &self.config,
            ));
        }
        for _ in 0..INITIAL_FISH_COUNT {
            self.organisms.push(Self::create_organism(
                &mut self.rng,
                self.window_size,
                OrganismType::Fish,
                &self.config,
            ));
        }
        for _ in 0..INITIAL_BUG_COUNT {
            self.organisms.push(Self::create_organism(
                &mut self.rng,
                self.window_size,
                OrganismType::Bug,
                &self.config,
            ));
        }
    }
    fn create_organism(
        rng: &mut SimRng,
        window_size: PhysicalSize<u32>,
        kind: OrganismType,
        config: &SimulationConfig,
    ) -> Organism {
        let position = Vec2::new(
            rng.gen_range(0.0..window_size.width as f32),
            rng.gen_range(0.0..window_size.height as f32),
        );
        let (min_lt, max_lt, base_gr, vel, base_color) = match kind {
            OrganismType::Plant => (
                config.plant.min_lifetime,
                config.plant.max_lifetime,
                config.plant.base_growth_rate,
                Vec2::ZERO,
                PLANT_BASE_COLOR,
            ),
            OrganismType::Fish => (
                config.fish.min_lifetime,
                config.fish.max_lifetime,
                config.fish.base_growth_rate,
                {
                    let angle = rng.gen_range(0.0..std::f32::consts::TAU);
                    Vec2::new(angle.cos(), angle.sin()) * config.fish.movement_speed
                },
                FISH_BASE_COLOR,
            ),
            OrganismType::Bug => (
                config.bug.min_lifetime,
                config.bug.max_lifetime,
                config.bug.base_growth_rate,
                {
                    let angle = rng.gen_range(0.0..std::f32::consts::TAU);
                    Vec2::new(angle.cos(), angle.sin()) * config.bug.movement_speed
                },
                BUG_BASE_COLOR,
            ),
        };
        let lifetime = rng.gen_range(min_lt..max_lt);
        let growth_rate = base_gr;
        let color = mutate_color(base_color, rng, INITIAL_COLOR_MUTATION_MAX_DELTA);
        Organism {
            kind,
            position,
            velocity: vel,
            age: 0.0,
            lifetime,
            growth_rate,
            color,
        }
    }
    fn create_offspring(
        parent: &Organism,
        window_size: PhysicalSize<u32>,
        rng: &mut SimRng,
        config: &SimulationConfig,
    ) -> Organism {
        let angle_offset = rng.gen_range(0.0..std::f32::consts::TAU);
        let offset = Vec2::new(angle_offset.cos(), angle_offset.sin()) * SPAWN_OFFSET_RADIUS;
        let mut position = parent.position + offset;
        position.x = position.x.clamp(0.0, window_size.width as f32);
        position.y = position.y.clamp(0.0, window_size.height as f32);
        let kind = parent.kind;
        let (min_lt, max_lt, base_gr, vel) = match kind {
            OrganismType::Plant => (
                config.plant.min_lifetime,
                config.plant.max_lifetime,
                config.plant.base_growth_rate,
                Vec2::ZERO,
            ),
            OrganismType::Fish => (
                config.fish.min_lifetime,
                config.fish.max_lifetime,
                config.fish.base_growth_rate,
                {
                    let angle = rng.gen_range(0.0..std::f32::consts::TAU);
                    Vec2::new(angle.cos(), angle.sin()) * config.fish.movement_speed
                },
            ),
            OrganismType::Bug => (
                config.bug.min_lifetime,
                config.bug.max_lifetime,
                config.bug.base_growth_rate,
                {
                    let parent_dir = parent.velocity.normalize_or_zero();
                    let angle_dev =
                        rng.gen_range(-std::f32::consts::PI / 4.0..std::f32::consts::PI / 4.0);
                    let new_dir = Vec2::from_angle(parent_dir.to_angle() + angle_dev);
                    new_dir * config.bug.movement_speed
                },
            ),
        };
        let lifetime = rng.gen_range(min_lt..max_lt);
        let growth_rate = base_gr;
        let color = mutate_color(parent.color, rng, OFFSPRING_COLOR_MUTATION_MAX_DELTA);
        Organism {
            kind,
            position,
            velocity: vel,
            age: 0.0,
            lifetime,
            growth_rate,
            color,
        }
    }

    #[inline]
    fn get_grid_key(&self, position: Vec2) -> GridKey {
        let cell_x = (position.x / GRID_CELL_SIZE).floor() as i32;
        let cell_y = (position.y / GRID_CELL_SIZE).floor() as i32;
        (
            cell_x.clamp(0, self.grid_width - 1),
            cell_y.clamp(0, self.grid_height - 1),
        )
    }
    fn build_grid(&mut self) {
        self.grid.clear();
        for (index, organism) in self.organisms.iter().enumerate() {
            let key = self.get_grid_key(organism.position);
            self.grid.entry(key).or_default().push(index);
        }
    }

    fn update(&mut self, delta_time: f32) {
        if self.is_paused {
            return;
        }
        let dt = delta_time * self.speed_multiplier;
        let mut new_organisms = Vec::new();
        let mut organisms_to_remove = Vec::new();
        let window_size = self.window_size;
        let current_organism_count = self.organisms.len();
        self.build_grid();
        let config = &self.config;
        let mut new_velocities: Vec<Option<Vec2>> = vec![None; current_organism_count];
        let mut plant_extra_aging: Vec<f32> = vec![0.0; current_organism_count];
        let mut fish_is_eating: Vec<bool> = vec![false; current_organism_count];
        let mut bug_is_eating: Vec<bool> = vec![false; current_organism_count];
        let mut plant_is_clustered: Vec<bool> = vec![false; current_organism_count];
        for i in 0..current_organism_count {
            let organism_i = &self.organisms[i];
            let pos_i = organism_i.position;
            let grid_key_i = self.get_grid_key(pos_i);
            let mut fish_influence_vector = Vec2::ZERO;
            let mut fish_neighbor_count = 0;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let check_key = (grid_key_i.0 + dx, grid_key_i.1 + dy);
                    if let Some(neighbor_indices) = self.grid.get(&check_key) {
                        for &j in neighbor_indices {
                            if i == j {
                                continue;
                            }
                            let organism_j = &self.organisms[j];
                            let pos_j = organism_j.position;
                            let vec_ij = pos_j - pos_i;
                            let dist_sq = vec_ij.length_squared();
                            match organism_i.kind {
                                OrganismType::Fish => {
                                    if dist_sq > 0.0 && dist_sq < config.fish.perception_radius_sq {
                                        let influence_factor = match organism_j.kind {
                                            OrganismType::Plant => config.fish.influence_plant,
                                            OrganismType::Fish => config.fish.influence_fish,
                                            OrganismType::Bug => config.fish.influence_bug,
                                        };
                                        if influence_factor != 0.0 {
                                            fish_influence_vector +=
                                                vec_ij.normalize_or_zero() * influence_factor;
                                            fish_neighbor_count += 1;
                                        }
                                    }
                                    if organism_j.kind == OrganismType::Plant
                                        && dist_sq < config.fish.eating_radius_sq
                                    {
                                        fish_is_eating[i] = true;
                                        plant_extra_aging[j] += config.plant.aging_rate_when_eaten;
                                    }
                                }
                                OrganismType::Plant => {
                                    if organism_j.kind == OrganismType::Plant
                                        && dist_sq < config.plant.clustering_radius_sq
                                    {
                                        plant_is_clustered[i] = true;
                                    }
                                }
                                OrganismType::Bug => {
                                    if organism_j.kind == OrganismType::Plant
                                        && dist_sq < config.bug.eating_radius_sq
                                    {
                                        bug_is_eating[i] = true;
                                        plant_extra_aging[j] += config.plant.aging_rate_when_eaten;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            match organism_i.kind {
                OrganismType::Fish => {
                    let random_angle = self.rng.gen_range(0.0..std::f32::consts::TAU);
                    let random_direction = Vec2::new(random_angle.cos(), random_angle.sin());
                    let normalized_influence = if fish_neighbor_count > 0 {
                        fish_influence_vector.normalize_or_zero()
                    } else {
                        Vec2::ZERO
                    };
                    let final_direction =
                        (random_direction + normalized_influence).normalize_or_zero();
                    new_velocities[i] = Some(final_direction * config.fish.movement_speed);
                }
                OrganismType::Bug => {
                    let current_velocity = organism_i.velocity;
                    let current_speed = current_velocity.length();
                    let current_dir = current_velocity.normalize_or_zero();
                    let max_turn_this_tick = config.bug.max_turn_angle_per_sec * dt as f32;
                    let turn_angle = self.rng.gen_range(-max_turn_this_tick..max_turn_this_tick);
                    let new_dir = Vec2::from_angle(current_dir.to_angle() + turn_angle);
                    new_velocities[i] = Some(new_dir * current_speed);
                }
                OrganismType::Plant => {}
            }
        }
        for i in 0..current_organism_count {
            let organism = &mut self.organisms[i];
            if let Some(new_vel) = new_velocities[i] {
                organism.velocity = new_vel;
            }
            organism.position += organism.velocity * dt;
            if organism.position.x < 0.0 {
                organism.position.x += window_size.width as f32;
            } else if organism.position.x >= window_size.width as f32 {
                organism.position.x -= window_size.width as f32;
            }
            if organism.position.y < 0.0 {
                organism.position.y += window_size.height as f32;
            } else if organism.position.y >= window_size.height as f32 {
                organism.position.y -= window_size.height as f32;
            }
            organism.age += dt;
            if organism.kind == OrganismType::Plant && plant_extra_aging[i] > 0.0 {
                organism.age += plant_extra_aging[i] * dt;
            }
            if organism.age >= organism.lifetime {
                organisms_to_remove.push(i);
                continue;
            }
            let base_prob_per_sec = organism.growth_rate / organism.lifetime;
            let mut spawn_prob_this_tick = base_prob_per_sec * dt;
            match organism.kind {
                OrganismType::Fish => {
                    if fish_is_eating[i] {
                        spawn_prob_this_tick *= config.fish.eating_spawn_boost_factor;
                    }
                }
                OrganismType::Plant => {
                    if plant_is_clustered[i] {
                        spawn_prob_this_tick *= config.plant.clustering_spawn_boost_factor;
                    }
                }
                OrganismType::Bug => {
                    if bug_is_eating[i] {
                        spawn_prob_this_tick *= config.bug.eating_spawn_boost_factor;
                    }
                }
            }
            if spawn_prob_this_tick > 0.0
                && self
                    .rng
                    .gen_bool(spawn_prob_this_tick.clamp(0.0, 1.0) as f64)
            {
                if current_organism_count + new_organisms.len() < MAX_ORGANISMS {
                    let parent_ref = &self.organisms[i];
                    new_organisms.push(Self::create_offspring(
                        parent_ref,
                        window_size,
                        &mut self.rng,
                        config,
                    ));
                }
            }
        }
        organisms_to_remove.sort_unstable_by(|a, b| b.cmp(a));
        for i in organisms_to_remove {
            if i < self.organisms.len() {
                self.organisms.swap_remove(i);
            } else {
                eprintln!("Warn: Attempted to remove organism at invalid index {}", i);
            }
        }
        self.organisms.extend(new_organisms);
    }

    // RESTORED function body
    fn adjust_speed(&mut self, increase: bool) {
        self.speed_multiplier = if increase {
            (self.speed_multiplier + SPEED_ADJUST_FACTOR).min(MAX_SPEED_MULTIPLIER)
        } else {
            (self.speed_multiplier - SPEED_ADJUST_FACTOR).max(MIN_SPEED_MULTIPLIER)
        };
        println!("Speed Multiplier: {:.2}", self.speed_multiplier);
    }
    // RESTORED function body
    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.window_size = new_size;
            self.grid_width = (new_size.width as f32 / GRID_CELL_SIZE).ceil() as i32;
            self.grid_height = (new_size.height as f32 / GRID_CELL_SIZE).ceil() as i32;
            println!("Resized grid to {}x{}", self.grid_width, self.grid_height);
        }
    }
    // RESTORED function body
    fn toggle_pause(&mut self) {
        self.is_paused = !self.is_paused;
        println!(
            "Simulation {}",
            if self.is_paused { "Paused" } else { "Resumed" }
        );
    }
    // RESTORED function body
    fn restart(&mut self) {
        println!("Restarting simulation with new seed...");
        self.rng = SimRng::from_entropy();
        self.initialize_organisms();
        self.speed_multiplier = INITIAL_SPEED_MULTIPLIER;
        self.is_paused = false;
    }
}

// --- Renderer ---
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
    // RESTORED function body
    async fn new(window: Arc<Window>) -> Self {
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
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();
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
        let shader_source = include_str!("shader.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader Module"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[size.width as f32, size.height as f32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("Uniform Bind Group Layout"),
            });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("Uniform Bind Group"),
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), InstanceData::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let max_instances = (INITIAL_PLANT_COUNT + INITIAL_FISH_COUNT + INITIAL_BUG_COUNT) * 10;
        let instance_data = Vec::with_capacity(max_instances);
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (max_instances * std::mem::size_of::<InstanceData>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            instance_data,
            uniform_buffer,
            uniform_bind_group,
            max_instances,
            window,
        }
    }
    // RESTORED function body
    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        let sim_new_size = PhysicalSize::new(new_size.width.max(1), new_size.height.max(1));
        if sim_new_size.width > 0 && sim_new_size.height > 0 && sim_new_size != self.size {
            self.size = sim_new_size;
            self.config.width = sim_new_size.width;
            self.config.height = sim_new_size.height;
            self.surface.configure(&self.device, &self.config);
            let resolution = [sim_new_size.width as f32, sim_new_size.height as f32];
            self.queue
                .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&resolution));
        }
    }
    // Render body was already complete
    fn render(
        &mut self,
        organisms: &[Organism],
        _config: &SimulationConfig,
    ) -> Result<(), wgpu::SurfaceError> {
        let output_texture = self.surface.get_current_texture()?;
        let view = output_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        self.instance_data.clear();
        self.instance_data
            .extend(organisms.iter().map(|org| InstanceData {
                world_position: org.position.to_array(),
                radius: ORGANISM_RADIUS,
                color: org.color.to_array(),
            }));
        let current_instance_count = self.instance_data.len();
        let required_buffer_size =
            (current_instance_count * std::mem::size_of::<InstanceData>()) as wgpu::BufferAddress;
        if current_instance_count > self.max_instances {
            let new_max_instances = (current_instance_count * 2).next_power_of_two();
            let new_buffer_size =
                (new_max_instances * std::mem::size_of::<InstanceData>()) as wgpu::BufferAddress;
            println!(
                "Resizing instance buffer from {} to {} instances ({} bytes)",
                self.max_instances, new_max_instances, new_buffer_size
            );
            self.instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Instance Buffer (Resized)"),
                size: new_buffer_size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.max_instances = new_max_instances;
        }
        if !self.instance_data.is_empty() {
            self.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instance_data),
            );
        }
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            if current_instance_count > 0 {
                render_pass
                    .set_vertex_buffer(1, self.instance_buffer.slice(..required_buffer_size));
                render_pass
                    .set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(
                    0..QUAD_INDICES.len() as u32,
                    0,
                    0..current_instance_count as u32,
                );
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output_texture.present();
        Ok(())
    }
}

// --- Main Function ---
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Urkaos Life Simulation")
            .with_inner_size(PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .build(&event_loop)?,
    );
    let mut renderer = pollster::block_on(Renderer::new(window.clone()));
    let simulation_config = SimulationConfig::default();
    let mut simulation_state = SimulationState::new(renderer.size, simulation_config);
    let mut last_sim_update_time = Instant::now();
    let mut time_accumulator = 0.0;
    let mut last_fps_update_time = Instant::now();
    let mut frames_since_last_fps_update = 0;
    let mut current_fps = 0.0;

    event_loop.run(move |event, elwt: &EventLoopWindowTarget<()>| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            Event::AboutToWait => {
                if !simulation_state.is_paused {
                    let now = Instant::now();
                    let delta_time = now.duration_since(last_sim_update_time).as_secs_f64();
                    last_sim_update_time = now;
                    time_accumulator += delta_time;
                    while time_accumulator >= FIXED_TIMESTEP {
                        simulation_state.update(FIXED_TIMESTEP as f32);
                        time_accumulator -= FIXED_TIMESTEP;
                    }
                } else {
                    last_sim_update_time = Instant::now();
                    time_accumulator = 0.0;
                }
                window.request_redraw();
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
                WindowEvent::KeyboardInput {
                    event: key_event, ..
                } => {
                    if key_event.state == ElementState::Pressed && !key_event.repeat {
                        match key_event.physical_key {
                            PhysicalKey::Code(KeyCode::ArrowUp) => {
                                simulation_state.adjust_speed(true)
                            }
                            PhysicalKey::Code(KeyCode::ArrowDown) => {
                                simulation_state.adjust_speed(false)
                            }
                            PhysicalKey::Code(KeyCode::Space) => simulation_state.toggle_pause(),
                            PhysicalKey::Code(KeyCode::KeyR) => simulation_state.restart(),
                            PhysicalKey::Code(KeyCode::Escape) => elwt.exit(),
                            _ => {}
                        }
                    }
                }
                WindowEvent::RedrawRequested => {
                    frames_since_last_fps_update += 1;
                    let now = Instant::now();
                    let elapsed_secs = now.duration_since(last_fps_update_time).as_secs_f64();
                    if elapsed_secs >= FPS_UPDATE_INTERVAL_SECS {
                        current_fps = frames_since_last_fps_update as f64 / elapsed_secs;
                        last_fps_update_time = now;
                        frames_since_last_fps_update = 0;
                    }
                    match renderer.render(&simulation_state.organisms, &simulation_state.config) {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            eprintln!("WGPU Error: OutOfMemory");
                            elwt.exit();
                        }
                        Err(e) => eprintln!("WGPU Error: {:?}", e),
                    }
                    let mut plant_count = 0;
                    let mut fish_count = 0;
                    let mut bug_count = 0;
                    for org in &simulation_state.organisms {
                        match org.kind {
                            OrganismType::Plant => plant_count += 1,
                            OrganismType::Fish => fish_count += 1,
                            OrganismType::Bug => bug_count += 1,
                        }
                    }
                    let paused_text = if simulation_state.is_paused {
                        " [PAUSED]"
                    } else {
                        ""
                    };
                    window.set_title(&format!(
                        "Urkaos - P: {}, F: {}, B: {} - Speed: {:.1}x - FPS: {:.1}{}",
                        plant_count,
                        fish_count,
                        bug_count,
                        simulation_state.speed_multiplier,
                        current_fps,
                        paused_text
                    ));
                }
                _ => {}
            },
            _ => {}
        }
    })?;
    Ok(())
}

// --- Helper Functions ---

trait Vec2Angle: Sized {
    fn to_angle(self) -> f32;
    fn from_angle(angle: f32) -> Self;
}
impl Vec2Angle for Vec2 {
    fn to_angle(self) -> f32 {
        self.y.atan2(self.x)
    }
    fn from_angle(angle: f32) -> Self {
        Vec2::new(angle.cos(), angle.sin())
    }
}

fn mutate_color(base_color: Vec4, rng: &mut SimRng, max_delta: f32) -> Vec4 {
    let r_delta = rng.gen_range(-max_delta..max_delta);
    let g_delta = rng.gen_range(-max_delta..max_delta);
    let b_delta = rng.gen_range(-max_delta..max_delta);
    let new_r = (base_color.x + r_delta).clamp(0.0, 1.0);
    let new_g = (base_color.y + g_delta).clamp(0.0, 1.0);
    let new_b = (base_color.z + b_delta).clamp(0.0, 1.0);
    Vec4::new(new_r, new_g, new_b, base_color.w)
}
