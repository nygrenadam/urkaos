// --- File: simulation.rs ---
// File: simulation.rs
use crate::config::{
    OrganismConfig, SimulationConfig, ABSOLUTE_MAX_MUTATION_RATE, ABSOLUTE_MIN_MUTATION_RATE,
};
// Import new constants
use crate::constants::*;
use crate::utils::mutate_color;
// Import helpers
use glam::{Vec2, Vec4};
// --- Use thread_rng for parallel random generation ---
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
// --- Import Rayon ---
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    f32::consts::TAU,
    sync::atomic::{AtomicU64, Ordering}, // For unique IDs
};
use winit::dpi::PhysicalSize;

// --- GPU Data Structure ---
// This struct MUST match the layout in the WGSL shader
// `repr(C)` ensures Rust uses a C-compatible memory layout.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OrganismGpuData {
    pub world_position: [f32; 2], // Use fixed-size arrays for Pod/Zeroable compatibility
    pub radius: f32,              // Current interpolated radius for rendering
    pub _padding1: f32,           // Ensure padding matches shader
    pub color: [f32; 4],          // Use fixed-size arrays
}

// --- Core Data Structures ---

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum OrganismType {
    Plant,
    Fish,
    Bug,
}

// --- NEW: Organism State Enum ---
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum OrganismState {
    Alive,
    CorpseA, // From stationary organism (movement_speed_factor == 0)
    CorpseB, // From mobile organism (movement_speed_factor > 0)
}
// --- End NEW ---

#[derive(Debug, Clone)] // Clone needed for create_offspring config copy
pub struct Organism {
    pub id: u64,                // Unique identifier
    pub parent_id: Option<u64>, // ID of the parent, if any
    pub kind: OrganismType,
    pub position: Vec2,
    pub velocity: Vec2,
    pub age: f32,
    pub lifetime: f32,      // Derived from config.min/max_lifetime
    pub growth_rate: f32,   // Derived from config.base_growth_rate
    pub color: Vec4,        // Derived from config.base_color
    pub radius: f32,        // Current interpolated radius (starts at 0)
    pub target_radius: f32, // The radius this organism aims for based on DNA/state
    // --- MODIFIED: State instead of is_dying ---
    pub state: OrganismState,
    pub original_type: Option<OrganismType>, // Stores kind when becoming a corpse
    pub decay_timer: f32,                    // Countdown for corpse removal
    // --- End MODIFIED ---
    pub config: OrganismConfig, // Holds the unique, potentially mutated DNA
}

pub type GridKey = (i32, i32);
pub type SimRng = StdRng; // Main RNG for seeding if needed, use thread_rng in parallel loops

// --- Constant for push-away force magnitude ---
const PUSH_AWAY_FORCE_MAGNITUDE: f32 = 64.0; // Tunable
// --- Minimum value clamps for mutated DNA ---
const MIN_LIFETIME_CLAMP: f32 = 1.0;
const MIN_RADIUS_CLAMP: f32 = 0.1;
const MIN_FACTOR_CLAMP: f32 = 0.0; // For rates, speeds, factors etc.

// Global atomic counter for organism IDs
static NEXT_ORGANISM_ID: AtomicU64 = AtomicU64::new(1);

// Helper function to get a new unique ID
fn get_new_organism_id() -> u64 {
    NEXT_ORGANISM_ID.fetch_add(1, Ordering::Relaxed)
}

pub struct SimulationState {
    pub organisms: Vec<Organism>,
    rng: SimRng, // Main RNG for initialization/restart
    window_size: PhysicalSize<u32>,
    speed_multiplier: f32,
    is_paused: bool,
    pub config: SimulationConfig, // Base config for creating new organisms
    grid: HashMap<GridKey, Vec<usize>>, // CPU grid for simulation logic
    grid_width: i32,
    grid_height: i32,
    grid_cell_size: f32,
    // OPTIMIZATION: Buffers for reuse in update loop
    new_organism_buffer: Vec<Organism>,
    removal_indices_buffer: Vec<usize>,
    removal_indices_set: HashSet<usize>, // Track indices marked for removal this frame
    interaction_radii_buffer: Vec<InteractionRadii>, // Reuse radii buffer
    accumulated_aging_rate_buffer: Vec<f32>, // Aging from being eaten (only applies to Alive)
    plant_is_clustered_buffer: Vec<bool>, // Only relevant for Alive plants

    // --- GPU grid data ---
    gpu_grid_indices: Vec<u32>, // Flat list of organism indices per cell
    gpu_grid_offsets: Vec<[u32; 2]>, // [offset, count] for each cell into gpu_grid_indices
}

#[derive(Debug, Default)]
struct OrganismCalculationResult {
    index: usize,
    new_velocity: Option<Vec2>,
    // Store aging effects to apply later serially (only applies *to* Alive organisms)
    prey_aging_effects: Vec<(usize, f32)>,
}

#[derive(Default, Clone, Copy)]
struct InteractionRadii {
    perception_sq: f32,
    eating_sq: f32,
    clustering_sq: f32, // Only used by Plants
    pushaway_sq: f32,
}

// --- DNA Mutation Helper Functions ---

fn mutate_value<R: Rng + ?Sized>(rng: &mut R, value: f32, rate: f32, min_clamp: f32) -> f32 {
    let multiplier = 1.0 + rng.gen_range(-rate..=rate);
    (value * multiplier).max(min_clamp)
}

fn mutate_influence<R: Rng + ?Sized>(rng: &mut R, value: f32, rate: f32) -> f32 {
    let multiplier = 1.0 + rng.gen_range(-rate..=rate);
    value * multiplier
}

// --- MODIFIED: Mutate new corpse eating factors and corpse influence factors ---
fn mutate_config_dna<R: Rng + ?Sized>(
    config: &mut OrganismConfig,
    rng: &mut R,
    mutation_rate_for_traits: f32,
    meta_mutation_rate: f32,
) {
    if mutation_rate_for_traits <= 0.0 && meta_mutation_rate <= 0.0 {
        return;
    }

    // --- 1. Mutate the standard traits using mutation_rate_for_traits ---
    if mutation_rate_for_traits > 0.0 {
        // Lifetimes
        config.min_lifetime = mutate_value(
            rng,
            config.min_lifetime,
            mutation_rate_for_traits,
            MIN_LIFETIME_CLAMP,
        );
        config.max_lifetime = mutate_value(
            rng,
            config.max_lifetime,
            mutation_rate_for_traits,
            MIN_LIFETIME_CLAMP,
        );
        if config.min_lifetime > config.max_lifetime {
            std::mem::swap(&mut config.min_lifetime, &mut config.max_lifetime);
        }
        // Factors/Rates
        config.base_growth_rate = mutate_value(
            rng,
            config.base_growth_rate,
            mutation_rate_for_traits,
            MIN_FACTOR_CLAMP,
        );
        if config.movement_speed_factor > 0.0 {
            config.movement_speed_factor = mutate_value(
                rng,
                config.movement_speed_factor,
                mutation_rate_for_traits,
                MIN_FACTOR_CLAMP,
            );
        }
        config.perception_radius_factor = mutate_value(
            rng,
            config.perception_radius_factor,
            mutation_rate_for_traits,
            MIN_FACTOR_CLAMP,
        );
        config.eating_radius_factor = mutate_value(
            rng,
            config.eating_radius_factor,
            mutation_rate_for_traits,
            MIN_FACTOR_CLAMP,
        );
        config.influence_weight = mutate_value(
            rng,
            config.influence_weight,
            mutation_rate_for_traits,
            MIN_FACTOR_CLAMP,
        )
            .clamp(0.0, 1.0);
        config.max_turn_angle_per_sec = mutate_value(
            rng,
            config.max_turn_angle_per_sec,
            mutation_rate_for_traits,
            MIN_FACTOR_CLAMP,
        );
        config.eating_spawn_boost_factor_plant = mutate_value(
            rng,
            config.eating_spawn_boost_factor_plant,
            mutation_rate_for_traits,
            1.0, // Base boost is 1.0
        );
        config.eating_spawn_boost_factor_fish = mutate_value(
            rng,
            config.eating_spawn_boost_factor_fish,
            mutation_rate_for_traits,
            1.0,
        );
        config.eating_spawn_boost_factor_bug = mutate_value(
            rng,
            config.eating_spawn_boost_factor_bug,
            mutation_rate_for_traits,
            1.0,
        );
        // --- Mutate corpse eating factors ---
        config.eating_spawn_boost_factor_corpse_a = mutate_value(
            rng,
            config.eating_spawn_boost_factor_corpse_a,
            mutation_rate_for_traits,
            0.0, // Allow zero boost for corpses
        );
        config.eating_spawn_boost_factor_corpse_b = mutate_value(
            rng,
            config.eating_spawn_boost_factor_corpse_b,
            mutation_rate_for_traits,
            0.0,
        );
        config.eating_spawn_boost_factor_corpse_kin = mutate_value(
            rng,
            config.eating_spawn_boost_factor_corpse_kin,
            mutation_rate_for_traits,
            0.0,
        );
        config.clustering_radius_factor = mutate_value(
            rng,
            config.clustering_radius_factor,
            mutation_rate_for_traits,
            MIN_FACTOR_CLAMP,
        );
        config.aging_rate_when_eaten = mutate_value(
            rng,
            config.aging_rate_when_eaten,
            mutation_rate_for_traits,
            MIN_FACTOR_CLAMP,
        );
        config.organism_min_pushaway_radius = mutate_value(
            rng,
            config.organism_min_pushaway_radius,
            mutation_rate_for_traits,
            MIN_FACTOR_CLAMP,
        );
        // Influences (Alive)
        config.influence_plant =
            mutate_influence(rng, config.influence_plant, mutation_rate_for_traits);
        config.influence_fish =
            mutate_influence(rng, config.influence_fish, mutation_rate_for_traits);
        config.influence_bug =
            mutate_influence(rng, config.influence_bug, mutation_rate_for_traits);
        config.influence_offspring =
            mutate_influence(rng, config.influence_offspring, mutation_rate_for_traits);
        // --- NEW: Influences (Corpses) ---
        config.influence_corpse_a =
            mutate_influence(rng, config.influence_corpse_a, mutation_rate_for_traits);
        config.influence_corpse_b =
            mutate_influence(rng, config.influence_corpse_b, mutation_rate_for_traits);
        config.influence_corpse_kin =
            mutate_influence(rng, config.influence_corpse_kin, mutation_rate_for_traits);
        // --- End NEW ---
        // Radii
        config.min_radius = mutate_value(
            rng,
            config.min_radius,
            mutation_rate_for_traits,
            MIN_RADIUS_CLAMP,
        );
        config.max_radius = mutate_value(
            rng,
            config.max_radius,
            mutation_rate_for_traits,
            MIN_RADIUS_CLAMP,
        );
        if config.min_radius > config.max_radius {
            std::mem::swap(&mut config.min_radius, &mut config.max_radius);
        }
        // Color
        let color_mutation_delta = mutation_rate_for_traits * config.color_mutation_sensitivity;
        config.base_color =
            mutate_color(Vec4::from(config.base_color), rng, color_mutation_delta).into();
    }

    // --- 2. Mutate the mutation rates themselves using meta_mutation_rate ---
    if meta_mutation_rate > 0.0 {
        config.current_dna_mutation_rate = mutate_value(
            rng,
            config.current_dna_mutation_rate,
            meta_mutation_rate,
            ABSOLUTE_MIN_MUTATION_RATE,
        )
            .clamp(config.min_dna_mutation_rate, config.max_dna_mutation_rate)
            .clamp(ABSOLUTE_MIN_MUTATION_RATE, ABSOLUTE_MAX_MUTATION_RATE);

        config.current_dna_spawn_mutation_rate = mutate_value(
            rng,
            config.current_dna_spawn_mutation_rate,
            meta_mutation_rate,
            ABSOLUTE_MIN_MUTATION_RATE,
        )
            .clamp(
                config.min_dna_spawn_mutation_rate,
                config.max_dna_spawn_mutation_rate,
            )
            .clamp(ABSOLUTE_MIN_MUTATION_RATE, ABSOLUTE_MAX_MUTATION_RATE);
    }
}

impl SimulationState {
    pub fn new(window_size: PhysicalSize<u32>, config: SimulationConfig) -> Self {
        let avg_radius = config.get_avg_base_radius();
        let grid_cell_size = (avg_radius * GRID_CELL_SIZE_FACTOR).max(1.0);
        let grid_width = ((window_size.width as f32 / grid_cell_size).ceil() as i32).max(1);
        let grid_height = ((window_size.height as f32 / grid_cell_size).ceil() as i32).max(1);
        let num_grid_cells = (grid_width * grid_height) as usize;

        let initial_capacity =
            (INITIAL_PLANT_COUNT + INITIAL_FISH_COUNT + INITIAL_BUG_COUNT).max(256);
        let removal_capacity = initial_capacity / 10;
        let new_org_capacity = initial_capacity / 5;

        NEXT_ORGANISM_ID.store(1, Ordering::Relaxed);

        let mut state = Self {
            organisms: Vec::with_capacity(MAX_ORGANISMS / 2),
            rng: SimRng::from_entropy(),
            window_size,
            speed_multiplier: INITIAL_SPEED_MULTIPLIER,
            is_paused: false,
            config,
            grid: HashMap::new(),
            grid_width,
            grid_height,
            grid_cell_size,
            new_organism_buffer: Vec::with_capacity(new_org_capacity),
            removal_indices_buffer: Vec::with_capacity(removal_capacity),
            removal_indices_set: HashSet::with_capacity(removal_capacity),
            interaction_radii_buffer: Vec::with_capacity(initial_capacity),
            accumulated_aging_rate_buffer: Vec::with_capacity(initial_capacity),
            plant_is_clustered_buffer: Vec::with_capacity(initial_capacity),
            gpu_grid_indices: Vec::with_capacity(initial_capacity * 2), // Pre-allocate more for indices
            gpu_grid_offsets: vec![[0, 0]; num_grid_cells],
        };
        state.initialize_organisms();
        state
    }

    fn initialize_organisms(&mut self) {
        self.organisms.clear();
        let total_initial_count = INITIAL_PLANT_COUNT + INITIAL_FISH_COUNT + INITIAL_BUG_COUNT;
        self.organisms.reserve(total_initial_count.max(100));
        let mut local_rng =
            SimRng::from_rng(&mut self.rng).unwrap_or_else(|_| SimRng::from_entropy());
        let base_sim_config = &self.config;

        NEXT_ORGANISM_ID.store(1, Ordering::Relaxed);

        for _ in 0..INITIAL_PLANT_COUNT {
            self.organisms.push(Self::create_organism(
                &mut local_rng,
                self.window_size,
                OrganismType::Plant,
                base_sim_config,
                None,
            ));
        }
        for _ in 0..INITIAL_FISH_COUNT {
            self.organisms.push(Self::create_organism(
                &mut local_rng,
                self.window_size,
                OrganismType::Fish,
                base_sim_config,
                None,
            ));
        }
        for _ in 0..INITIAL_BUG_COUNT {
            self.organisms.push(Self::create_organism(
                &mut local_rng,
                self.window_size,
                OrganismType::Bug,
                base_sim_config,
                None,
            ));
        }
        self.resize_internal_buffers(self.organisms.len());
        let num_grid_cells = (self.grid_width * self.grid_height) as usize;
        if self.gpu_grid_offsets.len() != num_grid_cells {
            self.gpu_grid_offsets.resize(num_grid_cells, [0, 0]);
        }
    }

    fn resize_internal_buffers(&mut self, capacity: usize) {
        if self.interaction_radii_buffer.capacity() < capacity {
            self.interaction_radii_buffer
                .reserve(capacity - self.interaction_radii_buffer.len());
        }
        self.interaction_radii_buffer
            .resize(capacity, Default::default());

        if self.accumulated_aging_rate_buffer.capacity() < capacity {
            self.accumulated_aging_rate_buffer
                .reserve(capacity - self.accumulated_aging_rate_buffer.len());
        }
        self.accumulated_aging_rate_buffer.resize(capacity, 0.0);

        if self.plant_is_clustered_buffer.capacity() < capacity {
            self.plant_is_clustered_buffer
                .reserve(capacity - self.plant_is_clustered_buffer.len());
        }
        self.plant_is_clustered_buffer.resize(capacity, false);
    }

    // --- MODIFIED: Initialize state fields ---
    fn create_organism(
        rng: &mut SimRng,
        window_size: PhysicalSize<u32>,
        kind: OrganismType,
        base_sim_config: &SimulationConfig,
        parent_id: Option<u64>,
    ) -> Organism {
        let mut organism_config = match kind {
            OrganismType::Plant => base_sim_config.plant.clone(),
            OrganismType::Fish => base_sim_config.fish.clone(),
            OrganismType::Bug => base_sim_config.bug.clone(),
        };

        // For *initial* organisms, apply only the *initial* mutation rate (0.0 currently)
        // to avoid starting with heavily mutated defaults.
        // Let spawn mutation handle the initial diversification.
        // If you *want* initial mutation, use: organism_config.current_dna_mutation_rate
        let initial_mutation_rate = 0.0; // Usually 0 for initial population
        let initial_meta_mutation_rate = 0.0; // Meta-mutation usually starts at 0

        mutate_config_dna(
            &mut organism_config,
            rng,
            initial_mutation_rate,
            initial_meta_mutation_rate,
        );

        let position = Vec2::new(
            rng.gen_range(0.0..window_size.width as f32),
            rng.gen_range(0.0..window_size.height as f32),
        );

        let min_radius_dna = organism_config.min_radius;
        let max_radius_dna = organism_config.max_radius;
        let target_radius_base = if min_radius_dna < max_radius_dna {
            rng.gen_range(min_radius_dna..=max_radius_dna)
        } else {
            min_radius_dna
        };
        let target_radius = (target_radius_base * GLOBAL_RADIUS_SCALE_FACTOR).max(MIN_RADIUS_CLAMP);

        let radius = 0.0; // Start at zero radius

        let velocity = if organism_config.movement_speed_factor > 0.0 {
            let angle = rng.gen_range(0.0..TAU);
            Vec2::from_angle(angle) * organism_config.movement_speed_factor * target_radius
        } else {
            Vec2::ZERO
        };

        let min_lifetime_dna = organism_config.min_lifetime;
        let max_lifetime_dna = organism_config.max_lifetime;
        let lifetime = if min_lifetime_dna < max_lifetime_dna {
            rng.gen_range(min_lifetime_dna..max_lifetime_dna)
        } else {
            min_lifetime_dna
        };

        let growth_rate = organism_config.base_growth_rate;
        let color = Vec4::from(organism_config.base_color);

        Organism {
            id: get_new_organism_id(),
            parent_id,
            kind,
            position,
            velocity,
            age: 0.0,
            lifetime,
            growth_rate,
            color,
            radius,
            target_radius,
            state: OrganismState::Alive, // Start as Alive
            original_type: None,         // No original type initially
            decay_timer: 0.0,            // No decay initially
            config: organism_config,
        }
    }

    // --- MODIFIED: Initialize state fields for offspring ---
    fn create_offspring<R: Rng>(
        parent: &Organism,
        window_size: PhysicalSize<u32>,
        rng: &mut R,
        _base_sim_config: &SimulationConfig, // Still passed but not directly used
    ) -> Organism {
        let mut offspring_config = parent.config.clone();

        // Apply the *parent's* current spawn mutation rate and meta-mutation rate
        let mutation_rate_for_traits = parent.config.current_dna_spawn_mutation_rate;
        let meta_mutation_rate = parent.config.current_dna_spawn_mutation_rate;

        mutate_config_dna(
            &mut offspring_config,
            rng,
            mutation_rate_for_traits,
            meta_mutation_rate,
        );

        let spawn_offset_dist = parent.radius * SPAWN_OFFSET_RADIUS_FACTOR;
        let angle_offset = rng.gen_range(0.0..TAU);
        let offset = Vec2::from_angle(angle_offset) * spawn_offset_dist;
        let mut position = parent.position + offset;
        // Clamp position to window bounds immediately after offset
        position.x = position.x.clamp(0.0, (window_size.width as f32) - 1.0);
        position.y = position.y.clamp(0.0, (window_size.height as f32) - 1.0);

        let min_radius_dna = offspring_config.min_radius;
        let max_radius_dna = offspring_config.max_radius;
        let offspring_target_radius_base = if min_radius_dna < max_radius_dna {
            rng.gen_range(min_radius_dna..=max_radius_dna)
        } else {
            min_radius_dna
        };
        let target_radius =
            (offspring_target_radius_base * GLOBAL_RADIUS_SCALE_FACTOR).max(MIN_RADIUS_CLAMP);

        let radius = 0.0; // Start offspring at zero radius

        let velocity = if offspring_config.movement_speed_factor > 0.0 {
            let angle = rng.gen_range(0.0..TAU);
            Vec2::from_angle(angle) * offspring_config.movement_speed_factor * target_radius
        } else {
            Vec2::ZERO
        };

        let min_lifetime_dna = offspring_config.min_lifetime;
        let max_lifetime_dna = offspring_config.max_lifetime;
        let lifetime = if min_lifetime_dna < max_lifetime_dna {
            rng.gen_range(min_lifetime_dna..max_lifetime_dna)
        } else {
            min_lifetime_dna
        };

        let growth_rate = offspring_config.base_growth_rate;
        let color = Vec4::from(offspring_config.base_color);

        Organism {
            id: get_new_organism_id(),
            parent_id: Some(parent.id),
            kind: parent.kind,
            position,
            velocity,
            age: 0.0,
            lifetime,
            growth_rate,
            color,
            radius,
            target_radius,
            state: OrganismState::Alive, // Start as Alive
            original_type: None,         // No original type initially
            decay_timer: 0.0,            // No decay initially
            config: offspring_config,
        }
    }

    #[inline]
    fn get_grid_key(&self, position: Vec2) -> GridKey {
        let cell_x = (position.x / self.grid_cell_size).floor() as i32;
        let cell_y = (position.y / self.grid_cell_size).floor() as i32;
        (
            cell_x.clamp(0, self.grid_width - 1),
            cell_y.clamp(0, self.grid_height - 1),
        )
    }

    // --- MODIFIED: Build grid includes all organisms (Alive and Corpses) ---
    fn build_grid(&mut self) {
        self.grid.clear();
        self.gpu_grid_indices.clear(); // Indices are rebuilt completely

        // Estimate required capacity to reduce reallocations
        let estimated_indices = self.organisms.len().max(100); // Simple estimate
        if self.gpu_grid_indices.capacity() < estimated_indices {
            self.gpu_grid_indices.reserve(estimated_indices);
        }

        // Ensure gpu_grid_offsets has the correct size
        let num_grid_cells = (self.grid_width * self.grid_height) as usize;
        if self.gpu_grid_offsets.len() != num_grid_cells {
            log::warn!(
                "Resizing gpu_grid_offsets in build_grid ({} -> {})",
                self.gpu_grid_offsets.len(),
                num_grid_cells
            );
            self.gpu_grid_offsets.resize(num_grid_cells, [0, 0]);
        } else {
            // Clear existing offsets if size is already correct
            for offset in self.gpu_grid_offsets.iter_mut() {
                *offset = [0, 0]; // Reset count and offset
            }
        }

        // Use a temporary structure better suited for building the GPU grid data
        // Map: flat_cell_index -> list_of_organism_indices_in_that_cell
        let mut temp_gpu_grid: HashMap<usize, Vec<u32>> =
            HashMap::with_capacity(num_grid_cells / 4); // Guess capacity

        let grid_w = self.grid_width;

        for (index, organism) in self.organisms.iter().enumerate() {
            // Include Alive and Corpses in the grid
            // Skip only very small newly spawned organisms that haven't grown yet
            if organism.radius < REMOVAL_RADIUS_THRESHOLD && organism.state == OrganismState::Alive
            {
                continue;
            }

            let key @ (cell_x, cell_y) = self.get_grid_key(organism.position);

            // CPU Grid update (includes corpses)
            self.grid.entry(key).or_default().push(index);

            // GPU Grid update (includes corpses - they will be rendered)
            let flat_index = (cell_x + cell_y * grid_w) as usize;

            // Bounds check before accessing temp_gpu_grid or gpu_grid_offsets
            if flat_index < num_grid_cells {
                temp_gpu_grid
                    .entry(flat_index)
                    .or_default()
                    .push(index as u32);
            } else {
                log::error!(
                    "Calculated flat grid index {} out of bounds (size {}) for organism at {:?}. Check get_grid_key logic.",
                    flat_index,
                    num_grid_cells,
                    organism.position
                );
            }
        }

        // Now, populate gpu_grid_offsets and gpu_grid_indices from temp_gpu_grid
        let mut current_offset = 0u32;
        // Iterate through cell indices 0..num_cells
        for i in 0..num_grid_cells {
            if let Some(indices_in_cell) = temp_gpu_grid.get(&i) {
                let count = indices_in_cell.len() as u32;
                if count > 0 {
                    self.gpu_grid_offsets[i] = [current_offset, count];
                    self.gpu_grid_indices.extend_from_slice(indices_in_cell);
                    current_offset += count;
                } else {
                    self.gpu_grid_offsets[i] = [current_offset, 0];
                }
            } else {
                self.gpu_grid_offsets[i] = [current_offset, 0];
            }
        }
    }

    // --- MODIFIED: Core update logic includes corpse influence ---
    pub fn update(&mut self, delta_time: f32) {
        if self.is_paused {
            return;
        }
        let dt = delta_time * self.speed_multiplier;
        if dt <= 0.0 {
            return;
        }

        let current_organism_count = self.organisms.len();
        if current_organism_count == 0 {
            return;
        }

        // --- Prepare Buffers ---
        self.new_organism_buffer.clear();
        self.removal_indices_buffer.clear();
        self.removal_indices_set.clear();
        self.resize_internal_buffers(current_organism_count);

        // Clear per-organism temporary buffers
        self.accumulated_aging_rate_buffer.fill(0.0); // Only used for Alive prey
        self.plant_is_clustered_buffer.fill(false); // Only used for Alive plants

        self.build_grid(); // Build grid based on current positions (includes corpses)

        // Pre-calculate interaction radii (only needed for Alive organisms)
        for i in 0..current_organism_count {
            // Borrow immutably first to check state
            let org_state = self.organisms[i].state;
            if i < self.interaction_radii_buffer.len() {
                if org_state == OrganismState::Alive {
                    // Borrow immutably again for other fields
                    let org = &self.organisms[i];
                    let radius = org.radius;
                    let pushaway_radius = org.config.organism_min_pushaway_radius.max(0.0);
                    self.interaction_radii_buffer[i] = match org.kind {
                        OrganismType::Plant => InteractionRadii {
                            perception_sq: 0.0,
                            eating_sq: 0.0, // Plants don't eat
                            clustering_sq: (radius * org.config.clustering_radius_factor).powi(2),
                            pushaway_sq: pushaway_radius.powi(2),
                        },
                        OrganismType::Fish | OrganismType::Bug => InteractionRadii {
                            perception_sq: (radius * org.config.perception_radius_factor).powi(2),
                            eating_sq: (radius * org.config.eating_radius_factor).powi(2),
                            clustering_sq: 0.0,
                            pushaway_sq: pushaway_radius.powi(2),
                        },
                    };
                } else {
                    // Corpses don't interact actively, set radii to zero
                    self.interaction_radii_buffer[i] = Default::default();
                }
            } else {
                log::error!(
                    "Index {} out of bounds for interaction_radii_buffer (len {})",
                    i,
                    self.interaction_radii_buffer.len()
                );
            }
        }

        // --- Parallel Calculation (Movement Influence & Potential Prey Aging) ---
        // This part calculates influences between ALIVE organisms and towards CORPSES,
        // and potential aging effects applied *to* ALIVE prey.
        let organisms_ref = &self.organisms;
        let grid_ref = &self.grid;
        let radii_ref = &self.interaction_radii_buffer; // Contains radii only for Alive orgs
        let grid_width = self.grid_width;
        let grid_height = self.grid_height;
        let grid_cell_size = self.grid_cell_size;

        let calculation_results: Vec<OrganismCalculationResult> = (0..current_organism_count)
            .into_par_iter()
            .map(|i| {
                let mut result = OrganismCalculationResult {
                    index: i,
                    ..Default::default()
                };

                // Basic bounds check
                if i >= organisms_ref.len() {
                    log::error!(
                        "Parallel calc: Index {} out of bounds for organisms_ref (len {})",
                        i,
                        organisms_ref.len()
                    );
                    return result;
                }

                let organism_i = &organisms_ref[i];

                // --- Skip calculation entirely if not Alive or cannot move/perceive ---
                if organism_i.state != OrganismState::Alive {
                    return result;
                }
                if organism_i.config.movement_speed_factor == 0.0 &&
                    organism_i.config.organism_min_pushaway_radius == 0.0 &&
                    organism_i.config.perception_radius_factor == 0.0 {
                    return result; // Plants etc. don't need velocity calculated
                }


                let pos_i = organism_i.position;
                let radii_i = if i < radii_ref.len() {
                    radii_ref[i]
                } else {
                    Default::default()
                };

                // --- Skip if no movement, perception or pushaway capabilities ---
                if radii_i.pushaway_sq == 0.0 && radii_i.perception_sq == 0.0 {
                    return result;
                }


                let cell_x = (pos_i.x / grid_cell_size).floor() as i32;
                let cell_y = (pos_i.y / grid_cell_size).floor() as i32;
                let grid_key_i = (
                    cell_x.clamp(0, grid_width - 1),
                    cell_y.clamp(0, grid_height - 1),
                );

                let mut influence_vector = Vec2::ZERO;
                let mut _neighbor_count = 0;
                let mut thread_rng = thread_rng(); // Need for random velocity component
                let mut pushed_away_this_tick = false;

                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let check_key = (grid_key_i.0 + dx, grid_key_i.1 + dy);
                        let clamped_check_key = (
                            check_key.0.clamp(0, grid_width - 1),
                            check_key.1.clamp(0, grid_height - 1),
                        );

                        if let Some(neighbor_indices) = grid_ref.get(&clamped_check_key) {
                            for &j in neighbor_indices {
                                // Basic index checks
                                if i == j || j >= organisms_ref.len() {
                                    continue;
                                }

                                let organism_j = &organisms_ref[j];
                                let pos_j = organism_j.position;
                                let vec_ij = pos_j - pos_i;
                                let dist_sq = vec_ij.length_squared();

                                // --- 1. Push-away Check (Alive vs Alive Only) ---
                                if radii_i.pushaway_sq > 0.0
                                    && organism_j.state == OrganismState::Alive // Only push away from Alive
                                    && dist_sq < radii_i.pushaway_sq
                                    && dist_sq > 1e-9
                                {
                                    let push_direction = -vec_ij.normalize_or_zero();
                                    let dist = dist_sq.sqrt();
                                    let safe_dist = dist.max(1e-5);
                                    let force_scale = 1.0 / safe_dist;
                                    influence_vector +=
                                        push_direction * PUSH_AWAY_FORCE_MAGNITUDE * force_scale;
                                    pushed_away_this_tick = true;
                                }

                                // --- 2. Perception-based Influence (Alive towards Alive OR Corpse) ---
                                // Only if not pushed, only within perception radius
                                if !pushed_away_this_tick
                                    && radii_i.perception_sq > 0.0
                                    && dist_sq < radii_i.perception_sq
                                    && dist_sq > 1e-6
                                {
                                    let mut type_influence_factor = 0.0;
                                    let mut is_offspring_influence = false;

                                    match organism_j.state {
                                        OrganismState::Alive => {
                                            // Check if it's offspring first
                                            if let Some(parent_id_j) = organism_j.parent_id {
                                                if parent_id_j == organism_i.id {
                                                    type_influence_factor =
                                                        organism_i.config.influence_offspring;
                                                    is_offspring_influence = true;
                                                }
                                            }
                                            // If not offspring, use standard Alive influences
                                            if !is_offspring_influence {
                                                type_influence_factor = match organism_j.kind {
                                                    OrganismType::Plant => {
                                                        organism_i.config.influence_plant
                                                    }
                                                    OrganismType::Fish => {
                                                        organism_i.config.influence_fish
                                                    }
                                                    OrganismType::Bug => {
                                                        organism_i.config.influence_bug
                                                    }
                                                };
                                            }
                                        }
                                        OrganismState::CorpseA | OrganismState::CorpseB => {
                                            // Determine corpse influence
                                            let original_kind_j = organism_j.original_type.unwrap_or(organism_j.kind); // Fallback just in case
                                            let is_kin_corpse = organism_i.kind == original_kind_j;

                                            if is_kin_corpse {
                                                type_influence_factor = organism_i.config.influence_corpse_kin;
                                            } else if organism_j.state == OrganismState::CorpseA {
                                                type_influence_factor = organism_i.config.influence_corpse_a;
                                            } else { // CorpseB
                                                type_influence_factor = organism_i.config.influence_corpse_b;
                                            }
                                        }
                                    }

                                    // Apply the calculated influence
                                    if type_influence_factor.abs() > 1e-6 {
                                        influence_vector +=
                                            vec_ij.normalize_or_zero() * type_influence_factor;
                                        _neighbor_count += 1;
                                    }
                                }

                                // --- 3. Potential Eating -> Prey Aging (Alive eats Alive Only) ---
                                if organism_i.state == OrganismState::Alive && organism_j.state == OrganismState::Alive {
                                    if radii_i.eating_sq > 0.0 && dist_sq < radii_i.eating_sq {
                                        let prey_aging_rate = organism_j.config.aging_rate_when_eaten;
                                        if prey_aging_rate > 0.0 {
                                            result.prey_aging_effects.push((j, prey_aging_rate));
                                        }
                                    }
                                }


                                pushed_away_this_tick = false; // Reset for next neighbor
                            } // End neighbor loop
                        } // End if cell exists
                    } // End dy loop
                } // End dx loop

                // --- Calculate New Velocity (Only if capable of moving) ---
                if organism_i.config.movement_speed_factor > 0.0 {
                    let current_radius_i = organism_i.radius;
                    let random_angle = thread_rng.gen_range(0.0..TAU);
                    let random_direction = Vec2::from_angle(random_angle);

                    let normalized_influence = if influence_vector.length_squared() > 1e-6 {
                        influence_vector.normalize()
                    } else {
                        organism_i
                            .velocity
                            .try_normalize()
                            .unwrap_or(random_direction)
                    };

                    let influence_weight = organism_i.config.influence_weight.clamp(0.0, 1.0);
                    let desired_direction = (random_direction * (1.0 - influence_weight)
                        + normalized_influence * influence_weight)
                        .normalize_or_zero();
                    let current_dir = organism_i.velocity.normalize_or_zero();
                    let angle_diff = current_dir.angle_between(desired_direction);
                    let max_turn = organism_i.config.max_turn_angle_per_sec * dt;
                    let turn_amount = angle_diff.clamp(-max_turn, max_turn);

                    let final_direction = if current_dir.length_squared() > 1e-6 {
                        Vec2::from_angle(current_dir.to_angle() + turn_amount)
                    } else {
                        desired_direction
                    };
                    result.new_velocity = Some(
                        final_direction
                            * organism_i.config.movement_speed_factor
                            * current_radius_i.max(0.1),
                    );
                } else {
                    // If not capable of moving, ensure velocity remains zero
                    result.new_velocity = Some(Vec2::ZERO);
                }
                result
            })
            .collect();

        // --- Apply Results Serially ---

        // 1. Aggregate prey aging effects from parallel results
        for result in &calculation_results {
            for &(prey_index, aging_rate) in &result.prey_aging_effects {
                if prey_index < self.accumulated_aging_rate_buffer.len() {
                    if prey_index < self.organisms.len()
                        && self.organisms[prey_index].state == OrganismState::Alive
                    {
                        self.accumulated_aging_rate_buffer[prey_index] += aging_rate;
                    }
                } else {
                    log::warn!(
                        "Serial apply aging: Prey index {} out of bounds (len {})",
                        prey_index,
                        self.accumulated_aging_rate_buffer.len()
                    );
                }
            }
        }

        let organism_count_at_start_of_serial = self.organisms.len();
        let window_w = self.window_size.width as f32;
        let window_h = self.window_size.height as f32;
        let max_organisms = MAX_ORGANISMS;
        let sim_config_ref = &self.config;
        let mut spawn_rng = thread_rng();

        // --- 2. Serial Update Loop (Restructured) ---
        for i in 0..organism_count_at_start_of_serial {
            // Skip if out of bounds (due to swap_remove) or already marked for removal this frame
            if i >= self.organisms.len() || self.removal_indices_set.contains(&i) {
                continue;
            }

            let organism_state = self.organisms[i].state; // Immutable borrow to check state

            // --- Handle Corpses (Decay & Removal) ---
            if organism_state == OrganismState::CorpseA || organism_state == OrganismState::CorpseB
            {
                // Borrow mutably ONLY to update decay timer
                let decay_remaining = {
                    // Scope the mutable borrow
                    let corpse = &mut self.organisms[i];
                    corpse.decay_timer -= dt;
                    corpse.decay_timer
                };
                if decay_remaining <= 0.0 {
                    self.removal_indices_set.insert(i); // Mark for removal
                }
                continue; // Skip ALL other logic for corpses
            }

            // --- Organism is ALIVE ---
            let mut died_this_tick = false;

            // --- Aging & Death Transition ---
            {
                // Scope for mutable borrow
                let organism_mut = &mut self.organisms[i];
                let base_age_increase = dt;
                let extra_aging_rate = if i < self.accumulated_aging_rate_buffer.len() {
                    self.accumulated_aging_rate_buffer[i]
                } else {
                    0.0
                };
                let effective_age_increase = base_age_increase * (1.0 + extra_aging_rate);
                organism_mut.age += effective_age_increase;

                if organism_mut.age >= organism_mut.lifetime {
                    // Transition to Corpse
                    organism_mut.original_type = Some(organism_mut.kind);
                    let corpse_type = if organism_mut.config.movement_speed_factor == 0.0 {
                        OrganismState::CorpseA
                    } else {
                        OrganismState::CorpseB
                    };
                    organism_mut.state = corpse_type;
                    organism_mut.decay_timer = CORPSE_LIFETIME;
                    organism_mut.velocity = Vec2::ZERO; // Stop movement
                    // Set fixed corpse color
                    let corpse_color_const = if corpse_type == OrganismState::CorpseA {
                        CORPSE_A_COLOR
                    } else {
                        CORPSE_B_COLOR
                    };
                    organism_mut.color = Vec4::new(
                        corpse_color_const.r as f32,
                        corpse_color_const.g as f32,
                        corpse_color_const.b as f32,
                        corpse_color_const.a as f32,
                    );
                    died_this_tick = true;
                }
            } // Mutable borrow ends here

            if died_this_tick {
                continue; // Skip rest of alive logic for this tick
            }

            // --- Grow Radius ---
            {
                // Scope for mutable borrow
                let organism_mut = &mut self.organisms[i];
                if organism_mut.radius < organism_mut.target_radius {
                    let growth_amount = (organism_mut.target_radius - organism_mut.radius)
                        * SPAWN_GROWTH_FACTOR
                        * dt;
                    organism_mut.radius =
                        (organism_mut.radius + growth_amount).min(organism_mut.target_radius);
                }
            } // Mutable borrow ends here

            // --- Apply Velocity (from parallel results) ---
            if i < calculation_results.len() {
                if let Some(new_vel) = calculation_results[i].new_velocity {
                    // Scope for mutable borrow
                    let organism_mut = &mut self.organisms[i];
                    organism_mut.velocity = new_vel;
                }
            } else {
                log::warn!(
                    "Serial apply velocity: Index {} out of bounds for results (len {})",
                    i,
                    calculation_results.len()
                );
            }

            // --- Update Position ---
            {
                // Scope for mutable borrow
                let organism_mut = &mut self.organisms[i];
                if organism_mut.velocity != Vec2::ZERO {
                    organism_mut.position += organism_mut.velocity * dt;
                    // Wrap around screen edges
                    organism_mut.position.x = (organism_mut.position.x + window_w) % window_w;
                    organism_mut.position.y = (organism_mut.position.y + window_h) % window_h;
                    // Clamp just in case (floating point issues)
                    organism_mut.position.x = organism_mut.position.x.max(0.0).min(window_w - 1.0);
                    organism_mut.position.y = organism_mut.position.y.max(0.0).min(window_h - 1.0);
                }
            } // Mutable borrow ends here

            // --- Eating Action & Spawn Boost Calculation ---
            let mut actual_spawn_boost_obtained = 1.0f32; // Base boost
            let mut is_plant_clustered = false; // Reset per organism
            {
                // Scope for immutable borrows needed for calculations
                let eater = &self.organisms[i];
                let eater_kind = eater.kind;
                let eater_config = &eater.config; // Borrow config immutably
                let eater_pos = eater.position;
                let radii_i = if i < self.interaction_radii_buffer.len() {
                    self.interaction_radii_buffer[i]
                } else {
                    Default::default()
                };

                // --- Eating Check ---
                if radii_i.eating_sq > 0.0 {
                    let grid_key_i = self.get_grid_key(eater_pos);
                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            let check_key = (grid_key_i.0 + dx, grid_key_i.1 + dy);
                            let clamped_check_key = (
                                check_key.0.clamp(0, self.grid_width - 1),
                                check_key.1.clamp(0, self.grid_height - 1),
                            );

                            if let Some(neighbor_indices) = self.grid.get(&clamped_check_key) {
                                for &j in neighbor_indices {
                                    if i == j
                                        || j >= self.organisms.len()
                                        || self.removal_indices_set.contains(&j)
                                    {
                                        continue;
                                    }

                                    let prey = &self.organisms[j];
                                    let prey_pos = prey.position;
                                    let dist_sq = prey_pos.distance_squared(eater_pos);

                                    if dist_sq < radii_i.eating_sq {
                                        let prey_state = prey.state;
                                        let boost_factor = match prey_state {
                                            OrganismState::Alive => match prey.kind {
                                                OrganismType::Plant => {
                                                    eater_config.eating_spawn_boost_factor_plant
                                                }
                                                OrganismType::Fish => {
                                                    eater_config.eating_spawn_boost_factor_fish
                                                }
                                                OrganismType::Bug => {
                                                    eater_config.eating_spawn_boost_factor_bug
                                                }
                                            },
                                            OrganismState::CorpseA | OrganismState::CorpseB => {
                                                // Mark corpse 'j' for removal
                                                self.removal_indices_set.insert(j);
                                                // Calculate corpse eating boost
                                                let original_prey_kind =
                                                    prey.original_type.unwrap_or(prey.kind);
                                                if eater_kind == original_prey_kind {
                                                    eater_config
                                                        .eating_spawn_boost_factor_corpse_kin
                                                } else if prey_state == OrganismState::CorpseA {
                                                    eater_config.eating_spawn_boost_factor_corpse_a
                                                } else { // CorpseB
                                                    eater_config.eating_spawn_boost_factor_corpse_b
                                                }
                                            }
                                        };
                                        actual_spawn_boost_obtained =
                                            actual_spawn_boost_obtained.max(boost_factor);
                                    }
                                } // End neighbor loop
                            } // End if cell exists
                        } // End dy loop
                    } // End dx loop
                } // End if eating radius > 0

                // --- Plant Clustering Check ---
                if eater_kind == OrganismType::Plant && radii_i.clustering_sq > 0.0 {
                    let grid_key_i = self.get_grid_key(eater_pos);
                    if let Some(neighbor_indices) = self.grid.get(&grid_key_i) {
                        for &j in neighbor_indices {
                            if i == j || j >= self.organisms.len() { continue; } // Check bounds
                            let neighbor = &self.organisms[j];
                            if neighbor.state == OrganismState::Alive
                                && neighbor.kind == OrganismType::Plant
                            {
                                let dist_sq = neighbor.position.distance_squared(eater_pos);
                                if dist_sq < radii_i.clustering_sq {
                                    is_plant_clustered = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            } // End scope for immutable borrows for calculations

            // Update clustering buffer (safe here, uses index `i`)
            if i < self.plant_is_clustered_buffer.len() {
                self.plant_is_clustered_buffer[i] = is_plant_clustered;
            }

            // --- Spawning Check ---
            let (growth_rate, lifetime, kind) = {
                let organism = &self.organisms[i];
                (organism.growth_rate, organism.lifetime, organism.kind)
            };

            let base_prob_per_sec = growth_rate / lifetime.max(1.0);
            let spawn_prob_this_tick = base_prob_per_sec * dt * actual_spawn_boost_obtained;

            if spawn_prob_this_tick > 0.0
                && spawn_rng.gen_bool(spawn_prob_this_tick.clamp(0.0, 1.0) as f64)
            {
                let current_removals = self.removal_indices_set.len();
                let current_additions_buffered = self.new_organism_buffer.len();
                let potential_next_count =
                    self.organisms.len() - current_removals + current_additions_buffered + 1;

                if potential_next_count <= max_organisms {
                    let parent_organism = &self.organisms[i];
                    self.new_organism_buffer.push(Self::create_offspring(
                        parent_organism,
                        self.window_size,
                        &mut spawn_rng,
                        sim_config_ref,
                    ));
                }
            }
        } // End serial loop `for i`

        // --- Perform Removals and Additions ---
        self.removal_indices_buffer.clear();
        self.removal_indices_buffer
            .extend(self.removal_indices_set.iter());
        self.removal_indices_buffer
            .sort_unstable_by(|a, b| b.cmp(a)); // Sort descending

        // Perform swap_remove
        for &index_to_remove in &self.removal_indices_buffer {
            if index_to_remove < self.organisms.len() {
                self.organisms.swap_remove(index_to_remove);
            } else {
                log::warn!(
                    "Attempted swap_remove on index {} which was likely invalid or already removed (list len {}).",
                    index_to_remove, self.organisms.len()
                 );
            }
        }

        // Add new organisms
        self.organisms.extend(self.new_organism_buffer.drain(..));

        // Enforce maximum count
        if self.organisms.len() > max_organisms {
            log::warn!(
                "Exceeded MAX_ORGANISMS ({}), truncating to {}",
                self.organisms.len(),
                max_organisms
            );
            self.organisms.truncate(max_organisms);
        }

        // Ensure internal buffers match the *final* organism count for the next frame
        self.resize_internal_buffers(self.organisms.len());
    }

    pub fn adjust_speed(&mut self, increase: bool) {
        self.speed_multiplier = if increase {
            (self.speed_multiplier + SPEED_ADJUST_FACTOR).min(MAX_SPEED_MULTIPLIER)
        } else {
            (self.speed_multiplier - SPEED_ADJUST_FACTOR).max(MIN_SPEED_MULTIPLIER)
        };
        log::info!("Speed Multiplier: {:.2}", self.speed_multiplier);
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 && self.window_size != new_size {
            self.window_size = new_size;

            let avg_radius = self.config.get_avg_base_radius();
            self.grid_cell_size = (avg_radius * GRID_CELL_SIZE_FACTOR).max(1.0);
            self.grid_width = ((new_size.width as f32 / self.grid_cell_size).ceil() as i32).max(1);
            self.grid_height =
                ((new_size.height as f32 / self.grid_cell_size).ceil() as i32).max(1);

            self.grid.clear(); // Clear CPU grid
            let num_grid_cells = (self.grid_width * self.grid_height) as usize;

            if self.gpu_grid_offsets.len() != num_grid_cells {
                self.gpu_grid_offsets.resize(num_grid_cells, [0, 0]);
                self.gpu_grid_indices.clear();
                log::info!(
                    "Resized GPU grid offsets buffer to {} cells due to window resize.",
                    num_grid_cells
                );
            }

            log::info!(
                "Resized simulation area to {}x{}, grid {}x{} (cell size ~{:.1})",
                new_size.width,
                new_size.height,
                self.grid_width,
                self.grid_height,
                self.grid_cell_size
            );
        }
    }

    pub fn toggle_pause(&mut self) {
        self.is_paused = !self.is_paused;
        log::info!(
            "Simulation {}",
            if self.is_paused { "Paused" } else { "Resumed" }
        );
    }

    pub fn restart(&mut self) {
        log::info!("Restarting simulation...");
        self.rng = SimRng::from_entropy();
        NEXT_ORGANISM_ID.store(1, Ordering::Relaxed);

        // Recalculate grid params
        let avg_radius = self.config.get_avg_base_radius();
        self.grid_cell_size = (avg_radius * GRID_CELL_SIZE_FACTOR).max(1.0);
        self.grid_width =
            ((self.window_size.width as f32 / self.grid_cell_size).ceil() as i32).max(1);
        self.grid_height =
            ((self.window_size.height as f32 / self.grid_cell_size).ceil() as i32).max(1);

        self.initialize_organisms(); // Re-populates self.organisms

        self.speed_multiplier = INITIAL_SPEED_MULTIPLIER;
        self.is_paused = false;

        // Clear stateful buffers
        self.grid.clear();
        self.new_organism_buffer.clear();
        self.removal_indices_buffer.clear();
        self.removal_indices_set.clear();
        self.gpu_grid_indices.clear();

        // Ensure GPU grid offsets buffer matches and clear old offsets
        let num_grid_cells = (self.grid_width * self.grid_height) as usize;
        if self.gpu_grid_offsets.len() != num_grid_cells {
            self.gpu_grid_offsets.resize(num_grid_cells, [0, 0]);
        } else {
            self.gpu_grid_offsets.fill([0, 0]);
        }

        log::info!(
            "Restarted with grid {}x{} (cell size ~{:.1})",
            self.grid_width,
            self.grid_height,
            self.grid_cell_size
        );
        // Resize internal buffers to match the *new* initial organism count
        self.resize_internal_buffers(self.organisms.len());
    }

    pub fn speed_multiplier(&self) -> f32 {
        self.speed_multiplier
    }
    pub fn is_paused(&self) -> bool {
        self.is_paused
    }

    // --- MODIFIED: Only count Alive organisms for title bar ---
    pub fn get_organism_counts(&self) -> (usize, usize, usize) {
        let mut plant_count = 0;
        let mut fish_count = 0;
        let mut bug_count = 0;
        for org in &self.organisms {
            // Count only Alive organisms
            if org.state == OrganismState::Alive {
                match org.kind {
                    OrganismType::Plant => plant_count += 1,
                    OrganismType::Fish => fish_count += 1,
                    OrganismType::Bug => bug_count += 1,
                }
            }
        }
        (plant_count, fish_count, bug_count)
    }

    #[inline]
    pub fn get_gpu_grid_indices(&self) -> &[u32] {
        &self.gpu_grid_indices
    }
    #[inline]
    pub fn get_gpu_grid_offsets(&self) -> &[[u32; 2]] {
        &self.gpu_grid_offsets
    }
    #[inline]
    pub fn get_grid_dimensions(&self) -> (i32, i32) {
        (self.grid_width, self.grid_height)
    }
    #[inline]
    pub fn get_grid_cell_size(&self) -> f32 {
        self.grid_cell_size
    }
}