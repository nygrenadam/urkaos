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
// --- NEW: Import Rayon ---
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    f32::consts::TAU,
    sync::atomic::{AtomicU64, Ordering}, // <<< NEW: For unique IDs
};
use winit::dpi::PhysicalSize;

// --- GPU Data Structure ---
// This struct MUST match the layout in the WGSL shader
// `repr(C)` ensures Rust uses a C-compatible memory layout.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OrganismGpuData {
    pub world_position: [f32; 2], // Use fixed-size arrays for Pod/Zeroable compatibility
    pub radius: f32,
    pub _padding1: f32,  // Ensure padding matches shader
    pub color: [f32; 4], // Use fixed-size arrays
}

// --- Core Data Structures ---

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum OrganismType {
    Plant,
    Fish,
    Bug,
}

#[derive(Debug, Clone)] // Clone needed for create_offspring config copy
pub struct Organism {
    pub id: u64,                // <<< NEW: Unique identifier
    pub parent_id: Option<u64>, // <<< NEW: ID of the parent, if any
    pub kind: OrganismType,
    pub position: Vec2,
    pub velocity: Vec2,
    pub age: f32,
    pub lifetime: f32,          // This is now derived from config.min/max_lifetime
    pub growth_rate: f32,       // This is now derived from config.base_growth_rate
    pub color: Vec4,            // This is now derived from config.base_color
    pub radius: f32, // This radius is used for simulation logic (Globally Scaled from config.min/max_radius)
    pub config: OrganismConfig, // Holds the unique, potentially mutated DNA
}

pub type GridKey = (i32, i32);
pub type SimRng = StdRng; // Keep main RNG for seeding if needed, but use thread_rng in parallel loops

// --- NEW: Constant for push-away force magnitude ---
const PUSH_AWAY_FORCE_MAGNITUDE: f32 = 64.0;
// --- NEW: Minimum value clamps for mutated DNA ---
const MIN_LIFETIME_CLAMP: f32 = 1.0;
const MIN_RADIUS_CLAMP: f32 = 0.1;
const MIN_FACTOR_CLAMP: f32 = 0.0; // For rates, speeds, factors etc.

// <<< NEW: Global atomic counter for organism IDs
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
    removal_indices_set: HashSet<usize>, // Also reuse hashset
    interaction_radii_buffer: Vec<InteractionRadii>, // Reuse radii buffer
    accumulated_aging_rate_buffer: Vec<f32>,
    plant_is_clustered_buffer: Vec<bool>,
    eating_boost_buffer: Vec<f32>,

    // --- GPU grid data ---
    gpu_grid_indices: Vec<u32>, // Flat list of organism indices per cell
    gpu_grid_offsets: Vec<[u32; 2]>, // [offset, count] for each cell into gpu_grid_indices
}

#[derive(Debug, Default)]
struct OrganismCalculationResult {
    index: usize,
    new_velocity: Option<Vec2>,
    prey_aging_effects: Vec<(usize, f32)>,
    spawn_boost_obtained: f32,
    is_clustered: bool,
}

#[derive(Default, Clone, Copy)]
struct InteractionRadii {
    perception_sq: f32,
    eating_sq: f32,
    clustering_sq: f32,
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

// --- MODIFIED: mutate_config_dna now also mutates the mutation rates themselves ---
fn mutate_config_dna<R: Rng + ?Sized>(
    config: &mut OrganismConfig,
    rng: &mut R,
    // This is the rate used to mutate *other* traits
    mutation_rate_for_traits: f32,
    // This is the rate used to mutate the mutation rates *themselves*
    // Typically, this might be the spawn mutation rate, so rates evolve slowly.
    meta_mutation_rate: f32,
) {
    if mutation_rate_for_traits <= 0.0 && meta_mutation_rate <= 0.0 {
        return; // Skip if nothing would change
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
            1.0,
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
        // Influences
        config.influence_plant =
            mutate_influence(rng, config.influence_plant, mutation_rate_for_traits);
        config.influence_fish =
            mutate_influence(rng, config.influence_fish, mutation_rate_for_traits);
        config.influence_bug =
            mutate_influence(rng, config.influence_bug, mutation_rate_for_traits);
        // <<< NEW: Mutate offspring influence
        config.influence_offspring =
            mutate_influence(rng, config.influence_offspring, mutation_rate_for_traits);
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
        // Mutate current_dna_mutation_rate and clamp within its own min/max bounds AND absolute bounds
        config.current_dna_mutation_rate = mutate_value(
            rng,
            config.current_dna_mutation_rate,
            meta_mutation_rate,         // Use the meta-rate here
            ABSOLUTE_MIN_MUTATION_RATE, // Absolute minimum clamp
        )
        .clamp(config.min_dna_mutation_rate, config.max_dna_mutation_rate) // Clamp within specific bounds
        .clamp(ABSOLUTE_MIN_MUTATION_RATE, ABSOLUTE_MAX_MUTATION_RATE); // Re-clamp with absolute bounds

        // Mutate current_dna_spawn_mutation_rate and clamp within its own min/max bounds AND absolute bounds
        config.current_dna_spawn_mutation_rate = mutate_value(
            rng,
            config.current_dna_spawn_mutation_rate,
            meta_mutation_rate,         // Use the meta-rate here
            ABSOLUTE_MIN_MUTATION_RATE, // Absolute minimum clamp
        )
        .clamp(
            config.min_dna_spawn_mutation_rate,
            config.max_dna_spawn_mutation_rate,
        ) // Clamp within specific bounds
        .clamp(ABSOLUTE_MIN_MUTATION_RATE, ABSOLUTE_MAX_MUTATION_RATE); // Re-clamp with absolute bounds

        // Optional: Mutate the min/max bounds themselves? (Could lead to instability)
        // For now, let's keep the min/max bounds fixed per organism type's initial setup.
    }
}

impl SimulationState {
    pub fn new(window_size: PhysicalSize<u32>, config: SimulationConfig) -> Self {
        let avg_radius = config.get_avg_base_radius();
        let grid_cell_size = avg_radius * GRID_CELL_SIZE_FACTOR;
        let grid_width = (window_size.width as f32 / grid_cell_size).ceil() as i32;
        let grid_height = (window_size.height as f32 / grid_cell_size).ceil() as i32;
        let num_grid_cells = (grid_width * grid_height).max(1) as usize;

        let initial_capacity =
            (INITIAL_PLANT_COUNT + INITIAL_FISH_COUNT + INITIAL_BUG_COUNT).max(256);
        let removal_capacity = initial_capacity / 10;
        let new_org_capacity = initial_capacity / 5;

        // <<< NEW: Reset ID counter on new simulation
        NEXT_ORGANISM_ID.store(1, Ordering::Relaxed);

        let mut state = Self {
            organisms: Vec::with_capacity(MAX_ORGANISMS / 2),
            rng: SimRng::from_entropy(),
            window_size,
            speed_multiplier: INITIAL_SPEED_MULTIPLIER,
            is_paused: false,
            config, // Store the base config
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
            eating_boost_buffer: Vec::with_capacity(initial_capacity),
            gpu_grid_indices: Vec::with_capacity(initial_capacity),
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

        // <<< NEW: Reset ID counter just before creating initial organisms
        NEXT_ORGANISM_ID.store(1, Ordering::Relaxed);

        for _ in 0..INITIAL_PLANT_COUNT {
            self.organisms.push(Self::create_organism(
                &mut local_rng,
                self.window_size,
                OrganismType::Plant,
                base_sim_config,
                None, // Initial organisms have no parent
            ));
        }
        for _ in 0..INITIAL_FISH_COUNT {
            self.organisms.push(Self::create_organism(
                &mut local_rng,
                self.window_size,
                OrganismType::Fish,
                base_sim_config,
                None, // Initial organisms have no parent
            ));
        }
        for _ in 0..INITIAL_BUG_COUNT {
            self.organisms.push(Self::create_organism(
                &mut local_rng,
                self.window_size,
                OrganismType::Bug,
                base_sim_config,
                None, // Initial organisms have no parent
            ));
        }
        self.resize_internal_buffers(self.organisms.len());
        let num_grid_cells = (self.grid_width * self.grid_height).max(1) as usize;
        if self.gpu_grid_offsets.len() != num_grid_cells {
            self.gpu_grid_offsets.resize(num_grid_cells, [0, 0]);
        }
    }

    fn resize_internal_buffers(&mut self, capacity: usize) {
        if self.interaction_radii_buffer.len() < capacity {
            self.interaction_radii_buffer
                .resize(capacity, Default::default());
        }
        if self.accumulated_aging_rate_buffer.len() < capacity {
            self.accumulated_aging_rate_buffer.resize(capacity, 0.0);
        }
        if self.plant_is_clustered_buffer.len() < capacity {
            self.plant_is_clustered_buffer.resize(capacity, false);
        }
        if self.eating_boost_buffer.len() < capacity {
            self.eating_boost_buffer.resize(capacity, 1.0);
        }
        // No need to truncate buffers; resizing handles capacity, and loops use current organism count
        // self.interaction_radii_buffer.truncate(capacity);
        // self.accumulated_aging_rate_buffer.truncate(capacity);
        // self.plant_is_clustered_buffer.truncate(capacity);
        // self.eating_boost_buffer.truncate(capacity);
    }

    // MODIFIED: Create organism uses its initial rate for mutation and accepts parent ID
    fn create_organism(
        rng: &mut SimRng,
        window_size: PhysicalSize<u32>,
        kind: OrganismType,
        base_sim_config: &SimulationConfig,
        parent_id: Option<u64>, // <<< NEW parameter
    ) -> Organism {
        let mut organism_config = match kind {
            OrganismType::Plant => base_sim_config.plant.clone(),
            OrganismType::Fish => base_sim_config.fish.clone(),
            OrganismType::Bug => base_sim_config.bug.clone(),
        };

        // Read the *initial* mutation rate from the config to use for the first mutation round
        let initial_mutation_rate = organism_config.current_dna_mutation_rate;
        // The meta-mutation rate for the first organism could be 0 or a small value.
        // Using 0 means the initial rate doesn't change in the very first mutation step.
        // Using the spawn rate might be interesting to slightly vary initial rates. Let's use 0 for simplicity.
        let initial_meta_mutation_rate = 0.0; // Don't mutate the rate itself on initial creation

        mutate_config_dna(
            &mut organism_config,
            rng,
            initial_mutation_rate,
            initial_meta_mutation_rate, // Use the rate we just read
        );

        let position = Vec2::new(
            rng.gen_range(0.0..window_size.width as f32),
            rng.gen_range(0.0..window_size.height as f32),
        );

        let min_radius_dna = organism_config.min_radius;
        let max_radius_dna = organism_config.max_radius;
        let initial_radius_base = if min_radius_dna < max_radius_dna {
            rng.gen_range(min_radius_dna..=max_radius_dna)
        } else {
            min_radius_dna
        };

        let radius = (initial_radius_base * GLOBAL_RADIUS_SCALE_FACTOR).max(MIN_RADIUS_CLAMP);

        let velocity = if organism_config.movement_speed_factor > 0.0 {
            let angle = rng.gen_range(0.0..TAU);
            Vec2::from_angle(angle) * organism_config.movement_speed_factor * radius
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
            id: get_new_organism_id(), // <<< NEW: Assign unique ID
            parent_id,                 // <<< NEW: Store parent ID
            kind,
            position,
            velocity,
            age: 0.0,
            lifetime,
            growth_rate,
            color,
            radius,
            config: organism_config, // Store the mutated config
        }
    }

    // MODIFIED: Create offspring uses parent's spawn rate for mutation and sets parent ID
    fn create_offspring<R: Rng>(
        parent: &Organism,
        window_size: PhysicalSize<u32>,
        rng: &mut R,
        _base_sim_config: &SimulationConfig, // Still needed if create_organism was called directly
    ) -> Organism {
        let mut offspring_config = parent.config.clone();

        // Read the *parent's current spawn mutation rate* to use for mutating traits
        let mutation_rate_for_traits = parent.config.current_dna_spawn_mutation_rate;
        // The meta-mutation rate (mutating the rates themselves) should also be the spawn rate
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
        position.x = position.x.clamp(0.0, (window_size.width as f32) - 1.0);
        position.y = position.y.clamp(0.0, (window_size.height as f32) - 1.0);

        let min_radius_dna = offspring_config.min_radius;
        let max_radius_dna = offspring_config.max_radius;
        let offspring_radius_base = if min_radius_dna < max_radius_dna {
            rng.gen_range(min_radius_dna..=max_radius_dna)
        } else {
            min_radius_dna
        };
        let radius = (offspring_radius_base * GLOBAL_RADIUS_SCALE_FACTOR).max(MIN_RADIUS_CLAMP);

        let velocity = if offspring_config.movement_speed_factor > 0.0 {
            let angle = rng.gen_range(0.0..TAU);
            Vec2::from_angle(angle) * offspring_config.movement_speed_factor * radius
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
            id: get_new_organism_id(),  // <<< NEW: Assign unique ID
            parent_id: Some(parent.id), // <<< NEW: Set parent ID
            kind: parent.kind,
            position,
            velocity,
            age: 0.0,
            lifetime,
            growth_rate,
            color,
            radius,
            config: offspring_config, // Store the mutated offspring config
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

    fn build_grid(&mut self) {
        self.grid.clear();
        self.gpu_grid_indices.clear();

        let mut temp_gpu_grid: HashMap<usize, Vec<u32>> = HashMap::new();
        let grid_w = self.grid_width;

        for (index, organism) in self.organisms.iter().enumerate() {
            let key @ (cell_x, cell_y) = self.get_grid_key(organism.position);
            self.grid.entry(key).or_default().push(index);

            let flat_index = (cell_x + cell_y * grid_w) as usize;
            let num_grid_cells = (self.grid_width * self.grid_height).max(1) as usize;
            if self.gpu_grid_offsets.len() != num_grid_cells {
                log::warn!(
                    "Resizing gpu_grid_offsets in build_grid ({} -> {})",
                    self.gpu_grid_offsets.len(),
                    num_grid_cells
                );
                self.gpu_grid_offsets.resize(num_grid_cells, [0, 0]);
            }

            if flat_index < self.gpu_grid_offsets.len() {
                temp_gpu_grid
                    .entry(flat_index)
                    .or_default()
                    .push(index as u32);
            } else {
                log::warn!(
                    "Calculated flat grid index {} out of bounds (size {}) for organism at {:?}. Skipping.",
                    flat_index,
                    self.gpu_grid_offsets.len(),
                    organism.position
                );
            }
        }

        let mut current_offset = 0u32;
        // Iterate using the guaranteed length after potential resize
        let num_cells = self.gpu_grid_offsets.len();
        for i in 0..num_cells {
            if let Some(indices) = temp_gpu_grid.get(&i) {
                let count = indices.len() as u32;
                self.gpu_grid_offsets[i] = [current_offset, count];
                self.gpu_grid_indices.extend_from_slice(indices);
                current_offset += count;
            } else {
                // Ensure entry exists even if empty, and reset offset/count
                if i < self.gpu_grid_offsets.len() {
                    self.gpu_grid_offsets[i] = [current_offset, 0];
                } else {
                    // This case should ideally not happen if resize logic is correct
                    log::error!(
                        "Grid offset index {} out of bounds during construction (size {})",
                        i,
                        self.gpu_grid_offsets.len()
                    );
                }
            }
        }
    }

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

        self.new_organism_buffer.clear();
        self.removal_indices_buffer.clear();
        self.removal_indices_set.clear();
        self.resize_internal_buffers(current_organism_count); // Ensure buffers are large enough

        // Clear buffers for the current count
        for i in 0..current_organism_count {
            self.accumulated_aging_rate_buffer[i] = 0.0;
            self.plant_is_clustered_buffer[i] = false;
            self.eating_boost_buffer[i] = 1.0; // Reset to base 1.0
        }

        self.build_grid(); // Build grid based on current positions

        // Pre-calculate interaction radii for the current count
        for i in 0..current_organism_count {
            if i >= self.organisms.len() {
                continue;
            } // Safety check
            let org = &self.organisms[i];
            let radius = org.radius;
            let pushaway_radius = org.config.organism_min_pushaway_radius.max(0.0);
            self.interaction_radii_buffer[i] = match org.kind {
                OrganismType::Plant => InteractionRadii {
                    perception_sq: 0.0,
                    eating_sq: 0.0,
                    clustering_sq: (radius * org.config.clustering_radius_factor).powi(2),
                    pushaway_sq: pushaway_radius.powi(2),
                },
                OrganismType::Fish | OrganismType::Bug => InteractionRadii {
                    perception_sq: (radius * org.config.perception_radius_factor).powi(2),
                    eating_sq: (radius * org.config.eating_radius_factor).powi(2),
                    clustering_sq: 0.0, // Fish/Bugs don't cluster (currently)
                    pushaway_sq: pushaway_radius.powi(2),
                },
            };
        }

        // --- Parallel Calculation ---
        let organisms_ref = &self.organisms; // Immutable borrow for parallel access
        let grid_ref = &self.grid;
        let radii_ref = &self.interaction_radii_buffer; // Use pre-calculated radii

        let calculation_results: Vec<OrganismCalculationResult> = (0..current_organism_count)
            .into_par_iter() // Parallel iteration over indices
            .map(|i| {
                let mut result = OrganismCalculationResult {
                    index: i,
                    ..Default::default()
                };
                // Safety check: ensure index is valid within the *current* organism slice
                if i >= organisms_ref.len() {
                    return result; // Return default if index is out of bounds
                }

                let organism_i = &organisms_ref[i];
                let pos_i = organism_i.position;
                let grid_key_i = self.get_grid_key(pos_i); // Use state's method (read-only)
                // Use pre-calculated radii, with safety check
                let radii_i = if i < radii_ref.len() { radii_ref[i] } else { Default::default() };


                let mut influence_vector = Vec2::ZERO;
                let mut neighbor_count = 0; // Count influential neighbors (non-pushaway)
                let mut max_boost_factor_for_i = 1.0f32; // Start with base boost
                let mut thread_rng = thread_rng(); // RNG for this thread
                let mut pushed_away_this_tick = false; // Track if push-away happened

                // Iterate over 3x3 grid neighborhood
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let check_key = (grid_key_i.0 + dx, grid_key_i.1 + dy);
                        // Access grid via immutable reference
                        if let Some(neighbor_indices) = grid_ref.get(&check_key) {
                            for &j in neighbor_indices {
                                // Skip self or invalid indices
                                if i == j || j >= organisms_ref.len() {
                                    continue;
                                }

                                let organism_j = &organisms_ref[j]; // Access neighbor data
                                let pos_j = organism_j.position;
                                let vec_ij = pos_j - pos_i;
                                let dist_sq = vec_ij.length_squared();

                                // --- 1. Push-away Check (Applies to all types) ---
                                // Use organism_i's pushaway radius
                                if radii_i.pushaway_sq > 0.0 && dist_sq < radii_i.pushaway_sq && dist_sq > 1e-6 {
                                    let push_direction = -vec_ij.normalize_or_zero();
                                    let force_scale = 1.0 / (dist_sq + 1e-4); // Stronger closer
                                    influence_vector += push_direction * PUSH_AWAY_FORCE_MAGNITUDE * force_scale;
                                    pushed_away_this_tick = true; // Mark push-away happened
                                    // Don't increment neighbor_count for push-away only
                                }

                                // --- 2. Perception-based Influence (Fish & Bugs) ---
                                if (organism_i.kind == OrganismType::Fish || organism_i.kind == OrganismType::Bug)
                                    && dist_sq < radii_i.perception_sq // Check perception range
                                    && dist_sq > 1e-6 // Avoid division by zero / self-influence
                                {
                                    let mut type_influence_factor = 0.0;
                                    let mut is_offspring_influence = false;

                                    // --- <<< NEW: Check for Offspring Influence FIRST >>> ---
                                    if let Some(parent_id_j) = organism_j.parent_id {
                                        if parent_id_j == organism_i.id {
                                            type_influence_factor = organism_i.config.influence_offspring;
                                            is_offspring_influence = true;
                                        }
                                    }

                                    // --- Standard Type Influence (If not offspring) ---
                                    if !is_offspring_influence {
                                        type_influence_factor = match organism_j.kind {
                                            OrganismType::Plant => organism_i.config.influence_plant,
                                            OrganismType::Fish => organism_i.config.influence_fish,
                                            OrganismType::Bug => organism_i.config.influence_bug,
                                        };
                                    }


                                    if type_influence_factor.abs() > 1e-6 {
                                        influence_vector += vec_ij.normalize_or_zero() * type_influence_factor;
                                        // Only count non-pushaway interactions towards weighted avg direction
                                        if !pushed_away_this_tick {
                                            neighbor_count += 1;
                                        }
                                    }
                                }

                                // --- 3. Eating Check (Fish & Bugs) ---
                                if (organism_i.kind == OrganismType::Fish || organism_i.kind == OrganismType::Bug)
                                    && dist_sq < radii_i.eating_sq // Check eating range
                                {
                                    // Determine boost factor based on prey type
                                    let boost_factor = match organism_j.kind {
                                        OrganismType::Plant => organism_i.config.eating_spawn_boost_factor_plant,
                                        OrganismType::Fish => organism_i.config.eating_spawn_boost_factor_fish,
                                        OrganismType::Bug => organism_i.config.eating_spawn_boost_factor_bug,
                                    };
                                    // Keep track of the highest boost obtained from any eaten neighbor
                                    if boost_factor > max_boost_factor_for_i {
                                        max_boost_factor_for_i = boost_factor;
                                    }

                                    // Apply aging effect to the prey (organism_j)
                                    let prey_aging_rate = organism_j.config.aging_rate_when_eaten;
                                    if prey_aging_rate > 0.0 {
                                        // Store effect to be applied later (atomically or aggregated)
                                        result.prey_aging_effects.push((j, prey_aging_rate));
                                    }
                                }

                                // --- 4. Clustering Check (Plants only) ---
                                if organism_i.kind == OrganismType::Plant && organism_j.kind == OrganismType::Plant
                                    && dist_sq < radii_i.clustering_sq // Check clustering range
                                {
                                    // Mark this plant as clustered (used later for potential spawn rate modification)
                                    result.is_clustered = true;
                                    // Note: Clustering doesn't directly affect movement/influence vector here
                                }
                            } // End loop over neighbors in cell
                        } // End if cell exists
                    } // End loop dy
                } // End loop dx (neighbor grid cells)

                // Store max boost factor obtained
                result.spawn_boost_obtained = max_boost_factor_for_i;

                // --- Calculate New Velocity (Fish & Bugs) ---
                if organism_i.kind == OrganismType::Fish || organism_i.kind == OrganismType::Bug {
                    let radius_i = organism_i.radius;
                    // Random direction component
                    let random_angle = thread_rng.gen_range(0.0..TAU);
                    let random_direction = Vec2::from_angle(random_angle);

                    // Normalize influence vector (if non-zero) or keep current velocity direction
                    let normalized_influence = if influence_vector.length_squared() > 1e-6 {
                        influence_vector.normalize() // Normalize the combined influence + pushaway
                    } else {
                        organism_i.velocity.normalize_or_zero() // Keep moving straight if no influence
                    };


                    // Weighted average between random direction and influence direction
                    let influence_weight = organism_i.config.influence_weight; // How much influence matters
                    let desired_direction = (
                        random_direction * (1.0 - influence_weight) +
                            normalized_influence * influence_weight
                    ).normalize_or_zero(); // Ensure final direction is normalized


                    // Smooth turning towards desired direction
                    let current_dir = organism_i.velocity.normalize_or_zero();
                    let angle_diff = current_dir.angle_between(desired_direction); // Angle difference
                    let max_turn = organism_i.config.max_turn_angle_per_sec * dt; // Max turn angle this tick
                    let turn_amount = angle_diff.clamp(-max_turn, max_turn); // Actual turn amount

                    // Calculate final direction after turning
                    let final_direction = if current_dir.length_squared() > 1e-6 {
                        // Rotate current direction by turn_amount
                        Vec2::from_angle(current_dir.to_angle() + turn_amount)
                    } else {
                        // If not moving, instantly face desired direction
                        desired_direction
                    };

                    // Calculate new velocity vector
                    result.new_velocity = Some(final_direction * organism_i.config.movement_speed_factor * radius_i);
                }

                result // Return the calculated results for this organism
            })
            .collect(); // Collect results from all threads

        // --- Apply Results Serially ---

        // 1. Aggregate prey aging effects (serial to avoid race conditions on shared buffer)
        for result in &calculation_results {
            for &(prey_index, aging_rate) in &result.prey_aging_effects {
                // Check bounds before accessing buffer
                if prey_index < self.accumulated_aging_rate_buffer.len() {
                    self.accumulated_aging_rate_buffer[prey_index] += aging_rate;
                }
            }
        }

        // Get current organism count *before* potentially adding/removing
        let organism_count_before_mutation = self.organisms.len();

        // Prepare for aging, movement, and spawning
        let window_w = self.window_size.width as f32;
        let window_h = self.window_size.height as f32;
        let max_organisms = MAX_ORGANISMS;
        let sim_config_ref = &self.config; // Reference to base config for spawning
        let mut spawn_rng = thread_rng(); // Re-seed thread_rng for spawning

        // 2. Apply movement, aging, check death, check spawning (serial iteration)
        for i in 0..current_organism_count {
            // Skip if index became invalid due to prior removals (shouldn't happen with this structure, but safe)
            if i >= calculation_results.len() || i >= organism_count_before_mutation {
                continue;
            }

            let result = &calculation_results[i];
            debug_assert_eq!(result.index, i, "Result index mismatch!"); // Ensure results align

            // Get mutable access to the organism for this index
            let organism = &mut self.organisms[i];

            // --- Apply Results to Organism ---
            self.eating_boost_buffer[i] = result.spawn_boost_obtained;
            self.plant_is_clustered_buffer[i] = result.is_clustered;

            // Apply calculated velocity change
            if let Some(new_vel) = result.new_velocity {
                organism.velocity = new_vel;
            }

            // Update position
            organism.position += organism.velocity * dt;

            // Wrap around window boundaries
            organism.position.x = (organism.position.x + window_w) % window_w;
            organism.position.y = (organism.position.y + window_h) % window_h;

            // --- Aging ---
            let base_age_increase = dt;
            let extra_aging_rate = self.accumulated_aging_rate_buffer[i]; // Already aggregated
            let effective_age_increase = base_age_increase * (1.0 + extra_aging_rate);
            organism.age += effective_age_increase;

            // --- Death Check ---
            if organism.age >= organism.lifetime {
                // Use HashSet to avoid duplicate removals if marked by multiple events
                if self.removal_indices_set.insert(i) {
                    self.removal_indices_buffer.push(i);
                }
                continue; // Skip spawning if dead
            }

            // --- Spawning Check ---
            let base_prob_per_sec = organism.growth_rate / organism.lifetime.max(1.0);
            // Apply boost obtained from eating
            let mut spawn_prob_this_tick = base_prob_per_sec * dt * self.eating_boost_buffer[i];

            // Apply plant clustering effect (if any) - currently does nothing, placeholder
            if organism.kind == OrganismType::Plant && self.plant_is_clustered_buffer[i] {
                // Example: spawn_prob_this_tick *= PLANT_CLUSTERING_SPAWN_FACTOR;
            }

            // Check probability and capacity
            if spawn_prob_this_tick > 0.0
                && spawn_rng.gen_bool(spawn_prob_this_tick.clamp(0.0, 1.0) as f64)
            {
                // Estimate count *after* removals and *before* adding this new one
                let potential_next_count = organism_count_before_mutation // Start with count before this loop
                    - self.removal_indices_set.len() // Account for removals marked so far
                    + self.new_organism_buffer.len() // Account for spawns added so far
                    + 1; // Account for the one we *might* add now

                if potential_next_count <= max_organisms {
                    // Create offspring using parent's data and add to buffer
                    self.new_organism_buffer.push(Self::create_offspring(
                        organism, // Pass immutable ref here, necessary data is cloned inside
                        self.window_size,
                        &mut spawn_rng,
                        sim_config_ref,
                    ));
                }
            }
        } // End serial loop for applying results

        // --- Perform Removals and Additions ---

        // Sort removal indices descending for safe swap_remove
        self.removal_indices_buffer
            .sort_unstable_by(|a, b| b.cmp(a));
        self.removal_indices_buffer.dedup(); // Should be redundant if using HashSet, but safe

        // Perform swap_remove
        for &index_to_remove in &self.removal_indices_buffer {
            if index_to_remove < self.organisms.len() {
                // Check bounds *before* removing
                self.organisms.swap_remove(index_to_remove);
            } else {
                // This indicates a logic error if it happens
                log::warn!(
                    "Attempted swap_remove with invalid index {} (current len {})",
                    index_to_remove,
                    self.organisms.len()
                );
            }
        }

        // Add new organisms from the buffer
        self.organisms.extend(self.new_organism_buffer.drain(..)); // Efficiently move organisms

        // Enforce maximum organism limit *after* additions
        if self.organisms.len() > max_organisms {
            log::warn!(
                "Exceeded MAX_ORGANISMS, truncating from {} to {}",
                self.organisms.len(),
                max_organisms
            );
            self.organisms.truncate(max_organisms);
        }

        // Ensure internal buffers match the final organism count for the *next* frame
        self.resize_internal_buffers(self.organisms.len());
    }

    pub fn adjust_speed(&mut self, increase: bool) {
        self.speed_multiplier = if increase {
            (self.speed_multiplier + SPEED_ADJUST_FACTOR).min(MAX_SPEED_MULTIPLIER)
        } else {
            (self.speed_multiplier - SPEED_ADJUST_FACTOR).max(MIN_SPEED_MULTIPLIER)
        };
        println!("Speed Multiplier: {:.2}", self.speed_multiplier);
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.window_size = new_size;
            let avg_radius = self.config.get_avg_base_radius();
            self.grid_cell_size = avg_radius * GRID_CELL_SIZE_FACTOR;
            self.grid_width = (new_size.width as f32 / self.grid_cell_size).ceil() as i32;
            self.grid_height = (new_size.height as f32 / self.grid_cell_size).ceil() as i32;
            self.grid.clear(); // Clear old grid data
            let num_grid_cells = (self.grid_width * self.grid_height).max(1) as usize;
            // Resize GPU offset buffer to match new grid dimensions
            if self.gpu_grid_offsets.len() != num_grid_cells {
                self.gpu_grid_offsets.resize(num_grid_cells, [0, 0]);
            }
            self.gpu_grid_indices.clear(); // Clear old indices (will be rebuilt)
            println!(
                "Resized simulation area to {}x{}, grid to {}x{} (cell size ~{:.1})",
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
        println!(
            "Simulation {}",
            if self.is_paused { "Paused" } else { "Resumed" }
        );
    }

    pub fn restart(&mut self) {
        println!("Restarting simulation with new seed...");
        self.rng = SimRng::from_entropy(); // Re-seed main RNG
        // <<< NEW: Reset ID counter on restart
        NEXT_ORGANISM_ID.store(1, Ordering::Relaxed);

        // Recalculate grid parameters based on potentially changed config/window size
        let avg_radius = self.config.get_avg_base_radius();
        self.grid_cell_size = avg_radius * GRID_CELL_SIZE_FACTOR;
        self.grid_width = (self.window_size.width as f32 / self.grid_cell_size).ceil() as i32;
        self.grid_height = (self.window_size.height as f32 / self.grid_cell_size).ceil() as i32;

        // Initialize organisms (clears existing, creates new ones with new IDs)
        self.initialize_organisms();

        // Reset simulation state variables
        self.speed_multiplier = INITIAL_SPEED_MULTIPLIER;
        self.is_paused = false;

        // Clear runtime data structures
        self.grid.clear();
        self.new_organism_buffer.clear();
        self.removal_indices_buffer.clear();
        self.removal_indices_set.clear();
        self.gpu_grid_indices.clear();

        // Ensure GPU offsets buffer matches the new grid size
        let num_grid_cells = (self.grid_width * self.grid_height).max(1) as usize;
        if self.gpu_grid_offsets.len() != num_grid_cells {
            self.gpu_grid_offsets.resize(num_grid_cells, [0, 0]);
        }

        println!(
            "Restarted with grid {}x{} (cell size ~{:.1})",
            self.grid_width, self.grid_height, self.grid_cell_size
        );
    }

    pub fn speed_multiplier(&self) -> f32 {
        self.speed_multiplier
    }
    pub fn is_paused(&self) -> bool {
        self.is_paused
    }

    pub fn get_organism_counts(&self) -> (usize, usize, usize) {
        let mut plant_count = 0;
        let mut fish_count = 0;
        let mut bug_count = 0;
        for org in &self.organisms {
            match org.kind {
                OrganismType::Plant => plant_count += 1,
                OrganismType::Fish => fish_count += 1,
                OrganismType::Bug => bug_count += 1,
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
// --- End of File: simulation.rs ---
