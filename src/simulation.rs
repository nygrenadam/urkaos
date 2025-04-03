// --- File: simulation.rs ---
// File: simulation.rs
use crate::config::{OrganismConfig, SimulationConfig};
use crate::constants::*;
use crate::utils::mutate_color;
// Import helpers
use glam::{Vec2, Vec4};
// --- Use thread_rng for parallel random generation ---
// REMOVED: unused SliceRandom
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
// --- NEW: Import Rayon ---
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    f32::consts::TAU,
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
    pub kind: OrganismType,
    pub position: Vec2,
    pub velocity: Vec2,
    pub age: f32,
    pub lifetime: f32,
    pub growth_rate: f32,
    pub color: Vec4,
    pub radius: f32, // This radius is used for simulation logic (Globally Scaled)
    pub config: OrganismConfig,
}

pub type GridKey = (i32, i32);
// --- Change SimRng to be Send + Sync if needed globally, but we'll use thread_rng mostly ---
// pub type SimRng = StdRng;
pub type SimRng = StdRng; // Keep main RNG for seeding if needed, but use thread_rng in parallel loops

// --- NEW: Constant for push-away force magnitude ---
const PUSH_AWAY_FORCE_MAGNITUDE: f32 = 20.0;

pub struct SimulationState {
    pub organisms: Vec<Organism>,
    rng: SimRng, // Main RNG for initialization/restart
    window_size: PhysicalSize<u32>,
    speed_multiplier: f32,
    is_paused: bool,
    pub config: SimulationConfig,
    grid: HashMap<GridKey, Vec<usize>>, // CPU grid for simulation logic
    grid_width: i32,
    grid_height: i32,
    grid_cell_size: f32,
    // OPTIMIZATION: Buffers for reuse in update loop
    new_organism_buffer: Vec<Organism>,
    removal_indices_buffer: Vec<usize>,
    removal_indices_set: HashSet<usize>, // Also reuse hashset
    interaction_radii_buffer: Vec<InteractionRadii>, // Reuse radii buffer
    // REMOVED: new_velocities_buffer (will be part of collected results)
    // Renamed for clarity: This accumulates the aging *rate* to apply later
    accumulated_aging_rate_buffer: Vec<f32>,
    plant_is_clustered_buffer: Vec<bool>,
    // Stores the maximum *beneficial* ( > 1.0) boost factor obtained from eating this tick
    eating_boost_buffer: Vec<f32>,

    // --- GPU grid data ---
    gpu_grid_indices: Vec<u32>, // Flat list of organism indices per cell
    gpu_grid_offsets: Vec<[u32; 2]>, // [offset, count] for each cell into gpu_grid_indices
}

// --- NEW: Structure to hold results from the parallel calculation phase ---
#[derive(Debug, Default)]
struct OrganismCalculationResult {
    index: usize, // Index of the organism this result is for
    new_velocity: Option<Vec2>,
    // List of prey affected by this organism eating them: (prey_index, aging_rate_to_apply)
    prey_aging_effects: Vec<(usize, f32)>,
    // Max spawn boost factor obtained by THIS organism from eating prey
    spawn_boost_obtained: f32,
    // Clustering status for THIS organism (if plant)
    is_clustered: bool,
}

// Helper struct for pre-calculated radii (used in simulation logic)
#[derive(Default, Clone, Copy)] // Add Default/Clone/Copy for easy vec resizing
struct InteractionRadii {
    perception_sq: f32,
    eating_sq: f32,
    clustering_sq: f32,
    // --- NEW: Squared push-away radius ---
    pushaway_sq: f32,
}

impl SimulationState {
    pub fn new(window_size: PhysicalSize<u32>, config: SimulationConfig) -> Self {
        let avg_radius = config.get_avg_base_radius();
        let grid_cell_size = avg_radius * GRID_CELL_SIZE_FACTOR;
        let grid_width = (window_size.width as f32 / grid_cell_size).ceil() as i32;
        let grid_height = (window_size.height as f32 / grid_cell_size).ceil() as i32;
        let num_grid_cells = (grid_width * grid_height).max(1) as usize; // Ensure non-zero

        let initial_capacity =
            (INITIAL_PLANT_COUNT + INITIAL_FISH_COUNT + INITIAL_BUG_COUNT).max(256);
        let removal_capacity = initial_capacity / 10;
        let new_org_capacity = initial_capacity / 5;

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
            // REMOVED: new_velocities_buffer initialization
            accumulated_aging_rate_buffer: Vec::with_capacity(initial_capacity),
            plant_is_clustered_buffer: Vec::with_capacity(initial_capacity),
            eating_boost_buffer: Vec::with_capacity(initial_capacity),
            // --- Initialize GPU grid vectors ---
            gpu_grid_indices: Vec::with_capacity(initial_capacity), // Estimate capacity
            gpu_grid_offsets: vec![[0, 0]; num_grid_cells],         // Initialize with zeros
        };
        state.initialize_organisms();
        state
    }

    fn initialize_organisms(&mut self) {
        self.organisms.clear();
        let total_initial_count = INITIAL_PLANT_COUNT + INITIAL_FISH_COUNT + INITIAL_BUG_COUNT;
        self.organisms.reserve(total_initial_count.max(100));
        // Use a local RNG derived from the main one for initialization consistency
        let mut local_rng =
            SimRng::from_rng(&mut self.rng).unwrap_or_else(|_| SimRng::from_entropy());
        for _ in 0..INITIAL_PLANT_COUNT {
            self.organisms.push(Self::create_organism(
                &mut local_rng, // Use local RNG
                self.window_size,
                OrganismType::Plant,
                &self.config,
            ));
        }
        for _ in 0..INITIAL_FISH_COUNT {
            self.organisms.push(Self::create_organism(
                &mut local_rng, // Use local RNG
                self.window_size,
                OrganismType::Fish,
                &self.config,
            ));
        }
        for _ in 0..INITIAL_BUG_COUNT {
            self.organisms.push(Self::create_organism(
                &mut local_rng, // Use local RNG
                self.window_size,
                OrganismType::Bug,
                &self.config,
            ));
        }
        // OPTIMIZATION: Ensure buffer capacities match initial organisms
        self.resize_internal_buffers(self.organisms.len());
        // --- NEW: Ensure GPU grid offset buffer has correct size ---
        let num_grid_cells = (self.grid_width * self.grid_height).max(1) as usize;
        if self.gpu_grid_offsets.len() != num_grid_cells {
            self.gpu_grid_offsets.resize(num_grid_cells, [0, 0]);
        }
        // --- End NEW ---
    }

    // OPTIMIZATION: Helper to resize all internal buffers consistently
    fn resize_internal_buffers(&mut self, capacity: usize) {
        // Resize buffers that need to match organism count exactly
        if self.interaction_radii_buffer.len() < capacity {
            self.interaction_radii_buffer
                .resize(capacity, Default::default());
        }
        // REMOVED: new_velocities_buffer resize
        if self.accumulated_aging_rate_buffer.len() < capacity {
            self.accumulated_aging_rate_buffer.resize(capacity, 0.0);
        }
        if self.plant_is_clustered_buffer.len() < capacity {
            self.plant_is_clustered_buffer.resize(capacity, false);
        }
        if self.eating_boost_buffer.len() < capacity {
            self.eating_boost_buffer.resize(capacity, 1.0);
        }

        // Truncate buffers if they became longer than organisms list
        self.interaction_radii_buffer.truncate(capacity);
        // REMOVED: new_velocities_buffer truncate
        self.accumulated_aging_rate_buffer.truncate(capacity);
        self.plant_is_clustered_buffer.truncate(capacity);
        self.eating_boost_buffer.truncate(capacity);

        // Buffers that just need clearing don't need resizing here (e.g., new_organism_buffer).
    }

    // MODIFIED: Apply GLOBAL_RADIUS_SCALE_FACTOR
    fn create_organism(
        rng: &mut SimRng, // Still takes SimRng for deterministic initialization
        window_size: PhysicalSize<u32>,
        kind: OrganismType,
        config: &SimulationConfig,
    ) -> Organism {
        let organism_config = match kind {
            OrganismType::Plant => &config.plant,
            OrganismType::Fish => &config.fish,
            OrganismType::Bug => &config.bug,
        };

        let position = Vec2::new(
            rng.gen_range(0.0..window_size.width as f32),
            rng.gen_range(0.0..window_size.height as f32),
        );

        let min_radius = organism_config.min_radius;
        let max_radius = organism_config.max_radius;
        let initial_radius = if min_radius < max_radius {
            rng.gen_range(min_radius..=max_radius)
        } else {
            min_radius
        };

        // --- Apply Global Scale ---
        let radius = (initial_radius * GLOBAL_RADIUS_SCALE_FACTOR).max(0.1); // Ensure non-zero radius

        let velocity = if organism_config.movement_speed_factor > 0.0 {
            let angle = rng.gen_range(0.0..TAU);
            // Velocity depends on scaled radius
            Vec2::from_angle(angle) * organism_config.movement_speed_factor * radius
        } else {
            Vec2::ZERO
        };

        let min_lifetime = organism_config.min_lifetime;
        let max_lifetime = organism_config.max_lifetime;
        let lifetime = if min_lifetime < max_lifetime {
            rng.gen_range(min_lifetime..max_lifetime)
        } else {
            min_lifetime
        };

        let growth_rate = organism_config.base_growth_rate;
        let color = mutate_color(
            Vec4::from(organism_config.base_color),
            rng, // Use the passed RNG
            INITIAL_COLOR_MUTATION_MAX_DELTA,
        );

        Organism {
            kind,
            position,
            velocity,
            age: 0.0,
            lifetime,
            growth_rate,
            color,
            radius, // Store the globally scaled simulation radius
            config: organism_config.clone(),
        }
    }

    // MODIFIED: Apply GLOBAL_RADIUS_SCALE_FACTOR
    // Needs to accept Rng generic bound for thread_rng
    fn create_offspring<R: Rng>(
        // Accept generic Rng
        parent: &Organism,
        window_size: PhysicalSize<u32>,
        rng: &mut R,                // Use generic Rng
        _config: &SimulationConfig, // Keep parameter prefixed if potentially used later
    ) -> Organism {
        // Offset depends on parent's (scaled) simulation radius
        let spawn_offset_dist = parent.radius * SPAWN_OFFSET_RADIUS_FACTOR;
        let angle_offset = rng.gen_range(0.0..TAU);
        let offset = Vec2::from_angle(angle_offset) * spawn_offset_dist;
        let mut position = parent.position + offset;
        position.x = position.x.clamp(0.0, (window_size.width as f32) - 1.0);
        position.y = position.y.clamp(0.0, (window_size.height as f32) - 1.0);

        let min_radius = parent.config.min_radius;
        let max_radius = parent.config.max_radius;
        let radius_delta = rng
            .gen_range(-OFFSPRING_RADIUS_MUTATION_MAX_DELTA..=OFFSPRING_RADIUS_MUTATION_MAX_DELTA);
        // Calculate mutated radius based on parent's scaled radius
        let mutated_radius_before_scale = (parent.radius / GLOBAL_RADIUS_SCALE_FACTOR.max(0.01)) + radius_delta;
        // Clamp based on *config* limits, then apply global scale
        let clamped_radius_before_scale = mutated_radius_before_scale.clamp(min_radius, max_radius);

        // --- Apply Global Scale ---
        let radius = (clamped_radius_before_scale * GLOBAL_RADIUS_SCALE_FACTOR).max(0.1); // Ensure non-zero radius

        let velocity = if parent.config.movement_speed_factor > 0.0 {
            let angle = rng.gen_range(0.0..TAU);
            // Velocity depends on scaled radius
            Vec2::from_angle(angle) * parent.config.movement_speed_factor * radius
        } else {
            Vec2::ZERO
        };

        let min_lifetime = parent.config.min_lifetime;
        let max_lifetime = parent.config.max_lifetime;
        let lifetime = if min_lifetime < max_lifetime {
            rng.gen_range(min_lifetime..max_lifetime)
        } else {
            min_lifetime
        };

        let growth_rate = parent.config.base_growth_rate;
        let color = mutate_color(parent.color, rng, OFFSPRING_COLOR_MUTATION_MAX_DELTA);

        Organism {
            kind: parent.kind,
            position,
            velocity,
            age: 0.0,
            lifetime,
            growth_rate,
            color,
            radius, // Store the globally scaled simulation radius
            config: parent.config.clone(),
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

    // Build grid remains sequential
    fn build_grid(&mut self) {
        self.grid.clear();
        self.gpu_grid_indices.clear();

        let mut temp_gpu_grid: HashMap<usize, Vec<u32>> = HashMap::new();
        let grid_w = self.grid_width;

        for (index, organism) in self.organisms.iter().enumerate() {
            let key @ (cell_x, cell_y) = self.get_grid_key(organism.position);
            self.grid.entry(key).or_default().push(index);

            let flat_index = (cell_x + cell_y * grid_w) as usize;
            // Ensure gpu_grid_offsets is correctly sized before access
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
                    "Calculated flat grid index {} out of bounds (size {}) for organism at {:?}. Skipping.", // Changed message slightly
                    flat_index,
                    self.gpu_grid_offsets.len(),
                    organism.position
                );
            }
        }

        let mut current_offset = 0u32;
        // Iterate using the guaranteed length
        for i in 0..self.gpu_grid_offsets.len() {
            if let Some(indices) = temp_gpu_grid.get(&i) {
                let count = indices.len() as u32;
                self.gpu_grid_offsets[i] = [current_offset, count];
                self.gpu_grid_indices.extend_from_slice(indices);
                current_offset += count;
            } else {
                self.gpu_grid_offsets[i] = [current_offset, 0];
            }
        }
    }

    // --- Simulation Update - MAJOR REFACTOR ---
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

        // --- Clear volatile buffers ---
        self.new_organism_buffer.clear();
        self.removal_indices_buffer.clear();
        self.removal_indices_set.clear();
        self.resize_internal_buffers(current_organism_count);

        // --- Reset State Buffers (Sequential OK) ---
        for i in 0..current_organism_count {
            self.accumulated_aging_rate_buffer[i] = 0.0;
            self.plant_is_clustered_buffer[i] = false;
            self.eating_boost_buffer[i] = 1.0;
        }

        // --- Build Grid (Sequential) ---
        self.build_grid();

        // --- Calculate Interaction Radii (Sequential) ---
        // Interaction radii depend on the organism's current (potentially scaled) radius
        // Pushaway radius uses the ABSOLUTE config value.
        for (i, org) in self.organisms.iter().enumerate() {
            let radius = org.radius; // Use the stored (potentially scaled) radius
            let pushaway_radius = org.config.organism_min_pushaway_radius.max(0.0);
            self.interaction_radii_buffer[i] = match org.kind {
                OrganismType::Plant => InteractionRadii {
                    perception_sq: 0.0,
                    eating_sq: 0.0,
                    clustering_sq: (radius * org.config.clustering_radius_factor).powi(2),
                    pushaway_sq: pushaway_radius.powi(2), // Based on config
                },
                OrganismType::Fish => InteractionRadii {
                    perception_sq: (radius * org.config.perception_radius_factor).powi(2),
                    eating_sq: (radius * org.config.eating_radius_factor).powi(2),
                    clustering_sq: 0.0,
                    pushaway_sq: pushaway_radius.powi(2), // Based on config
                },
                OrganismType::Bug => InteractionRadii {
                    perception_sq: (radius * org.config.perception_radius_factor).powi(2),
                    eating_sq: (radius * org.config.eating_radius_factor).powi(2),
                    clustering_sq: 0.0,
                    pushaway_sq: pushaway_radius.powi(2), // Based on config
                },
            };
        }

        // --- Parallel Interaction Calculation ---
        // Capture necessary data by reference
        let organisms_ref = &self.organisms; // Immutable borrow starts here
        let grid_ref = &self.grid;
        let radii_ref = &self.interaction_radii_buffer;
        // REMOVED: unused config_ref capture

        let calculation_results: Vec<OrganismCalculationResult> = (0..current_organism_count)
            .into_par_iter()
            .map(|i| {
                // --- Start of parallel work for organism `i` ---
                let mut result = OrganismCalculationResult {
                    index: i,
                    ..Default::default()
                };
                // Bounds check (paranoid, but safe if current_organism_count somehow changes)
                if i >= organisms_ref.len() {
                    return result;
                }

                let organism_i = &organisms_ref[i];
                let pos_i = organism_i.position;
                // Call get_grid_key using `self` implicitly captured - OK as it only reads state
                let grid_key_i = self.get_grid_key(pos_i);
                // Bounds check for radii_ref
                let radii_i = if i < radii_ref.len() {
                    radii_ref[i]
                } else {
                    Default::default()
                };

                let mut influence_vector = Vec2::ZERO;
                let mut neighbor_count = 0;
                let mut max_boost_factor_for_i = 1.0f32;
                let mut thread_rng = thread_rng();
                let mut pushed_away_this_tick = false; // Track if push force applied

                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let check_key = (grid_key_i.0 + dx, grid_key_i.1 + dy);
                        if let Some(neighbor_indices) = grid_ref.get(&check_key) {
                            for &j in neighbor_indices {
                                // Combine checks for clarity and efficiency
                                if i == j || j >= organisms_ref.len() {
                                    continue;
                                }

                                let organism_j = &organisms_ref[j];
                                let pos_j = organism_j.position;
                                let vec_ij = pos_j - pos_i;
                                let dist_sq = vec_ij.length_squared();

                                // --- Push-away Check (High Priority) ---
                                // Check if organism J is within organism I's pushaway radius
                                if radii_i.pushaway_sq > 0.0 && dist_sq < radii_i.pushaway_sq && dist_sq > 1e-6 {
                                    let push_direction = -vec_ij.normalize_or_zero(); // Push i away from j
                                    // Add a strong repulsive force, potentially scaled by inverse distance squared
                                    let force_scale = 1.0 / (dist_sq + 1e-4); // Avoid division by zero, strengthen close push
                                    influence_vector += push_direction * PUSH_AWAY_FORCE_MAGNITUDE * force_scale;
                                    pushed_away_this_tick = true;
                                    neighbor_count += 1; // Count this as an interaction
                                    // Skip other influence/eating checks if being pushed away? Optional.
                                    // For now, let push force dominate by adding to influence_vector.
                                    // continue; // Uncomment this to completely ignore other influences if pushed
                                }

                                // --- Standard Influence (Only if not dominated by push-away maybe?) ---
                                // Let's allow standard influence to still be calculated, but the push vector will likely dominate.
                                // if !pushed_away_this_tick { // Optional: only apply standard influence if not pushed
                                if (organism_i.kind == OrganismType::Fish
                                    || organism_i.kind == OrganismType::Bug)
                                    && dist_sq < radii_i.perception_sq
                                    && dist_sq > 1e-6
                                {
                                    let influence_factor = match organism_j.kind {
                                        OrganismType::Plant => organism_i.config.influence_plant,
                                        OrganismType::Fish => organism_i.config.influence_fish,
                                        OrganismType::Bug => organism_i.config.influence_bug,
                                    };
                                    if influence_factor.abs() > 1e-6 {
                                        influence_vector +=
                                            vec_ij.normalize_or_zero() * influence_factor;
                                        if !pushed_away_this_tick { neighbor_count += 1; } // Don't double-count if pushed
                                    }
                                }
                                // }

                                // --- Eating & Aging Effects (Can happen even if pushing/pushed) ---
                                if (organism_i.kind == OrganismType::Fish
                                    || organism_i.kind == OrganismType::Bug)
                                    && dist_sq < radii_i.eating_sq
                                {
                                    let boost_factor = match organism_j.kind {
                                        OrganismType::Plant => {
                                            organism_i.config.eating_spawn_boost_factor_plant
                                        }
                                        OrganismType::Fish => {
                                            organism_i.config.eating_spawn_boost_factor_fish
                                        }
                                        OrganismType::Bug => {
                                            organism_i.config.eating_spawn_boost_factor_bug
                                        }
                                    };

                                    if boost_factor > 1.0 {
                                        max_boost_factor_for_i =
                                            max_boost_factor_for_i.max(boost_factor);
                                        let prey_aging_rate =
                                            organism_j.config.aging_rate_when_eaten;
                                        if prey_aging_rate > 0.0 {
                                            result.prey_aging_effects.push((j, prey_aging_rate));
                                        }
                                    }
                                }

                                // --- Plant Clustering ---
                                if organism_i.kind == OrganismType::Plant
                                    && organism_j.kind == OrganismType::Plant
                                    && dist_sq < radii_i.clustering_sq
                                {
                                    result.is_clustered = true;
                                }
                            } // End loop j
                        } // End if grid cell
                    } // End loop dy
                } // End loop dx

                result.spawn_boost_obtained = max_boost_factor_for_i;

                // --- Calculate New Velocity ---
                if organism_i.kind == OrganismType::Fish || organism_i.kind == OrganismType::Bug {
                    let radius_i = organism_i.radius; // Use stored (potentially scaled) radius
                    let random_angle = thread_rng.gen_range(0.0..TAU);
                    let random_direction = Vec2::from_angle(random_angle);

                    // --- Normalize influence vector: if push force was applied, it will dominate ---
                    let normalized_influence = if neighbor_count > 0 { // Use neighbor_count which includes push interactions
                        influence_vector.normalize_or_zero()
                    } else {
                        organism_i.velocity.normalize_or_zero() // Maintain direction if no interactions
                    };

                    let influence_weight = organism_i.config.influence_weight.clamp(0.0, 1.0);

                    // --- Blend desired direction ---
                    // If pushed away, influence_weight might be less important, or maybe set to 1.0?
                    // Let's keep the blend for now. Push force magnitude should handle it.
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
                        desired_direction // Start moving in desired direction if stopped
                    };
                    // Velocity depends on scaled radius
                    result.new_velocity =
                        Some(final_direction * organism_i.config.movement_speed_factor * radius_i);
                }
                result
            })
            .collect(); // Collect results. Immutable borrow `organisms_ref` potentially ends scope here *if not used later*.

        // --- Sequential Application Phase ---

        // 1. Apply aging effects from eating to the shared buffer
        for result in &calculation_results {
            for &(prey_index, aging_rate) in &result.prey_aging_effects {
                // Bounds check before modifying buffer
                if prey_index < self.accumulated_aging_rate_buffer.len() {
                    self.accumulated_aging_rate_buffer[prey_index] += aging_rate;
                }
            }
        }

        // --- FIX: Get organism count *before* the mutable loop ---
        let organism_count_before_mutation = self.organisms.len();

        // 2. Update state, apply velocity, handle aging, death, and spawning
        let window_w = self.window_size.width as f32;
        let window_h = self.window_size.height as f32;
        let max_organisms = MAX_ORGANISMS;
        let sim_config_ref = &self.config; // Capture config for create_offspring
        let mut spawn_rng = thread_rng();

        // Iterate using the count from the *start* of the update, before any removals
        for i in 0..current_organism_count {
            // Ensure the result index matches the loop index `i`
            // This relies on Rayon's collect preserving order for indexed iterators.
            if i >= calculation_results.len() {
                continue;
            } // Safety check
            let result = &calculation_results[i];
            // Basic sanity check, remove in release if perf sensitive
            debug_assert_eq!(
                result.index, i,
                "Result index mismatch! Expected {}, got {}",
                i, result.index
            );

            // --- Ensure organism index `i` is still valid in the main vector ---
            if i >= organism_count_before_mutation {
                continue;
            } // Skip if index out of bounds

            // Update buffers based on calculated results (safe before mutable borrow)
            self.eating_boost_buffer[i] = result.spawn_boost_obtained;
            self.plant_is_clustered_buffer[i] = result.is_clustered;

            // --- Update Organism State (Mutable borrow starts here) ---
            let organism = &mut self.organisms[i]; // This is now safe

            // Apply calculated velocity
            if let Some(new_vel) = result.new_velocity {
                organism.velocity = new_vel;
            }
            organism.position += organism.velocity * dt;

            // Wrap around screen edges
            organism.position.x = (organism.position.x + window_w) % window_w;
            organism.position.y = (organism.position.y + window_h) % window_h;

            // Calculate final aging
            let base_age_increase = dt;
            // Bounds check aging buffer access
            let extra_aging_rate = if i < self.accumulated_aging_rate_buffer.len() {
                self.accumulated_aging_rate_buffer[i]
            } else {
                0.0
            };
            let effective_age_increase = base_age_increase + extra_aging_rate * dt;
            organism.age += effective_age_increase;

            // Check for death by old age
            if organism.age >= organism.lifetime {
                if self.removal_indices_set.insert(i) {
                    self.removal_indices_buffer.push(i);
                }
                continue; // Skip spawning if dead
            }

            // --- Spawning Logic (Only if alive) ---
            let base_prob_per_sec = organism.growth_rate / organism.lifetime.max(1.0);
            let mut spawn_prob_this_tick = base_prob_per_sec * dt;
            // Bounds check boost buffer access
            spawn_prob_this_tick *= if i < self.eating_boost_buffer.len() {
                self.eating_boost_buffer[i]
            } else {
                1.0
            };
            // Bounds check cluster buffer access
            if organism.kind == OrganismType::Plant
                && i < self.plant_is_clustered_buffer.len()
                && self.plant_is_clustered_buffer[i]
            {
                // spawn_prob_this_tick *= 1.1; // Optional clustering boost
            }

            if spawn_prob_this_tick > 0.0
                && spawn_rng.gen_bool(spawn_prob_this_tick.clamp(0.0, 1.0) as f64)
            {
                // --- FIX: Use the pre-calculated count ---
                let potential_next_count = organism_count_before_mutation // Use count before this loop
                    - self.removal_indices_set.len() // Use current removal count
                    + self.new_organism_buffer.len() + 1; // Use current new count

                if potential_next_count <= max_organisms {
                    self.new_organism_buffer.push(Self::create_offspring(
                        organism,
                        self.window_size,
                        &mut spawn_rng,
                        sim_config_ref, // Pass the captured ref
                    ));
                }
            }
        } // --- End Sequential Application Loop ---

        // --- Apply Removals and Additions (Sequential - Unchanged) ---
        self.removal_indices_buffer
            .sort_unstable_by(|a, b| b.cmp(a));
        self.removal_indices_buffer.dedup();

        // Use a temporary vec for swap_remove indices to satisfy borrow checker if needed,
        // but direct iteration should be fine here as we don't borrow `self.organisms` elsewhere.
        for &index_to_remove in &self.removal_indices_buffer {
            // Check index validity *against the current length* before swap_remove
            if index_to_remove < self.organisms.len() {
                self.organisms.swap_remove(index_to_remove);
            } else {
                log::warn!(
                    "Attempted swap_remove with invalid index {} (current len {} after previous removals)", // Clarified log
                    index_to_remove,
                    self.organisms.len()
                );
            }
        }
        self.organisms.extend(self.new_organism_buffer.drain(..));

        // Final count check
        if self.organisms.len() > max_organisms {
            log::warn!(
                "Exceeded MAX_ORGANISMS, truncating from {} to {}",
                self.organisms.len(),
                max_organisms
            );
            self.organisms.truncate(max_organisms);
        }

        // Ensure internal sim buffers are sized correctly for the *next* frame
        self.resize_internal_buffers(self.organisms.len());
    } // --- End Update Method ---

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
            self.grid.clear();
            let num_grid_cells = (self.grid_width * self.grid_height).max(1) as usize;
            if self.gpu_grid_offsets.len() != num_grid_cells {
                self.gpu_grid_offsets.resize(num_grid_cells, [0, 0]);
            }
            self.gpu_grid_indices.clear();
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
        self.rng = SimRng::from_entropy();
        let avg_radius = self.config.get_avg_base_radius();
        self.grid_cell_size = avg_radius * GRID_CELL_SIZE_FACTOR;
        self.grid_width = (self.window_size.width as f32 / self.grid_cell_size).ceil() as i32;
        self.grid_height = (self.window_size.height as f32 / self.grid_cell_size).ceil() as i32;
        self.initialize_organisms();
        self.speed_multiplier = INITIAL_SPEED_MULTIPLIER;
        self.is_paused = false;
        self.grid.clear();
        self.new_organism_buffer.clear();
        self.removal_indices_buffer.clear();
        self.removal_indices_set.clear();
        self.gpu_grid_indices.clear();
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

    // No changes needed
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