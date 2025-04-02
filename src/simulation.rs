// File: simulation.rs
use crate::config::{OrganismConfig, SimulationConfig};
use crate::constants::*;
use crate::utils::mutate_color;
// Import helpers
use glam::{Vec2, Vec4};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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
    pub radius: f32, // This radius is used for simulation logic
    pub config: OrganismConfig,
}

pub type GridKey = (i32, i32);
pub type SimRng = StdRng;

pub struct SimulationState {
    pub organisms: Vec<Organism>,
    rng: SimRng,
    window_size: PhysicalSize<u32>,
    speed_multiplier: f32,
    is_paused: bool,
    pub config: SimulationConfig,
    grid: HashMap<GridKey, Vec<usize>>,
    grid_width: i32,
    grid_height: i32,
    grid_cell_size: f32,
    // OPTIMIZATION: Buffers for reuse in update loop
    new_organism_buffer: Vec<Organism>,
    removal_indices_buffer: Vec<usize>,
    removal_indices_set: HashSet<usize>, // Also reuse hashset
    interaction_radii_buffer: Vec<InteractionRadii>, // Reuse radii buffer
    new_velocities_buffer: Vec<Option<Vec2>>,
    plant_extra_aging_buffer: Vec<f32>,
    fish_is_eating_buffer: Vec<bool>,
    bug_is_eating_buffer: Vec<bool>,
    plant_is_clustered_buffer: Vec<bool>,
}

// Helper struct for pre-calculated radii (used in simulation logic)
#[derive(Default, Clone, Copy)] // Add Default/Clone/Copy for easy vec resizing
struct InteractionRadii {
    perception_sq: f32,
    eating_sq: f32,
    clustering_sq: f32,
}

impl SimulationState {
    pub fn new(window_size: PhysicalSize<u32>, config: SimulationConfig) -> Self {
        let avg_radius = config.get_avg_base_radius();
        // Use the *simulation* radius for grid sizing
        // GRID_CELL_SIZE_FACTOR constant was adjusted
        let grid_cell_size = avg_radius * GRID_CELL_SIZE_FACTOR;

        let grid_width = (window_size.width as f32 / grid_cell_size).ceil() as i32;
        let grid_height = (window_size.height as f32 / grid_cell_size).ceil() as i32;

        // Estimate initial capacities for reusable buffers
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
            // OPTIMIZATION: Initialize buffers
            new_organism_buffer: Vec::with_capacity(new_org_capacity),
            removal_indices_buffer: Vec::with_capacity(removal_capacity),
            removal_indices_set: HashSet::with_capacity(removal_capacity),
            interaction_radii_buffer: Vec::with_capacity(initial_capacity),
            new_velocities_buffer: Vec::with_capacity(initial_capacity),
            plant_extra_aging_buffer: Vec::with_capacity(initial_capacity),
            fish_is_eating_buffer: Vec::with_capacity(initial_capacity),
            bug_is_eating_buffer: Vec::with_capacity(initial_capacity),
            plant_is_clustered_buffer: Vec::with_capacity(initial_capacity),
        };
        state.initialize_organisms();
        state
    }

    fn initialize_organisms(&mut self) {
        self.organisms.clear();
        let total_initial_count = INITIAL_PLANT_COUNT + INITIAL_FISH_COUNT + INITIAL_BUG_COUNT;
        self.organisms.reserve(total_initial_count.max(100));
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
        // OPTIMIZATION: Ensure buffer capacities match initial organisms
        self.resize_internal_buffers(self.organisms.len());
    }

    // OPTIMIZATION: Helper to resize all internal buffers consistently
    fn resize_internal_buffers(&mut self, capacity: usize) {
        // Removed unused variable: let capacity_with_headroom = (capacity + capacity / 8).max(16);

        // Ensure buffers always have *at least* the required capacity
        // Using resize might shrink, use reserve/extend or check capacity first
        if self.interaction_radii_buffer.len() < capacity {
            self.interaction_radii_buffer
                .resize(capacity, Default::default());
        }
        if self.new_velocities_buffer.len() < capacity {
            self.new_velocities_buffer.resize(capacity, None);
        }
        if self.plant_extra_aging_buffer.len() < capacity {
            self.plant_extra_aging_buffer.resize(capacity, 0.0);
        }
        if self.fish_is_eating_buffer.len() < capacity {
            self.fish_is_eating_buffer.resize(capacity, false);
        }
        if self.bug_is_eating_buffer.len() < capacity {
            self.bug_is_eating_buffer.resize(capacity, false);
        }
        if self.plant_is_clustered_buffer.len() < capacity {
            self.plant_is_clustered_buffer.resize(capacity, false);
        }
        // Ensure length matches EXACTLY for safe direct indexing later if needed,
        // although direct indexing accesses self.<buffer_name>[i] anyway.
        // Truncate if somehow buffers became longer than organisms list
        // (e.g. after massive organism death) - prevents out-of-bounds reads
        // if we were to take slices later, although we removed that approach.
        // Direct indexing [i] relies on `i < buffer.len()`.
        self.interaction_radii_buffer.truncate(capacity);
        self.new_velocities_buffer.truncate(capacity);
        self.plant_extra_aging_buffer.truncate(capacity);
        self.fish_is_eating_buffer.truncate(capacity);
        self.bug_is_eating_buffer.truncate(capacity);
        self.plant_is_clustered_buffer.truncate(capacity);

        // Buffers that just need clearing don't need resizing here,
        // Vec/HashSet handle capacity internally. Ensure initial capacity was decent.
    }

    fn create_organism(
        rng: &mut SimRng,
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
        // This is the simulation radius
        let radius = if min_radius < max_radius {
            rng.gen_range(min_radius..=max_radius)
        } else {
            min_radius
        };

        let velocity = if organism_config.movement_speed_factor > 0.0 {
            let angle = rng.gen_range(0.0..TAU);
            // Velocity depends on simulation radius
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
            rng,
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
            radius, // Store the simulation radius
            config: organism_config.clone(),
        }
    }

    fn create_offspring(
        parent: &Organism,
        window_size: PhysicalSize<u32>,
        rng: &mut SimRng,
        _config: &SimulationConfig, // Prefix unused parameter with underscore
    ) -> Organism {
        // Offset depends on simulation radius
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
        // New simulation radius
        let radius = (parent.radius + radius_delta).clamp(min_radius, max_radius);

        let velocity = if parent.config.movement_speed_factor > 0.0 {
            let angle = rng.gen_range(0.0..TAU);
            // Velocity depends on simulation radius
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
            radius,                        // Store the simulation radius
            config: parent.config.clone(), // Offspring gets config from parent
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
        // Optimization: Reserve capacity in grid HashMap entries if possible (more complex)
        // For now, rely on HashMap's internal growth.
        let avg_organisms_per_cell = (self.organisms.len() as f32
            / (self.grid_width * self.grid_height).max(1) as f32)
            .max(1.0);
        let expected_capacity_per_cell = (avg_organisms_per_cell * 1.5).ceil() as usize + 1;

        for (index, organism) in self.organisms.iter().enumerate() {
            let key = self.get_grid_key(organism.position);
            self.grid
                .entry(key)
                // OPTIMIZATION: Use or_default and push, slightly simpler? or_insert_with is fine.
                .or_insert_with(|| Vec::with_capacity(expected_capacity_per_cell))
                .push(index);
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

        let window_size = self.window_size;
        let current_organism_count = self.organisms.len();

        // --- Clear and Resize Buffers ---
        self.new_organism_buffer.clear();
        self.removal_indices_buffer.clear();
        self.removal_indices_set.clear();
        self.resize_internal_buffers(current_organism_count); // Ensure buffers are correct size

        // --- Reset State Buffers ---
        // Note: We access these via self.<buffer_name>[i] later, no need for mutable slices here.
        // Ensure buffers have the correct length *before* accessing indices up to current_organism_count.
        // resize_internal_buffers should guarantee this.
        for i in 0..current_organism_count {
            self.new_velocities_buffer[i] = None;
            self.plant_extra_aging_buffer[i] = 0.0;
            self.fish_is_eating_buffer[i] = false;
            self.bug_is_eating_buffer[i] = false;
            self.plant_is_clustered_buffer[i] = false;
        }

        // --- Build Grid (Needs &mut self) ---
        // Do this *before* the interaction loop that reads the grid.
        self.build_grid();

        // Borrow config locally (needed in multiple places)
        let config = &self.config;

        // --- Calculate Interaction Radii ---
        // Needs &self.organisms (immutable) and writes to &mut self.interaction_radii_buffer
        for (i, org) in self.organisms.iter().enumerate() {
            let radius = org.radius; // Use simulation radius
            // Direct indexing is safe because resize_internal_buffers ensured length
            self.interaction_radii_buffer[i] = match org.kind {
                OrganismType::Fish => InteractionRadii {
                    perception_sq: (radius * config.fish.perception_radius_factor).powi(2),
                    eating_sq: (radius * config.fish.eating_radius_factor).powi(2),
                    clustering_sq: 0.0,
                },
                OrganismType::Bug => InteractionRadii {
                    perception_sq: 0.0, // Bug perception not used currently
                    eating_sq: (radius * config.bug.eating_radius_factor).powi(2),
                    clustering_sq: 0.0,
                },
                OrganismType::Plant => InteractionRadii {
                    perception_sq: 0.0,
                    eating_sq: 0.0,
                    clustering_sq: (radius * config.plant.clustering_radius_factor).powi(2),
                },
            };
        }

        // --- Interaction Loop ---
        // This loop reads self.organisms, self.grid, self.config (all immutable borrows).
        // It calls self.get_grid_key (immutable borrow).
        // It writes to self.<buffer_name>[i] (mutable borrows via direct indexing, scopes limited per index).
        for i in 0..current_organism_count {
            let organism_i = &self.organisms[i]; // Immutable borrow of one organism
            let pos_i = organism_i.position;
            let grid_key_i = self.get_grid_key(pos_i); // Immutable borrow of self
            let radii_i = &self.interaction_radii_buffer[i]; // Immutable borrow of radii buffer element

            let mut fish_influence_vector = Vec2::ZERO;
            let mut fish_neighbor_count = 0;

            for dx in -1..=1 {
                for dy in -1..=1 {
                    let check_key = (grid_key_i.0 + dx, grid_key_i.1 + dy);
                    // Grid lookup uses immutable borrow of self.grid
                    if let Some(neighbor_indices) = self.grid.get(&check_key) {
                        for &j in neighbor_indices {
                            if i == j {
                                continue;
                            }
                            // Bounds check: Ensure neighbor index is valid for current organisms list
                            if j >= current_organism_count {
                                continue;
                            }

                            let organism_j = &self.organisms[j]; // Immutable borrow
                            let pos_j = organism_j.position;
                            let vec_ij = pos_j - pos_i;
                            let dist_sq = vec_ij.length_squared();

                            match organism_i.kind {
                                OrganismType::Fish => {
                                    if dist_sq < radii_i.perception_sq {
                                        if dist_sq > 1e-6 {
                                            let influence_factor = match organism_j.kind {
                                                OrganismType::Plant => config.fish.influence_plant,
                                                OrganismType::Fish => config.fish.influence_fish,
                                                OrganismType::Bug => config.fish.influence_bug,
                                            };
                                            if influence_factor.abs() > 1e-6 {
                                                fish_influence_vector +=
                                                    vec_ij.normalize_or_zero() * influence_factor;
                                                fish_neighbor_count += 1;
                                            }
                                        }
                                        if organism_j.kind == OrganismType::Plant
                                            && dist_sq < radii_i.eating_sq
                                        {
                                            // Mutable borrow of single buffer element - SAFE
                                            self.fish_is_eating_buffer[i] = true;
                                            // Bounds check for index j before mutable borrow
                                            if j < current_organism_count {
                                                self.plant_extra_aging_buffer[j] +=
                                                    config.plant.aging_rate_when_eaten;
                                            }
                                        }
                                    }
                                }
                                OrganismType::Plant => {
                                    if organism_j.kind == OrganismType::Plant
                                        && dist_sq < radii_i.clustering_sq
                                    {
                                        // Mutable borrow of single buffer element - SAFE
                                        self.plant_is_clustered_buffer[i] = true;
                                    }
                                }
                                OrganismType::Bug => {
                                    if organism_j.kind == OrganismType::Plant
                                        && dist_sq < radii_i.eating_sq
                                    {
                                        // Mutable borrow of single buffer element - SAFE
                                        self.bug_is_eating_buffer[i] = true;
                                        // Bounds check for index j before mutable borrow
                                        if j < current_organism_count {
                                            self.plant_extra_aging_buffer[j] +=
                                                config.plant.aging_rate_when_eaten;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Calculate new velocity based on interactions
            let radius_i = organism_i.radius; // Use simulation radius for movement scaling
            match organism_i.kind {
                OrganismType::Fish => {
                    let random_angle = self.rng.gen_range(0.0..TAU);
                    let random_direction = Vec2::from_angle(random_angle);
                    let normalized_influence = if fish_neighbor_count > 0 {
                        fish_influence_vector.normalize_or_zero()
                    } else {
                        Vec2::ZERO
                    };
                    let influence_weight = config.fish.influence_weight.clamp(0.0, 1.0);
                    let desired_direction = (random_direction * (1.0 - influence_weight)
                        + normalized_influence * influence_weight)
                        .normalize_or_zero();

                    let current_dir = organism_i.velocity.normalize_or_zero();
                    let angle_diff = current_dir.angle_between(desired_direction);
                    let max_turn = config.fish.max_turn_angle_per_sec * dt;
                    let turn_amount = angle_diff.clamp(-max_turn, max_turn);

                    let final_direction = if current_dir.length_squared() > 1e-6 {
                        Vec2::from_angle(current_dir.to_angle() + turn_amount)
                    } else {
                        desired_direction
                    };

                    // Mutable borrow of single buffer element - SAFE
                    self.new_velocities_buffer[i] =
                        Some(final_direction * config.fish.movement_speed_factor * radius_i);
                }
                OrganismType::Bug => {
                    let current_velocity = organism_i.velocity;
                    let target_speed = config.bug.movement_speed_factor * radius_i;
                    let current_dir = current_velocity.normalize_or_zero();

                    let final_dir = if current_dir.length_squared() > 1e-6 {
                        let max_turn_this_tick = config.bug.max_turn_angle_per_sec * dt;
                        let turn_angle = if max_turn_this_tick > 1e-9 {
                            self.rng.gen_range(-max_turn_this_tick..max_turn_this_tick)
                        } else {
                            0.0
                        };
                        Vec2::from_angle(current_dir.to_angle() + turn_angle)
                    } else {
                        Vec2::from_angle(self.rng.gen_range(0.0..TAU))
                    };
                    // Mutable borrow of single buffer element - SAFE
                    self.new_velocities_buffer[i] = Some(final_dir * target_speed);
                }
                OrganismType::Plant => {} // Velocity doesn't change based on interactions
            }
        } // End Interaction Loop

        // --- Update State & Spawning Loop ---
        // This loop reads interaction buffers (immutable borrow via index).
        // It modifies self.organisms (mutable borrow via index).
        // It adds to self.removal* buffers and self.new_organism_buffer (mutable borrows).
        for i in 0..current_organism_count {
            if self.removal_indices_set.contains(&i) {
                continue;
            }

            // --- Update Organism State ---
            {
                // Scope for mutable borrow of self.organisms[i]
                let organism = &mut self.organisms[i];

                // Apply velocity changes calculated in previous loop
                if let Some(new_vel) = self.new_velocities_buffer[i] {
                    // Read buffer immutably
                    organism.velocity = new_vel;
                }
                organism.position += organism.velocity * dt;

                // Wrap around screen edges
                organism.position.x =
                    (organism.position.x + window_size.width as f32) % window_size.width as f32;
                organism.position.y =
                    (organism.position.y + window_size.height as f32) % window_size.height as f32;

                // Aging
                let mut effective_age_increase = dt;
                // Read aging buffer immutably
                if organism.kind == OrganismType::Plant && self.plant_extra_aging_buffer[i] > 0.0 {
                    effective_age_increase +=
                        self.plant_extra_aging_buffer[i].min(MAX_PLANT_AGING_RATE_BONUS) * dt;
                }
                organism.age += effective_age_increase;

                // Check lifetime -> Add to removal buffers (mutable borrows)
                if organism.age >= organism.lifetime {
                    self.removal_indices_buffer.push(i);
                    self.removal_indices_set.insert(i);
                    continue; // Skip spawning check for this organism
                }
            } // End mutable borrow of self.organisms[i]

            // --- Spawning Logic ---
            // Re-borrow immutably if needed (already checked lifetime, so organism still exists)
            let organism = &self.organisms[i];
            let base_prob_per_sec = organism.growth_rate / organism.lifetime.max(1.0);
            let mut spawn_prob_this_tick = base_prob_per_sec * dt;

            // Apply spawn boosts (read interaction buffers immutably)
            match organism.kind {
                OrganismType::Fish if self.fish_is_eating_buffer[i] => {
                    spawn_prob_this_tick *= config.fish.eating_spawn_boost_factor;
                }
                OrganismType::Plant if self.plant_is_clustered_buffer[i] => {
                    // spawn_prob_this_tick *= config.plant.clustering_spawn_boost_factor;
                }
                OrganismType::Bug if self.bug_is_eating_buffer[i] => {
                    spawn_prob_this_tick *= config.bug.eating_spawn_boost_factor;
                }
                _ => {}
            }

            // Check probability and limits -> Add to new organism buffer (mutable borrow)
            if spawn_prob_this_tick > 0.0
                && self
                    .rng
                    .gen_bool(spawn_prob_this_tick.clamp(0.0, 1.0) as f64)
            {
                if current_organism_count - self.removal_indices_set.len()
                    + self.new_organism_buffer.len()
                    < MAX_ORGANISMS
                {
                    self.new_organism_buffer.push(Self::create_offspring(
                        organism, // Immutable borrow of organism
                        window_size,
                        &mut self.rng, // Mutable borrow of rng
                        config,        // Pass config (although create_offspring ignores it now)
                    ));
                }
            }
        } // End Update State & Spawning Loop

        // --- Apply Removals and Additions ---
        // Needs mutable borrow of self.organisms, self.removal_indices_buffer, self.new_organism_buffer
        self.removal_indices_buffer
            .sort_unstable_by(|a, b| b.cmp(a));
        for &index_to_remove in &self.removal_indices_buffer {
            if index_to_remove < self.organisms.len() {
                self.organisms.swap_remove(index_to_remove);
            } else {
                log::warn!(
                    "Attempted swap_remove with invalid index {}",
                    index_to_remove
                );
            }
        }

        self.organisms.extend(self.new_organism_buffer.drain(..));

        if self.organisms.len() > MAX_ORGANISMS {
            log::warn!(
                "Exceeded MAX_ORGANISMS, truncating from {} to {}",
                self.organisms.len(),
                MAX_ORGANISMS
            );
            self.organisms.truncate(MAX_ORGANISMS);
        }

        // Ensure internal buffers are sized correctly for the *next* frame's count
        self.resize_internal_buffers(self.organisms.len());
    }

    // --- NEW METHOD for preparing GPU data ---
    /// Creates a vector of data suitable for uploading to the GPU storage buffer.
    /// Applies a visual scaling factor to the radius specifically for rendering.
    pub fn get_gpu_data(&self, visual_radius_multiplier: f32) -> Vec<OrganismGpuData> {
        self.organisms
            .iter()
            .map(|org| {
                // Apply visual scaling ONLY here
                let render_radius = org.radius * visual_radius_multiplier.max(0.1); // Ensure non-zero

                OrganismGpuData {
                    world_position: org.position.into(), // Convert Vec2 to [f32; 2]
                    radius: render_radius,               // Use the visually scaled radius
                    _padding1: 0.0,                      // Explicitly set padding
                    color: org.color.into(),             // Convert Vec4 to [f32; 4]
                }
            })
            .collect()
    }
    // --- END NEW METHOD ---

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
            // Grid sizing still uses the *simulation* average radius and adjusted factor
            let avg_radius = self.config.get_avg_base_radius();
            self.grid_cell_size = avg_radius * GRID_CELL_SIZE_FACTOR; // Use adjusted constant
            self.grid_width = (new_size.width as f32 / self.grid_cell_size).ceil() as i32;
            self.grid_height = (new_size.height as f32 / self.grid_cell_size).ceil() as i32;
            // Clear the grid as organism positions relative to cells change
            self.grid.clear();
            // Buffers will be resized in the next update based on organism count.
            // No need to explicitly resize here unless we want to pre-allocate grid Vecs.
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
        // Grid sizing uses adjusted factor
        let avg_radius = self.config.get_avg_base_radius();
        self.grid_cell_size = avg_radius * GRID_CELL_SIZE_FACTOR; // Use adjusted constant
        self.grid_width = (self.window_size.width as f32 / self.grid_cell_size).ceil() as i32;
        self.grid_height = (self.window_size.height as f32 / self.grid_cell_size).ceil() as i32;
        self.initialize_organisms(); // This now also resizes internal buffers
        self.speed_multiplier = INITIAL_SPEED_MULTIPLIER;
        self.is_paused = false;
        self.grid.clear();
        // Clear reusable buffers explicitly on restart
        self.new_organism_buffer.clear();
        self.removal_indices_buffer.clear();
        self.removal_indices_set.clear();
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
}
