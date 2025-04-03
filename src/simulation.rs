// --- File: simulation.rs ---
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
    grid: HashMap<GridKey, Vec<usize>>, // CPU grid for simulation logic
    grid_width: i32,
    grid_height: i32,
    grid_cell_size: f32,
    // OPTIMIZATION: Buffers for reuse in update loop
    new_organism_buffer: Vec<Organism>,
    removal_indices_buffer: Vec<usize>,
    removal_indices_set: HashSet<usize>, // Also reuse hashset
    interaction_radii_buffer: Vec<InteractionRadii>, // Reuse radii buffer
    new_velocities_buffer: Vec<Option<Vec2>>,
    // Renamed for clarity: This accumulates the aging *rate* to apply later
    accumulated_aging_rate_buffer: Vec<f32>,
    plant_is_clustered_buffer: Vec<bool>,
    // Stores the maximum *beneficial* ( > 1.0) boost factor obtained from eating this tick
    eating_boost_buffer: Vec<f32>,

    // --- GPU grid data ---
    gpu_grid_indices: Vec<u32>, // Flat list of organism indices per cell
    gpu_grid_offsets: Vec<[u32; 2]>, // [offset, count] for each cell into gpu_grid_indices
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
        let grid_cell_size = avg_radius * GRID_CELL_SIZE_FACTOR;
        let grid_width = (window_size.width as f32 / grid_cell_size).ceil() as i32;
        let grid_height = (window_size.height as f32 / grid_cell_size).ceil() as i32;
        let num_grid_cells = (grid_width * grid_height) as usize;

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
            new_velocities_buffer: Vec::with_capacity(initial_capacity),
            // Renamed buffer initialization
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
        // --- NEW: Ensure GPU grid offset buffer has correct size ---
        let num_grid_cells = (self.grid_width * self.grid_height) as usize;
        self.gpu_grid_offsets.resize(num_grid_cells, [0, 0]);
        // --- End NEW ---
    }

    // OPTIMIZATION: Helper to resize all internal buffers consistently
    fn resize_internal_buffers(&mut self, capacity: usize) {
        // Resize buffers that need to match organism count exactly
        if self.interaction_radii_buffer.len() < capacity {
            self.interaction_radii_buffer.resize(capacity, Default::default());
        }
        if self.new_velocities_buffer.len() < capacity {
            self.new_velocities_buffer.resize(capacity, None);
        }
        // Renamed buffer resize
        if self.accumulated_aging_rate_buffer.len() < capacity {
            self.accumulated_aging_rate_buffer.resize(capacity, 0.0);
        }
        if self.plant_is_clustered_buffer.len() < capacity {
            self.plant_is_clustered_buffer.resize(capacity, false);
        }
        // NEW: Resize eating boost buffer, initialize new elements to 1.0 (no boost)
        if self.eating_boost_buffer.len() < capacity {
            self.eating_boost_buffer.resize(capacity, 1.0);
        }

        // Truncate buffers if they became longer than organisms list
        self.interaction_radii_buffer.truncate(capacity);
        self.new_velocities_buffer.truncate(capacity);
        // Renamed buffer truncate
        self.accumulated_aging_rate_buffer.truncate(capacity);
        self.plant_is_clustered_buffer.truncate(capacity);
        self.eating_boost_buffer.truncate(capacity); // NEW

        // Buffers that just need clearing don't need resizing here (e.g., new_organism_buffer).
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
        let radius = if min_radius < max_radius {
            rng.gen_range(min_radius..=max_radius)
        } else {
            min_radius
        };

        let velocity = if organism_config.movement_speed_factor > 0.0 {
            let angle = rng.gen_range(0.0..TAU);
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
        let radius = (parent.radius + radius_delta).clamp(min_radius, max_radius);

        let velocity = if parent.config.movement_speed_factor > 0.0 {
            let angle = rng.gen_range(0.0..TAU);
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
            radius,
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

    // --- Build grid also populates GPU grid data (No changes needed here) ---
    fn build_grid(&mut self) {
        self.grid.clear();
        self.gpu_grid_indices.clear();

        // --- Prepare GPU grid structure (using HashMap as intermediate) ---
        let mut temp_gpu_grid: HashMap<usize, Vec<u32>> = HashMap::new();
        let grid_w = self.grid_width; // Cache locally

        // Populate temporary grid and simulation grid
        for (index, organism) in self.organisms.iter().enumerate() {
            let key @ (cell_x, cell_y) = self.get_grid_key(organism.position);

            // Add to CPU grid (for simulation logic)
            self.grid.entry(key).or_default().push(index);

            // Add to temporary GPU grid (flat index)
            let flat_index = (cell_x + cell_y * grid_w) as usize;
            if flat_index < self.gpu_grid_offsets.len() {
                temp_gpu_grid
                    .entry(flat_index)
                    .or_default()
                    .push(index as u32);
            } else {
                log::warn!(
                    "Calculated flat grid index {} out of bounds (size {}) for organism at {:?}. Clamping.",
                    flat_index,
                    self.gpu_grid_offsets.len(),
                    organism.position
                );
                // Attempt to add to the last valid cell as a fallback? Or just skip? Skipping is safer.
            }
        }

        // Flatten the temporary grid into gpu_grid_indices and calculate offsets
        let mut current_offset = 0u32;
        for i in 0..self.gpu_grid_offsets.len() {
            if let Some(indices) = temp_gpu_grid.get(&i) {
                let count = indices.len() as u32;
                self.gpu_grid_offsets[i] = [current_offset, count];
                self.gpu_grid_indices.extend_from_slice(indices);
                current_offset += count;
            } else {
                // No organisms in this cell
                self.gpu_grid_offsets[i] = [current_offset, 0];
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

        let window_size = self.window_size;
        let current_organism_count = self.organisms.len();

        // --- Clear and Resize Buffers ---
        self.new_organism_buffer.clear();
        self.removal_indices_buffer.clear();
        self.removal_indices_set.clear();
        self.resize_internal_buffers(current_organism_count); // Ensure sim buffers are correct size

        // --- Reset State Buffers ---
        for i in 0..current_organism_count {
            self.new_velocities_buffer[i] = None;
            // Renamed buffer reset
            self.accumulated_aging_rate_buffer[i] = 0.0;
            self.plant_is_clustered_buffer[i] = false;
            // Reset eating boost buffer to 1.0 (no boost)
            self.eating_boost_buffer[i] = 1.0;
        }

        // --- Build Grid (Populates both CPU HashMap and GPU vectors) ---
        self.build_grid();

        // --- Calculate Interaction Radii ---
        for (i, org) in self.organisms.iter().enumerate() {
            let radius = org.radius;
            self.interaction_radii_buffer[i] = match org.kind {
                OrganismType::Plant => InteractionRadii {
                    perception_sq: 0.0,
                    eating_sq: 0.0, // Plants don't eat
                    clustering_sq: (radius * org.config.clustering_radius_factor).powi(2),
                },
                OrganismType::Fish => InteractionRadii {
                    perception_sq: (radius * org.config.perception_radius_factor).powi(2),
                    eating_sq: (radius * org.config.eating_radius_factor).powi(2),
                    clustering_sq: 0.0, // Fish don't cluster this way
                },
                OrganismType::Bug => InteractionRadii {
                    perception_sq: (radius * org.config.perception_radius_factor).powi(2), // Bug perception needed for influence
                    eating_sq: (radius * org.config.eating_radius_factor).powi(2),
                    clustering_sq: 0.0, // Bugs don't cluster this way
                },
            };
        }

        // --- Interaction Loop (Uses CPU grid HashMap) ---
        for i in 0..current_organism_count {
            // Early exit if organism was already marked for removal (e.g., eaten in a previous inner loop)
            // This check is less likely to be needed now but good for safety if interactions become more complex.
            if self.removal_indices_set.contains(&i) { continue; }

            let organism_i = &self.organisms[i]; // Borrow immutably first
            let pos_i = organism_i.position;
            let grid_key_i = self.get_grid_key(pos_i);
            let radii_i = self.interaction_radii_buffer[i]; // Already calculated

            let mut influence_vector = Vec2::ZERO;
            let mut neighbor_count = 0; // Generic neighbor count for influence

            for dx in -1..=1 {
                for dy in -1..=1 {
                    let check_key = (grid_key_i.0 + dx, grid_key_i.1 + dy);
                    // Use CPU grid for simulation interactions
                    if let Some(neighbor_indices) = self.grid.get(&check_key) {
                        for &j in neighbor_indices {
                            // Skip self, already removed, or out of bounds
                            if i == j { continue; }
                            if self.removal_indices_set.contains(&j) { continue; } // Skip if prey already marked
                            if j >= current_organism_count { continue; } // Safety check

                            // It's safe to borrow organism_j now
                            let organism_j = &self.organisms[j];
                            let pos_j = organism_j.position;
                            let vec_ij = pos_j - pos_i;
                            let dist_sq = vec_ij.length_squared();

                            // --- Influence Calculation (Fish and Bugs influence perception) ---
                            if (organism_i.kind == OrganismType::Fish || organism_i.kind == OrganismType::Bug)
                                && dist_sq < radii_i.perception_sq && dist_sq > 1e-6 {
                                let influence_factor = match organism_j.kind {
                                    OrganismType::Plant => organism_i.config.influence_plant,
                                    OrganismType::Fish => organism_i.config.influence_fish,
                                    OrganismType::Bug => organism_i.config.influence_bug,
                                };
                                if influence_factor.abs() > 1e-6 {
                                    influence_vector += vec_ij.normalize_or_zero() * influence_factor;
                                    neighbor_count += 1;
                                }
                            }

                            // --- Eating Calculation & Conditional Aging (Fish and Bugs eat) ---
                            if (organism_i.kind == OrganismType::Fish || organism_i.kind == OrganismType::Bug)
                                && dist_sq < radii_i.eating_sq
                            {
                                // Determine the boost factor based on predator (i) eating prey (j)
                                let boost_factor = match organism_j.kind {
                                    OrganismType::Plant => organism_i.config.eating_spawn_boost_factor_plant,
                                    OrganismType::Fish => organism_i.config.eating_spawn_boost_factor_fish,
                                    OrganismType::Bug => organism_i.config.eating_spawn_boost_factor_bug,
                                };

                                // --- Apply boost and aging ONLY if boost_factor > 1.0 ---
                                if boost_factor > 1.0 {
                                    // Update predator's boost buffer (take max boost from all prey eaten this tick)
                                    self.eating_boost_buffer[i] = self.eating_boost_buffer[i].max(boost_factor);

                                    // Apply aging penalty accumulation to the prey
                                    // Accumulate the rate defined in the *prey's* config
                                    let prey_aging_rate = organism_j.config.aging_rate_when_eaten;
                                    if prey_aging_rate > 0.0 {
                                        // Check index j validity again just in case
                                        if j < self.accumulated_aging_rate_buffer.len() { // Use buffer len for safety
                                            self.accumulated_aging_rate_buffer[j] += prey_aging_rate;
                                        }
                                    }
                                    // Optional: Mark prey for removal immediately if eaten by certain predators?
                                    // Example: if organism_i.kind == OrganismType::Bug && organism_j.kind == OrganismType::Fish {
                                    //     self.removal_indices_set.insert(j); // Mark fish j for removal
                                    // }
                                }
                                // If boost_factor <= 1.0, do nothing for boost or aging for this interaction.
                            }

                            // --- Plant Clustering (Only plants check other plants) ---
                            if organism_i.kind == OrganismType::Plant
                                && organism_j.kind == OrganismType::Plant
                                && dist_sq < radii_i.clustering_sq
                            {
                                self.plant_is_clustered_buffer[i] = true;
                            }

                        } // End loop over neighbors j
                    } // End if grid cell exists
                } // End loop dy
            } // End loop dx

            // --- Calculate New Velocity (based on influence, if applicable) ---
            if organism_i.kind == OrganismType::Fish || organism_i.kind == OrganismType::Bug {
                let radius_i = organism_i.radius; // Use already borrowed organism_i
                let random_angle = self.rng.gen_range(0.0..TAU);
                let random_direction = Vec2::from_angle(random_angle);
                let normalized_influence = if neighbor_count > 0 {
                    influence_vector.normalize_or_zero()
                } else {
                    // If no influence, maintain current direction (normalized velocity) or pick random if stopped
                    organism_i.velocity.normalize_or_zero()
                };
                let influence_weight = organism_i.config.influence_weight.clamp(0.0, 1.0);

                // If no neighbors influenced, desired direction tends towards random walk based on weight
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

                self.new_velocities_buffer[i] =
                    Some(final_direction * organism_i.config.movement_speed_factor * radius_i);
            }
            // Plants don't calculate velocity based on influence

        } // End Interaction Loop (over i)

        // --- Apply Removals Marked During Interaction (e.g., if immediate eating removal was implemented) ---
        // This requires converting the HashSet to a sorted Vec first if immediate removal occurs.
        // If only aging happens, removals are handled later based on age.
        // Example if using immediate removal:
        // if !self.removal_indices_set.is_empty() {
        //     self.removal_indices_buffer.extend(self.removal_indices_set.iter());
        //     // Avoid duplicates if already pushed from age later
        //     self.removal_indices_buffer.sort_unstable_by(|a, b| b.cmp(a));
        //     self.removal_indices_buffer.dedup();
        // }


        // --- Update State & Spawning Loop ---
        for i in 0..current_organism_count {
            // Skip if already marked for removal (e.g. eaten or previously aged out)
            if self.removal_indices_set.contains(&i) {
                continue;
            }

            // --- Update Organism State ---
            { // Scope for mutable borrow of self.organisms[i]
                let organism = &mut self.organisms[i];

                // Apply calculated velocity
                if let Some(new_vel) = self.new_velocities_buffer[i] {
                    organism.velocity = new_vel;
                }
                organism.position += organism.velocity * dt;

                // Wrap around screen edges
                organism.position.x =
                    (organism.position.x + window_size.width as f32) % window_size.width as f32;
                organism.position.y =
                    (organism.position.y + window_size.height as f32) % window_size.height as f32;

                // Calculate aging
                let mut effective_age_increase = dt;
                // Apply accumulated extra aging rate (if any)
                let accumulated_rate = self.accumulated_aging_rate_buffer[i];
                if accumulated_rate > 0.0 {
                    // The accumulated value IS the total rate from all predators this tick
                    effective_age_increase += accumulated_rate * dt;
                }
                organism.age += effective_age_increase;

                // Check for death by old age
                if organism.age >= organism.lifetime {
                    // Use try_insert which is slightly cleaner if duplicates might occur
                    if self.removal_indices_set.insert(i) {
                        self.removal_indices_buffer.push(i);
                    }
                    continue; // Skip spawning logic if dead
                }
            } // End mutable borrow for state update

            // --- Spawning Logic (Only if alive) ---
            let organism = &self.organisms[i]; // Immutable borrow now
            let base_prob_per_sec = organism.growth_rate / organism.lifetime.max(1.0);
            let mut spawn_prob_this_tick = base_prob_per_sec * dt;

            // Apply boost/penalty based on what was beneficially eaten this tick
            spawn_prob_this_tick *= self.eating_boost_buffer[i]; // Use the calculated boost factor (>1.0)

            // Apply boost for clustering (if applicable, currently only plants)
            if organism.kind == OrganismType::Plant && self.plant_is_clustered_buffer[i] {
                // Example: spawn_prob_this_tick *= 1.1; // Small boost for clustering
            }

            if spawn_prob_this_tick > 0.0 // Check probability is positive
                && self.rng.gen_bool(spawn_prob_this_tick.clamp(0.0, 1.0) as f64) // Clamp prob to [0, 1]
            {
                // Check against MAX_ORGANISMS, accounting for pending removals and additions
                let potential_next_count = self.organisms.len() - self.removal_indices_set.len() + self.new_organism_buffer.len() + 1;
                if potential_next_count <= MAX_ORGANISMS {
                    self.new_organism_buffer.push(Self::create_offspring(
                        organism,
                        window_size,
                        &mut self.rng,
                        &self.config, // Pass the main config reference
                    ));
                }
            }
        } // End Update State & Spawning Loop

        // --- Apply Removals and Additions ---
        // Sort removals by index descending to allow swap_remove without index issues
        self.removal_indices_buffer.sort_unstable_by(|a, b| b.cmp(a));
        // Ensure no duplicates if items were added to set and buffer separately
        self.removal_indices_buffer.dedup();

        for &index_to_remove in &self.removal_indices_buffer {
            if index_to_remove < self.organisms.len() { // Check index is still valid
                self.organisms.swap_remove(index_to_remove);
            } else {
                log::warn!(
                    "Attempted swap_remove with invalid index {} (current len {})",
                    index_to_remove, self.organisms.len()
                );
            }
        }
        self.organisms.extend(self.new_organism_buffer.drain(..));

        // Final check, although the check during spawning should prevent this usually
        if self.organisms.len() > MAX_ORGANISMS {
            log::warn!(
                "Exceeded MAX_ORGANISMS after additions/removals, truncating from {} to {}",
                self.organisms.len(),
                MAX_ORGANISMS
            );
            self.organisms.truncate(MAX_ORGANISMS);
        }

        // Ensure internal sim buffers are sized correctly for the *next* frame
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
            self.grid.clear(); // Clear CPU grid
            // --- Resize GPU grid offset buffer ---
            let num_grid_cells = (self.grid_width * self.grid_height) as usize;
            self.gpu_grid_offsets.clear(); // Clear old offsets
            self.gpu_grid_offsets.resize(num_grid_cells, [0, 0]); // Resize and fill with default
            self.gpu_grid_indices.clear(); // Indices will be rebuilt in next `build_grid`
            // --- End Resize ---
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
        self.initialize_organisms(); // Resizes sim buffers and gpu_grid_offsets
        self.speed_multiplier = INITIAL_SPEED_MULTIPLIER;
        self.is_paused = false;
        self.grid.clear(); // Clear CPU grid
        // Clear reusable sim buffers
        self.new_organism_buffer.clear();
        self.removal_indices_buffer.clear();
        self.removal_indices_set.clear();
        // --- Clear GPU grid index buffer ---
        self.gpu_grid_indices.clear();
        // Offsets buffer was already resized in initialize_organisms
        // --- End Clear ---
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

    // --- Getters for GPU grid data (No changes needed) ---
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
    // --- End Getters ---
}
// --- End of File: simulation.rs ---