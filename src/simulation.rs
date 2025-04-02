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
}

// Helper struct for pre-calculated radii (used in simulation logic)
struct InteractionRadii {
    perception_sq: f32,
    eating_sq: f32,
    clustering_sq: f32,
}

impl SimulationState {
    pub fn new(window_size: PhysicalSize<u32>, config: SimulationConfig) -> Self {
        let avg_radius = config.get_avg_base_radius();
        // Use the *simulation* radius for grid sizing
        let grid_cell_size = avg_radius * GRID_CELL_SIZE_FACTOR;

        let grid_width = (window_size.width as f32 / grid_cell_size).ceil() as i32;
        let grid_height = (window_size.height as f32 / grid_cell_size).ceil() as i32;
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
        config: &SimulationConfig, // Changed parameter name for clarity
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
            radius, // Store the simulation radius
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

    fn build_grid(&mut self) {
        self.grid.clear();
        let avg_organisms_per_cell =
            self.organisms.len() as f32 / (self.grid_width * self.grid_height) as f32;
        let expected_capacity_per_cell = (avg_organisms_per_cell * 1.5).ceil() as usize + 1;

        for (index, organism) in self.organisms.iter().enumerate() {
            let key = self.get_grid_key(organism.position);
            self.grid
                .entry(key)
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

        // Interaction radii are calculated based on the *simulation* radius
        let interaction_radii: Vec<InteractionRadii> = self
            .organisms
            .iter()
            .map(|org| {
                let radius = org.radius; // Use simulation radius
                match org.kind {
                    OrganismType::Fish => InteractionRadii {
                        perception_sq: (radius * config.fish.perception_radius_factor).powi(2),
                        eating_sq: (radius * config.fish.eating_radius_factor).powi(2),
                        clustering_sq: 0.0,
                    },
                    OrganismType::Bug => InteractionRadii {
                        perception_sq: 0.0,
                        eating_sq: (radius * config.bug.eating_radius_factor).powi(2),
                        clustering_sq: 0.0,
                    },
                    OrganismType::Plant => InteractionRadii {
                        perception_sq: 0.0,
                        eating_sq: 0.0,
                        clustering_sq: (radius * config.plant.clustering_radius_factor).powi(2),
                    },
                }
            })
            .collect();

        for i in 0..current_organism_count {
            let organism_i = &self.organisms[i];
            let pos_i = organism_i.position;
            let grid_key_i = self.get_grid_key(pos_i);
            let radii_i = &interaction_radii[i]; // Use precalculated sim radii

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

                            // All interaction checks use the simulation radii (via radii_i)
                            match organism_i.kind {
                                OrganismType::Fish => {
                                    if dist_sq > 1e-6 && dist_sq < radii_i.perception_sq {
                                        let influence_factor = match organism_j.kind {
                                            OrganismType::Plant => config.fish.influence_plant,
                                            OrganismType::Fish => config.fish.influence_fish,
                                            OrganismType::Bug => config.fish.influence_bug,
                                        };
                                        if influence_factor.abs() > 1e-6 {
                                            if let Some(normalized_vec) = vec_ij.try_normalize() {
                                                fish_influence_vector +=
                                                    normalized_vec * influence_factor;
                                                fish_neighbor_count += 1;
                                            }
                                        }
                                    }
                                    if organism_j.kind == OrganismType::Plant
                                        && dist_sq < radii_i.eating_sq
                                    {
                                        fish_is_eating[i] = true;
                                        plant_extra_aging[j] += config.plant.aging_rate_when_eaten;
                                    }
                                }
                                OrganismType::Plant => {
                                    if organism_j.kind == OrganismType::Plant
                                        && dist_sq < radii_i.clustering_sq
                                    {
                                        plant_is_clustered[i] = true;
                                    }
                                }
                                OrganismType::Bug => {
                                    if organism_j.kind == OrganismType::Plant
                                        && dist_sq < radii_i.eating_sq
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
                    let final_direction = Vec2::from_angle(current_dir.to_angle() + turn_amount);
                    // Scale velocity by simulation radius
                    new_velocities[i] =
                        Some(final_direction * config.fish.movement_speed_factor * radius_i);
                }
                OrganismType::Bug => {
                    let current_velocity = organism_i.velocity;
                    // Scale speed by simulation radius
                    let target_speed = config.bug.movement_speed_factor * radius_i;
                    let current_dir = current_velocity.normalize_or_zero();
                    let max_turn_this_tick = config.bug.max_turn_angle_per_sec * dt;
                    let turn_angle = if max_turn_this_tick > 1e-9 {
                        self.rng.gen_range(-max_turn_this_tick..max_turn_this_tick)
                    } else {
                        0.0
                    };
                    let new_dir = Vec2::from_angle(current_dir.to_angle() + turn_angle);
                    new_velocities[i] = Some(new_dir * target_speed);
                }
                OrganismType::Plant => {}
            }
        }

        let mut removal_indices_set = HashSet::with_capacity(current_organism_count / 10);
        for i in 0..current_organism_count {
            if removal_indices_set.contains(&i) {
                continue;
            }

            {
                // Update logic uses simulation velocity/position
                let organism = &mut self.organisms[i];
                if let Some(new_vel) = new_velocities[i] {
                    organism.velocity = new_vel;
                }
                organism.position += organism.velocity * dt;
                organism.position.x =
                    (organism.position.x + window_size.width as f32) % window_size.width as f32;
                organism.position.y =
                    (organism.position.y + window_size.height as f32) % window_size.height as f32;

                let mut effective_age_increase = dt;
                if organism.kind == OrganismType::Plant && plant_extra_aging[i] > 0.0 {
                    effective_age_increase +=
                        plant_extra_aging[i].min(MAX_PLANT_AGING_RATE_BONUS) * dt;
                }
                organism.age += effective_age_increase;

                if organism.age >= organism.lifetime {
                    organisms_to_remove.push(i);
                    removal_indices_set.insert(i);
                    continue;
                }
            }

            // Spawning logic uses simulation radius/lifetime etc.
            let organism = &self.organisms[i];
            let base_prob_per_sec = organism.growth_rate / organism.lifetime.max(1.0);
            let mut spawn_prob_this_tick = base_prob_per_sec * dt;

            match organism.kind {
                OrganismType::Fish if fish_is_eating[i] => {
                    spawn_prob_this_tick *= config.fish.eating_spawn_boost_factor
                }
                OrganismType::Plant if plant_is_clustered[i] => { /* Maybe add clustering boost factor? */
                }
                OrganismType::Bug if bug_is_eating[i] => {
                    spawn_prob_this_tick *= config.bug.eating_spawn_boost_factor
                }
                _ => {}
            }

            if spawn_prob_this_tick > 0.0
                && self
                    .rng
                    .gen_bool(spawn_prob_this_tick.clamp(0.0, 1.0) as f64)
            {
                if current_organism_count - removal_indices_set.len() + new_organisms.len()
                    < MAX_ORGANISMS
                {
                    // create_offspring uses simulation radius logic internally
                    new_organisms.push(Self::create_offspring(
                        organism,
                        window_size,
                        &mut self.rng,
                        config,
                    ));
                }
            }
        }

        organisms_to_remove.sort_unstable_by(|a, b| b.cmp(a));
        for index_to_remove in organisms_to_remove {
            self.organisms.swap_remove(index_to_remove);
        }
        self.organisms.extend(new_organisms);
        if self.organisms.len() > MAX_ORGANISMS {
            self.organisms.truncate(MAX_ORGANISMS);
        }
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
            // Grid sizing still uses the *simulation* average radius
            let avg_radius = self.config.get_avg_base_radius();
            self.grid_cell_size = avg_radius * GRID_CELL_SIZE_FACTOR;
            self.grid_width = (new_size.width as f32 / self.grid_cell_size).ceil() as i32;
            self.grid_height = (new_size.height as f32 / self.grid_cell_size).ceil() as i32;
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
        // Grid sizing still uses the *simulation* average radius
        let avg_radius = self.config.get_avg_base_radius();
        self.grid_cell_size = avg_radius * GRID_CELL_SIZE_FACTOR;
        self.grid_width = (self.window_size.width as f32 / self.grid_cell_size).ceil() as i32;
        self.grid_height = (self.window_size.height as f32 / self.grid_cell_size).ceil() as i32;
        self.initialize_organisms();
        self.speed_multiplier = INITIAL_SPEED_MULTIPLIER;
        self.is_paused = false;
        self.grid.clear();
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
