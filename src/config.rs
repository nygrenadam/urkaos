// --- File: config.rs ---
use crate::constants::BASE_ORGANISM_RADIUS;
use std::f32::consts::PI;
// Keep PI here for config defaults if needed

// --- NEW: Constant to tune color mutation sensitivity relative to DNA rates ---
const COLOR_MUTATION_SENSITIVITY: f32 = 4.0; // Adjust as needed

// --- NEW: Define sensible absolute min/max for mutation rates ---
pub const ABSOLUTE_MIN_MUTATION_RATE: f32 = 0.0001; // Prevent rates from becoming zero or negative
pub const ABSOLUTE_MAX_MUTATION_RATE: f32 = 0.5; // Prevent extremely high mutation rates

#[derive(Debug, Clone)]
pub struct OrganismConfig {
    pub min_lifetime: f32,
    pub max_lifetime: f32,
    pub base_growth_rate: f32,
    pub movement_speed_factor: f32,
    pub perception_radius_factor: f32,
    pub eating_radius_factor: f32,
    pub influence_plant: f32,
    pub influence_fish: f32,
    pub influence_bug: f32,
    pub influence_offspring: f32, // <<< NEW: Influence towards own offspring
    pub influence_weight: f32,
    pub max_turn_angle_per_sec: f32,
    pub eating_spawn_boost_factor_plant: f32,
    pub eating_spawn_boost_factor_fish: f32,
    pub eating_spawn_boost_factor_bug: f32,
    pub min_radius: f32,
    pub max_radius: f32,
    pub base_color: [f32; 4],
    pub clustering_radius_factor: f32,
    pub aging_rate_when_eaten: f32,
    pub organism_min_pushaway_radius: f32,

    // --- MODIFIED: DNA Mutation Rate Ranges ---
    // The *current* rate used for initial mutation (this value itself mutates).
    pub current_dna_mutation_rate: f32,
    // Defines the bounds within which current_dna_mutation_rate can evolve.
    pub min_dna_mutation_rate: f32,
    pub max_dna_mutation_rate: f32,

    // The *current* rate used for spawn mutation (this value itself mutates).
    pub current_dna_spawn_mutation_rate: f32,
    // Defines the bounds within which current_dna_spawn_mutation_rate can evolve.
    pub min_dna_spawn_mutation_rate: f32,
    pub max_dna_spawn_mutation_rate: f32,

    pub color_mutation_sensitivity: f32,
}

impl Default for OrganismConfig {
    fn default() -> Self {
        let min_radius = BASE_ORGANISM_RADIUS * 0.5;
        let max_radius = BASE_ORGANISM_RADIUS * 1.6;

        // Sensible default ranges for mutation rates
        let default_min_mutation = 0.005; // 0.5%
        let default_max_mutation = 0.1; // 10%
        let default_min_spawn_mutation = 0.001; // 0.1%
        let default_max_spawn_mutation = 0.05; // 5%

        Self {
            min_lifetime: 15.0,
            max_lifetime: 45.0,
            base_growth_rate: 1.0,
            movement_speed_factor: 0.0,
            perception_radius_factor: 0.0,
            eating_radius_factor: 0.0,
            influence_plant: 0.0,
            influence_fish: 0.0,
            influence_bug: 0.0,
            influence_offspring: 0.0, // <<< NEW: Default influence towards offspring
            influence_weight: 0.0,
            max_turn_angle_per_sec: 0.0,
            eating_spawn_boost_factor_plant: 1.0,
            eating_spawn_boost_factor_fish: 1.0,
            eating_spawn_boost_factor_bug: 1.0,
            min_radius,
            max_radius,
            base_color: [0.0, 0.0, 0.0, 1.0],
            clustering_radius_factor: 0.0,
            aging_rate_when_eaten: 0.0,
            organism_min_pushaway_radius: 0.5,

            // --- MODIFIED: Default DNA mutation rate ranges ---
            // Start current rates somewhere within the default range
            current_dna_mutation_rate: (default_min_mutation + default_max_mutation) / 2.0,
            min_dna_mutation_rate: default_min_mutation,
            max_dna_mutation_rate: default_max_mutation,
            current_dna_spawn_mutation_rate: (default_min_spawn_mutation
                + default_max_spawn_mutation)
                / 2.0,
            min_dna_spawn_mutation_rate: default_min_spawn_mutation,
            max_dna_spawn_mutation_rate: default_max_spawn_mutation,

            color_mutation_sensitivity: COLOR_MUTATION_SENSITIVITY,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SimulationConfig {
    pub plant: OrganismConfig,
    pub fish: OrganismConfig,
    pub bug: OrganismConfig,
}

impl SimulationConfig {
    pub fn new() -> Self {
        let mut config = Self::default();

        // --- Plant Config ---
        config.plant.min_lifetime = 25.0;
        config.plant.max_lifetime = 45.0;
        config.plant.base_growth_rate = 1.1;
        config.plant.movement_speed_factor = 0.0; // Stays 0
        config.plant.base_color = [0.1, 0.7, 0.1, 1.0]; // Green
        config.plant.clustering_radius_factor = 64.0;
        config.plant.aging_rate_when_eaten = 1.4;
        config.plant.min_radius = 5.0;
        config.plant.max_radius = 6.0;
        config.plant.organism_min_pushaway_radius = 4.0;
        config.plant.influence_offspring = 0.0; // Plants don't care about offspring location
        // DNA rates for Plant
        config.plant.min_dna_mutation_rate = 0.01; // 1% min
        config.plant.max_dna_mutation_rate = 0.08; // 8% max
        config.plant.current_dna_mutation_rate = 0.00006; // Start value
        config.plant.min_dna_spawn_mutation_rate = 0.0005; // 0.05% min
        config.plant.max_dna_spawn_mutation_rate = 0.01; // 1% max
        config.plant.current_dna_spawn_mutation_rate = 0.0015; // Start value

        // --- Fish Config ---
        config.fish.min_lifetime = 15.0;
        config.fish.max_lifetime = 35.0;
        config.fish.base_growth_rate = 2.55;
        config.fish.movement_speed_factor = 4.3;
        config.fish.max_turn_angle_per_sec = PI * 2.0;
        config.fish.perception_radius_factor = 16.0;
        config.fish.aging_rate_when_eaten = 4.0;
        config.fish.eating_radius_factor = 5.5;
        config.fish.influence_plant = 0.1;
        config.fish.influence_fish = 0.00000005;
        config.fish.influence_bug = -0.8;
        config.fish.influence_offspring = 0.9; //
        config.fish.influence_weight = 0.8;
        config.fish.base_color = [0.2, 0.3, 0.9, 1.0]; // Blue
        config.fish.eating_spawn_boost_factor_plant = 2.4;
        config.fish.min_radius = 3.4;
        config.fish.max_radius = 4.0;
        config.fish.organism_min_pushaway_radius = 8.0;
        // DNA rates for Fish
        config.fish.min_dna_mutation_rate = 0.01; // 1% min
        config.fish.max_dna_mutation_rate = 0.12; // 12% max
        config.fish.current_dna_mutation_rate = 0.00005; // Start value
        config.fish.min_dna_spawn_mutation_rate = 0.001; // 0.1% min
        config.fish.max_dna_spawn_mutation_rate = 0.05; // 5% max
        config.fish.current_dna_spawn_mutation_rate = 0.01; //

        // --- Bug Config ---
        config.bug.min_lifetime = 5.0;
        config.bug.max_lifetime = 55.0;
        config.bug.base_growth_rate = 1.0;
        config.bug.movement_speed_factor = 4.2;
        config.bug.max_turn_angle_per_sec = PI / 32.0;
        config.bug.perception_radius_factor = 16.0;
        config.bug.eating_radius_factor = 2.0;
        config.bug.influence_plant = -0.3;
        config.bug.influence_fish = 0.9;
        config.bug.influence_bug = 0.5;
        config.bug.influence_offspring = 0.4; // Bugs have stronger attraction to offspring
        config.bug.influence_weight = 0.8;
        config.bug.base_color = [0.9, 0.6, 0.1, 1.0]; // Orange/Brown
        config.bug.eating_spawn_boost_factor_fish = 1.5;
        config.bug.min_radius = 3.0;
        config.bug.max_radius = 4.0;
        config.bug.organism_min_pushaway_radius = 2.0;
        // DNA rates for Bug
        config.bug.min_dna_mutation_rate = 0.02; // 2% min
        config.bug.max_dna_mutation_rate = 0.15; // 15% max
        config.bug.current_dna_mutation_rate = 0.07; // Start value
        config.bug.min_dna_spawn_mutation_rate = 0.002; // 0.2% min
        config.bug.max_dna_spawn_mutation_rate = 0.08; // 8% max
        config.bug.current_dna_spawn_mutation_rate = 0.012; // Start value

        config
    }

    // get_avg_base_radius remains the same, using the initial config values
    pub fn get_avg_base_radius(&self) -> f32 {
        (self.plant.min_radius
            + self.plant.max_radius
            + self.fish.min_radius
            + self.fish.max_radius
            + self.bug.min_radius
            + self.bug.max_radius)
            / 6.0
    }
}
// --- End of File: config.rs ---
