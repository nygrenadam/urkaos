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
    // --- Base Influences (towards Alive organisms) ---
    pub influence_plant: f32,
    pub influence_fish: f32,
    pub influence_bug: f32,
    pub influence_offspring: f32,
    // --- NEW: Corpse Influences ---
    pub influence_corpse_a: f32,   // Influence towards stationary corpses (e.g., dead plants)
    pub influence_corpse_b: f32,   // Influence towards mobile corpses (e.g., dead fish/bugs)
    pub influence_corpse_kin: f32, // Influence towards corpses of the same OrganismType
    // --- End NEW ---
    pub influence_weight: f32,
    pub max_turn_angle_per_sec: f32,
    pub eating_spawn_boost_factor_plant: f32,
    pub eating_spawn_boost_factor_fish: f32,
    pub eating_spawn_boost_factor_bug: f32,
    // --- Corpse Eating Boost Factors ---
    pub eating_spawn_boost_factor_corpse_a: f32, // For stationary corpses
    pub eating_spawn_boost_factor_corpse_b: f32, // For mobile corpses
    pub eating_spawn_boost_factor_corpse_kin: f32, // For corpses of the same OrganismType
    // --- End Corpse Eating ---
    pub min_radius: f32,
    pub max_radius: f32,
    pub base_color: [f32; 4],
    pub clustering_radius_factor: f32,
    pub aging_rate_when_eaten: f32,
    pub organism_min_pushaway_radius: f32,

    // --- DNA Mutation Rate Ranges ---
    pub current_dna_mutation_rate: f32,
    pub min_dna_mutation_rate: f32,
    pub max_dna_mutation_rate: f32,
    pub current_dna_spawn_mutation_rate: f32,
    pub min_dna_spawn_mutation_rate: f32,
    pub max_dna_spawn_mutation_rate: f32,

    pub color_mutation_sensitivity: f32,
}

impl Default for OrganismConfig {
    fn default() -> Self {
        let min_radius = BASE_ORGANISM_RADIUS * 0.8;
        let max_radius = BASE_ORGANISM_RADIUS * 1.2;

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
            influence_offspring: 0.0,
            // --- NEW: Default Corpse Influences ---
            influence_corpse_a: 0.0,
            influence_corpse_b: 0.0,
            influence_corpse_kin: 0.0,
            // --- End NEW ---
            influence_weight: 0.0,
            max_turn_angle_per_sec: 0.0,
            eating_spawn_boost_factor_plant: 1.0,
            eating_spawn_boost_factor_fish: 1.0,
            eating_spawn_boost_factor_bug: 1.0,
            // --- Corpse Eating Boost Factors ---
            eating_spawn_boost_factor_corpse_a: 1.0,
            eating_spawn_boost_factor_corpse_b: 1.0,
            eating_spawn_boost_factor_corpse_kin: 1.0,
            // --- End Corpse Eating ---
            min_radius,
            max_radius,
            base_color: [0.5, 0.5, 0.5, 1.0],
            clustering_radius_factor: 0.0,
            aging_rate_when_eaten: 0.0,
            organism_min_pushaway_radius: 1.0,

            // --- DNA mutation rate ranges ---
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
        config.plant.base_growth_rate = 1.9;
        config.plant.movement_speed_factor = 0.0; // Stays 0 -> Corpse Type A
        config.plant.base_color = [0.1, 0.7, 0.1, 1.0]; // Green
        config.plant.clustering_radius_factor = 64.0;
        config.plant.aging_rate_when_eaten = 1.2;
        config.plant.eating_spawn_boost_factor_fish = 2.4;
        // Plants don't eat, so corpse factors are irrelevant but set defaults
        config.plant.eating_spawn_boost_factor_corpse_a = 1.0;
        config.plant.eating_spawn_boost_factor_corpse_b = 1.0;
        config.plant.eating_spawn_boost_factor_corpse_kin = 1.0;
        // Plants don't perceive/move, corpse influences irrelevant
        config.plant.influence_corpse_a = 0.0;
        config.plant.influence_corpse_b = 0.0;
        config.plant.influence_corpse_kin = 0.0;

        config.plant.min_radius = 3.0;
        config.plant.max_radius = 4.0;
        config.plant.organism_min_pushaway_radius = 4.0;
        config.plant.influence_offspring = 0.0;
        // DNA rates for Plant
        config.plant.min_dna_mutation_rate = 0.01;
        config.plant.max_dna_mutation_rate = 0.08;
        config.plant.current_dna_mutation_rate = 0.06;
        config.plant.min_dna_spawn_mutation_rate = 0.0005;
        config.plant.max_dna_spawn_mutation_rate = 0.05;
        config.plant.current_dna_spawn_mutation_rate = 0.015;

        // --- Fish Config ---
        config.fish.min_lifetime = 5.0;
        config.fish.max_lifetime = 25.0;
        config.fish.base_growth_rate = 1.55;
        config.fish.movement_speed_factor = 5.3; // > 0 -> Corpse Type B
        config.fish.max_turn_angle_per_sec = PI * 1.61803;
        config.fish.perception_radius_factor = 32.0;
        config.fish.aging_rate_when_eaten = 4.0;
        config.fish.eating_radius_factor = 5.5;
        config.fish.influence_plant = 0.1;
        config.fish.influence_fish = -0.05;
        config.fish.influence_bug = -0.8;
        config.fish.influence_offspring = 0.9;
        // Fish Corpse Influences (Example values)
        config.fish.influence_corpse_a = 0.6; // Attracted to Plant corpses
        config.fish.influence_corpse_b = -1.0; // Slightly attracted to mobile corpses (bugs/other fish)
        config.fish.influence_corpse_kin = -0.1; // Slightly repelled by own kind's corpses
        config.fish.influence_weight = 0.9;
        config.fish.base_color = [0.9, 0.1, 0.1, 1.0]; // Reddish
        config.fish.eating_spawn_boost_factor_plant = 1.3;
        // Fish Corpse Eating Config
        config.fish.eating_spawn_boost_factor_corpse_a = 58.5;
        config.fish.eating_spawn_boost_factor_corpse_b = 1.0;
        config.fish.eating_spawn_boost_factor_corpse_kin = 0.8; // Less penalty than influence, but still less desirable
        config.fish.min_radius = 3.4;
        config.fish.max_radius = 4.0;
        config.fish.organism_min_pushaway_radius = 8.0;
        // DNA rates for Fish
        config.fish.min_dna_mutation_rate = 0.01;
        config.fish.max_dna_mutation_rate = 0.12;
        config.fish.current_dna_mutation_rate = 0.2;
        config.fish.min_dna_spawn_mutation_rate = 0.001;
        config.fish.max_dna_spawn_mutation_rate = 0.15;
        config.fish.current_dna_spawn_mutation_rate = 0.01;

        // --- Bug Config ---
        config.bug.min_lifetime = 55.0;
        config.bug.max_lifetime = 355.0;
        config.bug.base_growth_rate = 1.2;
        config.bug.movement_speed_factor = 2.2; // > 0 -> Corpse Type B
        config.bug.max_turn_angle_per_sec = PI / 32.0;
        config.bug.perception_radius_factor = 16.0;
        config.bug.eating_radius_factor = 2.0;
        config.bug.influence_plant = -0.3;
        config.bug.influence_fish = 0.9;
        config.bug.influence_bug = 0.5;
        config.bug.influence_offspring = 0.4;
        // Bug Corpse Influences (Example values)
        config.bug.influence_corpse_a = -0.2; // Repelled by Plant corpses
        config.bug.influence_corpse_b = 0.8; // Strongly attracted to Fish/Bug corpses
        config.bug.influence_corpse_kin = 0.3; // Attracted to own kind's corpses (scavenging)
        config.bug.influence_weight = 0.8;
        config.bug.base_color = [0.1, 0.1, 0.9, 1.0]; // Blueish
        config.bug.eating_spawn_boost_factor_fish = 1.5;
        // Bug Corpse Eating Config
        config.bug.eating_spawn_boost_factor_corpse_a = 1.0; // Neutral on Plant corpses
        config.bug.eating_spawn_boost_factor_corpse_b = 1.8;
        config.bug.eating_spawn_boost_factor_corpse_kin = 1.1;
        config.bug.min_radius = 5.0;
        config.bug.max_radius = 8.0;
        config.bug.organism_min_pushaway_radius = 2.0;
        // DNA rates for Bug
        config.bug.min_dna_mutation_rate = 0.02;
        config.bug.max_dna_mutation_rate = 0.15;
        config.bug.current_dna_mutation_rate = 0.07;
        config.bug.min_dna_spawn_mutation_rate = 0.002;
        config.bug.max_dna_spawn_mutation_rate = 0.08;
        config.bug.current_dna_spawn_mutation_rate = 0.012;

        config
    }

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