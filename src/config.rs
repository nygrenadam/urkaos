// --- File: config.rs ---
use crate::constants::BASE_ORGANISM_RADIUS;
use std::f32::consts::PI;
// Keep PI here for config defaults if needed

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
    // --- NEW: Minimum push-away radius ---
    pub organism_min_pushaway_radius: f32,
}

impl Default for OrganismConfig {
    fn default() -> Self {
        let min_radius = BASE_ORGANISM_RADIUS * 0.5;
        let max_radius = BASE_ORGANISM_RADIUS * 1.6;
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
            // --- NEW: Default push-away radius ---
            organism_min_pushaway_radius: 0.5, // Default small push-away
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

        // Plant Config
        config.plant.min_lifetime = 7.0;
        config.plant.max_lifetime = 45.0;
        config.plant.base_growth_rate = 1.1;
        config.plant.movement_speed_factor = 0.0;
        config.plant.base_color = [0.1, 0.7, 0.1, 1.0]; // Green
        config.plant.clustering_radius_factor = 64.0; // Plants cluster slightly
        config.plant.aging_rate_when_eaten = 1.4; // Age faster when eaten
        config.plant.eating_spawn_boost_factor_plant = 1.0;
        config.plant.eating_spawn_boost_factor_fish = 1.0;
        config.plant.eating_spawn_boost_factor_bug = 1.0;
        config.plant.min_radius = 1.0;
        config.plant.max_radius = 6.0;
        // --- NEW: Plant push-away ---
        config.plant.organism_min_pushaway_radius = 4.0; // Minimal push for plants

        // Fish Config
        config.fish.min_lifetime = 5.0;
        config.fish.max_lifetime = 65.0;
        config.fish.base_growth_rate = 1.05;
        config.fish.movement_speed_factor = 4.3;
        config.fish.max_turn_angle_per_sec = PI * 2.0; // Can turn quickly
        config.fish.perception_radius_factor = 32.0;
        config.fish.aging_rate_when_eaten = 4.0; // Age faster when eaten
        config.fish.eating_radius_factor = 5.5;
        config.fish.influence_plant = 0.1; // Attracted to plants
        config.fish.influence_fish = 0.00000105; // Slightly attracted to other fish
        config.fish.influence_bug = -0.8; // Avoids bugs
        config.fish.influence_weight = 0.8;
        config.fish.base_color = [0.2, 0.3, 0.9, 1.0]; // Blue
        config.fish.eating_spawn_boost_factor_plant = 1.4;
        config.fish.eating_spawn_boost_factor_fish = 1.0;
        config.fish.eating_spawn_boost_factor_bug = 1.0;
        config.fish.min_radius = 1.4;
        config.fish.max_radius = 4.0;
        // --- NEW: Fish push-away ---
        config.fish.organism_min_pushaway_radius = 8.0; // Fish push each other away more

        // Bug Config
        config.bug.min_lifetime = 5.0;
        config.bug.max_lifetime = 55.0;
        config.bug.base_growth_rate = 1.0;
        config.bug.movement_speed_factor = 4.2;
        config.bug.max_turn_angle_per_sec = PI / 32.0; //
        config.bug.perception_radius_factor = 16.0; //
        config.bug.eating_radius_factor = 2.0;
        config.bug.influence_plant = -0.3; //
        config.bug.influence_fish = 0.9; //
        config.bug.influence_bug = 0.5; //
        config.bug.influence_weight = 0.8;
        config.bug.base_color = [0.9, 0.6, 0.1, 1.0]; //
        config.bug.eating_spawn_boost_factor_plant = 1.1;
        config.bug.eating_spawn_boost_factor_fish = 1.5; //
        config.bug.eating_spawn_boost_factor_bug = 1.0; //
        config.bug.min_radius = 2.0;
        config.bug.max_radius = 4.0;
        // --- NEW: Bug push-away ---
        config.bug.organism_min_pushaway_radius = 2.0; // Bugs also push away

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
// --- End of File: config.rs ---