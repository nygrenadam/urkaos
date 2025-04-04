// --- File: constants.rs ---
// --- File: constants.rs ---
// --- Global Simulation Constants ---
pub const BACKGROUND_COLOR: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};
pub const BASE_ORGANISM_RADIUS: f32 = 8.0; // Use as a base for default configs

pub const INITIAL_SPEED_MULTIPLIER: f32 = 10.0;
pub const MIN_SPEED_MULTIPLIER: f32 = 0.0;
pub const MAX_SPEED_MULTIPLIER: f32 = 100.0;
pub const SPEED_ADJUST_FACTOR: f32 = 10.0;
pub const FIXED_TIMESTEP: f32 = 2.0 / 60.0;
pub const SPAWN_OFFSET_RADIUS_FACTOR: f32 = 3.0;
pub const INITIAL_PLANT_COUNT: usize = 150;
pub const INITIAL_FISH_COUNT: usize = 50;
pub const INITIAL_BUG_COUNT: usize = 5;
pub const WINDOW_WIDTH: u32 = 1024;
pub const WINDOW_HEIGHT: u32 = 1024;
pub const MAX_ORGANISMS: usize = 20_000;
pub const FPS_UPDATE_INTERVAL_SECS: f64 = 4.0;
// OPTIMIZATION: Grid cell size is crucial for shader performance now.
// Needs to be large enough to contain interaction radii but small enough for effective culling.
// Experiment with this value. A larger value might be better for the shader grid.
pub const GRID_CELL_SIZE_FACTOR: f32 = 8.0; // Start with the previous value, may need tuning.

// --- NEW: Global Scaling for Simulation Radius ---
// Applies *after* min/max from config, before simulation use.
// 1.0 = no change. > 1.0 = larger simulation radius, < 1.0 = smaller.
pub const GLOBAL_RADIUS_SCALE_FACTOR: f32 = 0.9;

pub const MAX_PLANT_AGING_RATE_BONUS: f32 = 10.0; // Keep this if it's a global cap unrelated to DNA

// --- NEW: Constants for Spawn/Death Radius Animation ---
// Factor controlling how fast organisms grow to target size (higher = faster)
pub const SPAWN_GROWTH_FACTOR: f32 = 1.5;
// Factor controlling how fast organisms shrink on death (higher = faster)
// pub const DEATH_SHRINK_FACTOR: f32 = 0.9; // Replaced by Corpse logic
// Radius threshold below which a shrinking organism is removed
pub const REMOVAL_RADIUS_THRESHOLD: f32 = 0.1; // Still useful for newly spawned/very small

// --- NEW: Constants for Corpses ---
pub const CORPSE_LIFETIME: f32 = 15.0; // How long corpses last before disappearing (in simulation seconds)
// pub const CORPSE_COLOR_FADE_FACTOR: f32 = 0.6; // Replaced by fixed corpse colors

// --- NEW: Corpse Colors ---
// Example: Greyish-Brown for stationary (Plant) corpses
pub const CORPSE_A_COLOR: wgpu::Color = wgpu::Color {
    r: 140.0 / 255.0, // 0x8C
    g: 120.0 / 255.0, // 0x78
    b: 100.0 / 255.0, // 0x64
    a: 0.8,           // Slightly transparent
};
// Example: Darker Grey for mobile (Fish, Bug) corpses
pub const CORPSE_B_COLOR: wgpu::Color = wgpu::Color {
    r: 80.0 / 255.0, // 0x50
    g: 80.0 / 255.0, // 0x50
    b: 85.0 / 255.0, // 0x55
    a: 0.85,         // Slightly more opaque
};
// --- End NEW ---

// --- End of File: constants.rs ---
