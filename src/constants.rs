
// --- Global Simulation Constants ---
pub const BACKGROUND_COLOR: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};
pub const BASE_ORGANISM_RADIUS: f32 = 3.0; // Use as a base for default configs

pub const INITIAL_SPEED_MULTIPLIER: f32 = 10.0;
pub const MIN_SPEED_MULTIPLIER: f32 = 0.0;
pub const MAX_SPEED_MULTIPLIER: f32 = 10.0;
pub const SPEED_ADJUST_FACTOR: f32 = 0.5;
pub const FIXED_TIMESTEP: f32 = 1.0 / 60.0;
pub const SPAWN_OFFSET_RADIUS_FACTOR: f32 = 3.0;
pub const INITIAL_PLANT_COUNT: usize = 700;
pub const INITIAL_FISH_COUNT: usize = 25;
pub const INITIAL_BUG_COUNT: usize = 50;
pub const WINDOW_WIDTH: u32 = 1024;
pub const WINDOW_HEIGHT: u32 = 1024;
pub const MAX_ORGANISMS: usize = 5_000;
pub const FPS_UPDATE_INTERVAL_SECS: f64 = 0.5;
// OPTIMIZATION: Increased factor to better accommodate largest interaction radii (fish perception * 10.0)
// Ensure cells are large enough so the 3x3 neighbor check is usually sufficient.
pub const GRID_CELL_SIZE_FACTOR: f32 = 32.0; // Was 10.0

pub const INITIAL_COLOR_MUTATION_MAX_DELTA: f32 = 0.15;
pub const OFFSPRING_COLOR_MUTATION_MAX_DELTA: f32 = 0.05;
pub const OFFSPRING_RADIUS_MUTATION_MAX_DELTA: f32 = 0.5;

// --- Geometry Configuration ---
// Removed NUM_POLYGON_SIDES as it's no longer used by the renderer
// pub const NUM_POLYGON_SIDES: u32 = 12;

pub const MAX_PLANT_AGING_RATE_BONUS: f32 = 10.0;

// --- NEW: Rendering Constants ---
// Scale factor for the internal rendering resolution (e.g., 0.5 for half resolution)
pub const RENDER_RESOLUTION_SCALE: f32 = 0.5; // Lower value = lower res, better perf
