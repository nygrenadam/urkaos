// WGSL Compute Shader for Organism Updates

// Data structure matching the Rust `OrganismGpu` struct (Pod/Zeroable version)
// NOTE: Order and types must match exactly! Add padding if needed for alignment.
struct OrganismGpu {
    position: vec2<f32>,
    velocity: vec2<f32>,
    color: vec4<f32>, // Keep color for potential future GPU use or simplicity
    age: f32,
    lifetime: f32,
    growth_rate: f32, // Keep for potential future GPU use
    radius: f32,
    kind: u32, // Use u32 for organism type (0=Plant, 1=Fish, 2=Bug)
    // Add explicit padding if necessary to meet alignment requirements (e.g., vec4)
    // Usually Rust's repr(C) handles this well with bytemuck. Check std140/std430 alignment.
    _padding: vec3<f32>, // Example padding to align to 16 bytes after kind (u32) if needed
};

// Array of organisms in a storage buffer (read/write)
@group(0) @binding(0) var<storage, read_write> organisms: array<OrganismGpu>;

// Uniforms containing simulation parameters
struct ComputeUniforms {
    delta_time: f32,
    window_width: f32,
    window_height: f32,
    num_organisms: u32, // Pass the current count to avoid out-of-bounds access
    // Add other necessary global parameters here (e.g., speed multiplier if applied on GPU)
};
@group(0) @binding(1) var<uniform> uniforms: ComputeUniforms;

const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;

// Bug specific config (simplified for GPU, pass more via uniforms if needed)
// These could be part of a larger config uniform struct
const BUG_MAX_TURN_ANGLE_PER_SEC: f32 = PI / 4.0;

// --- Helper Functions ---
// Basic pseudo-random number generator (replace with a better one if needed)
// Takes a seed (e.g., from global_id + frame count) and modifies it
var<private> rand_seed: u32;

fn init_rand(invocation_id: u32, frame: u32) {
    // Simple hash, combine invocation id and a frame counter (passed via uniform)
    rand_seed = invocation_id * 1103515245u + 12345u + frame * 137u;
}

fn random_u32() -> u32 {
    rand_seed = rand_seed * 1103515245u + 12345u;
    return rand_seed;
}

// Generate float in [0.0, 1.0)
fn random_float() -> f32 {
    return f32(random_u32() & 0x00FFFFFFu) / 16777216.0; // Use lower 24 bits
}

// Generate float in [-1.0, 1.0)
fn random_float_bilateral() -> f32 {
     return random_float() * 2.0 - 1.0;
}
// -----------------------


@compute @workgroup_size(64) // Adjust workgroup size based on GPU (e.g., 64, 128, 256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32> // ID of this specific shader instance
) {
    let index = global_id.x;

    // Bounds check: Don't process indices beyond the actual number of organisms
    if (index >= uniforms.num_organisms) {
        return;
    }

    // --- Read current state ---
    // Make a local copy to avoid excessive reads/writes to storage buffer
    var org = organisms[index];

    // --- Update Age ---
    org.age = org.age + uniforms.delta_time;

    // --- Update Position (using current velocity) ---
    org.position = org.position + org.velocity * uniforms.delta_time;

    // --- Boundary Conditions (Wrap around) ---
    // Use modulo operator (%) for wrapping
    // Add window dimension before modulo to handle negative results correctly
    org.position.x = (org.position.x % uniforms.window_width + uniforms.window_width) % uniforms.window_width;
    org.position.y = (org.position.y % uniforms.window_height + uniforms.window_height) % uniforms.window_height;


    // --- Optional: Simple Velocity Updates (Example: Bug Random Walk) ---
    // More complex interactions (fish steering) are harder and kept on CPU for now.
    // We need the organism kind. 0=Plant, 1=Fish, 2=Bug
    if (org.kind == 2u) { // If it's a Bug
        // This simple RNG needs a seed based on index and frame/time
        // init_rand(index, ???); // Need a frame counter uniform! (Add to ComputeUniforms)

        // Without frame counter, RNG will be the same every frame for the same bug
        // Let's just apply a *fixed* turn for demonstration (replace with proper RNG later)
        // let max_turn_this_tick = BUG_MAX_TURN_ANGLE_PER_SEC * uniforms.delta_time;
        // let turn_angle = sin(f32(index) * 0.1) * max_turn_this_tick; // Deterministic placeholder

        // A slightly better deterministic approach without frame counter:
        // Base turn on position to make it vary
        let combined_pos = org.position.x + org.position.y;
        let turn_factor = sin(combined_pos * 0.01 + f32(index) * 0.1); // Pseudo-randomish factor [-1, 1]
        let max_turn_this_tick = BUG_MAX_TURN_ANGLE_PER_SEC * uniforms.delta_time;
        let turn_angle = turn_factor * max_turn_this_tick;

        let current_speed = length(org.velocity);
        if (current_speed > 0.01) { // Avoid normalizing zero vector
            let current_dir = normalize(org.velocity);
            let current_angle = atan2(current_dir.y, current_dir.x);
            let new_angle = current_angle + turn_angle;
            let new_dir = vec2<f32>(cos(new_angle), sin(new_angle));
            org.velocity = new_dir * current_speed; // Maintain speed
        }
    }
    // --- End Simple Velocity Update ---


    // --- Write updated state back ---
    organisms[index] = org;
}