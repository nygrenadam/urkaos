// --- File: shader.wgsl ---
// Metaball/SDF Shader (Vertex Buffer Version) - Grid Optimized

// === Uniforms (Global Settings) ===
struct GlobalUniforms {
    // Renamed from screen_resolution -> render_resolution for clarity
    render_resolution: vec2<f32>,
    iso_level: f32,
    smoothness: f32,
    background_color: vec4<f32>,
    // NEW: Grid parameters
    grid_dims: vec2<i32>, // width, height
    grid_cell_size: f32,
    // padding? check alignment if needed
    _padding_grid: f32, // Added padding for alignment
};
@group(0) @binding(0) var<uniform> u_globals: GlobalUniforms;

// === Organism Data ===
struct OrganismGpuData {
    world_position: vec2<f32>,
    radius: f32,
    _padding1: f32,
    color: vec4<f32>,
};

// === Bind Group 1: Storage Buffers & Counts ===
@group(1) @binding(0) var<storage, read> organisms_buffer: array<OrganismGpuData>;
@group(1) @binding(1) var<uniform> u_organism_count: u32; // Still useful maybe? Or remove if unused. Keep for now.
// NEW: Grid Data Buffers
// Stores [offset, count] for each grid cell
@group(1) @binding(2) var<storage, read> grid_offsets: array<vec2<u32>>;
// Stores flat list of organism indices for all cells concatenated
@group(1) @binding(3) var<storage, read> grid_indices: array<u32>;


// === Vertex Shader === (No changes)
struct VertexInput {
    @location(0) position: vec2<f32>,
};
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    return out;
}


// === Fragment Shader === (Grid Optimized Version)
@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // Screen coordinate (pixels, origin top-left)
    let screen_coord = vec2<f32>(
        frag_coord.x,
        // Flip Y coordinate - Check if needed based on WGPU setup (often is)
        u_globals.render_resolution.y - frag_coord.y
    );

    var total_field: f32 = 0.0;
    var accumulated_color: vec4<f32> = vec4(0.0);

    // --- Grid Optimization ---
    let grid_dims = u_globals.grid_dims;
    let cell_size = u_globals.grid_cell_size;

    // Calculate the grid cell this fragment belongs to
    let frag_cell_xy = vec2<i32>(floor(screen_coord / cell_size));

    // Iterate through the 3x3 neighborhood around the fragment's cell
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let neighbor_cell_xy = frag_cell_xy + vec2(dx, dy);

            // Clamp cell coordinates to be within grid bounds
            // Use explicit i32 vector constructor syntax
            let clamped_cell_xy = clamp(neighbor_cell_xy, vec2<i32>(0, 0), grid_dims - vec2<i32>(1, 1));

            // Convert 2D cell coord to 1D index for buffer lookup
            let cell_flat_index = u32(clamped_cell_xy.x + clamped_cell_xy.y * grid_dims.x);

            // Check bounds before accessing grid_offsets
            // Note: This check adds overhead. If confident grid_dims is always correct
            // and matches the buffer size, this could potentially be removed after testing.
            if (cell_flat_index >= arrayLength(&grid_offsets)) {
                continue; // Skip invalid cell index
            }

            let offset_count = grid_offsets[cell_flat_index];
            let cell_offset = offset_count.x;
            let cell_count = offset_count.y;

            // Loop through organism indices stored for this cell
            for (var k: u32 = 0u; k < cell_count; k = k + 1u) {
                let index_in_indices_buffer = cell_offset + k;

                // Check bounds before accessing grid_indices
                if (index_in_indices_buffer >= arrayLength(&grid_indices)) {
                    continue; // Skip invalid index
                }
                let organism_index = grid_indices[index_in_indices_buffer];

                // Check bounds before accessing organisms_buffer
                if (organism_index >= arrayLength(&organisms_buffer)) {
                    continue; // Skip invalid index
                }
                let org = organisms_buffer[organism_index];

                // --- Metaball Calculation (Same as before, but only for relevant orgs) ---
                let radius = org.radius;
                if (radius <= 0.0) { continue; }
                let radius_sq = radius * radius;

                let dist_vec = screen_coord - org.world_position;
                let dist_sq = dot(dist_vec, dist_vec);

                if (dist_sq >= radius_sq) { continue; }

                // Calculate field contribution
                let falloff = max(0.0, 1.0 - dist_sq / radius_sq);
                let field_contrib = falloff * falloff; // Quadratic falloff

                total_field += field_contrib;
                // Weight color by contribution
                accumulated_color += org.color * field_contrib;
                // --- End Metaball Calculation ---
            } // End loop k over organisms in cell
        } // End loop dx
    } // End loop dy
    // --- End Grid Optimization ---

    // --- Final Color Calculation (Same as before) ---
    let safe_total_field = max(total_field, 0.0001);
    let final_rgb = accumulated_color.rgb / safe_total_field;

    let alpha = smoothstep(
        u_globals.iso_level - u_globals.smoothness,
        u_globals.iso_level + u_globals.smoothness,
        total_field
    );

    let final_color = mix(
        u_globals.background_color,
        vec4(final_rgb, 1.0),
        alpha
    );

    return final_color;
}