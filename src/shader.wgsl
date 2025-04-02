// Metaball/SDF Shader (Vertex Buffer Version) - Optimized

// === Uniforms (Global Settings) ===
struct GlobalUniforms {
    screen_resolution: vec2<f32>,
    iso_level: f32,
    smoothness: f32,
    background_color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u_globals: GlobalUniforms;

// === Organism Data (Storage Buffer) ===
struct OrganismGpuData {
    world_position: vec2<f32>,
    radius: f32,
    _padding1: f32, // Keep padding for alignment
    color: vec4<f32>,
};
@group(1) @binding(0) var<storage, read> organisms_buffer: array<OrganismGpuData>;
@group(1) @binding(1) var<uniform> u_organism_count: u32;


// === Vertex Shader === (No changes needed or allowed)

struct VertexInput {
    @location(0) position: vec2<f32>, // NDC position from buffer
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


// === Fragment Shader === (Optimized Version)

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let screen_coord = vec2<f32>(
        frag_coord.x,
        u_globals.screen_resolution.y - frag_coord.y
    );

    var total_field: f32 = 0.0;
    var accumulated_color: vec4<f32> = vec4(0.0);
    let num_orgs = u_organism_count;

    for (var i: u32 = 0u; i < num_orgs; i = i + 1u) {
        let org = organisms_buffer[i];
        let radius = org.radius;
        let radius_sq = radius * radius;
        if (radius_sq <= 0.0) { continue; }

        let dist_vec = screen_coord - org.world_position;
        let dist_sq = dot(dist_vec, dist_vec);
        if (dist_sq >= radius_sq) { continue; }

        let rcp_radius_sq = 1.0 / radius_sq;
        let normalized_dist_sq = dist_sq * rcp_radius_sq;
        let h = max(0.0, 1.0 - normalized_dist_sq);
        let field_contrib = h * h;

        total_field += field_contrib;
        accumulated_color += org.color * field_contrib;
    }

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
