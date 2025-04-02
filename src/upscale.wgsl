// --- File: upscale.wgsl ---
// Upscales the low-resolution rendered texture to the screen using bilinear filtering.

// Input texture (rendered metaballs)
@group(0) @binding(1) var t_source: texture_2d<f32>;
// Sampler for the input texture
@group(0) @binding(0) var s_sampler: sampler;

// Vertex Shader Input/Output
struct VertexInput {
    @location(0) position: vec2<f32>, // NDC vertex position [-1, 1]
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>, // Texture coordinates [0, 1]
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Pass through clip space position
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    // Calculate texture coordinates from NDC position
    // Map x from [-1, 1] to [0, 1]
    // Map y from [-1, 1] to [1, 0] (texture coords often have Y=0 at the top)
    out.tex_coords = vec2<f32>(in.position.x * 0.5 + 0.5, in.position.y * -0.5 + 0.5);
    return out;
}

// Fragment Shader Input/Output
struct FragmentInput {
    @location(0) tex_coords: vec2<f32>,
};

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    // Sample the source texture using the sampler and interpolated tex coords
    return textureSample(t_source, s_sampler, in.tex_coords);
}
// --- End of File: upscale.wgsl ---