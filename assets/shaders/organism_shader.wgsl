// We only need to define things relevant to our fragment shader and material uniforms.

// This struct must match the AsBindGroup layout defined in Rust
@group(1) @binding(0)
var<uniform> material: Material;

// Define the structure for our material uniform buffer
struct Material {
    // Corresponds to the `color: Color` field in Rust (which becomes vec4<f32> in WGSL)
    color: vec4<f32>,
};

// Define the input structure expected by *this* fragment shader.
// This must match the output provided by the default Material2d vertex shader.
// The default vertex shader interpolates UV coordinates at location 0.
struct FragmentInput {
    @location(0) uv: vec2<f32>,
    // Other inputs like world position/normal might be available but aren't needed here.
};


// --- Fragment Shader ---
// Takes the standard interpolated inputs from the default vertex shader.
@fragment
fn fragment(in: FragmentInput) -> @location(0) vec4<f32> {
    // UV coordinates range from (0,0) top-left to (1,1) bottom-right.
    // Center is (0.5, 0.5).
    let center = vec2<f32>(0.5, 0.5);

    // Calculate the distance from the fragment's UV to the center UV.
    let dist_from_center = distance(in.uv, center);

    // Normalize the distance. The radius of the inscribed circle in UV coordinates is 0.5.
    // We normalize the distance based on this radius (0.0 at center, 1.0 at edge).
    let normalized_dist = dist_from_center / 0.5; // Equivalent to dist_from_center * 2.0

    // Discard fragments outside the circular radius (normalized_dist > 1.0)
    // This makes the quad appear as a circle. Even with a Circle mesh, this ensures
    // that the alpha gradient is applied correctly within the expected UV bounds.
    if (normalized_dist > 1.0) {
        discard; // Stop processing this fragment, making it transparent
    }

    // Calculate the alpha: 1.0 at the center (normalized_dist = 0) and 0.0 at the edge (normalized_dist = 1).
    let alpha = 1.0 - clamp(normalized_dist, 0.0, 1.0); // Fades in (opaque center, transparent edge)

    // Combine the material's base color with the calculated alpha.
    // Multiply by the material's base alpha (which should be 1.0 unless intentionally set lower).
    return vec4<f32>(material.color.rgb, alpha * material.color.a);
}