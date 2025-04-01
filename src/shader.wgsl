// src/shader.wgsl

struct VertexInput {
    @location(0) position: vec2<f32>, // Quad vertex position (-1 to 1)
};

struct InstanceInput {
     @location(1) world_position: vec2<f32>, // Center of the organism in pixel coords
     @location(2) radius: f32,               // Radius in pixel coords
     @location(3) color: vec4<f32>,          // RGBA color
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>, // Output position in clip space (-1 to 1)
    @location(0) local_pos: vec2<f32>, // Pass local quad coord [-1, 1] to frag
    @location(1) color: vec4<f32>,     // Pass color to frag
};

struct Uniforms {
    screen_resolution: vec2<f32>, // width, height of the window in pixels
};
// Uniform buffer bound at group 0, binding 0
@group(0) @binding(0) var<uniform> uniforms: Uniforms;


@vertex
fn vs_main(
    model: VertexInput,     // Input from the vertex buffer (quad vertices)
    instance: InstanceInput // Input from the instance buffer (organism data)
) -> VertexOutput {
    // 1. Scale the quad vertex (initially -1 to 1) by the instance radius.
    // This gives us the vertex position relative to the instance center, in pixels.
    let scaled_local_pos = model.position * instance.radius;

    // 2. Add the instance's world position (in pixels) to get the final world position of the vertex.
    let world_pos = scaled_local_pos + instance.world_position;

    // 3. Convert world pixel coordinates (origin top-left) to Normalized Device Coordinates (NDC) (-1 to 1, origin center).
    // NDC.x = (world_pos.x / screen_width) * 2.0 - 1.0
    // NDC.y = (world_pos.y / screen_height) * 2.0 - 1.0
    // However, NDC Y increases upwards, while screen Y increases downwards. So we need to flip Y.
    // NDC.y = 1.0 - (world_pos.y / screen_height) * 2.0
    let zero_to_two = world_pos / uniforms.screen_resolution * 2.0;
    let inverted_y = vec2(zero_to_two.x, -zero_to_two.y); // Scale to [0, 2] range, flip Y
    let ndc_pos = inverted_y - vec2(1.0, -1.0); // Translate to [-1, 1] NDC range

    var out: VertexOutput;
    out.clip_position = vec4<f32>(ndc_pos, 0.0, 1.0); // Z = 0, W = 1
    out.local_pos = model.position; // Pass the original quad vertex position (-1 to 1)
    out.color = instance.color;     // Pass the instance color through
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
     // Calculate distance from the center of the quad (0,0 in local_pos)
     // local_pos is interpolated across the quad's surface.
     let dist = length(in.local_pos);

     // Discard fragment if its distance from the center is > 1.0
     // (since local_pos ranges from -1 to 1, length goes from 0 to sqrt(2),
     // but we only care about the distance relative to the circle's edge at 1.0)
     if dist > 1.0 {
         discard; // Don't draw this pixel
     }

    // Optional: Smooth the edge slightly for anti-aliasing
       // let smooth_edge = 1.0 - smoothstep(0.95, 1.0, dist);
     //return vec4<f32>(in.color.rgb, in.color.a * smooth_edge);

     // Return the solid color passed from the vertex shader
     return in.color;
}