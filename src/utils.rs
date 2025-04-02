// Removed unused Vec2 import
use glam::Vec4;
use rand::Rng;

// --- Helper Functions ---

// mutate_color
pub fn mutate_color<R: Rng + ?Sized>(base_color: Vec4, rng: &mut R, max_delta: f32) -> Vec4 {
    let r_delta = rng.gen_range(-max_delta..max_delta);
    let g_delta = rng.gen_range(-max_delta..max_delta);
    let b_delta = rng.gen_range(-max_delta..max_delta);
    let new_r = (base_color.x + r_delta).clamp(0.0, 1.0);
    let new_g = (base_color.y + g_delta).clamp(0.0, 1.0);
    let new_b = (base_color.z + b_delta).clamp(0.0, 1.0);
    Vec4::new(new_r, new_g, new_b, base_color.w)
}
