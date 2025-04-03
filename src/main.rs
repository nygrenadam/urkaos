// --- File: main.rs ---
// Declare modules
mod config;
mod constants;
mod renderer;
mod simulation;
mod utils;

// Use items from modules
use config::SimulationConfig;
use constants::*;
use renderer::Renderer;
use simulation::SimulationState;

// Keep necessary external crates used directly in main
use std::{sync::Arc, time::Instant};
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

// --- Constants for main loop ---
const MAX_UPDATES_PER_FRAME: u32 = 5;
const MAX_ACCUMULATED_TIME_CLAMP: f32 = MAX_UPDATES_PER_FRAME as f32 * FIXED_TIMESTEP;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .parse_default_env()
        .init();

    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Urkaos Life Simulation")
            .with_inner_size(PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .with_resizable(true)
            .build(&event_loop)?,
    );

    let mut renderer = pollster::block_on(Renderer::new(window.clone()));
    let simulation_config = SimulationConfig::new(); // Use new()
    let mut simulation_state = SimulationState::new(renderer.size, simulation_config);

    let mut last_sim_update_time = Instant::now();
    let mut time_accumulator: f32 = 0.0;

    let mut last_fps_update_time = Instant::now();
    let mut frames_since_last_fps_update = 0;

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::AboutToWait => {
                if !simulation_state.is_paused() {
                    let now = Instant::now();
                    let delta_time = now
                        .saturating_duration_since(last_sim_update_time)
                        .as_secs_f32();
                    last_sim_update_time = now;

                    time_accumulator += delta_time.min(MAX_ACCUMULATED_TIME_CLAMP);

                    let mut updates_this_frame = 0;
                    while time_accumulator >= FIXED_TIMESTEP
                        && updates_this_frame < MAX_UPDATES_PER_FRAME
                    {
                        simulation_state.update(FIXED_TIMESTEP);
                        time_accumulator -= FIXED_TIMESTEP;
                        updates_this_frame += 1;
                    }

                    // Keep a small residual time if updates finish early
                    // time_accumulator = time_accumulator.min(FIXED_TIMESTEP); // Keep or remove? Remove for less stutter if slightly behind.

                    if updates_this_frame == MAX_UPDATES_PER_FRAME {
                        // If lagging, reset accumulator fully to prevent spiral
                        time_accumulator = 0.0;
                        log::warn!("Simulation lagging, max updates per frame reached. Clamping accumulator.");
                    }
                } else {
                    // If paused, ensure time doesn't accumulate
                    last_sim_update_time = Instant::now();
                    time_accumulator = 0.0;
                }
                window.request_redraw(); // Request redraw regardless of pause state
            }

            Event::WindowEvent { window_id, event } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => elwt.exit(),

                WindowEvent::Resized(physical_size) => {
                    log::info!("Window resized to: {:?}", physical_size);
                    if physical_size.width > 0 && physical_size.height > 0 {
                        renderer.resize(physical_size);
                        simulation_state.resize(physical_size); // Simulation also needs resize for grid etc.
                    }
                }
                WindowEvent::ScaleFactorChanged { .. } => {
                    // On HiDPI systems, resizing might trigger ScaleFactorChanged before Resized.
                    // Handle resize based on the *new* inner_size after the scale factor changes.
                    let new_inner_size = window.inner_size();
                    if new_inner_size.width > 0 && new_inner_size.height > 0 {
                        log::info!("Scale factor changed, resizing to inner size: {:?}", new_inner_size);
                        renderer.resize(new_inner_size);
                        simulation_state.resize(new_inner_size);
                    }
                }

                WindowEvent::KeyboardInput { event: key_event, .. } => {
                    if key_event.state == ElementState::Pressed && !key_event.repeat {
                        match key_event.physical_key {
                            PhysicalKey::Code(KeyCode::ArrowUp) => simulation_state.adjust_speed(true),
                            PhysicalKey::Code(KeyCode::ArrowDown) => simulation_state.adjust_speed(false),
                            PhysicalKey::Code(KeyCode::Space) => simulation_state.toggle_pause(),
                            PhysicalKey::Code(KeyCode::KeyR) => {
                                simulation_state.restart();
                                // Reset time accumulator on restart
                                time_accumulator = 0.0;
                                last_sim_update_time = Instant::now();
                            }
                            PhysicalKey::Code(KeyCode::Escape) => elwt.exit(),
                            _ => {}
                        }
                    }
                }

                WindowEvent::RedrawRequested => {
                    frames_since_last_fps_update += 1;
                    let now = Instant::now();
                    let elapsed_secs = now.saturating_duration_since(last_fps_update_time).as_secs_f64();

                    if elapsed_secs >= FPS_UPDATE_INTERVAL_SECS {
                        let current_fps = frames_since_last_fps_update as f64 / elapsed_secs;
                        last_fps_update_time = now;
                        frames_since_last_fps_update = 0;

                        let (plant_count, fish_count, bug_count) = simulation_state.get_organism_counts();
                        let total_count = plant_count + fish_count + bug_count;
                        let paused_text = if simulation_state.is_paused() { " [PAUSED]" } else { "" };
                        window.set_title(&format!(
                            "Urkaos - Total: {} (P:{}, F:{}, B:{}) - Speed: {:.1}x - FPS: {:.1}{}",
                            total_count, plant_count, fish_count, bug_count,
                            simulation_state.speed_multiplier(), current_fps, paused_text
                        ));
                    }

                    // --- MODIFIED: Pass whole simulation_state ---
                    match renderer.render(&simulation_state, &simulation_state.config) {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            log::warn!("Surface lost or outdated, resizing renderer.");
                            let current_size = renderer.size;
                            if current_size.width > 0 && current_size.height > 0 {
                                renderer.resize(current_size);
                                // Simulation state resize happens via WindowEvent::Resized
                            } else {
                                log::warn!("Cannot resize renderer with zero dimensions.");
                            }
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("GPU OutOfMemory error!");
                            elwt.exit();
                        }
                        Err(e) => log::error!("Unhandled surface error: {:?}", e),
                    }
                }
                _ => {}
            },
            _ => {}
        }
    })?;

    Ok(())
}
