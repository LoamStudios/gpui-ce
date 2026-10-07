//! Pressure strokes drawn as paths against the same strokes drawn as meshes:
//! N stroke outlines of 300 points each, built once, then drawn frame after
//! frame in an offscreen window.
//!
//! For each, it reports the one-time cost of building the strokes (lyon
//! tessellation for paths, tessellation and the fringe for meshes), and per
//! frame: the window's draw on the CPU (painting every stroke and finishing
//! the scene), the renderer's encoding of that scene on the CPU, and the
//! GPU's time on it, from Metal's command-buffer timestamps.
//!
//! ```sh
//! GPUI_RUN_BENCHMARKS=1 cargo test --release -p gpui_ce_platform --test meshes_bench
//! GPUI_RUN_BENCHMARKS=1 GPUI_BENCH_STROKES=500 cargo test --release -p gpui_ce_platform --test meshes_bench
//! ```
//!
//! Runs only with `GPUI_RUN_BENCHMARKS` set, on Metal.

#[cfg(target_os = "macos")]
use gpui::{
    AnyWindowHandle, AppContext as _, Context, IntoElement, Mesh, ParentElement as _, Path,
    PathBuilder, Pixels, Point, Render, Styled as _, VisualTestAppContext, Window, canvas, div,
    peniko, point, px, rgb,
};
#[cfg(target_os = "macos")]
use std::{
    rc::Rc,
    sync::Arc,
    time::{Duration, Instant},
};

const POINTS_PER_STROKE: usize = 300;
const FRAMES: usize = 40;
const WARM_UP: usize = 5;

/// A pressure stroke's outline: a wandering centreline of half the points,
/// out along its left edge and back along its right, its width swelling and
/// tapering as a pen's pressure would.
#[cfg(target_os = "macos")]
fn stroke(seed: u64) -> Vec<Point<Pixels>> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut random = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as f32 / (1u64 << 31) as f32
    };
    let samples = POINTS_PER_STROKE / 2;
    let (mut x, mut y) = (40. + random() * 1150., 40. + random() * 700.);
    let mut heading = random() * std::f32::consts::TAU;
    let turn = (random() - 0.5) * 0.08;
    let mut centre = Vec::with_capacity(samples);
    for _ in 0..samples {
        centre.push((x, y, heading));
        heading += turn + (random() - 0.5) * 0.05;
        x += heading.cos() * 1.5;
        y += heading.sin() * 1.5;
    }
    let width = 1.5 + random() * 5.;
    let half_width = |index: usize| {
        let t = index as f32 / (samples - 1) as f32;
        width * (std::f32::consts::PI * t).sin().max(0.05)
    };
    let side = |index: usize, sign: f32| {
        let (x, y, heading) = centre[index];
        let (nx, ny) = (-heading.sin(), heading.cos());
        let half = half_width(index) * sign;
        point(px(x + nx * half), px(y + ny * half))
    };
    (0..samples)
        .map(|index| side(index, 1.))
        .chain((0..samples).rev().map(|index| side(index, -1.)))
        .collect()
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq)]
enum Draw {
    Paths,
    Meshes,
}

#[cfg(target_os = "macos")]
struct Strokes {
    draw: Draw,
    paths: Rc<Vec<Path<Pixels>>>,
    meshes: Rc<Vec<Arc<Mesh>>>,
}

#[cfg(target_os = "macos")]
impl Render for Strokes {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let (draw, paths, meshes) = (self.draw, self.paths.clone(), self.meshes.clone());
        div().size_full().bg(rgb(0xffffff)).child(
            canvas(
                |_, _, _| (),
                move |_, _, window, _| {
                    let ink = rgb(0x202040);
                    match draw {
                        Draw::Paths => {
                            for path in paths.iter() {
                                window.paint_path(path.clone(), ink);
                            }
                        }
                        Draw::Meshes => {
                            for mesh in meshes.iter() {
                                window.paint_mesh(mesh, Point::default(), ink);
                            }
                        }
                    }
                },
            )
            .size_full(),
        )
    }
}

fn main() {
    if std::env::var_os("GPUI_RUN_BENCHMARKS").is_none() {
        return;
    }
    #[cfg(target_os = "macos")]
    run();
}

#[cfg(target_os = "macos")]
fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.
}

#[cfg(target_os = "macos")]
fn median(mut values: Vec<Duration>) -> Duration {
    values.sort();
    values[values.len() / 2]
}

#[cfg(target_os = "macos")]
fn run() {
    let strokes: usize = std::env::var("GPUI_BENCH_STROKES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2000);
    let outlines: Vec<_> = (0..strokes as u64).map(stroke).collect();

    let start = Instant::now();
    let paths: Vec<Path<Pixels>> = outlines
        .iter()
        .map(|outline| {
            let mut builder = PathBuilder::fill();
            builder.add_polygon(outline, true);
            builder.build().expect("a stroke tessellates")
        })
        .collect();
    let path_build = start.elapsed();
    let start = Instant::now();
    let meshes: Vec<Arc<Mesh>> = outlines
        .iter()
        .map(|outline| Arc::new(Mesh::from_polygon(outline, peniko::Fill::NonZero)))
        .collect();
    let mesh_build = start.elapsed();
    let path_vertices: usize = paths.iter().map(|path| path.vertices.len()).sum();
    let mesh_vertices: usize = meshes.iter().map(|mesh| mesh.vertices().len()).sum();
    let mesh_bytes: usize = meshes.iter().map(|mesh| mesh.byte_len()).sum();
    println!(
        "meshes_bench: {strokes} strokes of {POINTS_PER_STROKE} points; built as paths in {:.1} ms \
         ({path_vertices} vertices), as meshes in {:.1} ms ({mesh_vertices} vertices, {:.1} MB)",
        millis(path_build),
        millis(mesh_build),
        mesh_bytes as f64 / 1e6,
    );

    let (paths, meshes) = (Rc::new(paths), Rc::new(meshes));
    let mut cx = VisualTestAppContext::new(gpui_ce_platform::current_platform(false));
    let pool = Arc::new(parking_lot::Mutex::new(
        gpui_apple::metal_renderer::InstanceBufferPool::default(),
    ));
    let mut renderer = gpui_apple::metal_renderer::MetalRenderer::new_headless(pool);
    println!(
        "meshes_bench: {:>7}  {:>14}  {:>14}  {:>10}   (medians of {FRAMES} frames)",
        "", "window draw", "encode", "GPU"
    );
    for draw in [Draw::Paths, Draw::Meshes] {
        let window: AnyWindowHandle = cx
            .open_offscreen_window_default({
                let (paths, meshes) = (paths.clone(), meshes.clone());
                move |_, cx| {
                    cx.new(|_| Strokes {
                        draw,
                        paths,
                        meshes,
                    })
                }
            })
            .expect("failed to create the offscreen window")
            .into();
        cx.run_until_parked();
        let (mut drawn, mut encoded, mut gpu) = (Vec::new(), Vec::new(), Vec::new());
        for frame in 0..WARM_UP + FRAMES {
            let (draw_time, encode_time, gpu_time) = cx
                .update_window(window, |root, window, cx| {
                    cx.notify(root.entity_id());
                    let start = Instant::now();
                    window.draw(cx).clear(cx);
                    let draw_time = start.elapsed();
                    let scene = window.rendered_scene();
                    let size = window.viewport_size().map(|length| {
                        gpui::DevicePixels((f32::from(length) * window.scale_factor()) as i32)
                    });
                    let start = Instant::now();
                    renderer
                        .render_scene(scene, size)
                        .expect("the scene renders");
                    let encode_time = start.elapsed();
                    renderer
                        .render_scene_to_image(scene, size)
                        .expect("the scene renders");
                    (draw_time, encode_time, renderer.last_gpu_time().unwrap())
                })
                .expect("failed to draw the window");
            if frame >= WARM_UP {
                drawn.push(draw_time);
                encoded.push(encode_time);
                gpu.push(gpu_time);
            }
        }
        println!(
            "meshes_bench: {:>7}  {:>11.2} ms  {:>11.2} ms  {:>7.2} ms",
            format!("{draw:?}").to_lowercase(),
            millis(median(drawn)),
            millis(median(encoded)),
            millis(median(gpu)),
        );
    }
    let stats = renderer.mesh_stats();
    println!(
        "meshes_bench: the renderer uploaded {} meshes ({:.1} MB) over {} frames of meshes",
        stats.uploads,
        stats.uploaded_bytes as f64 / 1e6,
        WARM_UP + FRAMES,
    );
    std::mem::forget(cx);
}
