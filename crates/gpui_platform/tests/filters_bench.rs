//! The GPU cost of a group's filters: a 1000×800 group (a gradient, and a
//! grid of squares over it) drawn plain, isolated, and through each kind of
//! filter, frame after frame in an offscreen window. Each frame draws the
//! group [`COPIES`] times, one over another, and its scene is rendered
//! [`REPEATS`] times back to back, so the GPU's clocks stay up for the work
//! measured; each render after the first is timed by Metal's command-buffer
//! timestamps, from the end of the one before to its own. It reports the
//! frame's median time, and each copy's cost over the plain group's.
//!
//! ```sh
//! GPUI_RUN_BENCHMARKS=1 cargo test --release -p gpui_ce_platform --test filters_bench
//! ```
//!
//! Runs only with `GPUI_RUN_BENCHMARKS` set, on Metal.

#[cfg(target_os = "macos")]
use gpui::{
    AnyWindowHandle, AppContext as _, Context, Filter, IntoElement, ParentElement as _, Render,
    Styled as _, VisualTestAppContext, Window, div, linear_color_stop, linear_gradient,
    prelude::FluentBuilder as _,
    px, rgb, rgba,
    shader::{self, Paint},
};
#[cfg(target_os = "macos")]
use std::{sync::Arc, time::Duration};

const FRAMES: usize = 60;
const COPIES: usize = 8;
const WARM_UP: usize = 10;
/// Each frame's scene is rendered this many times, back to back, so that the
/// GPU does not idle (and slow its clocks) between the renders timed.
const REPEATS: usize = 4;
/// Rounds over every case; the first warms the GPU's clocks and is not reported.
const ROUNDS: usize = 3;

#[cfg(target_os = "macos")]
struct Group {
    filters: Vec<Filter>,
    isolated: bool,
}

#[cfg(target_os = "macos")]
impl Render for Group {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let squares = (0..20 * 16).map(|index| {
            div()
                .absolute()
                .left(px((index % 20) as f32 * 50. + 10.))
                .top(px((index / 20) as f32 * 50. + 10.))
                .size(px(30.))
                .bg(rgb(0x2050a0 + index as u32 * 97))
        });
        let group = || {
            div()
                .absolute()
                .left(px(140.))
                .top(px(0.))
                .w(px(1000.))
                .h(px(800.))
                .bg(linear_gradient(
                    90.,
                    linear_color_stop(rgb(0xf0c080), 0.),
                    linear_color_stop(rgb(0x4080c0), 1.),
                ))
                .children(squares.clone())
                .filter(self.filters.clone())
                .when(self.isolated, |group| group.group_opacity(0.99))
        };
        div()
            .size_full()
            .relative()
            .bg(rgb(0xffffff))
            .children((0..COPIES).map(|_| group()))
    }
}

#[cfg(target_os = "macos")]
fn sharpen() -> Paint {
    shader::paint(|px| {
        let around = px.input_at(shader::vec2(1.0, 0.0)).rgba()
            + px.input_at(shader::vec2(-1.0, 0.0)).rgba()
            + px.input_at(shader::vec2(0.0, 1.0)).rgba()
            + px.input_at(shader::vec2(0.0, -1.0)).rgba();
        Paint::premultiplied((px.input().rgba() * 5.0 - around).clamp(0.0, 1.0))
    })
}

fn main() {
    if std::env::var_os("GPUI_RUN_BENCHMARKS").is_none() {
        return;
    }
    #[cfg(target_os = "macos")]
    run();
}

#[cfg(target_os = "macos")]
fn median(mut values: Vec<Duration>) -> Duration {
    values.sort();
    values[values.len() / 2]
}

#[cfg(target_os = "macos")]
fn run() {
    // SAFETY: single-threaded until here; the renderer reads it when made.
    unsafe { std::env::set_var("GPUI_LINK_PROGRAMS_SYNCHRONOUSLY", "1") };
    let mut cx = VisualTestAppContext::new(gpui_ce_platform::current_platform(false));
    let pool = Arc::new(parking_lot::Mutex::new(
        gpui_apple::metal_renderer::InstanceBufferPool::default(),
    ));
    let mut renderer = gpui_apple::metal_renderer::MetalRenderer::new_headless(pool);
    let cases: Vec<(&str, Vec<Filter>, bool)> = vec![
        ("plain", vec![], false),
        ("isolated, no filter", vec![], true),
        (
            "saturate (colour matrix)",
            vec![Filter::saturate(1.4)],
            false,
        ),
        (
            "blur 1.5 (full resolution)",
            vec![Filter::blur(px(1.5))],
            false,
        ),
        (
            "blur 4 (half resolution)",
            vec![Filter::blur(px(4.))],
            false,
        ),
        ("blur 16", vec![Filter::blur(px(16.))], false),
        (
            "drop shadow, blur 16",
            vec![Filter::drop_shadow(
                px(8.),
                px(12.),
                px(16.),
                rgba(0x00000080),
            )],
            false,
        ),
        (
            "program (3×3 sharpen)",
            vec![Filter::program(sharpen())],
            false,
        ),
    ];
    let mut plain = Duration::ZERO;
    for round in 0..ROUNDS {
        for (what, filters, isolated) in cases.clone() {
            let window: AnyWindowHandle = cx
                .open_offscreen_window_default(move |_, cx| cx.new(|_| Group { filters, isolated }))
                .expect("failed to create the offscreen window")
                .into();
            cx.run_until_parked();
            let mut gpu = Vec::new();
            let mut device_size = (0, 0);
            for frame in 0..WARM_UP + FRAMES {
                let time = cx
                    .update_window(window, |root, window, cx| {
                        cx.notify(root.entity_id());
                        window.draw(cx).clear(cx);
                        let scene = window.rendered_scene();
                        let size = window.viewport_size().map(|length| {
                            gpui::DevicePixels((f32::from(length) * window.scale_factor()) as i32)
                        });
                        device_size = (size.width.0, size.height.0);
                        renderer
                            .time_scene_renders(scene, size, REPEATS)
                            .expect("the scene renders")
                    })
                    .expect("failed to draw the window");
                if frame >= WARM_UP {
                    // The first follows an idle GPU; the rest follow each other.
                    gpu.extend_from_slice(&time[1..]);
                }
            }
            let time = median(gpu);
            if round == 0 {
                continue;
            }
            if what == "plain" {
                plain = time;
                println!(
                    "filters_bench: {COPIES} 1000×800 groups in a {}×{} viewport; GPU medians of {FRAMES}×{} renders; cost of each copy over the plain group's",
                    device_size.0,
                    device_size.1,
                    REPEATS - 1
                );
            }
            println!(
                "filters_bench: {what:<28} {:>6.2} ms a frame, +{:.3} ms a copy",
                time.as_secs_f64() * 1000.,
                time.saturating_sub(plain).as_secs_f64() * 1000. / COPIES as f64,
            );
        }
    }
    std::mem::forget(cx);
}
