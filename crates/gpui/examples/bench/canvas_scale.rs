//! How many live, positioned elements a frame holds.
//!
//! A grid of interactive items — each with an id, a hover style, a mouse
//! handler, a border, rounded corners, and a label once it is large enough to
//! read — laid out on a page larger than the window. A camera zooms from close
//! up to the whole grid and back, and only the items in view are built each
//! frame. Every frame records how many items were built and how long the
//! frame took on the CPU, from the start of `render` to the paint of the last
//! element, which covers building, layout, prepaint and paint but not the GPU.
//!
//! With `--paint`, the same items are painted as quads by one element, with
//! no per-item elements, layout or hit regions and no labels: the floor that
//! element overhead sits on.
//!
//! ```text
//! cargo run -p gpui-ce --release --example canvas_scale                    # 100,000 items
//! cargo run -p gpui-ce --release --example canvas_scale -- 20000           # a smaller grid
//! cargo run -p gpui-ce --release --example canvas_scale -- 100000 --paint  # quads only
//! ```
//!
//! The run lasts one zoom out and back, then prints frame times grouped by how
//! many items were visible, and quits.

#[path = "../example_support/fonts.rs"]
mod example_support;

use std::{
    cell::RefCell,
    rc::Rc,
    time::{Duration, Instant},
};

use gpui::{
    App, BorderStyle, Bounds, Context, MouseButton, Render, SharedString, Window, WindowBounds,
    WindowOptions, canvas, div, hsla, point, prelude::*, px, quad, size,
};
use gpui_platform::application;

/// Items per row of the grid.
const COLUMNS: usize = 400;
/// An item's size and the gap between items, in page units.
const ITEM_W: f32 = 120.;
const ITEM_H: f32 = 72.;
const GAP: f32 = 24.;
/// The closest and farthest zoom the camera reaches.
const ZOOM_IN: f32 = 4.0;
const ZOOM_OUT: f32 = 0.012;
/// One zoom out and back in.
const RUN: Duration = Duration::from_secs(30);
/// The smallest on-screen width at which an item shows its label.
const LABEL_WIDTH: f32 = 48.;

/// One frame's measurement: items built, CPU time, and the interval since the
/// previous frame.
#[derive(Clone, Copy)]
struct Sample {
    visible: usize,
    cpu: Duration,
    interval: Option<Duration>,
}

struct Stage {
    items: usize,
    /// Paint quads from one element instead of building an element per item.
    direct: bool,
    started: Instant,
    last_frame: Option<Instant>,
    samples: Rc<RefCell<Vec<Sample>>>,
    finished: bool,
}

impl Stage {
    fn new(items: usize, direct: bool) -> Self {
        Self {
            items,
            direct,
            started: Instant::now(),
            last_frame: None,
            samples: Rc::new(RefCell::new(Vec::new())),
            finished: false,
        }
    }

    /// The camera's zoom at `elapsed`: log-linear from `ZOOM_IN` to
    /// `ZOOM_OUT` over the first half of the run, and back over the second.
    fn zoom(elapsed: Duration) -> f32 {
        let phase = (elapsed.as_secs_f32() / RUN.as_secs_f32()).min(1.0);
        let out = if phase < 0.5 { phase * 2.0 } else { (1.0 - phase) * 2.0 };
        let (near, far) = (ZOOM_IN.ln(), ZOOM_OUT.ln());
        (near + (far - near) * out).exp()
    }

    fn report(&self) {
        let samples = self.samples.borrow();
        let buckets: [(usize, usize); 7] = [
            (0, 500),
            (500, 2_000),
            (2_000, 5_000),
            (5_000, 10_000),
            (10_000, 20_000),
            (20_000, 50_000),
            (50_000, usize::MAX),
        ];
        println!(
            "canvas_scale: {} items in the grid, {}, {} frames over {:.1}s",
            self.items,
            if self.direct { "painted as quads" } else { "one element each" },
            samples.len(),
            self.started.elapsed().as_secs_f32()
        );
        println!("canvas_scale: visible items  frames   CPU median   CPU p95   interval median");
        for (low, high) in buckets {
            let in_bucket: Vec<&Sample> = samples
                .iter()
                .filter(|sample| sample.visible >= low && sample.visible < high)
                .collect();
            if in_bucket.is_empty() {
                continue;
            }
            let mut cpu: Vec<f64> = in_bucket.iter().map(|s| millis(s.cpu)).collect();
            let mut interval: Vec<f64> =
                in_bucket.iter().filter_map(|s| s.interval.map(millis)).collect();
            let range = if high == usize::MAX {
                format!("{low}+")
            } else {
                format!("{low}–{high}")
            };
            println!(
                "canvas_scale: {range:>13}  {:>6}   {:>7.2} ms  {:>6.2} ms   {:>8.2} ms",
                in_bucket.len(),
                quantile(&mut cpu, 0.5),
                quantile(&mut cpu, 0.95),
                quantile(&mut interval, 0.5),
            );
        }
    }
}

impl Render for Stage {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let frame_start = Instant::now();
        let interval = self.last_frame.replace(frame_start).map(|last| frame_start - last);
        let elapsed = self.started.elapsed();

        if elapsed >= RUN {
            if !self.finished {
                self.finished = true;
                self.report();
                println!("drive: done");
                cx.defer(|cx| cx.quit());
            }
        } else {
            window.request_animation_frame();
        }

        let zoom = Self::zoom(elapsed);
        let viewport = window.viewport_size();
        let (view_w, view_h) = (f32::from(viewport.width), f32::from(viewport.height));
        let rows = self.items.div_ceil(COLUMNS);
        let (pitch_x, pitch_y) = (ITEM_W + GAP, ITEM_H + GAP);
        let page_center_x = COLUMNS as f32 * pitch_x / 2.0;
        let page_center_y = rows as f32 * pitch_y / 2.0;

        // The page rectangle in view, and the grid cells it covers.
        let left = page_center_x - view_w / zoom / 2.0;
        let top = page_center_y - view_h / zoom / 2.0;
        let first_column = (left / pitch_x).floor().max(0.0) as usize;
        let last_column = (((left + view_w / zoom) / pitch_x).ceil() as usize).min(COLUMNS);
        let first_row = (top / pitch_y).floor().max(0.0) as usize;
        let last_row = (((top + view_h / zoom) / pitch_y).ceil() as usize).min(rows);

        let item_w = ITEM_W * zoom;
        let item_h = ITEM_H * zoom;
        let labelled = item_w >= LABEL_WIDTH;

        if self.direct {
            let items = self.items;
            let samples = self.samples.clone();
            let painter = canvas(
                |_, _, _| {},
                move |bounds, _, window, _| {
                    let mut visible = 0;
                    for row in first_row..last_row {
                        for column in first_column..last_column {
                            let index = row * COLUMNS + column;
                            if index >= items {
                                break;
                            }
                            let x = (column as f32 * pitch_x - left) * zoom;
                            let y = (row as f32 * pitch_y - top) * zoom;
                            let hue = (index * 37 % 360) as f32 / 360.0;
                            window.paint_quad(quad(
                                Bounds::new(
                                    bounds.origin + point(px(x), px(y)),
                                    size(px(item_w), px(item_h)),
                                ),
                                px((6.0 * zoom).max(1.0)),
                                hsla(hue, 0.55, 0.62, 1.0),
                                px(1.0),
                                hsla(hue, 0.6, 0.32, 1.0),
                                BorderStyle::Solid,
                            ));
                            visible += 1;
                        }
                    }
                    samples.borrow_mut().push(Sample {
                        visible,
                        cpu: frame_start.elapsed(),
                        interval,
                    });
                },
            )
            .size_full();
            return div()
                .size_full()
                .overflow_hidden()
                .bg(hsla(0.12, 0.1, 0.96, 1.0))
                .child(painter)
                .into_any_element();
        }
        let mut children = Vec::new();
        for row in first_row..last_row {
            for column in first_column..last_column {
                let index = row * COLUMNS + column;
                if index >= self.items {
                    break;
                }
                let x = (column as f32 * pitch_x - left) * zoom;
                let y = (row as f32 * pitch_y - top) * zoom;
                let hue = (index * 37 % 360) as f32 / 360.0;
                let item = div()
                    .id(("item", index))
                    .absolute()
                    .left(px(x))
                    .top(px(y))
                    .w(px(item_w))
                    .h(px(item_h))
                    .bg(hsla(hue, 0.55, 0.62, 1.0))
                    .border_1()
                    .border_color(hsla(hue, 0.6, 0.32, 1.0))
                    .rounded(px((6.0 * zoom).max(1.0)))
                    .hover(|style| style.bg(hsla(hue, 0.7, 0.8, 1.0)))
                    .on_mouse_down(MouseButton::Left, |_, _, _| {})
                    .when(labelled, |item| {
                        item.p(px(4.0 * zoom))
                            .text_size(px(12.0 * zoom))
                            .text_color(hsla(0.0, 0.0, 0.1, 1.0))
                            .child(SharedString::from(format!("Item {index}")))
                    });
                children.push(item);
            }
        }
        let visible = children.len();

        let samples = self.samples.clone();
        let recorder = canvas(
            |_, _, _| {},
            move |_, _, _, _| {
                samples.borrow_mut().push(Sample {
                    visible,
                    cpu: frame_start.elapsed(),
                    interval,
                });
            },
        )
        .absolute()
        .size_0();

        div()
            .size_full()
            .relative()
            .overflow_hidden()
            .bg(hsla(0.12, 0.1, 0.96, 1.0))
            .children(children)
            .child(
                div()
                    .absolute()
                    .top_2()
                    .left_2()
                    .px_2()
                    .bg(hsla(0.0, 0.0, 1.0, 0.85))
                    .text_size(px(13.))
                    .child(format!("{visible} items · zoom {zoom:.3}")),
            )
            .child(recorder)
            .into_any_element()
    }
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn quantile(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * q).round() as usize]
}

fn run_example() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let items = args
        .iter()
        .find_map(|arg| arg.parse().ok())
        .unwrap_or(100_000);
    let direct = args.iter().any(|arg| arg == "--paint");
    application().run(move |cx: &mut App| {
        example_support::load_fonts(cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds::centered(
                    None,
                    size(px(1600.0), px(1000.0)),
                    cx,
                ))),
                ..Default::default()
            },
            |_, cx| cx.new(|_| Stage::new(items, direct)),
        )
        .unwrap();
        cx.activate(true);
    });
}

fn main() {
    run_example();
}
