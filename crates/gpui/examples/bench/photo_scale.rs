//! How many photos a frame can show.
//!
//! A grid of photos, 4000×3000 pixels each, on a page larger than the
//! window. A camera zooms from one photo filling the window out to the whole
//! grid and back, and only the photos in view are painted. Every frame
//! records how many photos were in view, how long the frame took on the CPU
//! from the start of `render` to the end of paint, how long choosing the
//! photos' levels and tiles took after that, and how many of the tiles drawn
//! were not yet at the level wanted.
//!
//! The photos' pixels are made as each tile is asked for, as a source with
//! its levels already made would provide them, so the run measures the
//! renderer and the tiles' residency rather than decoding. With
//! `--files=<dir>`, the grid repeats the image files in that directory
//! instead, decoded as they are needed.
//!
//! With `--flat`, each tile is one colour, made in next to no time, which
//! shows the cost of placing and uploading tiles as fast as they can come.
//! With `--quads`, quads are painted where the photos would be, to compare
//! against. With `--hold=<zoom>`, the camera holds that zoom instead.
//!
//! ```text
//! cargo run -p gpui-ce --release --example photo_scale                 # 10,000 photos
//! cargo run -p gpui-ce --release --example photo_scale -- 2000         # fewer
//! cargo run -p gpui-ce --release --example photo_scale -- --files=/Volumes/Annex/photos
//! ```
//!
//! The window is off the screen, where it never shows, and the run draws
//! and presents each frame itself, as fast as frames finish; it reports how
//! long each frame's draw and present took as well. A present includes
//! waiting for a drawable, which Metal hands out at the display's rate even
//! off the screen, so most of it is that wait. With `--onscreen`, the
//! window is on the screen and frames come at the display's rate.
//!
//! The run lasts one zoom out and back, then prints frame times grouped by
//! how many photos were in view, and quits.

use std::{
    cell::RefCell,
    path::PathBuf,
    rc::Rc,
    time::{Duration, Instant},
};

use gpui::{
    App, Bounds, Context, Corners, MacActivationPolicy, ObjectFit, Photo, PhotoSource, Render,
    Size, Window, WindowBounds, WindowKind, WindowOptions, canvas, div, hsla, point, prelude::*,
    px, size,
};
use gpui_platform::application;

/// Photos per row of the grid.
const COLUMNS: usize = 100;
/// A photo's size and the gap between photos, in page units.
const PHOTO_W: f32 = 400.;
const PHOTO_H: f32 = 300.;
const GAP: f32 = 20.;
/// A photo's size in pixels.
const PIXELS_W: u32 = 4000;
const PIXELS_H: u32 = 3000;
/// The closest and farthest zoom the camera reaches.
const ZOOM_IN: f32 = 4.0;
const ZOOM_OUT: f32 = 0.035;
/// One zoom out and back in.
const RUN: Duration = Duration::from_secs(30);

/// A photo whose pixels are made from where they are: a wash of colour from
/// its seed, with rings fine enough to show which level is drawn.
struct MadePhoto {
    seed: usize,
    /// Make each tile one colour, which takes next to no time.
    flat: bool,
}

impl PhotoSource for MadePhoto {
    fn size(&self) -> Size<u32> {
        size(PIXELS_W, PIXELS_H)
    }

    fn decode(&self, level: u32, region: Bounds<u32>) -> anyhow::Result<Vec<u8>> {
        let scale = (1u32 << level) as f32;
        let hue = (self.seed * 37 % 360) as f32;
        if self.flat {
            let [r, g, b] = hsl_to_rgb(hue, 0.6, 0.3 + 0.05 * level as f32);
            let pixel = [(r * 255.) as u8, (g * 255.) as u8, (b * 255.) as u8, 255];
            return Ok(pixel.repeat((region.size.width * region.size.height) as usize));
        }
        let mut pixels = Vec::with_capacity((region.size.width * region.size.height * 4) as usize);
        for y in region.origin.y..region.origin.y + region.size.height {
            for x in region.origin.x..region.origin.x + region.size.width {
                let u = (x as f32 + 0.5) * scale / PIXELS_W as f32;
                let v = (y as f32 + 0.5) * scale / PIXELS_H as f32;
                let (du, dv) = (u - 0.5, (v - 0.5) * 0.75);
                let ring = ((du * du + dv * dv).sqrt() * 120.).sin() * 0.5 + 0.5;
                let lightness = 0.35 + 0.3 * v + 0.15 * ring;
                let [r, g, b] = hsl_to_rgb((hue + 60. * u) % 360., 0.6, lightness);
                pixels.extend_from_slice(&[
                    (r * 255.) as u8,
                    (g * 255.) as u8,
                    (b * 255.) as u8,
                    255,
                ]);
            }
        }
        Ok(pixels)
    }
}

/// `hue` in degrees, the others from 0 to 1.
fn hsl_to_rgb(hue: f32, saturation: f32, lightness: f32) -> [f32; 3] {
    let chroma = (1. - (2. * lightness - 1.).abs()) * saturation;
    let sector = hue / 60.;
    let x = chroma * (1. - (sector % 2. - 1.).abs());
    let (r, g, b) = match sector as u32 {
        0 => (chroma, x, 0.),
        1 => (x, chroma, 0.),
        2 => (0., chroma, x),
        3 => (0., x, chroma),
        4 => (x, 0., chroma),
        _ => (chroma, 0., x),
    };
    let m = lightness - chroma / 2.;
    [r + m, g + m, b + m]
}

#[derive(Clone, Copy)]
struct Sample {
    visible: usize,
    cpu: Duration,
    interval: Option<Duration>,
    stats: gpui::PhotoStats,
}

struct Stage {
    photos: Rc<Vec<Photo>>,
    cells: usize,
    started: Instant,
    last_frame: Option<Instant>,
    samples: Rc<RefCell<Vec<Sample>>>,
    pending: Rc<RefCell<Option<Sample>>>,
    /// How long each frame's draw and present took, when the run draws its
    /// own frames.
    draws: Rc<RefCell<Vec<(Duration, Duration)>>>,
    finished: bool,
    source: String,
    /// Paint quads where the photos would be, to compare against.
    quads: bool,
    /// A zoom to hold for the whole run, instead of zooming.
    hold: Option<f32>,
}

impl Stage {
    fn zoom(elapsed: Duration) -> f32 {
        let phase = (elapsed.as_secs_f32() / RUN.as_secs_f32()).min(1.0);
        let out = if phase < 0.5 {
            phase * 2.0
        } else {
            (1.0 - phase) * 2.0
        };
        let (near, far) = (ZOOM_IN.ln(), ZOOM_OUT.ln());
        (near + (far - near) * out).exp()
    }

    fn report(&self) {
        let samples = self.samples.borrow();
        println!(
            "photo_scale: {} photos of {}×{} ({}), {} frames over {:.1}s",
            self.cells,
            PIXELS_W,
            PIXELS_H,
            self.source,
            samples.len(),
            self.started.elapsed().as_secs_f32()
        );
        let draws = self.draws.borrow();
        if !draws.is_empty() {
            let mut draw: Vec<f64> = draws.iter().map(|(draw, _)| millis(*draw)).collect();
            let mut present: Vec<f64> = draws.iter().map(|(_, present)| millis(*present)).collect();
            println!(
                "photo_scale: whole frames: draw median {:.2} ms, p95 {:.2} ms; present median {:.2} ms, p95 {:.2} ms",
                quantile(&mut draw, 0.5),
                quantile(&mut draw, 0.95),
                quantile(&mut present, 0.5),
                quantile(&mut present, 0.95),
            );
        }
        println!(
            "photo_scale: in view  frames  CPU median  CPU p95  tiles median  tiles p95  interval  lacking  resident  layers"
        );
        let buckets: [(usize, usize); 6] = [
            (0, 10),
            (10, 100),
            (100, 1_000),
            (1_000, 5_000),
            (5_000, 10_000),
            (10_000, usize::MAX),
        ];
        for (low, high) in buckets {
            let in_bucket: Vec<&Sample> = samples
                .iter()
                .filter(|sample| sample.visible >= low && sample.visible < high)
                .collect();
            if in_bucket.is_empty() {
                continue;
            }
            let mut cpu: Vec<f64> = in_bucket.iter().map(|s| millis(s.cpu)).collect();
            let mut prepare: Vec<f64> = in_bucket.iter().map(|s| millis(s.stats.prepare)).collect();
            let mut interval: Vec<f64> = in_bucket
                .iter()
                .filter_map(|s| s.interval.map(millis))
                .collect();
            let mut lacking: Vec<f64> = in_bucket
                .iter()
                .map(|s| 100. * s.stats.tiles_lacking as f64 / s.stats.tiles_drawn.max(1) as f64)
                .collect();
            let resident = in_bucket
                .iter()
                .map(|s| s.stats.resident)
                .max()
                .unwrap_or(0);
            let layers = in_bucket.iter().map(|s| s.stats.layers).max().unwrap_or(0);
            let range = if high == usize::MAX {
                format!("{low}+")
            } else {
                format!("{low}–{high}")
            };
            println!(
                "photo_scale: {range:>9}  {:>6}  {:>7.2} ms  {:>5.2} ms  {:>9.2} ms  {:>6.2} ms  {:>5.2} ms  {:>6.1}%  {:>8}  {:>6}",
                in_bucket.len(),
                quantile(&mut cpu, 0.5),
                quantile(&mut cpu, 0.95),
                quantile(&mut prepare, 0.5),
                quantile(&mut prepare, 0.95),
                quantile(&mut interval, 0.5),
                quantile(&mut lacking, 0.5),
                resident,
                layers,
            );
        }
    }
}

impl Render for Stage {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let frame_start = Instant::now();
        let interval = self
            .last_frame
            .replace(frame_start)
            .map(|last| frame_start - last);
        let elapsed = self.started.elapsed();

        // The last frame's photos were prepared after it painted: its sample
        // is complete now.
        if let Some(mut sample) = self.pending.borrow_mut().take() {
            sample.stats = window.photo_stats();
            self.samples.borrow_mut().push(sample);
        }

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

        let viewport = window.viewport_size();
        let (view_w, view_h) = (f32::from(viewport.width), f32::from(viewport.height));
        let rows = self.cells.div_ceil(COLUMNS);
        let (pitch_x, pitch_y) = (PHOTO_W + GAP, PHOTO_H + GAP);
        let zoom = self.hold.unwrap_or_else(|| Self::zoom(elapsed));
        let left = COLUMNS as f32 * pitch_x / 2.0 - view_w / zoom / 2.0;
        let top = rows as f32 * pitch_y / 2.0 - view_h / zoom / 2.0;
        let first_column = (left / pitch_x).floor().max(0.0) as usize;
        let last_column = (((left + view_w / zoom) / pitch_x).ceil() as usize).min(COLUMNS);
        let first_row = (top / pitch_y).floor().max(0.0) as usize;
        let last_row = (((top + view_h / zoom) / pitch_y).ceil() as usize).min(rows);

        let photos = self.photos.clone();
        let cells = self.cells;
        let pending = self.pending.clone();
        let quads = self.quads;
        let painter = canvas(
            |_, _, _| {},
            move |bounds, _, window, _| {
                let mut visible = 0;
                for row in first_row..last_row {
                    for column in first_column..last_column {
                        let index = row * COLUMNS + column;
                        if index >= cells {
                            break;
                        }
                        let x = (column as f32 * pitch_x - left) * zoom;
                        let y = (row as f32 * pitch_y - top) * zoom;
                        let cell = Bounds::new(
                            bounds.origin + point(px(x), px(y)),
                            size(px(PHOTO_W * zoom), px(PHOTO_H * zoom)),
                        );
                        if quads {
                            window.paint_quad(gpui::fill(
                                cell,
                                hsla((index * 37 % 360) as f32 / 360., 0.6, 0.5, 1.),
                            ));
                        } else {
                            window.paint_photo(
                                cell,
                                Corners::default(),
                                &photos[index % photos.len()],
                                ObjectFit::Fill,
                            );
                        }
                        visible += 1;
                    }
                }
                *pending.borrow_mut() = Some(Sample {
                    visible,
                    cpu: frame_start.elapsed(),
                    interval,
                    stats: Default::default(),
                });
            },
        )
        .size_full();
        div()
            .size_full()
            .overflow_hidden()
            .bg(hsla(0.6, 0.05, 0.12, 1.0))
            .child(painter)
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
    let cells = args
        .iter()
        .find_map(|arg| arg.parse().ok())
        .unwrap_or(10_000);
    let flat = args.iter().any(|arg| arg == "--flat");
    let files = args
        .iter()
        .find_map(|arg| arg.strip_prefix("--files="))
        .map(PathBuf::from);
    let (photos, source) = match files {
        Some(directory) => {
            let mut paths: Vec<PathBuf> = std::fs::read_dir(&directory)
                .expect("--files=<dir> must be a readable directory")
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| {
                    path.extension()
                        .and_then(|extension| extension.to_str())
                        .is_some_and(|extension| {
                            matches!(
                                extension.to_ascii_lowercase().as_str(),
                                "jpg" | "jpeg" | "png"
                            )
                        })
                })
                .collect();
            paths.sort();
            let photos: Vec<Photo> = paths
                .iter()
                .filter_map(|path| Photo::open(path).ok())
                .collect();
            assert!(
                !photos.is_empty(),
                "no JPEG or PNG files in {}",
                directory.display()
            );
            let source = format!("{} files from {}", photos.len(), directory.display());
            (photos, source)
        }
        None => (
            (0..cells)
                .map(|seed| Photo::new(MadePhoto { seed, flat }))
                .collect(),
            "pixels made per tile".to_string(),
        ),
    };
    let photos = Rc::new(photos);
    let quads = args.iter().any(|arg| arg == "--quads");
    let hold = args
        .iter()
        .find_map(|arg| arg.strip_prefix("--hold="))
        .map(|zoom| zoom.parse().expect("--hold=<zoom>, e.g. --hold=1"));
    let onscreen = args.iter().any(|arg| arg == "--onscreen");
    let draws: Rc<RefCell<Vec<(Duration, Duration)>>> = Rc::default();
    // A benchmark run never takes the keyboard: the process cannot be
    // activated, and the window has no focus.
    application()
        .with_activation_policy(MacActivationPolicy::Prohibited)
        .run(move |cx: &mut App| {
            let window_size = size(px(1600.0), px(1000.0));
            let window = cx
                .open_window(
                    WindowOptions {
                        window_bounds: Some(WindowBounds::Windowed(if onscreen {
                            Bounds::centered(None, window_size, cx)
                        } else {
                            // Off the screen, where it never shows.
                            Bounds::new(point(px(-10_000.), px(-10_000.)), window_size)
                        })),
                        // On the screen, it floats above others, so it keeps
                        // drawing without being brought forward.
                        kind: if onscreen {
                            WindowKind::PopUp
                        } else {
                            WindowKind::Normal
                        },
                        focus: false,
                        ..Default::default()
                    },
                    |_, cx| {
                        cx.new(|_| Stage {
                            photos,
                            cells,
                            started: Instant::now(),
                            last_frame: None,
                            samples: Rc::default(),
                            pending: Rc::default(),
                            draws: draws.clone(),
                            finished: false,
                            source,
                            quads,
                            hold,
                        })
                    },
                )
                .unwrap();
            let window: gpui::AnyWindowHandle = window.into();
            if !onscreen {
                // The platform doesn't ask a window off the screen to draw:
                // draw each frame here, and the next as soon as it's done.
                cx.spawn(async move |cx| {
                    loop {
                        let drawn = cx.update_window(window, |root, window, cx| {
                            cx.notify(root.entity_id());
                            let times = window.draw_and_present(cx);
                            draws.borrow_mut().push(times);
                        });
                        if drawn.is_err() {
                            break;
                        }
                        cx.background_executor()
                            .timer(Duration::from_millis(1))
                            .await;
                    }
                })
                .detach();
            }
        });
}

fn main() {
    run_example();
}
