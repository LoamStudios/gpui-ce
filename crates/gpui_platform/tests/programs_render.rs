//! Renders shader program paints through the platform renderer and checks
//! their pixels against the same paints evaluated on the CPU: grain filling
//! a plain box, a rotated box, a path, and a box in a cached view moved
//! under a camera, which is drawn as a chunk. Until the renderer has linked
//! a paint's program into its shaders, the paint's fallback colour is drawn
//! instead; the first window checks that the fallback is replaced by the
//! program once it is linked.
//!
//! It runs on Metal, and with `macos-wgpu` on wgpu.
//!
//! Runs only with `GPUI_RUN_RENDERING_TESTS` set, in offscreen windows;
//! `GPUI_RENDERING_TEST_OUTPUT=<path.png>` saves the image.

#[cfg(target_os = "macos")]
use gpui::{
    AnyWindowHandle, AppContext as _, Bounds, Context, Entity, Fill, IntoElement,
    ParentElement as _, PathBuilder, Position, Render, StyleRefinement, Styled as _,
    VisualTestAppContext, Window, canvas, div, kurbo, point, px, radians, rgb,
    shader::{self, Fragment, Paint, vec2f},
};
#[cfg(target_os = "macos")]
use std::time::{Duration, Instant};

/// Where the boxes are, in logical pixels: a plain one, and one turned by
/// [`TURN`] about its centre.
const PLAIN: (f32, f32) = (50., 50.);
const TURNED: (f32, f32) = (250., 50.);
const SIDE: f32 = 100.;
const TURN: f32 = 0.5;
/// A triangle filled with the paint, as a path, and the box the paint spans.
const TRIANGLE: [(f32, f32); 3] = [(500., 50.), (600., 150.), (450., 150.)];
const TRIANGLE_BOX: (f32, f32, f32, f32) = (450., 50., 150., 100.);
const FALLBACK: u32 = 0x336699;
/// A grid of cells enough to be drawn as a chunk, with a box filled with
/// the paint at its top left, and where the camera moves it to.
const CELLS: usize = 20;
const PITCH: f32 = 20.;
const CHUNK_BOX: f32 = 60.;
const CAMERA: (f32, f32) = (700., 60.);

/// Grain: value noise over the box's own logical pixels, tinted across it.
#[cfg(target_os = "macos")]
fn grain() -> Paint {
    shader::paint(|px| {
        let value = shader::noise::value(px.position() * 0.125);
        let uv = px.uv();
        shader::rgba(&value, uv.x(), uv.y() * 0.5 + &value * 0.5, 1.0)
    })
    .fallback(rgb(FALLBACK))
}

#[cfg(target_os = "macos")]
struct ProgramsFixture;

#[cfg(target_os = "macos")]
impl Render for ProgramsFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let square = |(left, top): (f32, f32)| {
            div()
                .absolute()
                .left(px(left))
                .top(px(top))
                .size(px(SIDE))
                .bg(Fill::program(grain()))
        };
        div()
            .size_full()
            .relative()
            .bg(rgb(0xffffff))
            .child(square(PLAIN))
            .child(square(TURNED).rotate(radians(TURN)))
            .child(
                canvas(
                    |_, _, _| (),
                    |_, _, window, _| {
                        let (left, top, width, height) = TRIANGLE_BOX;
                        let background = window.program(
                            &grain(),
                            Bounds::new(
                                point(px(left), px(top)),
                                gpui::size(px(width), px(height)),
                            ),
                        );
                        let mut builder = PathBuilder::fill();
                        let [first, rest @ ..] = TRIANGLE;
                        builder.move_to(point(px(first.0), px(first.1)));
                        for (x, y) in rest {
                            builder.line_to(point(px(x), px(y)));
                        }
                        builder.close();
                        window.paint_path(builder.build().unwrap(), background);
                    },
                )
                .absolute()
                .size_full(),
            )
    }
}

/// The grid, cached, under a camera that moves it.
#[cfg(target_os = "macos")]
struct Grid;

#[cfg(target_os = "macos")]
impl Render for Grid {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .relative()
            .children((0..CELLS * CELLS).map(|index| {
                let (row, column) = (index / CELLS, index % CELLS);
                div()
                    .absolute()
                    .left(px(column as f32 * PITCH))
                    .top(px(row as f32 * PITCH))
                    .size(px(PITCH / 2.))
                    .bg(rgb(0x808080))
            }))
            .child(
                div()
                    .absolute()
                    .size(px(CHUNK_BOX))
                    .bg(Fill::program(grain())),
            )
    }
}

#[cfg(target_os = "macos")]
struct ChunkFixture {
    grid: Entity<Grid>,
    camera: std::rc::Rc<std::cell::Cell<(f32, f32)>>,
}

#[cfg(target_os = "macos")]
impl Render for ChunkFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let (x, y) = self.camera.get();
        let mut style = StyleRefinement::default();
        style.position = Some(Position::Absolute);
        style.size.width = Some(px(CELLS as f32 * PITCH).into());
        style.size.height = Some(px(CELLS as f32 * PITCH).into());
        div().size_full().relative().bg(rgb(0xffffff)).child(
            div()
                .absolute()
                .size_0()
                .transform(kurbo::Affine::translate((f64::from(x), f64::from(y))))
                .child(self.grid.clone().cached(style)),
        )
    }
}

/// A grey of `level` from a program of its own for each `level`: the grey
/// multiplied by one `level` times. Its fallback is green.
#[cfg(target_os = "macos")]
fn grey(level: u8) -> Paint {
    let mut value = shader::Pixel.uv().x() * 0.0 + f32::from(level) * 40. / 255.;
    for _ in 0..level {
        value = value * 1.0;
    }
    shader::rgba(&value, &value, &value, 1.0).fallback(rgb(0x00ff00))
}

/// A box filled with [`grey`] of the level it is set to.
#[cfg(target_os = "macos")]
struct CycleFixture {
    level: std::rc::Rc<std::cell::Cell<u8>>,
}

#[cfg(target_os = "macos")]
impl Render for CycleFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().bg(rgb(0xffffff)).child(
            div()
                .absolute()
                .left(px(PLAIN.0))
                .top(px(PLAIN.1))
                .size(px(SIDE))
                .bg(Fill::program(grey(self.level.get()))),
        )
    }
}

fn main() {
    if std::env::var_os("GPUI_RUN_RENDERING_TESTS").is_none() {
        return;
    }
    #[cfg(target_os = "macos")]
    render();
}

#[cfg(target_os = "macos")]
fn open(cx: &mut VisualTestAppContext) -> AnyWindowHandle {
    let window = cx
        .open_offscreen_window_default(|_, cx| cx.new(|_| ProgramsFixture))
        .expect("failed to create the offscreen window");
    cx.run_until_parked();
    window.into()
}

#[cfg(target_os = "macos")]
fn capture(cx: &mut VisualTestAppContext, window: AnyWindowHandle) -> image::RgbaImage {
    cx.capture_screenshot(window)
        .expect("failed to capture the rendered window")
}

/// A sample point inside one of the shapes, in logical pixels, and where it
/// is in the box its paint spans.
#[cfg(target_os = "macos")]
struct Sample {
    what: &'static str,
    window: (f32, f32),
    local: (f32, f32),
    size: (f32, f32),
}

/// Points well inside each shape, away from their antialiased edges.
#[cfg(target_os = "macos")]
fn samples() -> Vec<Sample> {
    let mut samples = Vec::new();
    let steps = [0.1, 0.3, 0.5, 0.7, 0.9];
    for u in steps {
        for v in steps {
            let local = (u * SIDE, v * SIDE);
            samples.push(Sample {
                what: "plain box",
                window: (PLAIN.0 + local.0, PLAIN.1 + local.1),
                local,
                size: (SIDE, SIDE),
            });
            // Turned clockwise about the box's centre.
            let (sin, cos) = TURN.sin_cos();
            let (dx, dy) = (local.0 - SIDE / 2., local.1 - SIDE / 2.);
            samples.push(Sample {
                what: "turned box",
                window: (
                    TURNED.0 + SIDE / 2. + dx * cos - dy * sin,
                    TURNED.1 + SIDE / 2. + dx * sin + dy * cos,
                ),
                local,
                size: (SIDE, SIDE),
            });
        }
    }
    let (left, top, width, height) = TRIANGLE_BOX;
    for (x, y) in [
        (520., 120.),
        (500., 100.),
        (480., 140.),
        (560., 140.),
        (510., 80.),
        (530., 130.),
    ] {
        samples.push(Sample {
            what: "path",
            window: (x, y),
            local: (x - left, y - top),
            size: (width, height),
        });
    }
    samples
}

/// Device pixels per logical pixel in `image`, of a 1280-wide window.
#[cfg(target_os = "macos")]
fn scale_of(image: &image::RgbaImage) -> f32 {
    image.width() as f32 / 1280.
}

/// The device pixel holding the logical point `at`.
#[cfg(target_os = "macos")]
fn device_pixel(image: &image::RgbaImage, at: (f32, f32)) -> (u32, u32) {
    let scale = scale_of(image);
    ((at.0 * scale) as u32, (at.1 * scale) as u32)
}

/// What the paint is on the CPU at the centre of the device pixel holding
/// `sample`, as sRGB bytes.
#[cfg(target_os = "macos")]
fn expected(paint: &Paint, image: &image::RgbaImage, sample: &Sample) -> [u8; 3] {
    let scale = scale_of(image);
    let (x, y) = device_pixel(image, sample.window);
    // The pixel's centre, moved by as much as the sample point was.
    let shift = (
        (x as f32 + 0.5) / scale - sample.window.0,
        (y as f32 + 0.5) / scale - sample.window.1,
    );
    let local = if sample.what == "turned box" {
        let (sin, cos) = TURN.sin_cos();
        (
            sample.local.0 + shift.0 * cos + shift.1 * sin,
            sample.local.1 - shift.0 * sin + shift.1 * cos,
        )
    } else {
        (sample.local.0 + shift.0, sample.local.1 + shift.1)
    };
    let fragment = Fragment {
        uv: vec2f(local.0 / sample.size.0, local.1 / sample.size.1),
        position: vec2f(local.0, local.1),
        size: vec2f(sample.size.0, sample.size.1),
        origin: vec2f(0., 0.),
        scale,
        stroke: vec2f(0., 0.),
    };
    let rgba = paint.evaluate(fragment).expect("the paint runs on the CPU");
    let alpha = rgba.w.max(1e-6);
    [rgba.x, rgba.y, rgba.z].map(|channel| ((channel / alpha).clamp(0., 1.) * 255.).round() as u8)
}

/// How far each sample is from the CPU's colour, at most.
#[cfg(target_os = "macos")]
fn compare(image: &image::RgbaImage, samples: &[Sample], tolerance: u8) -> Vec<String> {
    let paint = grain();
    samples
        .iter()
        .filter_map(|sample| {
            let (x, y) = device_pixel(image, sample.window);
            let actual = image.get_pixel(x, y).0;
            let expected = expected(&paint, image, sample);
            let off = actual[..3]
                .iter()
                .zip(expected)
                .any(|(actual, expected)| actual.abs_diff(expected) > tolerance);
            off.then(|| {
                format!(
                    "{} at {:?} (local {:?}): expected {expected:?}, got {:?}",
                    sample.what, sample.window, sample.local, actual
                )
            })
        })
        .collect()
}

/// Whether every sample shows the fallback colour.
#[cfg(target_os = "macos")]
fn shows_fallback(image: &image::RgbaImage) -> bool {
    let [_, r, g, b] = FALLBACK.to_be_bytes();
    samples().iter().all(|sample| {
        let (x, y) = device_pixel(image, sample.window);
        let actual = image.get_pixel(x, y).0;
        actual[..3]
            .iter()
            .zip([r, g, b])
            .all(|(actual, expected)| actual.abs_diff(expected) < 3)
    })
}

#[cfg(target_os = "macos")]
fn render() {
    let mut cx = VisualTestAppContext::new(gpui_ce_platform::current_platform(false));

    // Linked in the background: the first frame draws the fallback colour,
    // and a later one, drawn again, the program.
    let window = open(&mut cx);
    let first = capture(&mut cx, window);
    assert!(
        shows_fallback(&first),
        "the first frame should draw the fallback colour while the program links"
    );
    let start = Instant::now();
    let linked = loop {
        let image = capture(&mut cx, window);
        if !shows_fallback(&image) {
            break image;
        }
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "the program was never linked"
        );
        std::thread::sleep(Duration::from_millis(10));
    };
    println!(
        "programs_render: the program replaced the fallback within {:.0} ms",
        start.elapsed().as_secs_f64() * 1000.
    );
    let failures = compare(&linked, &samples(), 3);
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    // Linked before the first frame is drawn.
    // SAFETY: the test is single-threaded until here, and the renderer reads
    // this when the next window opens.
    unsafe { std::env::set_var("GPUI_LINK_PROGRAMS_SYNCHRONOUSLY", "1") };
    let window = open(&mut cx);
    let image = capture(&mut cx, window);
    if let Some(output) = std::env::var_os("GPUI_RENDERING_TEST_OUTPUT") {
        image.save(output).expect("failed to save the image");
    }
    let mut failures = compare(&image, &samples(), 3);
    let white = [255, 255, 255];
    for outside in [(45., 100.), (155., 100.), (455., 60.), (590., 60.)] {
        let (x, y) = device_pixel(&image, outside);
        let actual = image.get_pixel(x, y).0;
        if actual[..3] != white {
            failures.push(format!("outside at {outside:?}: got {actual:?}"));
        }
    }

    assert!(failures.is_empty(), "{}", failures.join("\n"));

    // In a cached view reused where a camera moves it, as a chunk.
    let camera = std::rc::Rc::new(std::cell::Cell::new((0., 0.)));
    let window: AnyWindowHandle = cx
        .open_offscreen_window_default({
            let camera = camera.clone();
            move |_, cx| {
                cx.new(|cx| ChunkFixture {
                    grid: cx.new(|_| Grid),
                    camera,
                })
            }
        })
        .expect("failed to create the offscreen window")
        .into();
    cx.run_until_parked();
    for position in [(CAMERA.0 / 2., CAMERA.1 / 2.), CAMERA] {
        camera.set(position);
        cx.update_window(window, |root, window, cx| {
            root.downcast::<ChunkFixture>()
                .unwrap()
                .update(cx, |_, cx| cx.notify());
            window.draw(cx).clear(cx);
        })
        .expect("failed to draw the window");
    }
    let image = cx
        .update_window(window, |_, window, _| window.render_to_image())
        .expect("failed to capture the rendered window")
        .expect("failed to capture the rendered window");
    let steps = [0.1, 0.3, 0.5, 0.7, 0.9];
    let in_chunk: Vec<Sample> = steps
        .iter()
        .flat_map(|u| steps.iter().map(move |v| (u, v)))
        .map(|(u, v)| {
            let local = (u * CHUNK_BOX, v * CHUNK_BOX);
            Sample {
                what: "chunk",
                window: (CAMERA.0 + local.0, CAMERA.1 + local.1),
                local,
                size: (CHUNK_BOX, CHUNK_BOX),
            }
        })
        .collect();
    let failures = compare(&image, &in_chunk, 3);

    assert!(failures.is_empty(), "{}", failures.join("\n"));

    cycle_past_the_cap(&mut cx);
    std::mem::forget(cx);
}

/// With room for two linked programs, a box cycling through five programs
/// draws each one's colour, and, linked in the background, a program
/// evicted since it was drawn draws its fallback colour, then itself.
#[cfg(target_os = "macos")]
fn cycle_past_the_cap(cx: &mut VisualTestAppContext) {
    // SAFETY: the test is single-threaded, and the renderer reads these
    // when the next window opens.
    unsafe { std::env::set_var("GPUI_LINKED_PROGRAMS_CAP", "2") };
    let open_cycle = |cx: &mut VisualTestAppContext| {
        let level = std::rc::Rc::new(std::cell::Cell::new(1));
        let window: AnyWindowHandle = cx
            .open_offscreen_window_default({
                let level = level.clone();
                move |_, cx| cx.new(|_| CycleFixture { level })
            })
            .expect("failed to create the offscreen window")
            .into();
        cx.run_until_parked();
        (window, level)
    };
    let show = |cx: &mut VisualTestAppContext, window: AnyWindowHandle, level| {
        cx.update_window(window, |root, window, cx| {
            root.downcast::<CycleFixture>()
                .unwrap()
                .update(cx, |fixture, cx| {
                    fixture.level.set(level);
                    cx.notify()
                });
            window.draw(cx).clear(cx);
        })
        .expect("failed to draw the window");
    };
    let centre = |image: &image::RgbaImage| {
        let (x, y) = device_pixel(image, (PLAIN.0 + SIDE / 2., PLAIN.1 + SIDE / 2.));
        let pixel = image.get_pixel(x, y).0;
        [pixel[0], pixel[1], pixel[2]]
    };
    let close = |actual: [u8; 3], expected: [u8; 3]| {
        actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.abs_diff(expected) <= 3)
    };
    let grey_of = |level: u8| {
        let value = (f32::from(level) * 40.).round() as u8;
        [value; 3]
    };

    // Each frame waits for its program.
    let (window, _) = open_cycle(cx);
    let mut failures = Vec::new();
    for round in 0..2 {
        for level in 1..=5 {
            show(cx, window, level);
            let actual = centre(&capture(cx, window));
            if !close(actual, grey_of(level)) {
                failures.push(format!(
                    "round {round}, program {level}: expected {:?}, got {actual:?}",
                    grey_of(level)
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    // In the background.
    unsafe { std::env::remove_var("GPUI_LINK_PROGRAMS_SYNCHRONOUSLY") };
    let (window, _) = open_cycle(cx);
    let wait_for = |cx: &mut VisualTestAppContext, level| {
        let start = Instant::now();
        loop {
            if close(centre(&capture(cx, window)), grey_of(level)) {
                return;
            }
            assert!(
                start.elapsed() < Duration::from_secs(20),
                "program {level} was never linked"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    };
    for level in [1, 2, 3] {
        show(cx, window, level);
        wait_for(cx, level);
    }
    // The first was evicted.
    show(cx, window, 1);
    let actual = centre(&capture(cx, window));
    assert!(
        close(actual, [0, 255, 0]),
        "an evicted program should draw its fallback until it is linked again, got {actual:?}"
    );
    wait_for(cx, 1);
    unsafe { std::env::remove_var("GPUI_LINKED_PROGRAMS_CAP") };
}
