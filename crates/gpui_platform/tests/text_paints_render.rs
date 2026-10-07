//! Renders text filled with paints through the platform renderer and checks
//! its pixels: wherever a glyph covers a pixel fully, text filled with a
//! gradient shows the colour a box filled with the same gradient has there,
//! and text filled with a shader program the colour the program has on the
//! CPU; wherever no glyph covers it, the background shows. Text filled with
//! a plain colour draws as text of that colour does.
//!
//! Each fill spans its text element from its top left, in its own logical
//! pixels, and the paints depend only on where they are evaluated, so the
//! expected colour of a pixel follows from where it is in its line.
//!
//! Coverage comes from the same line drawn in white. Runs on Metal, and on
//! wgpu with `macos-wgpu`; only with `GPUI_RUN_RENDERING_TESTS` set, in an
//! offscreen window. `GPUI_RENDERING_TEST_OUTPUT=<path.png>` saves the image.

#[cfg(target_os = "macos")]
use gpui::{
    AppContext as _, Context, Fill, IntoElement, ParentElement as _, Render, Styled as _,
    VisualTestAppContext, Window, div, peniko, px, rgb,
    shader::{self, Fragment, Paint, vec2f},
};
#[cfg(target_os = "macos")]
use std::borrow::Cow;

#[cfg(target_os = "macos")]
#[allow(dead_code)]
#[path = "../../gpui_ce_parley/src/font_fixtures.rs"]
mod font_fixtures;

/// Where each line is, in logical pixels: its top left, in a column.
const LEFT: f32 = 40.;
const WHITE_TOP: f32 = 20.;
const GRADIENT_TOP: f32 = 160.;
const PROGRAM_TOP: f32 = 300.;
const PLAIN_TOP: f32 = 440.;
const SOLID_TOP: f32 = 580.;
/// The box filled with the gradient, to compare with: right of the text.
const BOX_LEFT: f32 = 700.;
/// The part of each line to check.
const WIDTH: f32 = 560.;
const HEIGHT: f32 = 130.;
const TEXT: &str = "HMWE";

/// Stripes of colour across and down, from the paint's own pixels only.
#[cfg(target_os = "macos")]
fn bands() -> Paint {
    shader::paint(|px| {
        let position = px.position();
        let across = (position.x() * (1. / 37.)).fract();
        let down = (position.y() * (1. / 23.)).fract();
        shader::rgba(across, down, 0.6, 1.0)
    })
    .fallback(rgb(0x00ff00))
}

/// Red to blue, across 400 logical pixels from the left.
#[cfg(target_os = "macos")]
fn gradient() -> Fill {
    Fill::gradient(
        peniko::Gradient::new_linear((0., 0.), (400., 0.)).with_stops([
            peniko::color::palette::css::RED,
            peniko::color::palette::css::BLUE,
        ]),
    )
}

#[cfg(target_os = "macos")]
struct TextPaintsFixture;

#[cfg(target_os = "macos")]
impl Render for TextPaintsFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let line = |top: f32| {
            div()
                .absolute()
                .left(px(LEFT))
                .top(px(top))
                .w(px(WIDTH))
                .h(px(HEIGHT))
                .child(TEXT)
        };
        div()
            .size_full()
            .relative()
            .bg(rgb(0x000000))
            .font_family("IBM Plex Sans")
            .text_size(px(100.))
            .text_color(rgb(0xffffff))
            .child(line(WHITE_TOP))
            .child(line(GRADIENT_TOP).text_fill(gradient()))
            .child(line(PROGRAM_TOP).text_fill(Fill::program(bands())))
            .child(line(PLAIN_TOP).text_color(rgb(0xff8800)))
            .child(line(SOLID_TOP).text_fill(rgb(0xff8800)))
            .child(
                div()
                    .absolute()
                    .left(px(BOX_LEFT))
                    .top(px(GRADIENT_TOP))
                    .w(px(WIDTH))
                    .h(px(HEIGHT))
                    .bg(gradient()),
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
fn render() {
    // The program is linked before the first frame.
    // SAFETY: single-threaded until the window opens, which reads it.
    unsafe { std::env::set_var("GPUI_LINK_PROGRAMS_SYNCHRONOUSLY", "1") };
    let mut cx = VisualTestAppContext::new(gpui_ce_platform::current_platform(false));
    cx.update(|cx| {
        cx.text_system()
            .add_fonts(vec![Cow::Borrowed(font_fixtures::IBM_PLEX.data)])
    })
    .expect("failed to load the fixture font");
    let window = cx
        .open_offscreen_window_default(|_, cx| cx.new(|_| TextPaintsFixture))
        .expect("failed to create the offscreen window")
        .into();
    cx.run_until_parked();
    let image = cx
        .capture_screenshot(window)
        .expect("failed to capture the rendered window");
    if let Some(output) = std::env::var_os("GPUI_RENDERING_TEST_OUTPUT") {
        image.save(output).expect("failed to save the image");
    }

    let scale = image.width() as f32 / 1280.;
    let pixel = |x: u32, y: u32| {
        let [r, g, b, _] = image.get_pixel(x, y).0;
        [r, g, b]
    };
    let device = |logical: f32| (logical * scale).round() as u32;
    // Each device pixel of a line, by where it is in it, with the pixel at
    // the same place in the white line: its coverage.
    let (columns, rows) = (device(WIDTH), device(HEIGHT));
    let places = (0..rows).flat_map(move |y| (0..columns).map(move |x| (x, y)));
    let full = |coverage: [u8; 3]| coverage == [255, 255, 255];
    let none = |coverage: [u8; 3]| coverage == [0, 0, 0];
    let close = |a: [u8; 3], b: [u8; 3], tolerance: u8| {
        a.iter().zip(b).all(|(a, b)| a.abs_diff(b) <= tolerance)
    };
    let paint = bands();
    let mut failures = Vec::new();
    let mut checked = [0usize; 4];
    for (x, y) in places {
        let coverage = pixel(device(LEFT) + x, device(WHITE_TOP) + y);
        let at = |top: f32| pixel(device(LEFT) + x, device(top) + y);
        let mut check = |what: &str, index: usize, actual: [u8; 3], expected: [u8; 3]| {
            // Gradients are dithered by where they are drawn, so text and
            // box differ by a few 255ths.
            let tolerance = if what == "gradient" { 6 } else { 3 };
            checked[index] += 1;
            if !close(actual, expected, tolerance) && failures.len() < 20 {
                failures.push(format!(
                    "{what} at ({x}, {y}): expected {expected:?}, got {actual:?}"
                ));
            }
        };
        if none(coverage) {
            for (what, top) in [
                ("gradient", GRADIENT_TOP),
                ("program", PROGRAM_TOP),
                ("plain", PLAIN_TOP),
                ("solid fill", SOLID_TOP),
            ] {
                check(&format!("{what} outside the glyphs"), 3, at(top), [0, 0, 0]);
            }
            continue;
        }
        // Plain colour, and a fill of the same colour, everywhere alike.
        check("solid fill", 2, at(SOLID_TOP), at(PLAIN_TOP));
        if !full(coverage) {
            continue;
        }
        check("plain", 2, at(PLAIN_TOP), [0xff, 0x88, 0x00]);
        let in_box = pixel(device(BOX_LEFT) + x, device(GRADIENT_TOP) + y);
        check("gradient", 0, at(GRADIENT_TOP), in_box);
        // The program at the pixel's centre, in the line's logical pixels.
        let position = ((x as f32 + 0.5) / scale, (y as f32 + 0.5) / scale);
        let rgba = paint
            .evaluate(Fragment {
                uv: vec2f(position.0 / WIDTH, position.1 / HEIGHT),
                position: vec2f(position.0, position.1),
                size: vec2f(WIDTH, HEIGHT),
                origin: vec2f(0., 0.),
                scale,
                stroke: vec2f(0., 0.),
            })
            .expect("the paint runs on the CPU");
        let expected = [rgba.x, rgba.y, rgba.z].map(|channel| (channel * 255.).round() as u8);
        check("program", 1, at(PROGRAM_TOP), expected);
    }
    std::mem::forget(cx);
    println!(
        "text_paints_render: checked {} gradient, {} program, {} plain and {} uncovered pixels",
        checked[0], checked[1], checked[2], checked[3]
    );
    assert!(
        checked[0] > 1000 && checked[1] > 1000,
        "too few fully covered pixels: {checked:?}"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
