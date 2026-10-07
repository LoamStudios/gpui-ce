//! Renders shader program paints through the platform renderer and checks
//! their pixels: until a renderer links a paint's program into its shaders,
//! the paint's fallback colour is drawn.
//!
//! Runs only with `GPUI_RUN_RENDERING_TESTS` set, in an offscreen window;
//! `GPUI_RENDERING_TEST_OUTPUT=<path.png>` saves the image.

#[cfg(target_os = "macos")]
use gpui::{
    AppContext as _, Context, Fill, IntoElement, ParentElement as _, Render, Styled as _,
    VisualTestAppContext, Window, div, px, rgb,
    shader::{self, Paint},
};

/// Film grain: value noise over the box's own logical pixels.
#[cfg(target_os = "macos")]
fn grain() -> Paint {
    shader::paint(|px| {
        let value = shader::noise::value(px.position() * 0.5);
        shader::rgba(&value, &value, &value, 1.0)
    })
}

#[cfg(target_os = "macos")]
struct ProgramsFixture;

#[cfg(target_os = "macos")]
impl Render for ProgramsFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().relative().bg(rgb(0xffffff)).child(
            div()
                .absolute()
                .left(px(50.))
                .top(px(50.))
                .size(px(100.))
                .bg(Fill::program(grain().fallback(rgb(0x336699)))),
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
    let mut cx = VisualTestAppContext::new(gpui_ce_platform::current_platform(false));
    let window = cx
        .open_offscreen_window_default(|_, cx| cx.new(|_| ProgramsFixture))
        .expect("failed to create the offscreen window");
    let window = window.into();
    cx.run_until_parked();
    let image = cx
        .capture_screenshot(window)
        .expect("failed to capture the rendered window");
    if let Some(output) = std::env::var_os("GPUI_RENDERING_TEST_OUTPUT") {
        image.save(output).expect("failed to save the image");
    }

    let scale = image.width() as f32 / 1280.;
    let pixel = |x: f32, y: f32| *image.get_pixel((x * scale) as u32, (y * scale) as u32);
    let mut failures = Vec::new();
    let mut expect = |what: &str, x: f32, y: f32, rgb: [u8; 3]| {
        let actual = pixel(x, y);
        if !actual.0[..3]
            .iter()
            .zip(rgb)
            .all(|(actual, expected)| actual.abs_diff(expected) < 4)
        {
            failures.push(format!(
                "{what}: at ({x}, {y}) expected {rgb:?}, got {:?}",
                actual.0
            ));
        }
    };

    for (x, y) in [(51., 51.), (100., 100.), (148., 120.)] {
        expect("the fallback", x, y, [0x33, 0x66, 0x99]);
    }
    expect("outside", 45., 100., [255, 255, 255]);
    expect("outside", 155., 100., [255, 255, 255]);

    std::mem::forget(cx);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
