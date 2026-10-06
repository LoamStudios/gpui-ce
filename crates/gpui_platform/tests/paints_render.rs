//! Renders gradients from the paint table through the platform renderer and
//! checks their pixels: linear, radial and sweep gradients filling elements,
//! a repeating gradient, a gradient filling a path, and a gradient on a
//! rotated element.
//!
//! Runs only with `GPUI_RUN_RENDERING_TESTS` set, in an offscreen window;
//! `GPUI_RENDERING_TEST_OUTPUT=<path.png>` saves the image.

#[cfg(target_os = "macos")]
use gpui::{
    AppContext as _, Context, Fill, IntoElement, ParentElement as _, PathBuilder, Render,
    Styled as _, VisualTestAppContext, Window, canvas, div, peniko, point, px, radians, rgb,
};

#[cfg(target_os = "macos")]
struct PaintsFixture;

#[cfg(target_os = "macos")]
fn red_to_blue(gradient: peniko::Gradient) -> Fill {
    Fill::gradient(gradient.with_stops([
        peniko::color::palette::css::RED,
        peniko::color::palette::css::BLUE,
    ]))
}

#[cfg(target_os = "macos")]
fn square(left: f32, top: f32) -> gpui::Div {
    div().absolute().left(px(left)).top(px(top)).size(px(100.))
}

#[cfg(target_os = "macos")]
impl Render for PaintsFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .relative()
            .bg(rgb(0xffffff))
            // At (50, 50): red on the left to blue on the right.
            .child(
                square(50., 50.).bg(red_to_blue(peniko::Gradient::new_linear(
                    (0., 0.),
                    (100., 0.),
                ))),
            )
            // At (200, 50): red at the centre to blue 50px out, and beyond.
            .child(square(200., 50.).bg(red_to_blue(peniko::Gradient::new_radial((50., 50.), 50.))))
            // At (350, 50): red to blue clockwise from the right of the centre.
            .child(
                square(350., 50.).bg(red_to_blue(peniko::Gradient::new_sweep(
                    (50., 50.),
                    0.,
                    std::f32::consts::TAU,
                ))),
            )
            // At (500, 50): red to blue every 20px, repeated.
            .child(
                square(500., 50.).bg(red_to_blue(
                    peniko::Gradient::new_linear((0., 0.), (20., 0.))
                        .with_extend(peniko::Extend::Repeat),
                )),
            )
            // At (650, 50): the left-to-right gradient turned a quarter
            // clockwise: red at the top, blue at the bottom.
            .child(
                square(650., 50.)
                    .bg(red_to_blue(peniko::Gradient::new_linear(
                        (0., 0.),
                        (100., 0.),
                    )))
                    .rotate(radians(std::f32::consts::FRAC_PI_2)),
            )
            // At (50, 200): a triangle filled red at its top to blue at its
            // base.
            .child(
                canvas(
                    |_, _, _| (),
                    |_, _, window, _| {
                        let gradient = window.gradient(
                            &peniko::Gradient::new_linear((0., 200.), (0., 300.)).with_stops([
                                peniko::color::palette::css::RED,
                                peniko::color::palette::css::BLUE,
                            ]),
                        );
                        let mut builder = PathBuilder::fill();
                        builder.move_to(point(px(100.), px(200.)));
                        builder.line_to(point(px(150.), px(300.)));
                        builder.line_to(point(px(50.), px(300.)));
                        builder.close();
                        window.paint_path(builder.build().unwrap(), gradient);
                    },
                )
                .absolute()
                .size_full(),
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
        .open_offscreen_window_default(|_, cx| cx.new(|_| PaintsFixture))
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
            .all(|(actual, expected)| actual.abs_diff(expected) < 16)
        {
            failures.push(format!(
                "{what}: at ({x}, {y}) expected {rgb:?}, got {:?}",
                actual.0
            ));
        }
    };

    expect("linear, start", 51., 100., [255, 0, 0]);
    expect("linear, middle", 100., 100., [128, 0, 128]);
    expect("linear, end", 149., 100., [0, 0, 255]);

    expect("radial, centre", 250., 100., [255, 0, 0]);
    expect("radial, halfway", 275., 100., [128, 0, 128]);
    expect("radial, corner", 203., 53., [0, 0, 255]);

    expect("sweep, just past the start", 440., 103., [252, 0, 3]);
    expect("sweep, a quarter turn", 400., 140., [191, 0, 64]);
    expect("sweep, just before the end", 440., 97., [3, 0, 252]);

    expect("repeat, a quarter in", 505., 100., [191, 0, 64]);
    expect("repeat, near the end", 519., 100., [13, 0, 242]);
    expect("repeat, a quarter in again", 525., 100., [191, 0, 64]);

    expect("rotated, top", 700., 52., [255, 0, 0]);
    expect("rotated, bottom", 700., 148., [0, 0, 255]);

    expect("path, top", 100., 210., [230, 0, 25]);
    expect("path, base", 100., 295., [13, 0, 242]);
    expect("path, outside", 55., 210., [255, 255, 255]);

    std::mem::forget(cx);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
