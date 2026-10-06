//! Renders groups through the platform renderer and checks their pixels: a
//! group faded as one picture, against an element faded piece by piece; a
//! group multiplied into what is beneath it; groups faded three deep; a
//! rotated group; a blurred group; overflow clipped to rounded corners; a
//! group clipped to a circle; and a group masked by luminance.
//!
//! Runs only with `GPUI_RUN_RENDERING_TESTS` set, in an offscreen window;
//! `GPUI_RENDERING_TEST_OUTPUT=<path.png>` saves the image.

#[cfg(target_os = "macos")]
use gpui::{
    AppContext as _, BlendMode, Bounds, Context, IntoElement, MaskMode, ParentElement as _, Render,
    Styled as _, VisualTestAppContext, Window, canvas, div, fill, kurbo, point, px, radians, rgb,
    size,
};

#[cfg(target_os = "macos")]
struct GroupsFixture;

/// A red square with a blue one overlapping its lower right quarter, in a
/// 150px box at (`left`, `top`).
#[cfg(target_os = "macos")]
fn overlapping_squares(left: f32, top: f32) -> gpui::Div {
    div()
        .absolute()
        .left(px(left))
        .top(px(top))
        .size(px(150.))
        .child(div().absolute().size(px(100.)).bg(rgb(0xff0000)))
        .child(
            div()
                .absolute()
                .left(px(50.))
                .top(px(50.))
                .size(px(100.))
                .bg(rgb(0x0000ff)),
        )
}

#[cfg(target_os = "macos")]
impl Render for GroupsFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .relative()
            .bg(rgb(0xffffff))
            // At (50, 50): faded as one picture. Where the squares overlap,
            // only the blue one shows, at half strength.
            .child(overlapping_squares(50., 50.).group_opacity(0.5))
            // At (250, 50): faded piece by piece. Where they overlap, the red
            // one shows through the blue.
            .child(overlapping_squares(250., 50.).opacity(0.5))
            // At (450, 50): a yellow square multiplied into a cyan one: green.
            .child(
                div()
                    .absolute()
                    .left(px(450.))
                    .top(px(50.))
                    .size(px(150.))
                    .bg(rgb(0x00ffff))
                    .child(
                        div()
                            .size(px(100.))
                            .bg(rgb(0xffff00))
                            .mix_blend_mode(BlendMode::Multiply),
                    ),
            )
            // At (50, 300): a black square in three groups, each faded to
            // half: an eighth of black over white.
            .child(
                div().group_opacity(0.5).child(
                    div().group_opacity(0.5).child(
                        div().group_opacity(0.5).child(
                            div()
                                .absolute()
                                .left(px(50.))
                                .top(px(300.))
                                .size(px(100.))
                                .bg(rgb(0x000000)),
                        ),
                    ),
                ),
            )
            // At (250, 300): the overlapping squares faded as one, turned an
            // eighth about their center (325, 375).
            .child(
                overlapping_squares(250., 300.)
                    .group_opacity(0.5)
                    .rotate(radians(std::f32::consts::FRAC_PI_4)),
            )
            // At (450, 300): a blurred black square, whose edge fades.
            .child(
                div()
                    .absolute()
                    .left(px(450.))
                    .top(px(300.))
                    .size(px(100.))
                    .bg(rgb(0x000000))
                    .blur(px(10.)),
            )
            // At (650, 50): a black square in a box with 50px corners that
            // clips its overflow to them.
            .child(
                div()
                    .absolute()
                    .left(px(650.))
                    .top(px(50.))
                    .size(px(150.))
                    .rounded(px(50.))
                    .overflow_hidden()
                    .clip_overflow_to_corners()
                    .child(div().size(px(150.)).bg(rgb(0x000000))),
            )
            // At (650, 300): a red square clipped to a circle of radius 60
            // about its center.
            .child(
                div()
                    .absolute()
                    .left(px(650.))
                    .top(px(300.))
                    .size(px(150.))
                    .bg(rgb(0xff0000))
                    .clip_path(kurbo::Shape::to_path(
                        &kurbo::Circle::new((75., 75.), 60.),
                        0.1,
                    )),
            )
            // At (850, 50): a blue square masked by white on its left half
            // and black on its right: only the left half shows.
            .child(
                canvas(
                    |_, _, _| (),
                    |_, _, window, _| {
                        let square =
                            Bounds::new(point(px(850.), px(50.)), size(px(150.), px(150.)));
                        let half = size(px(75.), px(150.));
                        window.with_mask(
                            square,
                            MaskMode::Luminance,
                            |window| {
                                window.paint_quad(fill(
                                    Bounds::new(point(px(850.), px(50.)), half),
                                    rgb(0xffffff),
                                ));
                                window.paint_quad(fill(
                                    Bounds::new(point(px(925.), px(50.)), half),
                                    rgb(0x000000),
                                ));
                            },
                            |window| window.paint_quad(fill(square, rgb(0x0000ff))),
                        );
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
        .open_offscreen_window_default(|_, cx| cx.new(|_| GroupsFixture))
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
            .all(|(actual, expected)| actual.abs_diff(expected) < 12)
        {
            failures.push(format!(
                "{what}: at ({x}, {y}) expected {rgb:?}, got {:?}",
                actual.0
            ));
        }
    };

    // Faded as one: red alone, the overlap blue alone, each at half.
    expect("group opacity, red", 70., 70., [255, 128, 128]);
    expect("group opacity, overlap", 125., 125., [128, 128, 255]);
    // Faded piece by piece: the overlap mixes red through blue.
    expect("element opacity, overlap", 325., 125., [128, 64, 191]);

    expect("multiply", 500., 100., [0, 255, 0]);
    expect("multiply leaves the rest", 580., 180., [0, 255, 255]);

    expect("three faded groups", 100., 350., [223, 223, 223]);

    // The rotated group: its center is the overlap, blue at half; the
    // unrotated top-left corner of the box is outside it.
    expect("rotated group, center", 325., 375., [128, 128, 255]);
    expect(
        "rotated group, turned-away corner",
        255.,
        305.,
        [255, 255, 255],
    );

    expect("rounded overflow, corner", 655., 55., [255, 255, 255]);
    expect("rounded overflow, middle", 725., 125., [0, 0, 0]);
    expect("rounded overflow, edge", 725., 52., [0, 0, 0]);

    expect("clip path, inside", 725., 375., [255, 0, 0]);
    expect("clip path, corner", 655., 305., [255, 255, 255]);
    expect("clip path, just outside", 725., 312., [255, 255, 255]);
    expect("clip path, just inside", 725., 320., [255, 0, 0]);

    expect("luminance mask, white", 880., 125., [0, 0, 255]);
    expect("luminance mask, black", 970., 125., [255, 255, 255]);

    // The blurred square: solid in the middle, fading across its edge.
    expect("blur, middle", 500., 350., [0, 0, 0]);
    let edge = pixel(450., 350.).0[0];
    if !(40..215).contains(&edge) {
        failures.push(format!("blur, edge: expected a midtone, got {edge}"));
    }

    std::mem::forget(cx);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
