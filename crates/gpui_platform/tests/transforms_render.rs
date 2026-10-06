//! Renders elements under transforms through the platform renderer and checks
//! where their pixels land: a rotated card, a rotated frame that clips its
//! overflowing child along its own edges, and a zoomed card whose text is
//! rasterized at the size it appears.
//!
//! Runs only with `GPUI_RUN_RENDERING_TESTS` set, in an offscreen window;
//! `GPUI_RENDERING_TEST_OUTPUT=<path.png>` saves the image.

#[cfg(target_os = "macos")]
use gpui::{
    AppContext as _, Context, IntoElement, ParentElement as _, Render, Styled as _,
    VisualTestAppContext, Window, div, kurbo, px, radians, rgb,
};
#[cfg(target_os = "macos")]
use std::f32::consts::FRAC_PI_4;

#[cfg(target_os = "macos")]
struct TransformsFixture;

#[cfg(target_os = "macos")]
impl Render for TransformsFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .relative()
            .bg(rgb(0xffffff))
            // A 200×100 red card at (100, 100), turned an eighth about its
            // center (200, 150).
            .child(
                div()
                    .absolute()
                    .left(px(100.))
                    .top(px(100.))
                    .w(px(200.))
                    .h(px(100.))
                    .bg(rgb(0xff0000))
                    .rotate(radians(FRAC_PI_4))
                    .text_color(rgb(0xffffff))
                    .child("Rotated card"),
            )
            // A 200×100 frame at (500, 100), turned an eighth about its center
            // (600, 150), clipping a blue child twice its size.
            .child(
                div()
                    .absolute()
                    .left(px(500.))
                    .top(px(100.))
                    .w(px(200.))
                    .h(px(100.))
                    .overflow_hidden()
                    .rotate(radians(-FRAC_PI_4))
                    .child(div().w(px(400.)).h(px(200.)).bg(rgb(0x0000ff))),
            )
            // A 100×40 green card at (100, 400), zoomed three times about its
            // center (150, 420), with text that must stay sharp.
            .child(
                div()
                    .absolute()
                    .left(px(100.))
                    .top(px(400.))
                    .w(px(100.))
                    .h(px(40.))
                    .bg(rgb(0x00aa00))
                    .text_color(rgb(0x000000))
                    .transform(kurbo::Affine::scale(3.))
                    .child("Zoomed"),
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
        .open_offscreen_window_default(|_, cx| cx.new(|_| TransformsFixture))
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
    let is = |x: f32, y: f32, rgb: [u8; 3]| {
        let pixel = pixel(x, y);
        pixel.0[..3]
            .iter()
            .zip(rgb)
            .all(|(actual, expected)| actual.abs_diff(expected) < 24)
    };
    const WHITE: [u8; 3] = [255, 255, 255];

    // The rotated card: its unrotated top-left corner is now white, and a
    // point above its unrotated top edge, on the turned diagonal, is red.
    assert!(
        is(104., 104., WHITE),
        "the turned-away corner is empty: {:?}",
        pixel(104., 104.)
    );
    assert!(
        is(200., 85., [255, 0, 0]),
        "the card turned into it: {:?}",
        pixel(200., 85.)
    );

    // The rotated frame clips along its own edges: the child fills the
    // frame, which reaches above the frame's unrotated top, and nothing of
    // the child shows beyond the frame's unrotated right end, where the
    // unclipped child would be.
    assert!(
        is(600., 90., [0, 0, 255]),
        "the frame shows its child: {:?}",
        pixel(600., 90.)
    );
    assert!(
        is(760., 150., WHITE),
        "the child is clipped by the frame: {:?}",
        pixel(760., 150.)
    );
    assert!(
        is(504., 104., WHITE),
        "the frame's turned-away corner is empty: {:?}",
        pixel(504., 104.)
    );

    // The zoomed card covers (0, 360)-(300, 480).
    assert!(
        is(10., 470., [0, 170, 0]),
        "the zoomed card covers its scaled bounds: {:?}",
        pixel(10., 470.)
    );
    assert!(is(10., 350., WHITE), "and no more: {:?}", pixel(10., 350.));

    std::mem::forget(cx);
}
