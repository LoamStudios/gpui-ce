//! Renders text through the platform renderer, plainly, in composited groups
//! and under a non-uniform scale, and checks its pixels: text in a group
//! composites exactly as it draws outside one, without colour fringes, and
//! text stretched three times as tall as it is wide keeps its edges sharp
//! without aliasing along the axis it is squeezed on.
//!
//! Runs only with `GPUI_RUN_RENDERING_TESTS` set, in an offscreen window;
//! `GPUI_RENDERING_TEST_OUTPUT=<path.png>` saves the image.

#[cfg(target_os = "macos")]
use gpui::{
    AppContext as _, BlendMode, Context, IntoElement, ParentElement as _, Render, Styled as _,
    VisualTestAppContext, Window, div, kurbo, px, rgb,
};
#[cfg(target_os = "macos")]
use std::borrow::Cow;

#[cfg(target_os = "macos")]
#[allow(dead_code)]
#[path = "../../gpui_ce_parley/src/font_fixtures.rs"]
mod font_fixtures;

/// A 300×40 line of text at (`left`, `top`).
#[cfg(target_os = "macos")]
fn line(left: f32, top: f32) -> gpui::Div {
    div()
        .absolute()
        .left(px(left))
        .top(px(top))
        .w(px(300.))
        .h(px(40.))
        .child("Hamburgefontsiv")
}

#[cfg(target_os = "macos")]
struct TextFixture;

#[cfg(target_os = "macos")]
impl Render for TextFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .relative()
            .bg(rgb(0xffffff))
            .text_color(rgb(0x0000ff))
            .font_family("IBM Plex Sans")
            .text_size(px(20.))
            // Plain, at (20, 20).
            .child(line(20., 20.))
            // In a group multiplied into white, which leaves it as it is.
            .child(line(20., 80.).mix_blend_mode(BlendMode::Multiply))
            // In a group faded to half.
            .child(line(20., 140.).group_opacity(0.5))
            // In a group clipped by a path around all of it.
            .child(line(20., 200.).clip_path(kurbo::Shape::to_path(
                &kurbo::Rect::new(-10., -10., 310., 50.),
                0.1,
            )))
            // Three times as tall, about its center (170, 580): its top
            // edge, y = 560, lands at y = 520.
            .child(line(20., 560.).transform(kurbo::Affine::scale_non_uniform(1., 3.)))
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
    cx.update(|cx| {
        cx.text_system()
            .add_fonts(vec![Cow::Borrowed(font_fixtures::IBM_PLEX.data)])
    })
    .expect("failed to load the fixture font");
    let window = cx
        .open_offscreen_window_default(|_, cx| cx.new(|_| TextFixture))
        .expect("failed to create the offscreen window");
    let window = window.into();
    cx.run_until_parked();
    let image = cx
        .capture_screenshot(window)
        .expect("failed to capture the rendered window");
    if let Some(output) = std::env::var_os("GPUI_RENDERING_TEST_OUTPUT") {
        image.save(output).expect("failed to save the image");
    }

    let scale = image.width() / 1280;
    let pixel = |x: u32, y: u32| image.get_pixel(x, y).0;
    // The device pixels of a 200×30 window region at (left, top).
    let region = |left: u32, top: u32| {
        (top * scale..(top + 30) * scale)
            .flat_map(move |y| (left * scale..(left + 200) * scale).map(move |x| pixel(x, y)))
    };
    let mut failures = Vec::new();

    // Blue text over white: an antialiased pixel is (c, c, 255). A subpixel
    // mask composited through a group's alpha would leave red and green
    // apart, or blue below white.
    let plain: Vec<_> = region(20, 20).collect();
    assert!(
        plain.iter().any(|pixel| pixel[0] < 64),
        "the plain text was drawn"
    );
    for (what, top, faded) in [
        ("multiplied group", 80, false),
        ("faded group", 140, true),
        ("clip path", 200, false),
    ] {
        let mut worst = 0;
        let mut fringes = 0;
        for (plain, grouped) in plain.iter().zip(region(20, top)) {
            // Faded to half over white: halfway from the plain text to white.
            let expected = if faded {
                255 - (255 - plain[0]) / 2
            } else {
                plain[0]
            };
            worst = worst.max(grouped[0].abs_diff(expected));
            if grouped[0].abs_diff(grouped[1]) > 2 || grouped[2] < 250 {
                fringes += 1;
            }
        }
        if worst > 3 {
            failures.push(format!(
                "text in a {what} differs from plain text by up to {worst}"
            ));
        }
        if fringes > 0 {
            failures.push(format!(
                "text in a {what} has {fringes} pixels with colour fringes"
            ));
        }
    }

    // The stretched text. Its horizontal edges are as sharp as the plain
    // text's, not magnified from a smaller raster: the steepest steps down
    // its columns are near full contrast.
    let luminance = |x: u32, y: u32| f32::from(pixel(x, y)[0]);
    let mut steps: Vec<f32> = (20 * scale..220 * scale)
        .flat_map(|x| {
            (500 * scale..640 * scale).map(move |y| (luminance(x, y + 1) - luminance(x, y)).abs())
        })
        .collect();
    steps.sort_by(|a, b| b.total_cmp(a));
    let steepest = steps[..400].iter().sum::<f32>() / 400.;
    if steepest < 170. {
        failures.push(format!(
            "the stretched text's horizontal edges are soft: steepest steps average {steepest:.0}"
        ));
    }
    // Squeezed back to its height, each three rows averaged, it matches the
    // plain text: its raster, drawn narrower than it was made, is filtered
    // across its width rather than sampled at points.
    let mut difference = 0.;
    let mut count = 0.;
    for row in 0..40 * scale {
        let top = 520 * scale + 3 * row;
        for x in 20 * scale..220 * scale {
            let squeezed = (luminance(x, top) + luminance(x, top + 1) + luminance(x, top + 2)) / 3.;
            difference += (luminance(x, 20 * scale + row) - squeezed).abs();
            count += 1.;
        }
    }
    let difference = difference / count;
    if difference > 2.6 {
        failures.push(format!(
            "the stretched text, squeezed back, differs from the plain text by {difference:.2} on average"
        ));
    }

    std::mem::forget(cx);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
