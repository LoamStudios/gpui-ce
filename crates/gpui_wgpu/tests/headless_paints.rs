//! Renders quads filled from the scene's paint table through the headless WGPU renderer,
//! on the device's native tier (storage buffers) and the downlevel tier (data textures),
//! and checks the gradients' pixels.
#![cfg(feature = "test-support")]

use gpui::{
    Bounds, ContentMask, DevicePixels, PlatformHeadlessRenderer, Point, Quad, ScaledPixels, Scene,
    Size, TransformationMatrix, peniko,
};
use gpui_ce_wgpu::WgpuHeadlessRenderer;

const TARGET: Size<DevicePixels> = Size {
    width: DevicePixels(100),
    height: DevicePixels(40),
};

const TOLERANCE: u8 = 4;

fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
    Bounds {
        origin: Point {
            x: ScaledPixels(x),
            y: ScaledPixels(y),
        },
        size: Size {
            width: ScaledPixels(w),
            height: ScaledPixels(h),
        },
    }
}

/// A linear gradient from `from` at x = 0 to `to` at x = 100, in device pixels.
fn horizontal(
    from: peniko::color::AlphaColor<peniko::color::Srgb>,
    to: peniko::color::AlphaColor<peniko::color::Srgb>,
) -> peniko::Gradient {
    peniko::Gradient::new_linear((0., 0.), (100., 0.)).with_stops([from, to])
}

/// Fills the whole target with paint-table entry `index`.
fn paint_quad(scene: &mut Scene, index: u32) {
    let target = bounds(0.0, 0.0, 100.0, 40.0);
    scene.insert_primitive(Quad {
        bounds: target,
        content_mask: ContentMask {
            bounds: target,
            ..Default::default()
        },
        background: gpui::ScenePaintRef {
            color: gpui::white().into(),
            paint: index,
        },
        ..Default::default()
    });
}

fn render_on_both_tiers(build: impl Fn() -> Scene, check: impl Fn(&str, &image::RgbaImage)) {
    let renderers = [
        (
            "native",
            WgpuHeadlessRenderer::new().expect("headless renderer"),
        ),
        (
            "downlevel",
            WgpuHeadlessRenderer::new_downlevel().expect("downlevel renderer"),
        ),
    ];
    for (tier, mut renderer) in renderers {
        for frame in ["first", "second"] {
            let mut scene = build();
            scene.finish();
            let image = renderer
                .render_scene_to_image(&scene, TARGET)
                .expect("headless render");
            check(&format!("{tier} tier, {frame} frame"), &image);
        }
    }
}

fn assert_rgb(image: &image::RgbaImage, what: &str, x: u32, y: u32, expected: [u8; 3]) {
    let actual = image.get_pixel(x, y).0;
    assert!(
        actual[..3]
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.abs_diff(expected) <= TOLERANCE),
        "{what} at ({x}, {y}): expected {expected:?}, got {actual:?}"
    );
}

#[test]
fn a_linear_gradient_from_the_paint_table_fills_a_quad() {
    use peniko::color::palette::css::{BLUE, RED};
    render_on_both_tiers(
        || {
            let mut scene = Scene::default();
            let index = scene
                .push_gradient(&horizontal(RED, BLUE), TransformationMatrix::UNIT)
                .expect("gradient entry");
            paint_quad(&mut scene, index);
            scene
        },
        |what, image| {
            assert_rgb(image, &format!("{what}, start"), 0, 20, [255, 0, 0]);
            // sRGB interpolation: halfway is (128, 0, 128).
            assert_rgb(image, &format!("{what}, middle"), 50, 20, [128, 0, 128]);
            assert_rgb(image, &format!("{what}, end"), 99, 20, [0, 0, 255]);
        },
    );
}

/// Enough entries and stops to grow both tables past their initial capacity, so the
/// group-0 bind group is rebuilt; the quad draws the last entry, whose stops are last.
#[test]
fn a_late_entry_of_a_grown_paint_table_reads_its_own_stops() {
    use peniko::color::palette::css::{BLACK, BLUE, LIME, RED, WHITE};
    render_on_both_tiers(
        || {
            let mut scene = Scene::default();
            for _ in 0..100 {
                scene
                    .push_gradient(&horizontal(BLACK, WHITE), TransformationMatrix::UNIT)
                    .expect("gradient entry");
            }
            let index = scene
                .push_gradient(
                    &peniko::Gradient::new_linear((0., 0.), (100., 0.)).with_stops([
                        (0.0, RED),
                        (0.5, LIME),
                        (1.0, BLUE),
                    ]),
                    TransformationMatrix::UNIT,
                )
                .expect("gradient entry");
            assert!(index >= 100, "the last entry follows the others");
            paint_quad(&mut scene, index);
            scene
        },
        |what, image| {
            assert_rgb(image, &format!("{what}, start"), 0, 20, [255, 0, 0]);
            // The pixel centre, 50.5, is 1% of the way from the lime stop to the blue one.
            assert_rgb(image, &format!("{what}, middle"), 50, 20, [0, 252, 3]);
            assert_rgb(image, &format!("{what}, end"), 99, 20, [0, 0, 255]);
        },
    );
}
