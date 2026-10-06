//! Renders groups on both WGPU tiers and checks their pixels: a group faded as one
//! picture, a group multiplied into what is beneath it, groups faded three deep, a
//! blurred group, and a backdrop filter inside a group. Each group sits away from the
//! viewport's origin, so its target does too, and a target drawn as if it covered the
//! viewport would put its pixels somewhere else.
#![cfg(feature = "test-support")]

use gpui::{
    BackdropFilter, BlendMode, Bounds, ContentMask, DevicePixels, GroupBoundary,
    PlatformHeadlessRenderer, Point, Quad, ScaledFilter, ScaledPixels, Scene, Size,
    solid_background,
};
use gpui_ce_wgpu::WgpuHeadlessRenderer;

const TARGET: Size<DevicePixels> = Size {
    width: DevicePixels(100),
    height: DevicePixels(100),
};

/// Rendering, blending and quantizing to 8 bits differ a little across backends.
const TOLERANCE: u8 = 3;

fn bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
    Bounds {
        origin: Point {
            x: ScaledPixels(x),
            y: ScaledPixels(y),
        },
        size: Size {
            width: ScaledPixels(width),
            height: ScaledPixels(height),
        },
    }
}

fn viewport_mask() -> ContentMask<ScaledPixels> {
    ContentMask {
        bounds: bounds(0.0, 0.0, 100.0, 100.0),
        ..Default::default()
    }
}

fn quad(scene: &mut Scene, bounds: Bounds<ScaledPixels>, color: u32) {
    scene.insert_primitive(Quad {
        bounds,
        content_mask: viewport_mask(),
        background: solid_background(gpui::rgb_to_hsla(gpui::rgb(color))),
        ..Default::default()
    });
}

/// Paints what `contents` paints as one group, made by an element at `bounds`.
fn group(
    scene: &mut Scene,
    bounds: Bounds<ScaledPixels>,
    opacity: f32,
    blend_mode: BlendMode,
    filters: &[ScaledFilter],
    contents: impl FnOnce(&mut Scene),
) {
    let start = GroupBoundary {
        order: 0,
        bounds,
        content_mask: viewport_mask(),
        filters: filters.iter().copied().collect(),
        opacity,
        blend_mode,
        masked: false,
        mask_mode: None,
        is_start: true,
    };
    scene.insert_primitive(start.clone());
    contents(scene);
    scene.insert_primitive(GroupBoundary {
        is_start: false,
        ..start
    });
}

fn render(renderer: &mut WgpuHeadlessRenderer, mut scene: Scene) -> image::RgbaImage {
    scene.finish();
    renderer
        .render_scene_to_image(&scene, TARGET)
        .expect("headless render")
}

/// Renders the scene `build` makes on the device's native tier and on the downlevel
/// tier, twice on each, so the second frame draws into targets the pool kept from the
/// first.
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
            let image = render(&mut renderer, build());
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
fn a_faded_group_fades_as_one_picture() {
    render_on_both_tiers(
        || {
            let mut scene = Scene::default();
            quad(&mut scene, bounds(0.0, 0.0, 100.0, 100.0), 0xffffff);
            // A red square, and a blue one over its lower right quarter.
            group(
                &mut scene,
                bounds(10.0, 10.0, 60.0, 60.0),
                0.5,
                BlendMode::Normal,
                &[],
                |scene| {
                    quad(scene, bounds(10.0, 10.0, 40.0, 40.0), 0xff0000);
                    quad(scene, bounds(30.0, 30.0, 40.0, 40.0), 0x0000ff);
                },
            );
            scene
        },
        |what, image| {
            assert_rgb(image, what, 15, 15, [255, 128, 128]);
            // Where the squares overlap only the blue one shows, at half strength;
            // fading each square would let the red show through.
            assert_rgb(image, what, 40, 40, [128, 128, 255]);
            assert_rgb(image, what, 65, 65, [128, 128, 255]);
            assert_rgb(image, what, 90, 90, [255, 255, 255]);
        },
    );
}

#[test]
fn a_multiplied_group_mixes_with_what_is_beneath_it() {
    render_on_both_tiers(
        || {
            let mut scene = Scene::default();
            quad(&mut scene, bounds(0.0, 0.0, 100.0, 100.0), 0x00ffff);
            group(
                &mut scene,
                bounds(20.0, 20.0, 40.0, 40.0),
                1.0,
                BlendMode::Multiply,
                &[],
                |scene| quad(scene, bounds(20.0, 20.0, 40.0, 40.0), 0xffff00),
            );
            scene
        },
        |what, image| {
            // Yellow multiplied into cyan is green; the rest stays cyan.
            assert_rgb(image, what, 22, 22, [0, 255, 0]);
            assert_rgb(image, what, 40, 40, [0, 255, 0]);
            assert_rgb(image, what, 80, 80, [0, 255, 255]);
            assert_rgb(image, what, 10, 40, [0, 255, 255]);
        },
    );
}

#[test]
fn nested_faded_groups_multiply_their_opacities() {
    render_on_both_tiers(
        || {
            let mut scene = Scene::default();
            quad(&mut scene, bounds(0.0, 0.0, 100.0, 100.0), 0xffffff);
            let square = bounds(30.0, 30.0, 40.0, 40.0);
            group(&mut scene, square, 0.5, BlendMode::Normal, &[], |scene| {
                group(scene, square, 0.5, BlendMode::Normal, &[], |scene| {
                    group(scene, square, 0.5, BlendMode::Normal, &[], |scene| {
                        quad(scene, square, 0x000000)
                    })
                })
            });
            scene
        },
        |what, image| {
            // An eighth of black over white.
            assert_rgb(image, what, 50, 50, [223, 223, 223]);
            assert_rgb(image, what, 10, 10, [255, 255, 255]);
        },
    );
}

#[test]
fn a_blurred_group_spreads_what_it_draws() {
    render_on_both_tiers(
        || {
            let mut scene = Scene::default();
            quad(&mut scene, bounds(0.0, 0.0, 100.0, 100.0), 0xffffff);
            let square = bounds(30.0, 30.0, 40.0, 40.0);
            group(
                &mut scene,
                square,
                1.0,
                BlendMode::Normal,
                &[ScaledFilter::Blur(ScaledPixels(4.0))],
                |scene| quad(scene, square, 0x000000),
            );
            scene
        },
        |what, image| {
            // Solid in the middle, fading across the edge, untouched far away.
            assert!(
                image.get_pixel(50, 50).0[0] < 20,
                "{what}: the blurred square's middle must stay dark, got {:?}",
                image.get_pixel(50, 50).0
            );
            for (x, y) in [(30, 50), (70, 50), (50, 30), (50, 70)] {
                let edge = image.get_pixel(x, y).0[0];
                assert!(
                    (40..215).contains(&edge),
                    "{what}: the blurred edge at ({x}, {y}) must be a midtone, got {edge}"
                );
            }
            // Outside the square, the blur spreads it.
            assert!(
                image.get_pixel(27, 50).0[0] < 250,
                "{what}: the blur must spread past the square"
            );
            assert_rgb(image, what, 5, 5, [255, 255, 255]);
        },
    );
}

#[test]
fn a_backdrop_filter_in_a_group_blurs_the_groups_target() {
    render_on_both_tiers(
        || {
            let mut scene = Scene::default();
            quad(&mut scene, bounds(0.0, 0.0, 100.0, 100.0), 0x00ff00);
            // A black and a white half, the edge between them blurred, the whole faded
            // to half over green.
            group(
                &mut scene,
                bounds(40.0, 40.0, 60.0, 60.0),
                0.5,
                BlendMode::Normal,
                &[],
                |scene| {
                    quad(scene, bounds(40.0, 40.0, 30.0, 60.0), 0x000000);
                    quad(scene, bounds(70.0, 40.0, 30.0, 60.0), 0xffffff);
                    scene.insert_primitive(BackdropFilter {
                        bounds: bounds(40.0, 40.0, 60.0, 60.0),
                        content_mask: viewport_mask(),
                        filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(3.0))],
                        opacity: 1.0,
                        ..Default::default()
                    });
                },
            );
            scene
        },
        |what, image| {
            assert_rgb(image, what, 10, 10, [0, 255, 0]);
            assert_rgb(image, what, 50, 70, [0, 128, 0]);
            assert_rgb(image, what, 90, 70, [128, 255, 128]);
            // Across the edge, the halves blend into each other: unblurred, x 68 would be
            // black and x 72 white.
            let left = image.get_pixel(68, 70).0[0];
            let right = image.get_pixel(72, 70).0[0];
            assert!(
                (10..64).contains(&left) && (64..118).contains(&right),
                "{what}: the edge must be blurred, got red {left} and {right} either side of it"
            );
        },
    );
}
