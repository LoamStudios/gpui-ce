//! Renders groups on both WGPU tiers and checks their pixels: a group faded as one
//! picture, a group multiplied into what is beneath it, groups faded three deep, a
//! blurred group, a backdrop filter inside a group, and groups masked by alpha and by
//! luminance. Each group sits away from the viewport's origin, so its target does too,
//! and a target drawn as if it covered the viewport would put its pixels somewhere else.
#![cfg(feature = "test-support")]

use gpui::{
    BackdropFilter, BlendMode, Bounds, ContentMask, DevicePixels, GroupBoundary, MaskMode,
    PlatformHeadlessRenderer, Point, Quad, ScaledFilter, ScaledPixels, Scene, Size,
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
        background: gpui::rgb_to_hsla(gpui::rgb(color)).into(),
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

/// Paints what `contents` paints as one group, made by an element at `bounds`, masked
/// in `mode` by what `mask` paints within `mask_bounds`: the group's first child group.
fn masked_group(
    scene: &mut Scene,
    bounds: Bounds<ScaledPixels>,
    mask_bounds: Bounds<ScaledPixels>,
    mode: MaskMode,
    mask: impl FnOnce(&mut Scene),
    contents: impl FnOnce(&mut Scene),
) {
    let start = GroupBoundary {
        order: 0,
        bounds,
        content_mask: viewport_mask(),
        filters: Default::default(),
        opacity: 1.0,
        blend_mode: BlendMode::Normal,
        masked: true,
        mask_mode: None,
        is_start: true,
    };
    let mask_start = GroupBoundary {
        bounds: mask_bounds,
        masked: false,
        mask_mode: Some(mode),
        ..start.clone()
    };
    scene.insert_primitive(start.clone());
    scene.insert_primitive(mask_start.clone());
    mask(scene);
    scene.insert_primitive(GroupBoundary {
        is_start: false,
        ..mask_start
    });
    contents(scene);
    scene.insert_primitive(GroupBoundary {
        is_start: false,
        ..start
    });
}

#[test]
fn an_alpha_mask_shows_the_group_only_where_it_is_opaque() {
    render_on_both_tiers(
        || {
            let mut scene = Scene::default();
            quad(&mut scene, bounds(0.0, 0.0, 100.0, 100.0), 0xffffff);
            masked_group(
                &mut scene,
                bounds(10.0, 10.0, 80.0, 80.0),
                bounds(30.0, 30.0, 40.0, 40.0),
                MaskMode::Alpha,
                // Its colour does not matter, only its coverage.
                |scene| quad(scene, bounds(30.0, 30.0, 40.0, 40.0), 0x000000),
                |scene| quad(scene, bounds(10.0, 10.0, 80.0, 80.0), 0xff0000),
            );
            scene
        },
        |what, image| {
            assert_rgb(image, what, 50, 50, [255, 0, 0]);
            assert_rgb(image, what, 32, 68, [255, 0, 0]);
            assert_rgb(image, what, 15, 15, [255, 255, 255]);
            assert_rgb(image, what, 85, 50, [255, 255, 255]);
            assert_rgb(image, what, 50, 27, [255, 255, 255]);
        },
    );
}

#[test]
fn a_luminance_mask_shows_the_group_where_it_is_white() {
    render_on_both_tiers(
        || {
            let mut scene = Scene::default();
            quad(&mut scene, bounds(0.0, 0.0, 100.0, 100.0), 0xffffff);
            let square = bounds(20.0, 20.0, 60.0, 60.0);
            masked_group(
                &mut scene,
                square,
                square,
                MaskMode::Luminance,
                |scene| {
                    quad(scene, bounds(20.0, 20.0, 30.0, 60.0), 0xffffff);
                    quad(scene, bounds(50.0, 20.0, 30.0, 60.0), 0x000000);
                },
                |scene| quad(scene, square, 0x0000ff),
            );
            scene
        },
        |what, image| {
            assert_rgb(image, what, 30, 50, [0, 0, 255]);
            assert_rgb(image, what, 70, 50, [255, 255, 255]);
            assert_rgb(image, what, 10, 50, [255, 255, 255]);
        },
    );
}

#[test]
fn a_mask_partly_outside_its_group_shows_only_their_overlap() {
    render_on_both_tiers(
        || {
            let mut scene = Scene::default();
            quad(&mut scene, bounds(0.0, 0.0, 100.0, 100.0), 0xffffff);
            masked_group(
                &mut scene,
                bounds(20.0, 20.0, 40.0, 40.0),
                bounds(40.0, 40.0, 50.0, 50.0),
                MaskMode::Alpha,
                |scene| quad(scene, bounds(40.0, 40.0, 50.0, 50.0), 0x000000),
                |scene| quad(scene, bounds(20.0, 20.0, 40.0, 40.0), 0x00ff00),
            );
            scene
        },
        |what, image| {
            assert_rgb(image, what, 50, 50, [0, 255, 0]);
            assert_rgb(image, what, 42, 58, [0, 255, 0]);
            // The group without the mask, and the mask without the group.
            assert_rgb(image, what, 30, 30, [255, 255, 255]);
            assert_rgb(image, what, 50, 30, [255, 255, 255]);
            assert_rgb(image, what, 70, 70, [255, 255, 255]);
            assert_rgb(image, what, 85, 50, [255, 255, 255]);
        },
    );
}

#[test]
fn a_group_masked_out_of_view_is_not_drawn() {
    render_on_both_tiers(
        || {
            let mut scene = Scene::default();
            quad(&mut scene, bounds(0.0, 0.0, 100.0, 100.0), 0xffffff);
            masked_group(
                &mut scene,
                bounds(20.0, 20.0, 40.0, 40.0),
                bounds(200.0, 200.0, 40.0, 40.0),
                MaskMode::Alpha,
                |scene| quad(scene, bounds(200.0, 200.0, 40.0, 40.0), 0x000000),
                |scene| quad(scene, bounds(20.0, 20.0, 40.0, 40.0), 0xff0000),
            );
            // What follows the hidden group is still drawn.
            quad(&mut scene, bounds(70.0, 70.0, 20.0, 20.0), 0x0000ff);
            scene
        },
        |what, image| {
            assert_rgb(image, what, 40, 40, [255, 255, 255]);
            assert_rgb(image, what, 80, 80, [0, 0, 255]);
        },
    );
}
