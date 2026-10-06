//! Renders primitives that refer to the scene's transform and clip tables, on both WGPU
//! tiers: storage buffers on the device's native tier, and the downlevel data textures.
//! Each scene would paint somewhere else, or nothing, if a table were not bound or not
//! uploaded, so these prove the tables reach the shaders.
#![cfg(feature = "test-support")]

use gpui::{
    Bounds, ContentMask, Corners, DevicePixels, Hsla, PlatformHeadlessRenderer, Point, Quad,
    ScaledPixels, Scene, SceneClip, SceneTransform, Size, TransformationMatrix, solid_background,
};
use gpui_ce_wgpu::WgpuHeadlessRenderer;

const TARGET: Size<DevicePixels> = Size {
    width: DevicePixels(100),
    height: DevicePixels(100),
};
const BLACK: [u8; 4] = [0, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];

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

fn green() -> Hsla {
    gpui::rgb_to_hsla(gpui::rgb(0x00ff00))
}

/// A quarter turn clockwise about the viewport's centre, (50, 50): a point at
/// (x, y) lands at (100 - y, x).
fn quarter_turn() -> SceneTransform {
    let transformation = TransformationMatrix {
        rotation_scale: [[0.0, -1.0], [1.0, 0.0]],
        translation: [100.0, 0.0],
    };
    SceneTransform {
        transformation,
        inverse: transformation.inverse().expect("a rotation is invertible"),
    }
}

/// A horizontal bar, x 10..40 and y 40..60, turned about the centre into a vertical
/// bar, x 40..60 and y 10..40.
fn turned_bar(scene: &mut Scene, transform: u32) {
    scene.insert_primitive(Quad {
        transform,
        bounds: bounds(10.0, 40.0, 30.0, 20.0),
        content_mask: viewport_mask(),
        background: solid_background(green()),
        ..Default::default()
    });
}

fn assert_turned_bar(image: &image::RgbaImage) {
    for (x, y) in [(50, 15), (45, 25), (55, 35)] {
        assert_eq!(
            image.get_pixel(x, y).0,
            GREEN,
            "the transformed bar must cover ({x}, {y})"
        );
    }
    for (x, y) in [(15, 50), (25, 45), (35, 55), (50, 50)] {
        assert_eq!(
            image.get_pixel(x, y).0,
            BLACK,
            "the untransformed bar must not cover ({x}, {y})"
        );
    }
}

/// A full-viewport quad clipped to a rectangle x 30..70 and y 40..60 in the quarter
/// turn's space, which covers x 40..60 and y 30..70 in the viewport.
fn clipped_fill(scene: &mut Scene) {
    let transform = scene.push_transform(quarter_turn());
    let clip = scene.push_clip(SceneClip {
        bounds: bounds(30.0, 40.0, 40.0, 20.0),
        corner_radii: Corners::default(),
        transform,
        parent: 0,
    });
    scene.insert_primitive(Quad {
        clip,
        bounds: bounds(0.0, 0.0, 100.0, 100.0),
        content_mask: viewport_mask(),
        background: solid_background(green()),
        ..Default::default()
    });
}

fn assert_clipped_fill(image: &image::RgbaImage) {
    for (x, y) in [(50, 35), (45, 50), (55, 65)] {
        assert_eq!(
            image.get_pixel(x, y).0,
            GREEN,
            "the transformed clip must keep ({x}, {y})"
        );
    }
    // Inside the clip's rectangle before its transform, but outside it after.
    for (x, y) in [(35, 50), (65, 45), (10, 10), (90, 90)] {
        assert_eq!(
            image.get_pixel(x, y).0,
            BLACK,
            "the transformed clip must remove ({x}, {y})"
        );
    }
}

fn render(renderer: &mut WgpuHeadlessRenderer, mut scene: Scene) -> image::RgbaImage {
    scene.finish();
    renderer
        .render_scene_to_image(&scene, TARGET)
        .expect("headless render")
}

/// A downlevel renderer that has already drawn an unrelated frame, so a data texture
/// that lagged a frame behind its upload would show that frame instead of the scene
/// under test.
fn warmed_downlevel_renderer() -> WgpuHeadlessRenderer {
    let mut renderer = WgpuHeadlessRenderer::new_downlevel().expect("downlevel renderer");
    let mut warmup = Scene::default();
    warmup.insert_primitive(Quad {
        bounds: bounds(0.0, 0.0, 1.0, 1.0),
        content_mask: viewport_mask(),
        background: solid_background(green()),
        ..Default::default()
    });
    render(&mut renderer, warmup);
    renderer
}

fn transformed_quad_scene() -> Scene {
    let mut scene = Scene::default();
    let transform = scene.push_transform(quarter_turn());
    turned_bar(&mut scene, transform);
    scene
}

fn clipped_quad_scene() -> Scene {
    let mut scene = Scene::default();
    clipped_fill(&mut scene);
    scene
}

/// More entries than the tables first hold, so they grow, and the group-0 bind group is
/// rebuilt, mid-run; downlevel, the entries span several data-texture rows.
fn large_table_scene() -> Scene {
    let mut scene = Scene::default();
    let decoy = SceneTransform {
        transformation: TransformationMatrix {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [1000.0, 1000.0],
        },
        inverse: TransformationMatrix {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [-1000.0, -1000.0],
        },
    };
    for _ in 0..2000 {
        scene.push_transform(decoy);
    }
    let transform = scene.push_transform(quarter_turn());
    turned_bar(&mut scene, transform);
    for _ in 0..300 {
        scene.push_clip(SceneClip::default());
    }
    clipped_fill(&mut scene);
    scene
}

#[test]
fn transformed_quad_lands_where_its_transform_puts_it() {
    let mut renderer = WgpuHeadlessRenderer::new().expect("headless renderer");
    assert_turned_bar(&render(&mut renderer, transformed_quad_scene()));
}

#[test]
fn transformed_clip_removes_pixels_outside_it() {
    let mut renderer = WgpuHeadlessRenderer::new().expect("headless renderer");
    assert_clipped_fill(&render(&mut renderer, clipped_quad_scene()));
}

#[test]
fn scene_tables_grow_and_stay_addressable() {
    let mut renderer = WgpuHeadlessRenderer::new().expect("headless renderer");
    // A small frame first, so the large one replaces storage the renderer already bound.
    assert_turned_bar(&render(&mut renderer, transformed_quad_scene()));
    let image = render(&mut renderer, large_table_scene());
    assert_eq!(image.get_pixel(50, 15).0, GREEN);
    assert_eq!(image.get_pixel(15, 50).0, BLACK);
    assert_eq!(image.get_pixel(55, 65).0, GREEN);
    assert_eq!(image.get_pixel(90, 90).0, BLACK);
}

#[test]
fn downlevel_transformed_quad_lands_where_its_transform_puts_it() {
    let mut renderer = warmed_downlevel_renderer();
    assert_turned_bar(&render(&mut renderer, transformed_quad_scene()));
}

#[test]
fn downlevel_transformed_clip_removes_pixels_outside_it() {
    let mut renderer = warmed_downlevel_renderer();
    assert_clipped_fill(&render(&mut renderer, clipped_quad_scene()));
}

#[test]
fn downlevel_scene_tables_grow_and_stay_addressable() {
    let mut renderer = warmed_downlevel_renderer();
    assert_turned_bar(&render(&mut renderer, transformed_quad_scene()));
    let image = render(&mut renderer, large_table_scene());
    assert_eq!(image.get_pixel(50, 15).0, GREEN);
    assert_eq!(image.get_pixel(15, 50).0, BLACK);
    assert_eq!(image.get_pixel(55, 65).0, GREEN);
    assert_eq!(image.get_pixel(90, 90).0, BLACK);
}
