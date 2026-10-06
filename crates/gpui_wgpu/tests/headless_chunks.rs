//! Renders scene chunks, each a finished scene of its own drawn as one unit at a
//! placement, on both WGPU tiers, and checks their pixels land where the placement puts
//! them: a grid moved and then zoomed, as a cached view under a camera is, a chunk inside
//! a chunk, clipped, whose tables differ from the window's, and paths inside chunks.
//!
//! The chunks are built here as gpui builds them from a reused cached view: a finished
//! scene, and its placement in the scene that draws it.
#![cfg(feature = "test-support")]

use gpui::{
    Bounds, ContentMask, DevicePixels, PlacedChunk, PlatformHeadlessRenderer, Point, Primitive,
    Quad, ScaledPixels, Scene, SceneChunk, SceneTransform, Size, TransformationMatrix,
    solid_background,
};
use gpui_ce_wgpu::WgpuHeadlessRenderer;
use std::rc::Rc;

/// Cells per side of the grid, and their pitch and size.
const CELLS: usize = 20;
const PITCH: f32 = 20.;
const CELL: f32 = 16.;
/// Where the grid sits in its chunk's viewport.
const GRID_ORIGIN: f32 = 50.;

const WHITE: [u8; 3] = [255, 255, 255];
const BLACK: [u8; 3] = [0, 0, 0];

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

fn size(width: i32, height: i32) -> Size<DevicePixels> {
    Size {
        width: DevicePixels(width),
        height: DevicePixels(height),
    }
}

fn mask(bounds: Bounds<ScaledPixels>) -> ContentMask<ScaledPixels> {
    ContentMask {
        bounds,
        ..Default::default()
    }
}

fn rgb(color: u32) -> [u8; 3] {
    [(color >> 16) as u8, (color >> 8) as u8, color as u8]
}

fn quad(bounds: Bounds<ScaledPixels>, color: u32) -> Quad {
    Quad {
        bounds,
        content_mask: mask(bounds),
        background: gpui::rgb_to_hsla(gpui::rgb(color)).into(),
        ..Default::default()
    }
}

/// Scale by `scale`, then move by (`x`, `y`).
fn camera(scale: f32, x: f32, y: f32) -> TransformationMatrix {
    TransformationMatrix {
        rotation_scale: [[scale, 0.], [0., scale]],
        translation: [x, y],
    }
}

/// `scene`, finished, as a chunk covering `covers` of its viewport.
fn chunk(mut scene: Scene, covers: Bounds<ScaledPixels>) -> Rc<SceneChunk> {
    scene.finish();
    Rc::new(SceneChunk {
        scene,
        bounds: covers,
    })
}

/// Draws `chunk` in `scene` at `placement`, clipped to `clip`, in `scene`'s viewport.
fn place(
    scene: &mut Scene,
    chunk: &Rc<SceneChunk>,
    placement: TransformationMatrix,
    clip: Bounds<ScaledPixels>,
) {
    let corners = [
        (0., 0.),
        (chunk.bounds.size.width.0, 0.),
        (0., chunk.bounds.size.height.0),
        (chunk.bounds.size.width.0, chunk.bounds.size.height.0),
    ]
    .map(|(x, y)| {
        placement.apply(gpui::point(
            gpui::px(chunk.bounds.origin.x.0 + x),
            gpui::px(chunk.bounds.origin.y.0 + y),
        ))
    });
    let left = corners
        .iter()
        .map(|c| f32::from(c.x))
        .fold(f32::MAX, f32::min);
    let top = corners
        .iter()
        .map(|c| f32::from(c.y))
        .fold(f32::MAX, f32::min);
    let right = corners
        .iter()
        .map(|c| f32::from(c.x))
        .fold(f32::MIN, f32::max);
    let bottom = corners
        .iter()
        .map(|c| f32::from(c.y))
        .fold(f32::MIN, f32::max);
    scene.insert_primitive(Primitive::Chunk(PlacedChunk {
        order: 0,
        chunk: chunk.clone(),
        placement,
        bounds: bounds(left, top, right - left, bottom - top),
        content_mask: mask(clip),
    }));
}

fn assert_color(image: &image::RgbaImage, x: f32, y: f32, expected: [u8; 3], what: &str) {
    let actual = image.get_pixel(x as u32, y as u32).0;
    assert!(
        actual[..3]
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.abs_diff(expected) < 12),
        "{what}: at ({x}, {y}) expected {expected:?}, got {actual:?}"
    );
}

/// The colour of grid cell (`row`, `column`).
fn cell_color(row: usize, column: usize) -> u32 {
    [0xff0000, 0x00c000, 0x0000ff, 0xffc000][(row + 2 * column) % 4]
}

fn grid_chunk() -> Rc<SceneChunk> {
    let mut scene = Scene::default();
    for row in 0..CELLS {
        for column in 0..CELLS {
            scene.insert_primitive(quad(
                bounds(
                    GRID_ORIGIN + column as f32 * PITCH,
                    GRID_ORIGIN + row as f32 * PITCH,
                    CELL,
                    CELL,
                ),
                cell_color(row, column),
            ));
        }
    }
    let extent = (CELLS - 1) as f32 * PITCH + CELL;
    chunk(scene, bounds(GRID_ORIGIN, GRID_ORIGIN, extent, extent))
}

const GRID_TARGET: (i32, i32) = (800, 720);

/// A white window with the grid drawn in it, as one chunk, under `camera`.
fn grid_window(grid: &Rc<SceneChunk>, camera: TransformationMatrix) -> Scene {
    let viewport = bounds(0., 0., GRID_TARGET.0 as f32, GRID_TARGET.1 as f32);
    let mut scene = Scene::default();
    scene.insert_primitive(quad(viewport, 0xffffff));
    place(&mut scene, grid, camera, viewport);
    scene.finish();
    scene
}

/// Mirrors `gpui_platform`'s `chunks_render`: the grid is drawn moved, then moved and
/// zoomed, and each cell lands where the camera puts it.
fn grid_moved_then_zoomed(renderer: &mut WgpuHeadlessRenderer) {
    let grid = grid_chunk();
    let target = size(GRID_TARGET.0, GRID_TARGET.1);
    renderer
        .render_scene_to_image(&grid_window(&grid, camera(1., 30., 20.)), target)
        .expect("moved grid");
    let (zoom, camera_x, camera_y) = (1.5, 120., 40.);
    let window = grid_window(&grid, camera(zoom, camera_x, camera_y));
    assert!(
        window.quads.len() < CELLS * CELLS,
        "the grid's quads must be drawn from its chunk, not the window's scene: {} quads",
        window.quads.len()
    );
    let image = renderer
        .render_scene_to_image(&window, target)
        .expect("zoomed grid");

    let on_screen = |page: f32, camera: f32| page * zoom + camera;
    for (row, column) in [(0, 0), (0, 19), (7, 3), (19, 0), (19, 19), (10, 11)] {
        let center = GRID_ORIGIN + CELL / 2.;
        assert_color(
            &image,
            on_screen(center + column as f32 * PITCH, camera_x),
            on_screen(center + row as f32 * PITCH, camera_y),
            rgb(cell_color(row, column)),
            &format!("cell ({row}, {column})"),
        );
    }
    let gap = on_screen(GRID_ORIGIN + CELL + 2., camera_x);
    assert_color(
        &image,
        gap,
        on_screen(GRID_ORIGIN + 8., camera_y),
        WHITE,
        "between cells",
    );
    assert_color(
        &image,
        GRID_ORIGIN + 5.,
        GRID_ORIGIN + 5.,
        WHITE,
        "where the grid first was",
    );
}

/// A chunk drawn in a chunk, clipped, whose transform table differs from the window's:
/// each must be drawn with its own tables, placements composed, and clips intersected.
fn nested_chunk(renderer: &mut WgpuHeadlessRenderer) {
    let viewport = bounds(0., 0., 300., 200.);
    // A 4 by 4 grid of 10px cells, at a pitch of 12.
    let mut inner = Scene::default();
    for row in 0..4 {
        for column in 0..4 {
            inner.insert_primitive(quad(
                bounds(column as f32 * 12., row as f32 * 12., 10., 10.),
                cell_color(row, column),
            ));
        }
    }
    let inner = chunk(inner, bounds(0., 0., 46., 46.));

    // The grid moved by (20, 10) and clipped to its first two columns, x 20..44, and a
    // quad its transform table moves down by 70.
    let mut outer = Scene::default();
    place(
        &mut outer,
        &inner,
        camera(1., 20., 10.),
        bounds(20., 0., 24., 100.),
    );
    let down = TransformationMatrix {
        rotation_scale: TransformationMatrix::unit().rotation_scale,
        translation: [0., 70.],
    };
    let transform = outer.push_transform(SceneTransform {
        transformation: down,
        inverse: down.inverse().unwrap(),
    });
    outer.insert_primitive(Quad {
        transform,
        content_mask: mask(bounds(0., 0., 100., 100.)),
        ..quad(bounds(0., 0., 10., 10.), 0x0000ff)
    });
    let outer = chunk(outer, bounds(0., 0., 66., 80.));

    // The window's own transform table moves its quad right by 200.
    let mut window = Scene::default();
    let right = TransformationMatrix {
        rotation_scale: TransformationMatrix::unit().rotation_scale,
        translation: [200., 0.],
    };
    let transform = window.push_transform(SceneTransform {
        transformation: right,
        inverse: right.inverse().unwrap(),
    });
    window.insert_primitive(Quad {
        transform,
        content_mask: mask(viewport),
        ..quad(bounds(0., 0., 10., 10.), 0x00ff00)
    });
    // Twice the size, moved by (10, 10).
    place(&mut window, &outer, camera(2., 10., 10.), viewport);
    window.finish();

    let image = renderer
        .render_scene_to_image(&window, size(300, 200))
        .expect("nested chunk");
    // An inner cell's centre, (c * 12 + 5, r * 12 + 5), is moved by (20, 10), then
    // doubled and moved by (10, 10).
    let on_screen = |row: usize, column: usize| {
        (
            (column as f32 * 12. + 5. + 20.) * 2. + 10.,
            (row as f32 * 12. + 5. + 10.) * 2. + 10.,
        )
    };
    for (row, column) in [(0, 0), (3, 1), (2, 0)] {
        let (x, y) = on_screen(row, column);
        assert_color(
            &image,
            x,
            y,
            rgb(cell_color(row, column)),
            &format!("inner cell ({row}, {column})"),
        );
    }
    for (row, column) in [(0, 2), (3, 3)] {
        let (x, y) = on_screen(row, column);
        assert_color(
            &image,
            x,
            y,
            BLACK,
            &format!("inner cell ({row}, {column}), clipped"),
        );
    }
    assert_color(
        &image,
        20.,
        160.,
        rgb(0x0000ff),
        "the outer chunk's quad, by its own transform table",
    );
    assert_color(
        &image,
        205.,
        5.,
        rgb(0x00ff00),
        "the window's quad, by the window's transform table",
    );
}

fn triangle(points: [(f32, f32); 3], color: u32) -> gpui::Path<ScaledPixels> {
    let mut builder = gpui::PathBuilder::fill();
    builder.move_to(gpui::point(gpui::px(points[0].0), gpui::px(points[0].1)));
    builder.line_to(gpui::point(gpui::px(points[1].0), gpui::px(points[1].1)));
    builder.line_to(gpui::point(gpui::px(points[2].0), gpui::px(points[2].1)));
    builder.close();
    let mut path = builder.build().expect("triangle").scale(1.0);
    path.content_mask = mask(bounds(0., 0., 100., 100.));
    path.color = solid_background(gpui::rgb_to_hsla(gpui::rgb(color)));
    path
}

/// Two disjoint triangles and a quad, in a chunk placed by `placement`, and a point in
/// the chunk's viewport where each is, or nothing is.
fn paths_in_chunk(
    renderer: &mut WgpuHeadlessRenderer,
    placement: TransformationMatrix,
    what: &str,
) {
    let mut scene = Scene::default();
    scene.insert_primitive(triangle([(20., 20.), (80., 20.), (50., 80.)], 0xffffff));
    scene.insert_primitive(triangle([(85., 20.), (99., 20.), (92., 60.)], 0xffff00));
    scene.insert_primitive(quad(bounds(60., 85., 30., 10.), 0xff0000));
    let paths = chunk(scene, bounds(0., 0., 100., 100.));
    let viewport = bounds(0., 0., 300., 220.);
    let mut window = Scene::default();
    place(&mut window, &paths, placement, viewport);
    window.finish();

    let image = renderer
        .render_scene_to_image(&window, size(300, 220))
        .expect("paths in a chunk");
    let at = |x: f32, y: f32| {
        let point = placement.apply(gpui::point(gpui::px(x), gpui::px(y)));
        (f32::from(point.x), f32::from(point.y))
    };
    for (x, y, color, label) in [
        (50., 40., WHITE, "the first triangle"),
        (92., 33., rgb(0xffff00), "the second triangle"),
        (75., 90., rgb(0xff0000), "the quad"),
        (25., 70., BLACK, "beside the first triangle"),
        (82., 60., BLACK, "between the triangles"),
    ] {
        let (px, py) = at(x, y);
        assert_color(&image, px, py, color, &format!("{what}: {label}"));
    }
}

fn scaled_paths_in_chunk(renderer: &mut WgpuHeadlessRenderer) {
    paths_in_chunk(renderer, camera(2., 100., 10.), "scaled");
}

fn turned_paths_in_chunk(renderer: &mut WgpuHeadlessRenderer) {
    // A quarter turn: (x, y) lands at (250 - y, x + 10).
    paths_in_chunk(
        renderer,
        TransformationMatrix {
            rotation_scale: [[0., -1.], [1., 0.]],
            translation: [250., 10.],
        },
        "turned",
    );
}

/// A downlevel renderer that has already drawn an unrelated frame, so a data texture
/// that lagged a frame behind its upload would show that frame instead.
fn downlevel_renderer() -> WgpuHeadlessRenderer {
    let mut renderer = WgpuHeadlessRenderer::new_downlevel().expect("downlevel renderer");
    let mut warmup = Scene::default();
    warmup.insert_primitive(quad(bounds(0., 0., 1., 1.), 0x00ff00));
    warmup.finish();
    renderer
        .render_scene_to_image(&warmup, size(10, 10))
        .expect("warm-up frame");
    renderer
}

fn renderer() -> WgpuHeadlessRenderer {
    WgpuHeadlessRenderer::new().expect("headless renderer")
}

#[test]
fn chunk_grid_lands_where_the_camera_puts_it() {
    grid_moved_then_zoomed(&mut renderer());
}

#[test]
fn nested_chunk_is_placed_clipped_and_drawn_with_its_own_tables() {
    nested_chunk(&mut renderer());
}

#[test]
fn paths_in_a_chunk_are_placed() {
    let mut renderer = renderer();
    scaled_paths_in_chunk(&mut renderer);
    turned_paths_in_chunk(&mut renderer);
}

#[test]
fn downlevel_chunk_grid_lands_where_the_camera_puts_it() {
    grid_moved_then_zoomed(&mut downlevel_renderer());
}

#[test]
fn downlevel_nested_chunk_is_placed_clipped_and_drawn_with_its_own_tables() {
    nested_chunk(&mut downlevel_renderer());
}

#[test]
fn downlevel_paths_in_a_chunk_are_placed() {
    let mut renderer = downlevel_renderer();
    scaled_paths_in_chunk(&mut renderer);
    turned_paths_in_chunk(&mut renderer);
}
