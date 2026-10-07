//! Renders meshes through the platform renderer and checks their pixels:
//!
//! - a star filled as a mesh matches the same star filled as a path, within
//!   what their different antialiasing allows;
//! - a square zoomed, and one turned and zoomed, keep edges one device pixel
//!   wide, where a fringe that scaled with them would be several;
//! - two overlapping meshes in a group with opacity fade as one picture;
//! - a strip filled with a shader program reading its stroke coordinates
//!   matches the program evaluated on the CPU;
//! - a mesh in a cached view reused under a camera, drawn as a chunk, lands
//!   where the camera puts it.
//!
//! It runs on Metal, and with `macos-wgpu` on wgpu.
//!
//! Runs only with `GPUI_RUN_RENDERING_TESTS` set, in offscreen windows;
//! `GPUI_RENDERING_TEST_OUTPUT=<path.png>` saves the first image.

#[cfg(target_os = "macos")]
use gpui::{
    AnyWindowHandle, AppContext as _, Bounds, Context, Entity, IntoElement, Mesh,
    ParentElement as _, PathBuilder, Pixels, Point, Position, Render, StyleRefinement, Styled as _,
    VisualTestAppContext, Window, canvas, div, kurbo, peniko, point, px, rgb,
    shader::{self, Fragment, Paint, vec2f},
};
#[cfg(target_os = "macos")]
use std::sync::Arc;

/// The star's centre in the mesh, its outer and inner radii, and how far to
/// the right the path is drawn.
const STAR: (f32, f32) = (130., 130.);
const STAR_RADII: (f32, f32) = (100., 42.);
const PATH_SHIFT: f32 = 240.;
/// A square of side [`SQUARE`] zoomed by [`ZOOM`] at [`ZOOMED`], and one
/// turned by [`TURN`] and zoomed at [`TURNED`].
const SQUARE: f32 = 20.;
const ZOOM: f32 = 6.;
const ZOOMED: (f32, f32) = (520., 30.);
const TURN: f64 = 0.3;
const TURNED: (f32, f32) = (820., 40.);
/// The strip: from `STRIP.0` to `STRIP.1` along y = `STRIP.2`, half
/// [`STRIP_WIDTH`] either side.
const STRIP: (f32, f32, f32) = (40., 440., 420.);
const STRIP_WIDTH: f32 = 60.;
/// The group of two overlapping squares, faded by half.
const GROUP: (f32, f32) = (560., 330.);
const GROUP_SQUARE: f32 = 120.;
const GROUP_OFFSET: f32 = 60.;

#[cfg(target_os = "macos")]
fn star() -> Vec<Point<Pixels>> {
    (0..10)
        .map(|index| {
            let angle = index as f32 * std::f32::consts::PI / 5. - std::f32::consts::FRAC_PI_2;
            let radius = if index % 2 == 0 {
                STAR_RADII.0
            } else {
                STAR_RADII.1
            };
            point(
                px(STAR.0 + radius * angle.cos()),
                px(STAR.1 + radius * angle.sin()),
            )
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn square(side: f32) -> Vec<Point<Pixels>> {
    [(0., 0.), (side, 0.), (side, side), (0., side)]
        .map(|(x, y)| point(px(x), px(y)))
        .to_vec()
}

/// Colours the strip by where a fragment is along it and across it.
#[cfg(target_os = "macos")]
fn stroke_paint() -> Paint {
    shader::paint(|px| {
        let length = STRIP.1 - STRIP.0;
        shader::rgba(
            px.along() / length,
            px.across() * 0.5 + 0.5,
            px.position().y() / STRIP_WIDTH,
            1.0,
        )
    })
    .fallback(rgb(0x00ff00))
}

#[cfg(target_os = "macos")]
struct Meshes {
    star: Arc<Mesh>,
    square: Arc<Mesh>,
    group_square: Arc<Mesh>,
    strip: Arc<Mesh>,
}

#[cfg(target_os = "macos")]
impl Meshes {
    fn new() -> Self {
        let half = STRIP_WIDTH / 2.;
        let ribs: Vec<_> = (0..=10)
            .map(|step| {
                let x = px(STRIP.0 + (STRIP.1 - STRIP.0) * step as f32 / 10.);
                (point(x, px(STRIP.2 - half)), point(x, px(STRIP.2 + half)))
            })
            .collect();
        Self {
            star: Arc::new(Mesh::from_polygon(&star(), peniko::Fill::NonZero)),
            square: Arc::new(Mesh::from_polygon(&square(SQUARE), peniko::Fill::NonZero)),
            group_square: Arc::new(Mesh::from_polygon(
                &square(GROUP_SQUARE),
                peniko::Fill::NonZero,
            )),
            strip: Arc::new(Mesh::strip(&ribs)),
        }
    }
}

#[cfg(target_os = "macos")]
struct MeshFixture {
    meshes: Arc<Meshes>,
}

#[cfg(target_os = "macos")]
impl Render for MeshFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let meshes = self.meshes.clone();
        let group_meshes = self.meshes.clone();
        div()
            .size_full()
            .relative()
            .bg(rgb(0xffffff))
            .child(
                canvas(
                    |_, _, _| (),
                    move |_, _, window, _| {
                        let black = rgb(0x000000);
                        window.paint_mesh(&meshes.star, Point::default(), black);
                        let mut path = PathBuilder::fill();
                        let shifted =
                            |point: Point<Pixels>| point + gpui::point(px(PATH_SHIFT), px(0.));
                        let star = star();
                        path.move_to(shifted(star[0]));
                        for point in &star[1..] {
                            path.line_to(shifted(*point));
                        }
                        path.close();
                        window.paint_path(path.build().unwrap(), black);

                        let zoomed =
                            kurbo::Affine::translate((f64::from(ZOOMED.0), f64::from(ZOOMED.1)))
                                * kurbo::Affine::scale(f64::from(ZOOM));
                        window.with_transform(zoomed, |window| {
                            window.paint_mesh(&meshes.square, Point::default(), black)
                        });
                        let turned =
                            kurbo::Affine::translate((f64::from(TURNED.0), f64::from(TURNED.1)))
                                * kurbo::Affine::rotate(TURN)
                                * kurbo::Affine::scale(f64::from(ZOOM));
                        window.with_transform(turned, |window| {
                            window.paint_mesh(&meshes.square, Point::default(), black)
                        });

                        let half = STRIP_WIDTH / 2.;
                        let paint = window.program(
                            &stroke_paint(),
                            Bounds::new(
                                point(px(STRIP.0), px(STRIP.2 - half)),
                                gpui::size(px(STRIP.1 - STRIP.0), px(STRIP_WIDTH)),
                            ),
                        );
                        window.paint_mesh(&meshes.strip, Point::default(), paint);
                    },
                )
                .absolute()
                .size_full(),
            )
            .child(
                div()
                    .absolute()
                    .left(px(GROUP.0))
                    .top(px(GROUP.1))
                    .size(px(GROUP_SQUARE + GROUP_OFFSET))
                    .group_opacity(0.5)
                    .child(
                        canvas(
                            |_, _, _| (),
                            move |bounds, _, window, _| {
                                let red = rgb(0xff0000);
                                window.paint_mesh(&group_meshes.group_square, bounds.origin, red);
                                window.paint_mesh(
                                    &group_meshes.group_square,
                                    bounds.origin + point(px(GROUP_OFFSET), px(GROUP_OFFSET)),
                                    red,
                                );
                            },
                        )
                        .size_full(),
                    ),
            )
    }
}

/// A grid of cells enough to be drawn as a chunk, with a blue mesh square
/// at its top left, and where the camera moves it to.
const CELLS: usize = 20;
const PITCH: f32 = 20.;
const CHUNK_SQUARE: f32 = 60.;
const CAMERA: (f32, f32) = (700., 500.);

#[cfg(target_os = "macos")]
struct Grid {
    square: Arc<Mesh>,
}

#[cfg(target_os = "macos")]
impl Render for Grid {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let square = self.square.clone();
        div()
            .size_full()
            .relative()
            .children((0..CELLS * CELLS).map(|index| {
                let (row, column) = (index / CELLS, index % CELLS);
                div()
                    .absolute()
                    .left(px(column as f32 * PITCH + PITCH / 2.))
                    .top(px(row as f32 * PITCH + PITCH / 2.))
                    .size(px(PITCH / 4.))
                    .bg(rgb(0x808080))
            }))
            .child(
                canvas(
                    |_, _, _| (),
                    move |bounds, _, window, _| {
                        window.paint_mesh(&square, bounds.origin, rgb(0x0000ff));
                    },
                )
                .absolute()
                .size_full(),
            )
    }
}

#[cfg(target_os = "macos")]
struct ChunkFixture {
    grid: Entity<Grid>,
    camera: std::rc::Rc<std::cell::Cell<(f32, f32)>>,
}

#[cfg(target_os = "macos")]
impl Render for ChunkFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let (x, y) = self.camera.get();
        let mut style = StyleRefinement::default();
        style.position = Some(Position::Absolute);
        style.size.width = Some(px(CELLS as f32 * PITCH).into());
        style.size.height = Some(px(CELLS as f32 * PITCH).into());
        div().size_full().relative().bg(rgb(0xffffff)).child(
            div()
                .absolute()
                .size_0()
                .transform(kurbo::Affine::translate((f64::from(x), f64::from(y))))
                .child(self.grid.clone().cached(style)),
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

/// Device pixels per logical pixel in `image`, of a 1280-wide window.
#[cfg(target_os = "macos")]
fn scale_of(image: &image::RgbaImage) -> f32 {
    image.width() as f32 / 1280.
}

/// How much of a device pixel is covered by black on white.
#[cfg(target_os = "macos")]
fn coverage(image: &image::RgbaImage, x: u32, y: u32) -> f32 {
    1. - f32::from(image.get_pixel(x, y).0[1]) / 255.
}

#[cfg(target_os = "macos")]
fn render() {
    let mut cx = VisualTestAppContext::new(gpui_ce_platform::current_platform(false));
    // SAFETY: the test is single-threaded, and the renderer reads this when
    // the window opens.
    unsafe { std::env::set_var("GPUI_LINK_PROGRAMS_SYNCHRONOUSLY", "1") };
    let meshes = Arc::new(Meshes::new());
    let window: AnyWindowHandle = cx
        .open_offscreen_window_default(move |_, cx| cx.new(|_| MeshFixture { meshes }))
        .expect("failed to create the offscreen window")
        .into();
    cx.run_until_parked();
    let image = cx
        .capture_screenshot(window)
        .expect("failed to capture the rendered window");
    if let Some(output) = std::env::var_os("GPUI_RENDERING_TEST_OUTPUT") {
        image.save(output).expect("failed to save the image");
    }
    let mut failures = Vec::new();
    star_matches_path(&image, &mut failures);
    edges_stay_a_pixel_wide(&image, &mut failures);
    group_fades_as_one(&image, &mut failures);
    program_reads_stroke(&image, &mut failures);
    mesh_in_a_chunk(&mut cx, &mut failures);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    println!("meshes_render: ok");
    std::mem::forget(cx);
}

/// The star as a mesh and as a path cover the same pixels: alike inside and
/// outside, and at their edges within what 4× multisampling and an analytic
/// fringe differ by.
#[cfg(target_os = "macos")]
fn star_matches_path(image: &image::RgbaImage, failures: &mut Vec<String>) {
    let scale = scale_of(image);
    let shift = (PATH_SHIFT * scale).round() as u32;
    let (left, top) = (
        ((STAR.0 - STAR_RADII.0 - 4.) * scale) as u32,
        ((STAR.1 - STAR_RADII.0 - 4.) * scale) as u32,
    );
    let side = ((STAR_RADII.0 * 2. + 8.) * scale) as u32;
    let (mut mesh_total, mut path_total) = (0f32, 0f32);
    let (mut edge_pixels, mut edge_difference, mut worst) = (0, 0f32, 0f32);
    let mut worst_at = (0, 0);
    for y in top..top + side {
        for x in left..left + side {
            let mesh = coverage(image, x, y);
            let path = coverage(image, x + shift, y);
            mesh_total += mesh;
            path_total += path;
            let difference = (mesh - path).abs();
            if difference > worst {
                worst = difference;
                worst_at = (x, y);
            }
            if (0.02..0.98).contains(&mesh) || (0.02..0.98).contains(&path) {
                edge_pixels += 1;
                edge_difference += difference;
            } else if difference > 0.02 {
                failures.push(format!(
                    "star at ({x}, {y}): mesh {mesh:.3}, path {path:.3}"
                ));
            }
        }
    }
    let mean = edge_difference / edge_pixels.max(1) as f32;
    println!(
        "meshes_render: star area mesh {mesh_total:.1} / path {path_total:.1} px², \
         {edge_pixels} edge pixels, mean difference {mean:.3}, worst {worst:.3} at \
         {:?} logical",
        (worst_at.0 as f32 / scale, worst_at.1 as f32 / scale)
    );
    if (mesh_total - path_total).abs() > path_total * 0.01 {
        failures.push(format!(
            "star area: mesh {mesh_total:.1}, path {path_total:.1}"
        ));
    }
    // Multisampling quantizes coverage to quarters, so edges differ by
    // about a twelfth on average; the most is at the star's sharp tips,
    // where a mitred fringe spreads its ramp along the tip.
    if mean > 0.1 || worst > 0.5 {
        failures.push(format!(
            "star edges: mean difference {mean:.3}, worst {worst:.3}"
        ));
    }
}

/// Across each square's edges, at most the pixels a one-pixel ramp can
/// partly cover are partly covered, though the squares are zoomed six times.
#[cfg(target_os = "macos")]
fn edges_stay_a_pixel_wide(image: &image::RgbaImage, failures: &mut Vec<String>) {
    let scale = scale_of(image);
    // Partly covered pixels along `row` from `from` to `to`, in logical
    // pixels, the coverage summed over them, and where they end, in device
    // pixels.
    let partial = |row: u32, from: f32, to: f32| -> (usize, f32, f32) {
        let (from, to) = ((from * scale) as u32, (to * scale) as u32);
        let mut count = 0;
        let mut sum = 0.;
        for x in from..to {
            let covered = coverage(image, x, row);
            if (0.03..0.97).contains(&covered) {
                count += 1;
            }
            sum += covered;
        }
        (count, sum, to as f32)
    };
    let side = SQUARE * ZOOM;
    // The zoomed square's left and right edges, along a row through its
    // middle.
    let row = ((ZOOMED.1 + side / 2.) * scale) as u32;
    for (what, from, to) in [
        ("zoomed left edge", ZOOMED.0 - 10., ZOOMED.0 + 10.),
        (
            "zoomed right edge",
            ZOOMED.0 + side - 10.,
            ZOOMED.0 + side + 10.,
        ),
    ] {
        let (count, _, _) = partial(row, from, to);
        println!("meshes_render: {what}: {count} partly covered pixels");
        if count > 1 {
            failures.push(format!("{what}: {count} partly covered pixels"));
        }
    }
    // The turned square: along a row through its middle, its left edge
    // crosses the row at a slant of TURN, so a pixel-wide ramp spans
    // 1 / cos(TURN) pixels of it.
    let (sin, cos) = (TURN as f32).sin_cos();
    let centre = (
        TURNED.0 + side / 2. * cos - side / 2. * sin,
        TURNED.1 + side / 2. * sin + side / 2. * cos,
    );
    let row = (centre.1 * scale) as u32;
    // Where the turned left edge (x = 0 in the square) crosses the row's
    // centre.
    let row_centre = (row as f32 + 0.5) / scale;
    let edge_x = TURNED.0 - (row_centre - TURNED.1) * sin / cos;
    let (count, sum, end) = partial(row, edge_x - 10., edge_x + 10.);
    // Coverage summed across an antialiased edge is the distance from the
    // edge to the end of the row.
    let offset = end - edge_x * scale - sum;
    println!(
        "meshes_render: turned left edge: {count} partly covered pixels, \
         its coverage puts it {offset:.2} device pixels from the edge"
    );
    if count > 2 {
        failures.push(format!("turned left edge: {count} partly covered pixels"));
    }
    if offset.abs() > 0.25 {
        failures.push(format!(
            "turned left edge: its coverage puts it {offset:.2} device pixels off"
        ));
    }
}

/// Two overlapping red squares faded by half as a group are one even pink.
#[cfg(target_os = "macos")]
fn group_fades_as_one(image: &image::RgbaImage, failures: &mut Vec<String>) {
    let scale = scale_of(image);
    let at = |x: f32, y: f32| {
        image
            .get_pixel(
                ((GROUP.0 + x) * scale) as u32,
                ((GROUP.1 + y) * scale) as u32,
            )
            .0
    };
    let single = at(GROUP_OFFSET / 2., GROUP_OFFSET / 2.);
    let overlap = at(GROUP_OFFSET * 1.5, GROUP_OFFSET * 1.5);
    let expected = [255, 128, 128];
    for (what, pixel) in [("single", single), ("overlap", overlap)] {
        if pixel[..3]
            .iter()
            .zip(expected)
            .any(|(actual, expected)| actual.abs_diff(expected) > 3)
        {
            failures.push(format!(
                "group {what}: expected {expected:?}, got {pixel:?}"
            ));
        }
    }
}

/// The strip's program paint, at pixels well inside it, matches the paint
/// evaluated on the CPU at their stroke coordinates.
#[cfg(target_os = "macos")]
fn program_reads_stroke(image: &image::RgbaImage, failures: &mut Vec<String>) {
    let scale = scale_of(image);
    let paint = stroke_paint();
    let half = STRIP_WIDTH / 2.;
    let length = STRIP.1 - STRIP.0;
    let mut checked = 0;
    for u in [0.05, 0.25, 0.5, 0.75, 0.95] {
        for v in [-0.8, -0.4, 0., 0.4, 0.8] {
            let (x, y) = (
                ((STRIP.0 + u * length) * scale) as u32,
                ((STRIP.2 + v * half) * scale) as u32,
            );
            let centre = ((x as f32 + 0.5) / scale, (y as f32 + 0.5) / scale);
            let position = (centre.0 - STRIP.0, centre.1 - (STRIP.2 - half));
            let fragment = Fragment {
                uv: vec2f(position.0 / length, position.1 / STRIP_WIDTH),
                position: vec2f(position.0, position.1),
                size: vec2f(length, STRIP_WIDTH),
                origin: vec2f(0., 0.),
                scale,
                stroke: vec2f(centre.0 - STRIP.0, (centre.1 - STRIP.2) / half),
            };
            let rgba = paint.evaluate(fragment).expect("the paint runs on the CPU");
            let expected = [rgba.x, rgba.y, rgba.z].map(|channel| (channel * 255.).round() as u8);
            let actual = image.get_pixel(x, y).0;
            checked += 1;
            if actual[..3]
                .iter()
                .zip(expected)
                .any(|(actual, expected)| actual.abs_diff(expected) > 3)
            {
                failures.push(format!(
                    "strip at along {u}, across {v}: expected {expected:?}, got {actual:?}"
                ));
            }
        }
    }
    println!("meshes_render: checked the strip's program at {checked} pixels");
}

/// A mesh in a cached view, reused under a camera as a chunk, is drawn
/// where the camera puts it, with sharp edges.
#[cfg(target_os = "macos")]
fn mesh_in_a_chunk(cx: &mut VisualTestAppContext, failures: &mut Vec<String>) {
    let camera = std::rc::Rc::new(std::cell::Cell::new((0., 0.)));
    let square = Arc::new(Mesh::from_polygon(
        &square(CHUNK_SQUARE),
        peniko::Fill::NonZero,
    ));
    let window: AnyWindowHandle = cx
        .open_offscreen_window_default({
            let camera = camera.clone();
            move |_, cx| {
                cx.new(|cx| ChunkFixture {
                    grid: cx.new(|_| Grid { square }),
                    camera,
                })
            }
        })
        .expect("failed to create the offscreen window")
        .into();
    cx.run_until_parked();
    for position in [(CAMERA.0 / 2., CAMERA.1 / 2.), CAMERA] {
        camera.set(position);
        cx.update_window(window, |root, window, cx| {
            root.downcast::<ChunkFixture>()
                .unwrap()
                .update(cx, |_, cx| cx.notify());
            window.draw(cx).clear(cx);
        })
        .expect("failed to draw the window");
    }
    let image = cx
        .update_window(window, |_, window, _| window.render_to_image())
        .expect("failed to capture the rendered window")
        .expect("failed to capture the rendered window");
    let scale = scale_of(&image);
    let pixel = |x: f32, y: f32| image.get_pixel((x * scale) as u32, (y * scale) as u32).0;
    for (u, v) in [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (0.1, 0.9)] {
        let inside = pixel(CAMERA.0 + u * CHUNK_SQUARE, CAMERA.1 + v * CHUNK_SQUARE);
        if inside[..3] != [0, 0, 255] {
            failures.push(format!("chunk mesh at ({u}, {v}): got {inside:?}"));
        }
    }
    for (x, y) in [
        (CAMERA.0 - 2., CAMERA.1 + 30.),
        (CAMERA.0 + CHUNK_SQUARE + 2., CAMERA.1 + 30.),
        (CAMERA.0 / 2. + 30., CAMERA.1 / 2. + 30.),
    ] {
        let outside = pixel(x, y);
        if outside[2] == 255 && outside[0] == 0 {
            failures.push(format!("chunk mesh drawn outside it at ({x}, {y})"));
        }
    }
    // Its left edge, across a row, is one pixel wide.
    let row = ((CAMERA.1 + CHUNK_SQUARE / 2.) * scale) as u32;
    let partial = ((CAMERA.0 - 4.) * scale) as u32..((CAMERA.0 + 4.) * scale) as u32;
    let count = partial
        .filter(|&x| {
            let red = image.get_pixel(x, row).0[0];
            (8..247).contains(&red)
        })
        .count();
    if count > 1 {
        failures.push(format!("chunk mesh edge: {count} partly covered pixels"));
    }
}
