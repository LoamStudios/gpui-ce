//! Renders a cached view large enough to be drawn as a chunk under a camera
//! that turns it, and checks that while the camera turns the chunk is placed
//! on the GPU, its text resampled and soft, and once the camera settles the
//! chunk is prepared at its angle: its text is rasterized turned, as sharp
//! as upright text, its quads land where the camera puts them, and a clip
//! inside it still clips along its turned edges.
//!
//! Runs only with `GPUI_RUN_RENDERING_TESTS` set, in an offscreen window;
//! `GPUI_RENDERING_TEST_OUTPUT=<path.png>` saves the image.

#[cfg(target_os = "macos")]
use gpui::{
    AppContext as _, Context, Entity, IntoElement, ParentElement as _, Position, Render,
    StyleRefinement, Styled as _, VisualTestAppContext, Window, div, kurbo, px, rgb,
};
#[cfg(target_os = "macos")]
use std::{borrow::Cow, cell::Cell, rc::Rc};

#[cfg(target_os = "macos")]
#[allow(dead_code)]
#[path = "../../gpui_ce_parley/src/font_fixtures.rs"]
mod font_fixtures;

/// Cells per side of the grid, and their pitch and size.
#[cfg(target_os = "macos")]
const CELLS: usize = 16;
#[cfg(target_os = "macos")]
const PITCH: f32 = 12.;
#[cfg(target_os = "macos")]
const CELL: f32 = 10.;
/// Where the lines of text sit, in the grid's coordinates, and how many.
#[cfg(target_os = "macos")]
const TEXT_ORIGIN: (f32, f32) = (0., 200.);
#[cfg(target_os = "macos")]
const TEXT_LINES: usize = 3;
/// A box that clips a wide red bar inside it to its own bounds, in the
/// grid's coordinates.
#[cfg(target_os = "macos")]
const CLIP: (f32, f32, f32, f32) = (220., 40., 60., 24.);
/// The grid's size.
#[cfg(target_os = "macos")]
const GRID: (f32, f32) = (320., 290.);
/// The angle the camera turns the page to, about the grid's center.
#[cfg(target_os = "macos")]
const DEGREES: f64 = 30.;

/// The colour of cell (`row`, `column`).
#[cfg(target_os = "macos")]
fn cell_color(row: usize, column: usize) -> u32 {
    [0xff0000, 0x00c000, 0x0000ff, 0xffc000][(row + 2 * column) % 4]
}

#[cfg(target_os = "macos")]
struct Grid;

#[cfg(target_os = "macos")]
impl Render for Grid {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .relative()
            .font_family("IBM Plex Sans")
            .text_size(px(16.))
            .text_color(rgb(0x000000))
            .children((0..CELLS * CELLS).map(|index| {
                let (row, column) = (index / CELLS, index % CELLS);
                div()
                    .absolute()
                    .left(px(column as f32 * PITCH))
                    .top(px(row as f32 * PITCH))
                    .size(px(CELL))
                    .bg(rgb(cell_color(row, column)))
            }))
            .children((0..TEXT_LINES).map(|line| {
                div()
                    .absolute()
                    .left(px(TEXT_ORIGIN.0))
                    .top(px(TEXT_ORIGIN.1 + 26. * line as f32))
                    .child("Hamburgefontsiv quickly")
            }))
            .child(
                div()
                    .absolute()
                    .left(px(CLIP.0))
                    .top(px(CLIP.1))
                    .w(px(CLIP.2))
                    .h(px(CLIP.3))
                    .overflow_hidden()
                    .child(div().w(px(200.)).h(px(CLIP.3)).bg(rgb(0xff0000))),
            )
    }
}

#[cfg(target_os = "macos")]
struct TurnedChunksFixture {
    grid: Entity<Grid>,
    camera: Rc<Cell<kurbo::Affine>>,
}

/// Where the grid sits on the page.
#[cfg(target_os = "macos")]
const GRID_ORIGIN: (f32, f32) = (400., 200.);

#[cfg(target_os = "macos")]
impl Render for TurnedChunksFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let mut style = StyleRefinement::default();
        style.position = Some(Position::Absolute);
        style.inset.left = Some(px(GRID_ORIGIN.0).into());
        style.inset.top = Some(px(GRID_ORIGIN.1).into());
        style.size.width = Some(px(GRID.0).into());
        style.size.height = Some(px(GRID.1).into());
        div().size_full().relative().bg(rgb(0xffffff)).child(
            // A page at the window's origin, so its transform maps page
            // points directly.
            div()
                .absolute()
                .size_0()
                .transform(self.camera.get())
                .child(self.grid.clone().cached(style)),
        )
    }
}

/// The camera turned `degrees` about the grid's center.
#[cfg(target_os = "macos")]
fn camera(degrees: f64) -> kurbo::Affine {
    let center = kurbo::Point::new(
        f64::from(GRID_ORIGIN.0 + GRID.0 / 2.),
        f64::from(GRID_ORIGIN.1 + GRID.1 / 2.),
    );
    kurbo::Affine::rotate_about(degrees.to_radians(), center)
}

fn main() {
    if std::env::var_os("GPUI_RUN_RENDERING_TESTS").is_none() {
        return;
    }
    #[cfg(target_os = "macos")]
    render();
}

/// The mean of the `count` largest forward-difference steps of `image`'s
/// green channel among the points of the grid's coordinates `region` covers
/// under `camera`: how steep its steepest edges are.
#[cfg(target_os = "macos")]
fn steepest(
    image: &image::RgbaImage,
    camera: kurbo::Affine,
    region: (f32, f32, f32, f32),
    count: usize,
) -> f32 {
    let scale = image.width() as f64 / 1280.;
    let inverse = camera.inverse();
    let value = |x: u32, y: u32| f32::from(image.get_pixel(x, y).0[1]);
    let mut steps = Vec::new();
    for y in 0..image.height() - 1 {
        for x in 0..image.width() - 1 {
            let page = inverse
                * kurbo::Point::new((f64::from(x) + 0.5) / scale, (f64::from(y) + 0.5) / scale);
            let (grid_x, grid_y) = (page.x as f32 - GRID_ORIGIN.0, page.y as f32 - GRID_ORIGIN.1);
            if grid_x < region.0
                || grid_y < region.1
                || grid_x > region.0 + region.2
                || grid_y > region.1 + region.3
            {
                continue;
            }
            let here = value(x, y);
            let (dx, dy) = (value(x + 1, y) - here, value(x, y + 1) - here);
            steps.push((dx * dx + dy * dy).sqrt());
        }
    }
    steps.sort_by(|a, b| b.total_cmp(a));
    steps[..count].iter().sum::<f32>() / count as f32
}

#[cfg(target_os = "macos")]
fn render() {
    let mut cx = VisualTestAppContext::new(gpui_ce_platform::current_platform(false));
    cx.update(|cx| {
        cx.text_system()
            .add_fonts(vec![Cow::Borrowed(font_fixtures::IBM_PLEX.data)])
    })
    .expect("failed to load the fixture font");
    let camera_cell = Rc::new(Cell::new(kurbo::Affine::IDENTITY));
    let window = cx
        .open_offscreen_window_default({
            let camera = camera_cell.clone();
            move |_, cx| {
                cx.new(|cx| TurnedChunksFixture {
                    grid: cx.new(|_| Grid),
                    camera,
                })
            }
        })
        .expect("failed to create the offscreen window");
    let window: gpui::AnyWindowHandle = window.into();
    cx.run_until_parked();
    let upright = cx
        .capture_screenshot(window)
        .expect("failed to capture the rendered window");

    // Moved a little, the grid is reused and made a chunk; then the camera
    // turns it, and the chunk is placed on the GPU.
    let mut draw = |camera: kurbo::Affine, capture: bool| {
        camera_cell.set(camera);
        cx.update_window(window, |root, window, cx| {
            root.downcast::<TurnedChunksFixture>()
                .unwrap()
                .update(cx, |_, cx| cx.notify());
            window.draw(cx).clear(cx);
            capture.then(|| window.render_to_image())
        })
        .expect("failed to draw the window")
    };
    draw(kurbo::Affine::translate((0.5, 0.)), false);
    draw(kurbo::Affine::IDENTITY, false);
    // Turned, captured before the frame the chunk asks for to be prepared
    // once its placement holds still.
    let turning = draw(camera(DEGREES), true)
        .expect("the frame was drawn")
        .expect("failed to capture the rendered window");
    // The camera holds still: the chunk is prepared at its angle.
    cx.run_until_parked();
    let (window_quads, ..) = cx
        .update_window(window, |_, window, _| window.rendered_primitive_counts())
        .unwrap();
    let image = cx
        .capture_screenshot(window)
        .expect("failed to capture the rendered window");
    if let Some(output) = std::env::var_os("GPUI_RENDERING_TEST_OUTPUT") {
        image.save(&output).expect("failed to save the image");
        let mut path = std::path::PathBuf::from(output);
        path.set_extension("turning.png");
        turning.save(path).expect("failed to save the image");
    }
    let mut failures = Vec::new();
    if window_quads >= CELLS * CELLS {
        failures.push(format!(
            "the grid was replayed into the window's scene, not drawn as a chunk: \
             {window_quads} quads"
        ));
    }

    // The text: resampled while the camera turned, rasterized turned once
    // it settled, as sharp as the upright text.
    let text = (TEXT_ORIGIN.0, TEXT_ORIGIN.1, 200., 26. * TEXT_LINES as f32);
    let count = 300;
    let upright_steepest = steepest(&upright, kurbo::Affine::IDENTITY, text, count);
    let turning_steepest = steepest(&turning, camera(DEGREES), text, count);
    let prepared_steepest = steepest(&image, camera(DEGREES), text, count);
    eprintln!(
        "steepest edges: upright {upright_steepest:.0}, turning {turning_steepest:.0}, \
         prepared {prepared_steepest:.0}"
    );
    if prepared_steepest < upright_steepest * 0.9 || prepared_steepest < turning_steepest * 1.1 {
        failures.push(format!(
            "the text is soft once the camera settles: its steepest edges average \
             {prepared_steepest:.0}, upright {upright_steepest:.0}, turning {turning_steepest:.0}"
        ));
    }

    // Cells, and the clipped bar, where the camera puts them.
    let scale = image.width() as f64 / 1280.;
    let pixel = |x: f32, y: f32| {
        let point = camera(DEGREES)
            * kurbo::Point::new(f64::from(GRID_ORIGIN.0 + x), f64::from(GRID_ORIGIN.1 + y));
        image
            .get_pixel((point.x * scale) as u32, (point.y * scale) as u32)
            .0
    };
    for (row, column) in [(0, 0), (0, 15), (7, 3), (15, 0), (15, 15)] {
        let center = CELL / 2.;
        let (x, y) = (center + column as f32 * PITCH, center + row as f32 * PITCH);
        let expected = cell_color(row, column);
        let expected = [
            (expected >> 16) as u8,
            (expected >> 8) as u8,
            expected as u8,
        ];
        let actual = pixel(x, y);
        if !actual[..3]
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.abs_diff(expected) < 12)
        {
            failures.push(format!(
                "cell ({row}, {column}): expected {expected:?}, got {actual:?}"
            ));
        }
    }
    let (left, top, width, height) = CLIP;
    for (what, x, y, red) in [
        (
            "inside the clip",
            left + width / 2.,
            top + height / 2.,
            true,
        ),
        // Within the turned clip's axis-aligned bounds, but outside it.
        (
            "past the clip's right edge",
            left + width + 6.,
            top + 4.,
            false,
        ),
        (
            "past the clip's bottom edge",
            left + 4.,
            top + height + 6.,
            false,
        ),
    ] {
        let actual = pixel(x, y);
        let is_red = actual[0] > 200 && actual[1] < 60 && actual[2] < 60;
        if is_red != red {
            failures.push(format!(
                "{what}: expected {}, got {actual:?}",
                if red { "red" } else { "not red" }
            ));
        }
    }

    std::mem::forget(cx);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
