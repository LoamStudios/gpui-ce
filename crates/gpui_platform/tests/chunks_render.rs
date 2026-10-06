//! Renders a cached view large enough to be drawn as a chunk, reused under a
//! camera transform that moves and zooms it, and checks its pixels land
//! where the transform puts them.
//!
//! Runs only with `GPUI_RUN_RENDERING_TESTS` set, in an offscreen window;
//! `GPUI_RENDERING_TEST_OUTPUT=<path.png>` saves the image.

#[cfg(target_os = "macos")]
use gpui::{
    AppContext as _, Context, Entity, IntoElement, ParentElement as _, Position, Render,
    StyleRefinement, Styled as _, VisualTestAppContext, Window, div, kurbo, px, rgb,
};
#[cfg(target_os = "macos")]
use std::{cell::Cell, rc::Rc};

/// Cells per side of the grid, and their pitch and size.
const CELLS: usize = 20;
const PITCH: f32 = 20.;
const CELL: f32 = 16.;
/// Where the grid sits on the page, away from the pointer at the origin.
const GRID_ORIGIN: f32 = 50.;

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
            .children((0..CELLS * CELLS).map(|index| {
                let (row, column) = (index / CELLS, index % CELLS);
                div()
                    .absolute()
                    .left(px(column as f32 * PITCH))
                    .top(px(row as f32 * PITCH))
                    .size(px(CELL))
                    .bg(rgb(cell_color(row, column)))
            }))
    }
}

#[cfg(target_os = "macos")]
struct ChunksFixture {
    grid: Entity<Grid>,
    /// The camera's scale and translation.
    camera: Rc<Cell<(f64, f64, f64)>>,
}

#[cfg(target_os = "macos")]
impl Render for ChunksFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let (scale, x, y) = self.camera.get();
        let mut style = StyleRefinement::default();
        style.position = Some(Position::Absolute);
        style.inset.left = Some(px(GRID_ORIGIN).into());
        style.inset.top = Some(px(GRID_ORIGIN).into());
        style.size.width = Some(px(CELLS as f32 * PITCH).into());
        style.size.height = Some(px(CELLS as f32 * PITCH).into());
        div().size_full().relative().bg(rgb(0xffffff)).child(
            // A page at the window's origin, so its transform maps page
            // points directly.
            div()
                .absolute()
                .size_0()
                .transform(kurbo::Affine::translate((x, y)) * kurbo::Affine::scale(scale))
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

#[cfg(target_os = "macos")]
fn render() {
    let mut cx = VisualTestAppContext::new(gpui_ce_platform::current_platform(false));
    let camera = Rc::new(Cell::new((1., 0., 0.)));
    let window = cx
        .open_offscreen_window_default({
            let camera = camera.clone();
            move |_, cx| {
                cx.new(|cx| ChunksFixture {
                    grid: cx.new(|_| Grid),
                    camera,
                })
            }
        })
        .expect("failed to create the offscreen window");
    let window: gpui::AnyWindowHandle = window.into();
    cx.run_until_parked();

    // Moved, the grid is reused and made a chunk; moved and zoomed, the
    // chunk is placed again.
    let final_camera = (1.5, 120., 40.);
    for next in [(1., 30., 20.), final_camera] {
        camera.set(next);
        cx.update_window(window, |root, window, cx| {
            root.downcast::<ChunksFixture>()
                .unwrap()
                .update(cx, |_, cx| cx.notify());
            window.draw(cx).clear(cx);
        })
        .expect("failed to draw the window");
    }
    // The grid's quads are drawn from its chunk, not the window's scene.
    let (window_quads, ..) = cx
        .update_window(window, |_, window, _| window.rendered_primitive_counts())
        .unwrap();
    assert!(
        window_quads < CELLS * CELLS,
        "the grid was replayed into the window's scene, not drawn as a chunk: {window_quads} quads"
    );
    let image = cx
        .capture_screenshot(window)
        .expect("failed to capture the rendered window");
    if let Some(output) = std::env::var_os("GPUI_RENDERING_TEST_OUTPUT") {
        image.save(output).expect("failed to save the image");
    }

    let scale = image.width() as f32 / 1280.;
    let pixel = |x: f32, y: f32| *image.get_pixel((x * scale) as u32, (y * scale) as u32);
    let (zoom, camera_x, camera_y) = final_camera;
    let on_screen = |page: f32, camera: f64| page * zoom as f32 + camera as f32;
    let mut failures = Vec::new();
    for (row, column) in [(0, 0), (0, 19), (7, 3), (19, 0), (19, 19), (10, 11)] {
        let center = GRID_ORIGIN + CELL / 2.;
        let x = on_screen(center + column as f32 * PITCH, camera_x);
        let y = on_screen(center + row as f32 * PITCH, camera_y);
        let expected = cell_color(row, column);
        let expected = [
            (expected >> 16) as u8,
            (expected >> 8) as u8,
            expected as u8,
        ];
        let actual = pixel(x, y);
        if !actual.0[..3]
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.abs_diff(expected) < 12)
        {
            failures.push(format!(
                "cell ({row}, {column}): at ({x}, {y}) expected {expected:?}, got {:?}",
                actual.0
            ));
        }
    }
    // Between cells, and where the grid was before it moved, is white.
    let gap = on_screen(GRID_ORIGIN + CELL + 2., camera_x);
    for (what, x, y) in [
        ("between cells", gap, on_screen(GRID_ORIGIN + 8., camera_y)),
        (
            "where the grid first was",
            GRID_ORIGIN + 5.,
            GRID_ORIGIN + 5.,
        ),
    ] {
        if pixel(x, y).0[..3] != [255, 255, 255] {
            failures.push(format!(
                "{what}: at ({x}, {y}) expected white, got {:?}",
                pixel(x, y).0
            ));
        }
    }

    std::mem::forget(cx);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
