//! Renders filtered groups through the platform renderer and checks their
//! pixels against CPU references: colour matrices (the CSS functions, and
//! exposure, temperature and tint in linear light) on two-tone boxes; the
//! profile of a blurred hard edge, at full and half resolution, unclipped by
//! the element's bounds; a blur under a zoom and a rotation, which keeps its
//! width in the element's own space; drop shadows, hard and blurred; shader
//! programs reading the picture (a sharpen and a duotone); filters on a
//! photo; and filtered groups in a cached view moved and zoomed by a camera.
//!
//! It runs on Metal, and with `macos-wgpu` on wgpu.
//!
//! Runs only with `GPUI_RUN_RENDERING_TESTS` set, in offscreen windows;
//! `GPUI_RENDERING_TEST_OUTPUT=<path.png>` saves the first image.

#[cfg(target_os = "macos")]
use gpui::{
    AnyWindowHandle, AppContext as _, ColorMatrix, Context, Entity, Filter, IntoElement, ObjectFit,
    ParentElement as _, Photo, Position, Render, Rgba, StyleRefinement, Styled as _,
    VisualTestAppContext, Window, blurred_edge_coverage, canvas, div, kurbo, px, radians, rgb,
    rgba,
    shader::{self, Fragment, Paint, vec2f, vec4f},
};
#[cfg(target_os = "macos")]
use image::RgbaImage;

/// The two colours of each colour-matrix swatch.
const SWATCH_LEFT: u32 = 0x3366cc;
const SWATCH_RIGHT: u32 = 0xe08030;
const SWATCH_TOP: f32 = 40.;
const SWATCH_SIZE: (f32, f32) = (80., 60.);
const SWATCH_PITCH: f32 = 100.;

/// Blurred black boxes: a large deviation (half resolution) and a small one
/// (full resolution).
const LARGE_BLUR: (f32, f32, f32, f32, f32) = (60., 180., 160., 100., 6.);
const SMALL_BLUR: (f32, f32, f32, f32, f32) = (300., 180., 100., 100., 1.5);
/// A 60×50 box blurred by 3, zoomed twice about its centre.
const ZOOMED: (f32, f32, f32, f32, f32) = (520., 180., 60., 50., 3.);
const ZOOM: f32 = 2.;
/// A 100 square blurred by 4, turned 30° about its centre.
const TURNED: (f32, f32, f32, f32, f32) = (700., 180., 100., 100., 4.);
const TURN: f32 = std::f32::consts::PI / 6.;

/// Drop shadows: a blue square with a hard shadow, and one with a blurred
/// shadow, each moved (20, 30), of half-transparent black.
const HARD_SHADOW: (f32, f32) = (60., 340.);
const SOFT_SHADOW: (f32, f32) = (260., 340.);
const SHADOW_BOX: f32 = 80.;
const SHADOW_OFFSET: (f32, f32) = (20., 30.);
const SHADOW_BLUR_RADIUS: f32 = 10.;

/// Stripes sharpened by a program.
const STRIPES: (f32, f32, f32, f32) = (480., 340., 120., 80.);
const STRIPE_WIDTH: f32 = 8.;
const STRIPE_COLORS: [u32; 3] = [0x204080, 0xd0a060, 0x60c040];

/// Photos: a duotone program, and exposure then saturation.
const DUOTONE_PHOTO: (f32, f32) = (650., 340.);
const ADJUSTED_PHOTO: (f32, f32) = (800., 340.);
const PHOTO_SIDE: f32 = 100.;
const QUADRANTS: [[u8; 3]; 4] = [[200, 60, 40], [40, 160, 90], [50, 80, 200], [230, 210, 120]];
const DUOTONE: ([f32; 3], [f32; 3]) = ([0.1, 0.05, 0.3], [1.0, 0.85, 0.4]);

/// The filters of the swatches, in order.
#[cfg(target_os = "macos")]
fn swatch_filters() -> Vec<(&'static str, Vec<Filter>)> {
    vec![
        ("brightness", vec![Filter::brightness(1.3)]),
        ("contrast", vec![Filter::contrast(1.4)]),
        ("saturate", vec![Filter::saturate(1.8)]),
        ("hue-rotate", vec![Filter::hue_rotate(90.)]),
        ("grayscale", vec![Filter::grayscale(1.)]),
        ("sepia", vec![Filter::sepia(0.8)]),
        ("invert", vec![Filter::invert(0.7)]),
        ("exposure", vec![Filter::exposure(1.)]),
        ("temperature", vec![Filter::temperature(0.8)]),
        ("tint", vec![Filter::tint(-0.6)]),
        // Folded into one pass: neither clamps.
        (
            "saturate then sepia",
            Filter::saturate(0.5).then(Filter::sepia(0.3)).into(),
        ),
        // Two passes, in different spaces, and an opacity.
        (
            "saturate, exposure, opacity",
            Filter::saturate(1.2)
                .then(Filter::exposure(0.5))
                .then(Filter::opacity(0.5))
                .into(),
        ),
    ]
}

#[cfg(target_os = "macos")]
fn color(hex: u32) -> [f32; 4] {
    let rgba: Rgba = rgb(hex);
    [
        rgba.color.red,
        rgba.color.green,
        rgba.color.blue,
        rgba.alpha,
    ]
}

/// What a chain of colour-matrix filters makes of opaque `input`, over
/// white, as 8-bit sRGB: each matrix applied as the renderer applies it,
/// with the pictures between passes held in 8 bits.
#[cfg(target_os = "macos")]
fn filtered_over_white(filters: &[Filter], input: [f32; 4]) -> [f32; 3] {
    let mut picture = input;
    let mut previous: Option<ColorMatrix> = None;
    for filter in filters {
        let Filter::ColorMatrix(matrix) = filter else {
            panic!("only colour matrices here");
        };
        // The plan folds a matrix into the one before when that one keeps
        // colours in range and works in the same space.
        previous = Some(match previous {
            Some(first) if first.color_space == matrix.color_space && first.keeps_unit_cube() => {
                first.then(matrix)
            }
            Some(first) => {
                picture = quantize(first.apply_premultiplied(picture));
                *matrix
            }
            None => *matrix,
        });
    }
    if let Some(last) = previous {
        picture = quantize(last.apply_premultiplied(picture));
    }
    let alpha = picture[3];
    [0, 1, 2].map(|channel| (picture[channel] + (1. - alpha)) * 255.)
}

#[cfg(target_os = "macos")]
fn quantize(rgba: [f32; 4]) -> [f32; 4] {
    rgba.map(|value| (value.clamp(0., 1.) * 255.).round() / 255.)
}

/// The stripes' colour at `x` logical pixels from their left.
#[cfg(target_os = "macos")]
fn stripe_color(x: f32) -> u32 {
    STRIPE_COLORS[(x / STRIPE_WIDTH).floor() as usize % STRIPE_COLORS.len()]
}

/// A 3×3 sharpen of the picture, a logical pixel to a tap.
#[cfg(target_os = "macos")]
fn sharpen() -> Paint {
    shader::paint(|px| {
        let center = px.input().rgba();
        let around = px.input_at(shader::vec2(1.0, 0.0)).rgba()
            + px.input_at(shader::vec2(-1.0, 0.0)).rgba()
            + px.input_at(shader::vec2(0.0, 1.0)).rgba()
            + px.input_at(shader::vec2(0.0, -1.0)).rgba();
        let sharpened = center * 5.0 - around;
        Paint::premultiplied(sharpened.clamp(0.0, 1.0))
    })
}

/// The picture's luminance mapped from a dark colour to a light one.
#[cfg(target_os = "macos")]
fn duotone() -> Paint {
    shader::paint(|px| {
        let input = px.input();
        let luminance = input.rgb().dot(shader::vec3(0.2126, 0.7152, 0.0722));
        let (dark, light) = DUOTONE;
        let toned = shader::vec3(dark[0], dark[1], dark[2])
            .mix(shader::vec3(light[0], light[1], light[2]), luminance);
        shader::rgba(toned.x(), toned.y(), toned.z(), input.alpha())
    })
}

#[cfg(target_os = "macos")]
fn quadrants() -> Photo {
    let side = 64;
    Photo::from_rgba(RgbaImage::from_fn(side, side, |x, y| {
        let [r, g, b] = QUADRANTS[(usize::from(y >= side / 2) * 2) + usize::from(x >= side / 2)];
        image::Rgba([r, g, b, 255])
    }))
}

#[cfg(target_os = "macos")]
struct FiltersFixture {
    photo: Photo,
}

#[cfg(target_os = "macos")]
fn boxed((left, top, width, height): (f32, f32, f32, f32)) -> gpui::Div {
    div()
        .absolute()
        .left(px(left))
        .top(px(top))
        .w(px(width))
        .h(px(height))
}

#[cfg(target_os = "macos")]
fn blurred((left, top, width, height, std_deviation): (f32, f32, f32, f32, f32)) -> gpui::Div {
    boxed((left, top, width, height))
        .bg(rgb(0x000000))
        .filter(Filter::blur(px(std_deviation)))
}

#[cfg(target_os = "macos")]
fn photo_box(photo: &Photo, (left, top): (f32, f32), filters: Vec<Filter>) -> gpui::Div {
    let photo = photo.clone();
    boxed((left, top, PHOTO_SIDE, PHOTO_SIDE))
        .filter(filters)
        .child(
            canvas(
                |_, _, _| (),
                move |bounds, _, window, _| {
                    window.paint_photo(bounds, gpui::Corners::default(), &photo, ObjectFit::Fill);
                },
            )
            .size_full(),
        )
}

#[cfg(target_os = "macos")]
impl Render for FiltersFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let shadow_color = rgba(0x00000080);
        let swatches = swatch_filters()
            .into_iter()
            .enumerate()
            .map(|(index, (_, filters))| {
                boxed((
                    20. + index as f32 * SWATCH_PITCH,
                    SWATCH_TOP,
                    SWATCH_SIZE.0,
                    SWATCH_SIZE.1,
                ))
                .flex()
                .filter(filters)
                .child(
                    div()
                        .w(px(SWATCH_SIZE.0 / 2.))
                        .h_full()
                        .bg(rgb(SWATCH_LEFT)),
                )
                .child(
                    div()
                        .w(px(SWATCH_SIZE.0 / 2.))
                        .h_full()
                        .bg(rgb(SWATCH_RIGHT)),
                )
            });
        let stripes = (0..(STRIPES.2 / STRIPE_WIDTH) as usize).map(|index| {
            div()
                .w(px(STRIPE_WIDTH))
                .h_full()
                .bg(rgb(stripe_color(index as f32 * STRIPE_WIDTH)))
        });
        div()
            .size_full()
            .relative()
            .bg(rgb(0xffffff))
            .children(swatches)
            .child(blurred(LARGE_BLUR))
            .child(blurred(SMALL_BLUR))
            .child(blurred(ZOOMED).scale(ZOOM))
            .child(blurred(TURNED).rotate(radians(TURN)))
            .child(
                boxed((HARD_SHADOW.0, HARD_SHADOW.1, SHADOW_BOX, SHADOW_BOX))
                    .bg(rgb(0x0000ff))
                    .drop_shadow(
                        px(SHADOW_OFFSET.0),
                        px(SHADOW_OFFSET.1),
                        px(0.),
                        shadow_color,
                    ),
            )
            .child(
                boxed((SOFT_SHADOW.0, SOFT_SHADOW.1, SHADOW_BOX, SHADOW_BOX))
                    .bg(rgb(0x0000ff))
                    .drop_shadow(
                        px(SHADOW_OFFSET.0),
                        px(SHADOW_OFFSET.1),
                        px(SHADOW_BLUR_RADIUS),
                        shadow_color,
                    ),
            )
            .child(
                boxed(STRIPES)
                    .flex()
                    .filter(Filter::program(sharpen()))
                    .children(stripes),
            )
            .child(photo_box(
                &self.photo,
                DUOTONE_PHOTO,
                vec![Filter::program(duotone())],
            ))
            .child(photo_box(
                &self.photo,
                ADJUSTED_PHOTO,
                Filter::exposure(0.7).then(Filter::saturate(1.4)).into(),
            ))
    }
}

fn main() {
    if std::env::var_os("GPUI_RUN_RENDERING_TESTS").is_none() {
        return;
    }
    #[cfg(target_os = "macos")]
    render();
}

/// The image's pixels, read at logical positions.
#[cfg(target_os = "macos")]
struct Picture {
    image: RgbaImage,
    scale: f32,
    failures: Vec<String>,
}

#[cfg(target_os = "macos")]
impl Picture {
    fn new(image: RgbaImage) -> Self {
        let scale = image.width() as f32 / 1280.;
        Self {
            image,
            scale,
            failures: Vec::new(),
        }
    }

    /// The device pixel at logical `(x, y)`, and the logical position of
    /// its centre.
    fn pixel(&self, x: f32, y: f32) -> ([u8; 4], (f32, f32)) {
        let (column, row) = ((x * self.scale) as u32, (y * self.scale) as u32);
        let center = (
            (column as f32 + 0.5) / self.scale,
            (row as f32 + 0.5) / self.scale,
        );
        (self.image.get_pixel(column, row).0, center)
    }

    /// Checks the pixel at `(x, y)` is `expected` within `tolerance`, and
    /// returns how far it is.
    fn expect(
        &mut self,
        what: &str,
        (x, y): (f32, f32),
        expected: [f32; 3],
        tolerance: f32,
    ) -> f32 {
        let (actual, _) = self.pixel(x, y);
        let error = actual[..3]
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (f32::from(*actual) - expected).abs())
            .fold(0., f32::max);
        if error > tolerance {
            self.failures.push(format!(
                "{what}: at ({x}, {y}) expected {:?}, got {actual:?}",
                expected.map(|value| value.round())
            ));
        }
        error
    }

    /// Checks the profile of a black edge blurred by `std_deviation`
    /// logical pixels, from `edge` along the unit `normal` (pointing out of
    /// the box), against the Gaussian's, and returns the largest error.
    fn expect_edge(
        &mut self,
        what: &str,
        edge: (f32, f32),
        normal: (f32, f32),
        std_deviation: f32,
        tolerance: f32,
    ) -> f32 {
        let mut largest: f32 = 0.;
        let steps = 12;
        for step in -steps..=steps {
            let distance = step as f32 * 2.5 * std_deviation / steps as f32;
            let (x, y) = (edge.0 + normal.0 * distance, edge.1 + normal.1 * distance);
            // Measured from the centre of the pixel sampled.
            let (_, center) = self.pixel(x, y);
            let along = (center.0 - edge.0) * normal.0 + (center.1 - edge.1) * normal.1;
            let coverage = blurred_edge_coverage(along, std_deviation);
            let grey = 255. * (1. - coverage);
            largest = largest.max(self.expect(
                &format!("{what} at {distance:.1} from the edge"),
                (x, y),
                [grey; 3],
                tolerance,
            ));
        }
        largest
    }
}

#[cfg(target_os = "macos")]
fn draw_until_photos_arrive(cx: &mut VisualTestAppContext, window: AnyWindowHandle) {
    for _ in 0..40 {
        cx.run_until_parked();
        std::thread::sleep(std::time::Duration::from_millis(10));
        cx.update_window(window, |_, window, cx| {
            window.refresh();
            window.draw(cx).clear(cx);
        })
        .expect("failed to draw the window");
    }
}

#[cfg(target_os = "macos")]
fn render() {
    // SAFETY: the test is single-threaded until here, and the renderer reads
    // this when the window opens. Programs are linked before frames draw.
    unsafe { std::env::set_var("GPUI_LINK_PROGRAMS_SYNCHRONOUSLY", "1") };
    let mut cx = VisualTestAppContext::new(gpui_ce_platform::current_platform(false));
    let window: AnyWindowHandle = cx
        .open_offscreen_window_default(|_, cx| cx.new(|_| FiltersFixture { photo: quadrants() }))
        .expect("failed to create the offscreen window")
        .into();
    draw_until_photos_arrive(&mut cx, window);
    let image = cx
        .capture_screenshot(window)
        .expect("failed to capture the rendered window");
    if let Some(output) = std::env::var_os("GPUI_RENDERING_TEST_OUTPUT") {
        image.save(output).expect("failed to save the image");
    }
    let mut picture = Picture::new(image);

    // Colour matrices, against the CPU reference.
    let mut matrix_error: f32 = 0.;
    for (index, (what, filters)) in swatch_filters().iter().enumerate() {
        let left = 20. + index as f32 * SWATCH_PITCH;
        let middle = SWATCH_TOP + SWATCH_SIZE.1 / 2.;
        for (half, hex) in [(0.25, SWATCH_LEFT), (0.75, SWATCH_RIGHT)] {
            let expected = filtered_over_white(filters, color(hex));
            matrix_error = matrix_error.max(picture.expect(
                what,
                (left + SWATCH_SIZE.0 * half, middle),
                expected,
                3.,
            ));
        }
    }

    // Blurred edges: their profiles, out past the element's bounds.
    let (left, top, _, height, std_deviation) = LARGE_BLUR;
    let large = picture.expect_edge(
        "large blur",
        (left, top + height / 2.),
        (-1., 0.),
        std_deviation,
        6.,
    );
    let (left, top, _, height, std_deviation) = SMALL_BLUR;
    let small = picture.expect_edge(
        "small blur",
        (left, top + height / 2.),
        (-1., 0.),
        std_deviation,
        6.,
    );
    // Zoomed twice about its centre: its edge moves out, and its blur is
    // twice as wide.
    let (left, top, width, height, std_deviation) = ZOOMED;
    let center = (left + width / 2., top + height / 2.);
    let zoomed = picture.expect_edge(
        "zoomed blur",
        (center.0 - width / 2. * ZOOM, center.1),
        (-1., 0.),
        std_deviation * ZOOM,
        6.,
    );
    // Turned: across its turned left edge, as wide as unturned.
    let (left, top, width, height, std_deviation) = TURNED;
    let center = (left + width / 2., top + height / 2.);
    let (sin, cos) = TURN.sin_cos();
    let normal = (-cos, -sin);
    let turned = picture.expect_edge(
        "turned blur",
        (
            center.0 + normal.0 * width / 2.,
            center.1 + normal.1 * height / 2.,
        ),
        normal,
        std_deviation,
        6.,
    );
    // The large blur reaches past the element's bounds: at one deviation
    // out, it is 16% covered, not white.
    let (left, top, _, height, std_deviation) = LARGE_BLUR;
    picture.expect(
        "large blur, unclipped",
        (left - std_deviation, top + height / 2.),
        [255. * (1. - 0.1587); 3],
        6.,
    );

    // Drop shadows. Hard: the square, the shadow beside and below it, and
    // white past them.
    let blue = [0., 0., 255.];
    let shadow = [127.5; 3];
    let white = [255.; 3];
    let (x, y) = HARD_SHADOW;
    let (dx, dy) = SHADOW_OFFSET;
    let mut shadow_error: f32 = 0.;
    for (what, at, expected) in [
        ("hard shadow: the square", (x + 40., y + 40.), blue),
        (
            "hard shadow: right of the square",
            (x + SHADOW_BOX + dx / 2., y + dy + 40.),
            shadow,
        ),
        (
            "hard shadow: below the square",
            (x + 40., y + SHADOW_BOX + dy / 2.),
            shadow,
        ),
        (
            "hard shadow: above it",
            (x + SHADOW_BOX + dx / 2., y + dy / 2.),
            white,
        ),
        (
            "hard shadow: left of it",
            (x + dx / 2., y + SHADOW_BOX + dy / 2.),
            white,
        ),
    ] {
        shadow_error = shadow_error.max(picture.expect(what, at, expected, 3.));
    }
    // Blurred: across the shadow's right edge, half black at most.
    let (x, y) = SOFT_SHADOW;
    let edge = (x + SHADOW_BOX + dx, y + dy + SHADOW_BOX / 2.);
    let std_deviation = SHADOW_BLUR_RADIUS / 2.;
    for step in -4..=4 {
        let at = (edge.0 + step as f32 * 3., edge.1);
        let (_, center) = picture.pixel(at.0, at.1);
        let coverage = blurred_edge_coverage(center.0 - edge.0, std_deviation);
        shadow_error = shadow_error.max(picture.expect(
            "blurred shadow",
            at,
            [255. * (1. - 0.5 * coverage); 3],
            6.,
        ));
    }

    // Its left edge, below the square, is as blurred: the shadow is blurred
    // where it is shown, not cut off where it was cast.
    let edge = (x + dx, y + SHADOW_BOX + dy / 2.);
    for step in -4..=4 {
        let at = (edge.0 + step as f32 * 3., edge.1);
        let (_, center) = picture.pixel(at.0, at.1);
        let coverage = blurred_edge_coverage(edge.0 - center.0, std_deviation);
        shadow_error = shadow_error.max(picture.expect(
            "blurred shadow, left edge",
            at,
            [255. * (1. - 0.5 * coverage); 3],
            6.,
        ));
    }

    // Programs: the sharpened stripes against the program on the CPU,
    // reading the stripes as drawn.
    let (left, top, width, height) = STRIPES;
    let sharpen = sharpen();
    let mut program_error: f32 = 0.;
    for (u, v) in [
        (0.05, 0.5),
        (0.2, 0.3),
        (0.33, 0.5),
        (0.4, 0.7),
        (0.55, 0.5),
        (0.68, 0.4),
        (0.8, 0.6),
        (0.93, 0.5),
    ] {
        let (_, center) = picture.pixel(left + u * width, top + v * height);
        let position = vec2f(center.0 - left, center.1 - top);
        let fragment = fragment(position, (width, height), picture.scale);
        let input = |at: shader::Vec2f| {
            if at.x < 0. || at.y < 0. || at.x >= width || at.y >= height {
                return vec4f(0., 0., 0., 0.);
            }
            let [r, g, b, a] = color(stripe_color(at.x));
            vec4f(r, g, b, a)
        };
        let out = sharpen
            .evaluate_filter(fragment, &input)
            .expect("the sharpen evaluates");
        let expected = [out.x, out.y, out.z].map(|channel| (channel + 1. - out.w) * 255.);
        program_error = program_error.max(picture.expect(
            "sharpened stripes",
            (left + u * width, top + v * height),
            expected,
            3.,
        ));
    }

    // On photos: the duotone, and exposure then saturation, at each
    // quadrant's centre.
    let duotone = duotone();
    let adjustments: Vec<Filter> = Filter::exposure(0.7).then(Filter::saturate(1.4)).into();
    let mut photo_error: f32 = 0.;
    for (index, [r, g, b]) in QUADRANTS.iter().enumerate() {
        let (u, v) = (
            0.25 + 0.5 * (index % 2) as f32,
            0.25 + 0.5 * (index / 2) as f32,
        );
        let input = [*r, *g, *b].map(|channel| f32::from(channel) / 255.);
        let input = [input[0], input[1], input[2], 1.];

        let position = vec2f(u * PHOTO_SIDE, v * PHOTO_SIDE);
        let out = duotone
            .evaluate_filter(
                fragment(position, (PHOTO_SIDE, PHOTO_SIDE), picture.scale),
                &|_| vec4f(input[0], input[1], input[2], 1.),
            )
            .expect("the duotone evaluates");
        photo_error = photo_error.max(picture.expect(
            "duotone photo",
            (
                DUOTONE_PHOTO.0 + u * PHOTO_SIDE,
                DUOTONE_PHOTO.1 + v * PHOTO_SIDE,
            ),
            [out.x, out.y, out.z].map(|channel| channel * 255.),
            3.,
        ));
        photo_error = photo_error.max(picture.expect(
            "adjusted photo",
            (
                ADJUSTED_PHOTO.0 + u * PHOTO_SIDE,
                ADJUSTED_PHOTO.1 + v * PHOTO_SIDE,
            ),
            filtered_over_white(&adjustments, input),
            3.,
        ));
    }

    println!(
        "filters_render: largest errors, out of 255: colour matrices {matrix_error:.1}, \
         blurs {large:.1} (half resolution) {small:.1} (full) {zoomed:.1} (zoomed) \
         {turned:.1} (turned), shadows {shadow_error:.1}, programs {program_error:.1}, \
         photos {photo_error:.1}"
    );
    let failures = std::mem::take(&mut picture.failures);
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    in_a_cached_view(&mut cx);
    std::mem::forget(cx);
}

#[cfg(target_os = "macos")]
fn fragment(position: shader::Vec2f, (width, height): (f32, f32), scale: f32) -> Fragment {
    Fragment {
        uv: vec2f(position.x / width, position.y / height),
        position,
        size: vec2f(width, height),
        origin: vec2f(0., 0.),
        scale,
        stroke: vec2f(0., 0.),
    }
}

/// Cells enough for the cached view to be drawn as a chunk, were it not for
/// its filtered groups, and where they are.
const CELLS: usize = 20;
const PITCH: f32 = 20.;
const CACHED_SHADOW: (f32, f32) = (40., 40.);
const CACHED_BLUR: (f32, f32, f32, f32, f32) = (200., 40., 120., 80., 4.);

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
                    .top(px(row as f32 * PITCH + 200.))
                    .size(px(PITCH / 4.))
                    .bg(rgb(0xe0e0e0))
            }))
            .child(
                boxed((CACHED_SHADOW.0, CACHED_SHADOW.1, SHADOW_BOX, SHADOW_BOX))
                    .bg(rgb(0x0000ff))
                    .drop_shadow(
                        px(SHADOW_OFFSET.0),
                        px(SHADOW_OFFSET.1),
                        px(0.),
                        rgba(0x00000080),
                    ),
            )
            .child(blurred(CACHED_BLUR))
    }
}

#[cfg(target_os = "macos")]
struct CameraFixture {
    grid: Entity<Grid>,
    camera: std::rc::Rc<std::cell::Cell<kurbo::Affine>>,
}

#[cfg(target_os = "macos")]
impl Render for CameraFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let mut style = StyleRefinement::default();
        style.position = Some(Position::Absolute);
        style.size.width = Some(px(CELLS as f32 * PITCH).into());
        style.size.height = Some(px(CELLS as f32 * PITCH + 200.).into());
        div().size_full().relative().bg(rgb(0xffffff)).child(
            div()
                .absolute()
                .size_0()
                .transform(self.camera.get())
                .child(self.grid.clone().cached(style)),
        )
    }
}

/// Filtered groups in a cached view, which a camera moves, then zooms: the
/// view's recording is reused, placed by the camera, with its groups'
/// shadows moved and blurs scaled with it.
#[cfg(target_os = "macos")]
fn in_a_cached_view(cx: &mut VisualTestAppContext) {
    let camera = std::rc::Rc::new(std::cell::Cell::new(kurbo::Affine::IDENTITY));
    let window: AnyWindowHandle = cx
        .open_offscreen_window_default({
            let camera = camera.clone();
            move |_, cx| {
                cx.new(|cx| CameraFixture {
                    grid: cx.new(|_| Grid),
                    camera,
                })
            }
        })
        .expect("failed to create the offscreen window")
        .into();
    cx.run_until_parked();
    let mut failures = Vec::new();
    for (what, zoom, origin) in [
        ("moved", 1., (300., 150.)),
        ("moved again", 1., (500., 260.)),
        ("zoomed", 1.5, (400., 200.)),
    ] {
        camera.set(
            kurbo::Affine::translate((f64::from(origin.0), f64::from(origin.1)))
                * kurbo::Affine::scale(f64::from(zoom)),
        );
        cx.update_window(window, |root, window, cx| {
            root.downcast::<CameraFixture>()
                .unwrap()
                .update(cx, |_, cx| cx.notify());
            window.draw(cx).clear(cx);
        })
        .expect("failed to draw the window");
        let image = cx
            .update_window(window, |_, window, _| window.render_to_image())
            .expect("failed to capture the rendered window")
            .expect("failed to capture the rendered window");
        let mut picture = Picture::new(image);
        let at = |x: f32, y: f32| (origin.0 + x * zoom, origin.1 + y * zoom);
        let (x, y) = CACHED_SHADOW;
        let (dx, dy) = SHADOW_OFFSET;
        picture.expect(
            &format!("{what}: the square"),
            at(x + 40., y + 40.),
            [0., 0., 255.],
            3.,
        );
        picture.expect(
            &format!("{what}: its shadow"),
            at(x + SHADOW_BOX + dx / 2., y + dy + 40.),
            [127.5; 3],
            3.,
        );
        picture.expect(
            &format!("{what}: above its shadow"),
            at(x + SHADOW_BOX + dx / 2., y + dy / 2.),
            [255.; 3],
            3.,
        );
        let (left, top, _, height, std_deviation) = CACHED_BLUR;
        picture.expect_edge(
            &format!("{what}: the blur"),
            at(left, top + height / 2.),
            (-1., 0.),
            std_deviation * zoom,
            6.,
        );
        failures.extend(picture.failures);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
