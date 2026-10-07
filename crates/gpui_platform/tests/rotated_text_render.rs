//! Renders a line of text plainly and turned or skewed about its center, and
//! checks the turned text is rasterized as it is drawn rather than resampled:
//! its edges are as steep as the plain text's, and it matches the plain text
//! turned on the CPU.
//!
//! Runs only with `GPUI_RUN_RENDERING_TESTS` set, in an offscreen window;
//! `GPUI_RENDERING_TEST_OUTPUT=<path.png>` saves the image.

#[cfg(target_os = "macos")]
use gpui::{
    AppContext as _, Context, IntoElement, ParentElement as _, Render, Styled as _,
    VisualTestAppContext, Window, div, kurbo, px, rgb,
};
#[cfg(target_os = "macos")]
use std::borrow::Cow;

#[cfg(target_os = "macos")]
#[allow(dead_code)]
#[path = "../../gpui_ce_parley/src/font_fixtures.rs"]
mod font_fixtures;

/// The size of each line's box.
#[cfg(target_os = "macos")]
const LINE: (f32, f32) = (300., 40.);
/// The plain line's box's center.
#[cfg(target_os = "macos")]
const PLAIN: (f32, f32) = (190., 60.);

/// Each turned or skewed line: what it is, its transform about its box's
/// center, and that center.
#[cfg(target_os = "macos")]
fn cases() -> Vec<(&'static str, kurbo::Affine, (f32, f32))> {
    let degrees = |angle: f64| kurbo::Affine::rotate(angle.to_radians());
    vec![
        ("turned 7°", degrees(7.), (230., 330.)),
        ("turned 30°", degrees(30.), (640., 330.)),
        ("turned 45°", degrees(45.), (1050., 330.)),
        ("turned -60°", degrees(-60.), (230., 610.)),
        ("skewed", kurbo::Affine::skew(0.4, 0.), (640., 610.)),
        (
            "turned 20° and stretched",
            degrees(20.) * kurbo::Affine::scale_non_uniform(1.5, 1.),
            (1050., 610.),
        ),
    ]
}

/// A line of text in a box centered at `center`.
#[cfg(target_os = "macos")]
fn line(center: (f32, f32)) -> gpui::Div {
    div()
        .absolute()
        .left(px(center.0 - LINE.0 / 2.))
        .top(px(center.1 - LINE.1 / 2.))
        .w(px(LINE.0))
        .h(px(LINE.1))
        .flex()
        .items_center()
        .justify_center()
        .child("Hamburgefontsiv")
}

#[cfg(target_os = "macos")]
struct RotatedTextFixture;

#[cfg(target_os = "macos")]
impl Render for RotatedTextFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let mut root = div()
            .size_full()
            .relative()
            .bg(rgb(0xffffff))
            .text_color(rgb(0x000000))
            .font_family("IBM Plex Sans")
            .text_size(px(20.))
            .child(line(PLAIN));
        for (_, transform, center) in cases() {
            root = root.child(line(center).transform(transform));
        }
        root
    }
}

fn main() {
    if std::env::var_os("GPUI_RUN_RENDERING_TESTS").is_none() {
        return;
    }
    #[cfg(target_os = "macos")]
    render();
}

/// A gray image, as darkness: 0 for white paper, 255 for black ink.
#[cfg(target_os = "macos")]
struct Ink {
    width: i32,
    height: i32,
    values: Vec<f32>,
}

#[cfg(target_os = "macos")]
impl Ink {
    fn of(image: &image::RgbaImage) -> Self {
        Self {
            width: image.width() as i32,
            height: image.height() as i32,
            values: image
                .pixels()
                .map(|pixel| 255. - f32::from(pixel.0[1]))
                .collect(),
        }
    }

    fn at(&self, x: i32, y: i32) -> f32 {
        if x < 0 || y < 0 || x >= self.width || y >= self.height {
            return 0.;
        }
        self.values[(y * self.width + x) as usize]
    }

    /// Bilinearly interpolated, at a position in pixel units where pixel
    /// centers are at half-integers.
    fn sample(&self, x: f32, y: f32) -> f32 {
        let (x, y) = (x - 0.5, y - 0.5);
        let (left, top) = (x.floor(), y.floor());
        let (fx, fy) = (x - left, y - top);
        let (left, top) = (left as i32, top as i32);
        let row = |y| self.at(left, y) * (1. - fx) + self.at(left + 1, y) * fx;
        row(top) * (1. - fy) + row(top + 1) * fy
    }

    /// The mean of the `count` largest forward-difference gradient
    /// magnitudes in `region`: how steep the steepest edges are.
    fn steepest(&self, region: (i32, i32, i32, i32), count: usize) -> f32 {
        let (left, top, right, bottom) = region;
        let mut gradients = Vec::new();
        for y in top..bottom {
            for x in left..right {
                let here = self.at(x, y);
                let (dx, dy) = (self.at(x + 1, y) - here, self.at(x, y + 1) - here);
                gradients.push((dx * dx + dy * dy).sqrt());
            }
        }
        gradients.sort_by(|a, b| b.total_cmp(a));
        gradients[..count].iter().sum::<f32>() / count as f32
    }

    /// The total ink in `region`.
    fn total(&self, region: (i32, i32, i32, i32)) -> f32 {
        let (left, top, right, bottom) = region;
        (top..bottom)
            .flat_map(|y| (left..right).map(move |x| (x, y)))
            .map(|(x, y)| self.at(x, y))
            .sum()
    }

    /// The image with each pixel of `region` averaged with its eight
    /// neighbours, and nothing outside it.
    fn blurred(&self, region: (i32, i32, i32, i32)) -> Self {
        let (left, top, right, bottom) = region;
        let mut values = vec![0.; self.values.len()];
        for y in top..bottom {
            for x in left..right {
                let mut sum = 0.;
                for (dx, dy) in (-1..=1).flat_map(|dy| (-1..=1).map(move |dx| (dx, dy))) {
                    sum += self.at(x + dx, y + dy);
                }
                values[(y * self.width + x) as usize] = sum / 9.;
            }
        }
        Self {
            width: self.width,
            height: self.height,
            values,
        }
    }
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
        .open_offscreen_window_default(|_, cx| cx.new(|_| RotatedTextFixture))
        .expect("failed to create the offscreen window");
    let window = window.into();
    cx.run_until_parked();
    let image = cx
        .capture_screenshot(window)
        .expect("failed to capture the rendered window");
    if let Some(output) = std::env::var_os("GPUI_RENDERING_TEST_OUTPUT") {
        image.save(output).expect("failed to save the image");
    }

    let scale = (image.width() / 1280) as f32;
    let ink = Ink::of(&image);
    let plain_region = (
        ((PLAIN.0 - LINE.0 / 2.) * scale) as i32,
        ((PLAIN.1 - LINE.1 / 2.) * scale) as i32,
        ((PLAIN.0 + LINE.0 / 2.) * scale) as i32,
        ((PLAIN.1 + LINE.1 / 2.) * scale) as i32,
    );
    let plain_ink = ink.total(plain_region);
    assert!(plain_ink > 0., "the plain text was drawn");
    // The steepest edges: about as many pixels as the text's outline has
    // edge pixels, a fraction of its ink.
    let count = (plain_ink / 255. * 0.3) as usize;
    let plain_steepest = ink.steepest(plain_region, count);
    let mut failures = Vec::new();

    for (what, transform, center) in cases() {
        let reach = 170. * scale;
        let region = (
            (center.0 * scale - reach) as i32,
            (center.1 * scale - reach) as i32,
            (center.0 * scale + reach) as i32,
            (center.1 * scale + reach) as i32,
        );
        // The plain text turned on the CPU: each pixel of the turned line's
        // region sampled from the plain text where the transform takes it
        // from, moved by `shift` device pixels there.
        let inverse = transform.inverse();
        let turned_on_cpu = |shift: (f32, f32)| {
            let mut reference = Ink {
                width: ink.width,
                height: ink.height,
                values: vec![0.; ink.values.len()],
            };
            for y in region.1..region.3 {
                for x in region.0..region.2 {
                    let local = kurbo::Point::new(
                        f64::from((x as f32 + 0.5) / scale - center.0),
                        f64::from((y as f32 + 0.5) / scale - center.1),
                    );
                    let source = inverse * local;
                    reference.values[(y * ink.width + x) as usize] = ink.sample(
                        (source.x as f32 + PLAIN.0) * scale + shift.0,
                        (source.y as f32 + PLAIN.1) * scale + shift.1,
                    );
                }
            }
            reference
        };
        // How far the turned text is from the reference, each blurred a
        // little so edges a pixel softer or sharper count for little.
        let blurred = ink.blurred(region);
        let difference = |reference: &Ink| {
            let reference = reference.blurred(region);
            let (mut difference, mut covered) = (0., 0.);
            for y in region.1..region.3 {
                for x in region.0..region.2 {
                    let (turned, expected) = (blurred.at(x, y), reference.at(x, y));
                    if turned > 0. || expected > 0. {
                        difference += (turned - expected).abs();
                        covered += 1.;
                    }
                }
            }
            difference / covered
        };
        // The plain text's glyphs are snapped to whole pixels down and
        // quarter pixels across, the turned text's are not: compare it with
        // the reference shifted by up to that much, at its best.
        let difference = [-0.125, 0., 0.125]
            .iter()
            .flat_map(|&dx| [-0.5, -0.25, 0., 0.25, 0.5].map(move |dy| (dx, dy)))
            .map(|shift| difference(&turned_on_cpu(shift)))
            .fold(f32::INFINITY, f32::min);

        let steepest = ink.steepest(region, count);
        let reference_steepest = turned_on_cpu((0., 0.)).steepest(region, count);
        let weight = ink.total(region) / plain_ink * (transform.determinant().abs() as f32).recip();
        eprintln!(
            "{what}: steepest edges {steepest:.0} (plain {plain_steepest:.0}, turned on the CPU \
             {reference_steepest:.0}), differs from the CPU's by {difference:.2}, ink {weight:.3}"
        );
        if steepest < plain_steepest * 0.9 {
            failures.push(format!(
                "the text {what} is soft: its steepest edges average {steepest:.0}, \
                 the plain text's {plain_steepest:.0}"
            ));
        }
        if difference > 10. {
            failures.push(format!(
                "the text {what} differs from the plain text turned on the CPU by {difference:.2}"
            ));
        }
        if (weight - 1.).abs() > 0.05 {
            failures.push(format!(
                "the text {what} has {weight:.3} times the plain text's ink"
            ));
        }
    }

    std::mem::forget(cx);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
