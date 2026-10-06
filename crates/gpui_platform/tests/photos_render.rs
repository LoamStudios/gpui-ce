//! Renders photos through the platform renderer and checks their pixels: a
//! fine pattern drawn small, from a coarser level; a gradient drawn a pixel
//! to a pixel across the edges of its tiles; a rotated photo; a photo
//! repeated, with transparency; and a photo cropped to cover a box.
//!
//! Runs only with `GPUI_RUN_RENDERING_TESTS` set, in an offscreen window;
//! `GPUI_RENDERING_TEST_OUTPUT=<path.png>` saves the image.

#[cfg(target_os = "macos")]
use gpui::{
    AppContext as _, Bounds, Context, IntoElement, ObjectFit, ParentElement as _, Photo, Render,
    Styled as _, VisualTestAppContext, Window, canvas, div, fill, kurbo, peniko, point, px, rgb,
    size,
};
#[cfg(target_os = "macos")]
use image::{Rgba, RgbaImage};

#[cfg(target_os = "macos")]
struct PhotosFixture {
    /// 2048 pixels square: a checkerboard of single black and white pixels
    /// on the left, red on the right.
    pattern: Photo,
    /// 1024×64: red rising by one every four pixels, across four tiles.
    ramp: Photo,
    /// 200 pixels square: blue on the left, green on the right.
    halves: Photo,
    /// 32 pixels square: a red square in the top left quarter, the rest
    /// transparent.
    corner: Photo,
    /// 400×100: cyan on the left, magenta on the right.
    wide: Photo,
}

#[cfg(target_os = "macos")]
impl PhotosFixture {
    fn new() -> Self {
        Self {
            pattern: Photo::from_rgba(RgbaImage::from_fn(2048, 2048, |x, y| {
                if x >= 1024 {
                    Rgba([255, 0, 0, 255])
                } else if (x + y) % 2 == 0 {
                    Rgba([255, 255, 255, 255])
                } else {
                    Rgba([0, 0, 0, 255])
                }
            })),
            ramp: Photo::from_rgba(RgbaImage::from_fn(1024, 64, |x, _| {
                Rgba([(x / 4) as u8, 0, 0, 255])
            })),
            halves: Photo::from_rgba(RgbaImage::from_fn(200, 200, |x, _| {
                if x < 100 {
                    Rgba([0, 0, 255, 255])
                } else {
                    Rgba([0, 255, 0, 255])
                }
            })),
            corner: Photo::from_rgba(RgbaImage::from_fn(32, 32, |x, y| {
                if x < 16 && y < 16 {
                    Rgba([255, 0, 0, 255])
                } else {
                    Rgba([0, 0, 0, 0])
                }
            })),
            wide: Photo::from_rgba(RgbaImage::from_fn(400, 100, |x, _| {
                if x < 200 {
                    Rgba([0, 255, 255, 255])
                } else {
                    Rgba([255, 0, 255, 255])
                }
            })),
        }
    }
}

#[cfg(target_os = "macos")]
impl Render for PhotosFixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let pattern = self.pattern.clone();
        let ramp = self.ramp.clone();
        let halves = self.halves.clone();
        let corner = self.corner.clone();
        let wide = self.wide.clone();
        div().size_full().relative().bg(rgb(0xffffff)).child(
            canvas(
                |_, _, _| (),
                move |_, _, window, _| {
                    let sampler = peniko::ImageSampler::default();

                    // At (50, 50), 256 square: the 2048-pixel pattern, eight
                    // photo pixels to a point, four to a device pixel.
                    let bounds = Bounds::new(point(px(50.), px(50.)), size(px(256.), px(256.)));
                    let background = window.transformed_photo(
                        &pattern,
                        &sampler,
                        kurbo::Affine::translate((50., 50.)) * kurbo::Affine::scale(0.125),
                    );
                    window.paint_quad(fill(bounds, background));

                    // At (350, 50), 512×32: the ramp, a photo pixel to a
                    // device pixel.
                    let bounds = Bounds::new(point(px(350.), px(50.)), size(px(512.), px(32.)));
                    let background = window.transformed_photo(
                        &ramp,
                        &sampler,
                        kurbo::Affine::translate((350., 50.)) * kurbo::Affine::scale(0.5),
                    );
                    window.paint_quad(fill(bounds, background));

                    // At (50, 350), 200 square: the halves turned a quarter
                    // clockwise about their centre, blue on top.
                    let bounds = Bounds::new(point(px(50.), px(350.)), size(px(200.), px(200.)));
                    let background = window.transformed_photo(
                        &halves,
                        &sampler,
                        kurbo::Affine::rotate_about(
                            std::f64::consts::FRAC_PI_2,
                            kurbo::Point::new(150., 450.),
                        ) * kurbo::Affine::translate((50., 350.)),
                    );
                    window.paint_quad(fill(bounds, background));

                    // At (300, 350), 200×100: the corner repeated every 32
                    // points.
                    let bounds = Bounds::new(point(px(300.), px(350.)), size(px(200.), px(100.)));
                    let background = window.transformed_photo(
                        &corner,
                        &peniko::ImageSampler::default().with_extend(peniko::Extend::Repeat),
                        kurbo::Affine::translate((300., 350.)),
                    );
                    window.paint_quad(fill(bounds, background));

                    // At (550, 350), 100 square: the wide photo's middle,
                    // covering it.
                    window.paint_photo(
                        Bounds::new(point(px(550.), px(350.)), size(px(100.), px(100.))),
                        gpui::Corners::default(),
                        &wide,
                        ObjectFit::Cover,
                    );
                },
            )
            .absolute()
            .size_full(),
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
    let window = cx
        .open_offscreen_window_default(|_, cx| cx.new(|_| PhotosFixture::new()))
        .expect("failed to create the offscreen window");
    let window: gpui::AnyWindowHandle = window.into();

    // Tiles are decoded in the background, and each frame draws what has
    // arrived: draw until they all have.
    for _ in 0..40 {
        cx.run_until_parked();
        std::thread::sleep(std::time::Duration::from_millis(10));
        cx.update_window(window, |_, window, cx| {
            window.refresh();
            window.draw(cx).clear(cx);
        })
        .expect("failed to draw the window");
    }
    let image = cx
        .capture_screenshot(window)
        .expect("failed to capture the rendered window");
    if let Some(output) = std::env::var_os("GPUI_RENDERING_TEST_OUTPUT") {
        image.save(output).expect("failed to save the image");
    }

    let scale = image.width() as f32 / 1280.;
    let pixel = |x: f32, y: f32| *image.get_pixel((x * scale) as u32, (y * scale) as u32);
    let mut failures = Vec::new();
    let mut expect = |what: &str, x: f32, y: f32, rgb: [u8; 3]| {
        let actual = pixel(x, y);
        if !actual.0[..3]
            .iter()
            .zip(rgb)
            .all(|(actual, expected)| actual.abs_diff(expected) < 12)
        {
            failures.push(format!(
                "{what}: at ({x}, {y}) expected {rgb:?}, got {:?}",
                actual.0
            ));
        }
    };

    // Drawn from a level where each pixel averages the checkerboard: grey,
    // not black or white.
    expect("pattern, checkerboard", 100., 150., [128, 128, 128]);
    expect(
        "pattern, checkerboard elsewhere",
        151.3,
        201.7,
        [128, 128, 128],
    );
    expect("pattern, red", 250., 150., [255, 0, 0]);

    // The ramp at the edges of its tiles, at 256, 512 and 768 photo pixels:
    // 128, 256 and 384 points in.
    for edge in [128., 256., 384.] {
        for offset in [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25] {
            let x = 350. + edge + offset;
            let expected = ((x - 350.) * 2. / 4.) as u8;
            expect("ramp across a tile edge", x, 66., [expected, 0, 0]);
        }
    }

    expect("rotated photo, top", 150., 380., [0, 0, 255]);
    expect("rotated photo, bottom", 150., 520., [0, 255, 0]);

    expect(
        "repeated photo, a red square",
        300. + 8. + 32. * 3.,
        350. + 8. + 32.,
        [255, 0, 0],
    );
    expect(
        "repeated photo, transparent",
        300. + 24. + 32. * 2.,
        360.,
        [255, 255, 255],
    );

    expect("covering photo, left", 575., 400., [0, 255, 255]);
    expect("covering photo, right", 625., 400., [255, 0, 255]);

    std::mem::forget(cx);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
