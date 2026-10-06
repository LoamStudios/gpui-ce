//! Renders photos through the headless WGPU renderer, on the device's native tier and the
//! downlevel tier, and checks their pixels: a fine pattern drawn small, from a coarser
//! level; a gradient drawn a pixel to a pixel across the edges of its tiles; a rotated
//! photo; a photo repeated, with transparency; a photo cropped to cover a box; and enough
//! tiles to grow the photo tile array, keeping the tiles it held.
//!
//! Tiles are decoded inline, before each frame, by gpui's synchronous photo residency.
#![cfg(feature = "test-support")]

use gpui::{
    Bounds, ContentMask, DevicePixels, ObjectFit, Photo, PlatformHeadlessRenderer, Point, Quad,
    ScaledPixels, Scene, Size, SynchronousPhotoResidency, TransformationMatrix, kurbo, peniko,
};
use gpui_ce_wgpu::WgpuHeadlessRenderer;
use image::{Rgba, RgbaImage};

const TOLERANCE: u8 = 12;

fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
    Bounds {
        origin: Point {
            x: ScaledPixels(x),
            y: ScaledPixels(y),
        },
        size: Size {
            width: ScaledPixels(w),
            height: ScaledPixels(h),
        },
    }
}

fn target(width: i32, height: i32) -> Size<DevicePixels> {
    Size {
        width: DevicePixels(width),
        height: DevicePixels(height),
    }
}

/// Fills `bounds` with `color`, or, if `paint` is not 0, with that paint-table entry.
fn fill(scene: &mut Scene, bounds: Bounds<ScaledPixels>, color: gpui::Hsla, paint: u32) {
    scene.insert_primitive(Quad {
        bounds,
        content_mask: ContentMask {
            bounds,
            ..Default::default()
        },
        background: gpui::ScenePaintRef {
            color: color.into(),
            paint,
        },
        ..Default::default()
    });
}

/// Adds a paint of `photo`, placed by `to_viewport`, from its pixels to device pixels.
fn photo_paint(
    scene: &mut Scene,
    photo: &Photo,
    sampler: &peniko::ImageSampler,
    to_viewport: kurbo::Affine,
    region: Option<kurbo::Rect>,
) -> u32 {
    SynchronousPhotoResidency::push_photo(
        scene,
        photo,
        TransformationMatrix::from(to_viewport.inverse()),
        sampler,
        region,
    )
}

fn renderers() -> [(&'static str, WgpuHeadlessRenderer); 2] {
    [
        (
            "native",
            WgpuHeadlessRenderer::new().expect("headless renderer"),
        ),
        (
            "downlevel",
            WgpuHeadlessRenderer::new_downlevel().expect("downlevel renderer"),
        ),
    ]
}

/// Draws the scene `build` makes, its photos prepared by `residency` with every tile
/// decoded, into a target `size`.
fn render(
    renderer: &mut WgpuHeadlessRenderer,
    residency: &mut SynchronousPhotoResidency,
    size: Size<DevicePixels>,
    build: impl FnOnce(&mut Scene),
) -> RgbaImage {
    let mut scene = Scene::default();
    build(&mut scene);
    scene.finish();
    residency.prepare(&mut scene, size);
    renderer
        .render_scene_to_image(&scene, size)
        .expect("headless render")
}

#[derive(Default)]
struct Failures(Vec<String>);

impl Failures {
    fn expect(&mut self, image: &RgbaImage, what: &str, x: u32, y: u32, expected: [u8; 3]) {
        let actual = image.get_pixel(x, y).0;
        if !actual[..3]
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.abs_diff(expected) <= TOLERANCE)
        {
            self.0.push(format!(
                "{what} at ({x}, {y}): expected {expected:?}, got {actual:?}"
            ));
        }
    }

    fn assert_none(self) {
        assert!(self.0.is_empty(), "{}", self.0.join("\n"));
    }
}

struct Fixture {
    /// 2048 pixels square: a checkerboard of single black and white pixels on the
    /// left, red on the right.
    pattern: Photo,
    /// 1024×64: red rising by one every four pixels, across four tiles.
    ramp: Photo,
    /// 200 pixels square: blue on the left, green on the right.
    halves: Photo,
    /// 32 pixels square: a red square in the top left quarter, the rest transparent.
    corner: Photo,
    /// 400×100: cyan on the left, magenta on the right.
    wide: Photo,
}

impl Fixture {
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

    fn build(&self, scene: &mut Scene) {
        let sampler = peniko::ImageSampler::default();
        fill(scene, bounds(0., 0., 600., 360.), gpui::white(), 0);

        // At (0, 0), 128 square: the 2048-pixel pattern, sixteen photo pixels to a
        // device pixel.
        let paint = photo_paint(
            scene,
            &self.pattern,
            &sampler,
            kurbo::Affine::scale(1. / 16.),
            None,
        );
        fill(scene, bounds(0., 0., 128., 128.), gpui::white(), paint);

        // At (140, 0), 400×32: the ramp from photo pixel 200, a photo pixel to a device
        // pixel, across the tile edges at 256 and 512.
        let paint = photo_paint(
            scene,
            &self.ramp,
            &sampler,
            kurbo::Affine::translate((-60., 0.)),
            None,
        );
        fill(scene, bounds(140., 0., 400., 32.), gpui::white(), paint);

        // At (0, 150), 200 square: the halves turned a quarter clockwise about their
        // centre, blue on top.
        let paint = photo_paint(
            scene,
            &self.halves,
            &sampler,
            kurbo::Affine::rotate_about(std::f64::consts::FRAC_PI_2, kurbo::Point::new(100., 250.))
                * kurbo::Affine::translate((0., 150.)),
            None,
        );
        fill(scene, bounds(0., 150., 200., 200.), gpui::white(), paint);

        // At (220, 150), 200×100: the corner repeated every 32 pixels.
        let paint = photo_paint(
            scene,
            &self.corner,
            &peniko::ImageSampler::default().with_extend(peniko::Extend::Repeat),
            kurbo::Affine::translate((220., 150.)),
            None,
        );
        fill(scene, bounds(220., 150., 200., 100.), gpui::white(), paint);

        // At (440, 150), 100 square: the wide photo's middle, covering it, fitted as
        // `Window::paint_photo` fits it.
        let fitted = ObjectFit::Cover.get_bounds(
            Bounds::new(
                gpui::point(gpui::px(440.), gpui::px(150.)),
                gpui::size(gpui::px(100.), gpui::px(100.)),
            ),
            gpui::size(DevicePixels(400), DevicePixels(100)),
        );
        let scale = f64::from(fitted.size.height) / 100.;
        let left = f64::from(fitted.origin.x);
        let top = f64::from(fitted.origin.y);
        let paint = photo_paint(
            scene,
            &self.wide,
            &sampler,
            kurbo::Affine::translate((left, top)) * kurbo::Affine::scale(scale),
            Some(kurbo::Rect::new(
                (440. - left) / scale,
                (150. - top) / scale,
                (540. - left) / scale,
                (250. - top) / scale,
            )),
        );
        fill(scene, bounds(440., 150., 100., 100.), gpui::white(), paint);
    }
}

#[test]
fn photos_draw_from_their_resident_tiles() {
    let fixture = Fixture::new();
    for (tier, mut renderer) in renderers() {
        let mut residency = SynchronousPhotoResidency::default();
        // The second frame draws from the tiles the first uploaded.
        for frame in ["first", "second"] {
            let image = render(&mut renderer, &mut residency, target(600, 360), |scene| {
                fixture.build(scene)
            });
            let mut failures = Failures::default();
            let what = |name: &str| format!("{tier} tier, {frame} frame, {name}");

            // Drawn from a level where each pixel averages the checkerboard: grey, not
            // black or white.
            failures.expect(&image, &what("checkerboard"), 30, 60, [128, 128, 128]);
            failures.expect(&image, &what("checkerboard"), 41, 97, [128, 128, 128]);
            failures.expect(&image, &what("pattern's red"), 100, 60, [255, 0, 0]);

            // The ramp about its tile edges at 256 and 512 photo pixels: device pixel x
            // shows photo pixel x + 60.
            for edge in [256u32, 512] {
                for photo_x in edge - 3..edge + 3 {
                    let x = photo_x - 60;
                    failures.expect(
                        &image,
                        &what("ramp across a tile edge"),
                        x,
                        16,
                        [(photo_x / 4) as u8, 0, 0],
                    );
                }
            }

            failures.expect(&image, &what("rotated photo, top"), 100, 180, [0, 0, 255]);
            failures.expect(
                &image,
                &what("rotated photo, bottom"),
                100,
                320,
                [0, 255, 0],
            );

            failures.expect(
                &image,
                &what("repeated photo, a red square"),
                220 + 8 + 32 * 3,
                150 + 8 + 32,
                [255, 0, 0],
            );
            failures.expect(
                &image,
                &what("repeated photo, transparent"),
                220 + 24 + 32 * 2,
                160,
                [255, 255, 255],
            );

            failures.expect(
                &image,
                &what("covering photo, left"),
                465,
                200,
                [0, 255, 255],
            );
            failures.expect(
                &image,
                &what("covering photo, right"),
                515,
                200,
                [255, 0, 255],
            );
            failures.assert_none();
        }
    }
}

/// 4096 pixels square, sixteen tiles each way, blue: red rising with x and green with y, by
/// one every sixteen pixels.
fn grid_photo() -> Photo {
    Photo::from_rgba(RgbaImage::from_fn(4096, 4096, |x, y| {
        Rgba([(x / 16) as u8, (y / 16) as u8, 255, 255])
    }))
}

/// An 8-pixel square at (8 `x`, 8 `y`) showing the middle of tile (`x`, `y`) of the grid
/// photo, a pixel to a pixel; only that tile is loaded.
fn grid_square(scene: &mut Scene, photo: &Photo, x: u32, y: u32) {
    let origin = (8. * x as f64, 8. * y as f64);
    let shown = (256. * x as f64 + 124., 256. * y as f64 + 124.);
    let paint = photo_paint(
        scene,
        photo,
        &peniko::ImageSampler::default(),
        kurbo::Affine::translate((origin.0 - shown.0, origin.1 - shown.1)),
        Some(kurbo::Rect::new(
            shown.0,
            shown.1,
            shown.0 + 8.,
            shown.1 + 8.,
        )),
    );
    fill(
        scene,
        bounds(origin.0 as f32, origin.1 as f32, 8., 8.),
        gpui::white(),
        paint,
    );
}

/// The first frame places a row of tiles; the second places the rest, more than a layer
/// holds, so the array grows, copying the first frame's tiles into the new one; the third
/// draws them all from the grown array, uploading nothing.
#[test]
fn the_photo_tile_array_grows_keeping_its_tiles() {
    let photo = grid_photo();
    let size = target(128, 128);
    for (tier, mut renderer) in renderers() {
        let mut residency = SynchronousPhotoResidency::default();
        let first = render(&mut renderer, &mut residency, size, |scene| {
            for x in 0..16 {
                grid_square(scene, &photo, x, 0);
            }
        });
        let mut failures = Failures::default();
        for x in 0..16 {
            failures.expect(
                &first,
                &format!("{tier} tier, first frame, tile ({x}, 0)"),
                8 * x + 4,
                4,
                [(16 * x + 8) as u8, 8, 255],
            );
        }
        for frame in ["grown", "after growing"] {
            let image = render(&mut renderer, &mut residency, size, |scene| {
                for y in 0..16 {
                    for x in 0..16 {
                        grid_square(scene, &photo, x, y);
                    }
                }
            });
            for y in 0..16 {
                for x in 0..16 {
                    failures.expect(
                        &image,
                        &format!("{tier} tier, {frame} frame, tile ({x}, {y})"),
                        8 * x + 4,
                        8 * y + 4,
                        [(16 * x + 8) as u8, (16 * y + 8) as u8, 255],
                    );
                }
            }
        }
        failures.assert_none();
    }
}
