//! Photos: images drawn from a pyramid of levels, each half the size of the
//! one before, cut into tiles that are decoded off the main thread and kept
//! on the GPU only while they are on screen. A photo of any size draws as
//! one paint, at the level its size on screen calls for, so a page can hold
//! thousands of them.

use crate::{Bounds, Size, point, size};
use anyhow::{Context as _, Result};
use collections::FxHashMap;
use image::RgbaImage;
use parking_lot::Mutex;
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, OnceLock, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

mod residency;

pub(crate) use residency::PhotoResidency;
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub use residency::SynchronousPhotoResidency;
pub use residency::{PHOTO_MEMORY_BUDGET, PhotoTileUpload, PhotoUploads};

/// The side of a photo tile, in pixels of its level.
pub const PHOTO_TILE_SIZE: u32 = 256;

/// The pixels each tile repeats from its neighbours, or from its own edge at
/// the photo's, on every side, so that filtering never reaches outside it.
pub const PHOTO_TILE_GUTTER: u32 = 1;

/// The side of each layer of the texture array photo tiles are kept in.
pub const PHOTO_LAYER_SIZE: u32 = 4096;

/// Identifies a [`Photo`] for as long as the process runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhotoId(u64);

/// Where a photo's pixels come from: anything that can decode a region of
/// one of its levels.
///
/// Level 0 is the photo at full size, and each level after it is half the
/// size of the one before, rounded up, down to a single pixel: see
/// [`photo_level_size`]. A source is asked for regions of levels from
/// background threads, as the photo's tiles are needed.
pub trait PhotoSource: Send + Sync + 'static {
    /// The photo's size at level 0, in pixels.
    fn size(&self) -> Size<u32>;

    /// The pixels of `region` of `level`, as rows of premultiplied RGBA,
    /// eight bits a channel, `region.size.width` pixels long.
    fn decode(&self, level: u32, region: Bounds<u32>) -> Result<Vec<u8>>;
}

/// A photo, cheap to clone: draw it with
/// [`Window::photo`](crate::Window::photo) or
/// [`Window::paint_photo`](crate::Window::paint_photo).
#[derive(Clone)]
pub struct Photo(Arc<PhotoState>);

struct PhotoState {
    id: PhotoId,
    size: Size<u32>,
    source: Box<dyn PhotoSource>,
}

impl std::fmt::Debug for Photo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Photo")
            .field("id", &self.0.id)
            .field("size", &self.0.size)
            .finish()
    }
}

impl PartialEq for Photo {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

impl Eq for Photo {}

impl Photo {
    /// A photo whose pixels come from `source`.
    pub fn new(source: impl PhotoSource) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        Self(Arc::new(PhotoState {
            id: PhotoId(NEXT_ID.fetch_add(1, Ordering::Relaxed)),
            size: source.size(),
            source: Box::new(source),
        }))
    }

    /// A photo of `image`, whose pixels are not premultiplied. Its smaller
    /// levels are made from it when first needed.
    pub fn from_rgba(image: RgbaImage) -> Self {
        Self::new(PixelPhoto::new(image))
    }

    /// The photo in the image file at `path`, decoded when its pixels are
    /// first needed. Only its size is read now.
    ///
    /// Its levels are kept in memory while it is drawn, within a budget all
    /// such photos share, and decoded again if they were let go.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let (width, height) = image::image_dimensions(&path)
            .with_context(|| format!("reading the size of {}", path.display()))?;
        Ok(Self::new(FilePhoto {
            path,
            size: size(width, height),
            levels: Mutex::new(Weak::new()),
        }))
    }

    /// The photo's identity.
    pub fn id(&self) -> PhotoId {
        self.0.id
    }

    /// The photo's size at level 0, in pixels.
    pub fn size(&self) -> Size<u32> {
        self.0.size
    }

    /// How many levels the photo has: down to one pixel.
    pub fn level_count(&self) -> u32 {
        photo_level_count(self.0.size)
    }

    /// The pixels of tile (`x`, `y`) of `level`, with its gutter: rows of
    /// premultiplied RGBA, and their size.
    pub(crate) fn decode_tile(&self, level: u32, x: u32, y: u32) -> Result<(Size<u32>, Vec<u8>)> {
        let level_size = photo_level_size(self.0.size, level);
        let content = photo_tile_bounds(level_size, x, y);
        let left = content.origin.x.saturating_sub(PHOTO_TILE_GUTTER);
        let top = content.origin.y.saturating_sub(PHOTO_TILE_GUTTER);
        let right =
            (content.origin.x + content.size.width + PHOTO_TILE_GUTTER).min(level_size.width);
        let bottom =
            (content.origin.y + content.size.height + PHOTO_TILE_GUTTER).min(level_size.height);
        let region = Bounds {
            origin: point(left, top),
            size: size(right - left, bottom - top),
        };
        let pixels = self.0.source.decode(level, region)?;
        anyhow::ensure!(
            pixels.len() == (region.size.width * region.size.height * 4) as usize,
            "a photo source returned {} bytes for a {}×{} region",
            pixels.len(),
            region.size.width,
            region.size.height
        );

        // Lay the region out in the tile, repeating its edge into the
        // gutter where the photo ends.
        let tile_size = size(
            content.size.width + 2 * PHOTO_TILE_GUTTER,
            content.size.height + 2 * PHOTO_TILE_GUTTER,
        );
        let mut tile = vec![0u8; (tile_size.width * tile_size.height * 4) as usize];
        let tile_origin = point(
            content.origin.x as i64 - PHOTO_TILE_GUTTER as i64,
            content.origin.y as i64 - PHOTO_TILE_GUTTER as i64,
        );
        for row in 0..tile_size.height {
            let source_y =
                (tile_origin.y + row as i64).clamp(top as i64, bottom as i64 - 1) as u32 - top;
            for column in 0..tile_size.width {
                let source_x = (tile_origin.x + column as i64).clamp(left as i64, right as i64 - 1)
                    as u32
                    - left;
                let from = ((source_y * region.size.width + source_x) * 4) as usize;
                let to = ((row * tile_size.width + column) * 4) as usize;
                tile[to..to + 4].copy_from_slice(&pixels[from..from + 4]);
            }
        }
        Ok((tile_size, tile))
    }
}

/// The size of `level` of a photo `size` at level 0: halved `level` times,
/// rounding up.
pub fn photo_level_size(size: Size<u32>, level: u32) -> Size<u32> {
    let halve = |length: u32| {
        if level >= 32 {
            1
        } else {
            (length.div_ceil(1 << level)).max(1)
        }
    };
    crate::size(halve(size.width), halve(size.height))
}

/// How many levels a photo `size` at level 0 has: down to one pixel.
pub fn photo_level_count(size: Size<u32>) -> u32 {
    let longest = size.width.max(size.height).max(1);
    32 - (longest - 1).leading_zeros() + 1
}

/// The pixels of `level`, `level_size`, that tile (`x`, `y`) holds, without
/// its gutter.
pub fn photo_tile_bounds(level_size: Size<u32>, x: u32, y: u32) -> Bounds<u32> {
    let left = x * PHOTO_TILE_SIZE;
    let top = y * PHOTO_TILE_SIZE;
    Bounds {
        origin: point(left, top),
        size: size(
            PHOTO_TILE_SIZE.min(level_size.width.saturating_sub(left)),
            PHOTO_TILE_SIZE.min(level_size.height.saturating_sub(top)),
        ),
    }
}

/// The levels of an image: level 0 premultiplied, then each made from the
/// one before by averaging squares of four pixels.
fn pyramid(mut image: RgbaImage) -> Vec<RgbaImage> {
    for pixel in image.pixels_mut() {
        let alpha = pixel.0[3] as u32;
        for channel in &mut pixel.0[..3] {
            *channel = ((*channel as u32 * alpha + 127) / 255) as u8;
        }
    }
    let count = photo_level_count(size(image.width(), image.height()));
    let mut levels = Vec::with_capacity(count as usize);
    levels.push(image);
    for _ in 1..count {
        let previous = levels.last().expect("level 0 is pushed first");
        let (width, height) = (previous.width(), previous.height());
        let next = RgbaImage::from_fn(width.div_ceil(2), height.div_ceil(2), |x, y| {
            let mut sum = [0u32; 4];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let pixel =
                    previous.get_pixel((2 * x + dx).min(width - 1), (2 * y + dy).min(height - 1));
                for (total, channel) in sum.iter_mut().zip(pixel.0) {
                    *total += channel as u32;
                }
            }
            image::Rgba(sum.map(|total| ((total + 2) / 4) as u8))
        });
        levels.push(next);
    }
    levels
}

/// The pixels of `region` of `levels[level]`, as rows.
fn region_of(levels: &[RgbaImage], level: u32, region: Bounds<u32>) -> Result<Vec<u8>> {
    let image = levels
        .get(level as usize)
        .with_context(|| format!("a photo has no level {level}"))?;
    anyhow::ensure!(
        region.origin.x + region.size.width <= image.width()
            && region.origin.y + region.size.height <= image.height(),
        "region {region:?} is outside level {level}"
    );
    let row_bytes = (region.size.width * 4) as usize;
    let mut pixels = Vec::with_capacity(row_bytes * region.size.height as usize);
    for y in region.origin.y..region.origin.y + region.size.height {
        let start = ((y * image.width() + region.origin.x) * 4) as usize;
        pixels.extend_from_slice(&image.as_raw()[start..start + row_bytes]);
    }
    Ok(pixels)
}

/// A photo of pixels in memory.
struct PixelPhoto {
    image: Mutex<Option<RgbaImage>>,
    size: Size<u32>,
    levels: OnceLock<Vec<RgbaImage>>,
}

impl PixelPhoto {
    fn new(image: RgbaImage) -> Self {
        Self {
            size: size(image.width(), image.height()),
            image: Mutex::new(Some(image)),
            levels: OnceLock::new(),
        }
    }
}

impl PhotoSource for PixelPhoto {
    fn size(&self) -> Size<u32> {
        self.size
    }

    fn decode(&self, level: u32, region: Bounds<u32>) -> Result<Vec<u8>> {
        let levels = self.levels.get_or_init(|| {
            let image = self
                .image
                .lock()
                .take()
                .unwrap_or_else(|| RgbaImage::new(self.size.width, self.size.height));
            pyramid(image)
        });
        region_of(levels, level, region)
    }
}

/// A photo in an image file, whose levels are decoded when first needed and
/// kept while [`DECODED_FILE_PHOTOS`] holds them.
struct FilePhoto {
    path: PathBuf,
    size: Size<u32>,
    /// The decoded levels, while they are held. Locked while decoding, so the
    /// file is decoded once however many tiles ask for it.
    levels: Mutex<Weak<Vec<RgbaImage>>>,
}

impl FilePhoto {
    fn levels(&self) -> Result<Arc<Vec<RgbaImage>>> {
        let mut held = self.levels.lock();
        if let Some(levels) = held.upgrade() {
            DECODED_FILE_PHOTOS.touch(&self.path);
            return Ok(levels);
        }
        let image = decode_file(&self.path)?;
        let levels = Arc::new(pyramid(image));
        *held = Arc::downgrade(&levels);
        DECODED_FILE_PHOTOS.hold(self.path.clone(), levels.clone());
        Ok(levels)
    }
}

fn decode_file(path: &Path) -> Result<RgbaImage> {
    Ok(image::ImageReader::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("decoding {}", path.display()))?
        .into_rgba8())
}

impl PhotoSource for FilePhoto {
    fn size(&self) -> Size<u32> {
        self.size
    }

    fn decode(&self, level: u32, region: Bounds<u32>) -> Result<Vec<u8>> {
        region_of(&self.levels()?, level, region)
    }
}

/// The decoded levels of file photos, the most recently used first, within
/// [`DecodedFilePhotos::BUDGET`] bytes.
static DECODED_FILE_PHOTOS: DecodedFilePhotos = DecodedFilePhotos {
    held: Mutex::new(None),
};

struct DecodedFilePhotos {
    held: Mutex<Option<HeldFilePhotos>>,
}

#[derive(Default)]
struct HeldFilePhotos {
    levels: FxHashMap<PathBuf, (Arc<Vec<RgbaImage>>, u64)>,
    bytes: usize,
    clock: u64,
}

impl DecodedFilePhotos {
    /// The most memory decoded file photos are kept in.
    const BUDGET: usize = 1 << 30;

    fn hold(&self, path: PathBuf, levels: Arc<Vec<RgbaImage>>) {
        let mut held = self.held.lock();
        let held = held.get_or_insert_with(HeldFilePhotos::default);
        held.clock += 1;
        held.bytes += levels
            .iter()
            .map(|level| level.as_raw().len())
            .sum::<usize>();
        if let Some((previous, _)) = held.levels.insert(path, (levels, held.clock)) {
            held.bytes -= previous
                .iter()
                .map(|level| level.as_raw().len())
                .sum::<usize>();
        }
        while held.bytes > Self::BUDGET && held.levels.len() > 1 {
            let Some(oldest) = held
                .levels
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(path, _)| path.clone())
            else {
                break;
            };
            if let Some((levels, _)) = held.levels.remove(&oldest) {
                held.bytes -= levels
                    .iter()
                    .map(|level| level.as_raw().len())
                    .sum::<usize>();
            }
        }
    }

    fn touch(&self, path: &Path) {
        let mut held = self.held.lock();
        let Some(held) = held.as_mut() else {
            return;
        };
        held.clock += 1;
        let clock = held.clock;
        if let Some((_, used)) = held.levels.get_mut(path) {
            *used = clock;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_halve_rounding_up_down_to_a_pixel() {
        let photo = size(1000, 600);
        assert_eq!(photo_level_size(photo, 0), size(1000, 600));
        assert_eq!(photo_level_size(photo, 1), size(500, 300));
        assert_eq!(photo_level_size(photo, 3), size(125, 75));
        assert_eq!(photo_level_size(photo, 4), size(63, 38));
        assert_eq!(photo_level_count(photo), 11);
        assert_eq!(photo_level_size(photo, 10), size(1, 1));
        assert_eq!(photo_level_count(size(1, 1)), 1);
        assert_eq!(photo_level_count(size(256, 1)), 9);
    }

    #[test]
    fn tiles_repeat_their_neighbours_and_the_photos_edge_in_their_gutter() {
        // Each pixel's red is its x and green its y.
        let image = RgbaImage::from_fn(300, 10, |x, y| image::Rgba([x as u8, y as u8, 0, 255]));
        let photo = Photo::from_rgba(image);

        let (first_size, first) = photo.decode_tile(0, 0, 0).unwrap();
        assert_eq!(first_size, size(258, 12));
        let at = |pixels: &[u8], width: u32, x: u32, y: u32| {
            let index = ((y * width + x) * 4) as usize;
            [pixels[index], pixels[index + 1]]
        };
        // The gutter repeats the photo's edge above and to the left...
        assert_eq!(at(&first, 258, 0, 0), [0, 0]);
        assert_eq!(at(&first, 258, 1, 1), [0, 0]);
        // ...and the next tile's first column to the right.
        assert_eq!(at(&first, 258, 257, 5), [0, 4]);
        assert_eq!(at(&first, 258, 256, 5), [255, 4]);

        let (second_size, second) = photo.decode_tile(0, 1, 0).unwrap();
        assert_eq!(second_size, size(46, 12));
        // Its gutter on the left is the previous tile's last column.
        assert_eq!(at(&second, 46, 0, 1), [255, 0]);
        assert_eq!(at(&second, 46, 45, 11), [43, 9]);
    }

    #[test]
    fn smaller_levels_average_premultiplied_pixels() {
        let image = RgbaImage::from_fn(2, 2, |x, _| {
            if x == 0 {
                image::Rgba([255, 0, 0, 255])
            } else {
                image::Rgba([0, 255, 0, 0])
            }
        });
        let levels = pyramid(image);
        assert_eq!(levels.len(), 2);
        // The transparent green adds nothing but transparency.
        assert_eq!(levels[1].get_pixel(0, 0).0, [128, 0, 0, 128]);
    }
}
