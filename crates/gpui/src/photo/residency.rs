//! Which photo tiles a window keeps on the GPU: those its frames draw, decoded
//! off the main thread when first needed, and let go of, least recently
//! drawn first, when the window's photo memory is full.

use super::{
    PHOTO_LAYER_SIZE, PHOTO_TILE_GUTTER, PHOTO_TILE_SIZE, Photo, PhotoId, photo_level_size,
};
use crate::{
    AnyWindowHandle, App, AppContext as _, DevicePixels, PaintExtend, PaintWord, PhotoPaint, Point,
    Scene, ScenePaint, Size, point,
};
use anyhow::Result;
use collections::{FxHashMap, FxHashSet};
use etagere::{AllocId, AtlasAllocator};
use parking_lot::Mutex;
use std::sync::Arc;

/// The most texture memory a window keeps photo tiles in.
pub const PHOTO_MEMORY_BUDGET: usize = 512 << 20;

/// The most tiles one photo paint draws from: a photo that would need more
/// is drawn from a coarser level.
const MAX_TILES_PER_PAINT: u32 = 1024;

/// Tile pixels a window's renderer copies into its photo texture array before
/// it next draws, in order, and how many layers the array needs.
#[derive(Default)]
pub struct PhotoUploads {
    state: Mutex<PhotoUploadState>,
}

#[derive(Default)]
struct PhotoUploadState {
    layers: u32,
    tiles: Vec<PhotoTileUpload>,
}

/// A tile to copy into a layer of the photo texture array: rows of
/// premultiplied RGBA, eight bits a channel, `size.width` pixels long, with
/// their top left at `origin`.
pub struct PhotoTileUpload {
    #[allow(missing_docs)]
    pub layer: u32,
    #[allow(missing_docs)]
    pub origin: Point<u32>,
    #[allow(missing_docs)]
    pub size: Size<u32>,
    #[allow(missing_docs)]
    pub pixels: Vec<u8>,
}

impl PhotoUploads {
    /// How many layers of [`PHOTO_LAYER_SIZE`] pixels square the photo
    /// texture array needs, and the tiles to copy into it, which are taken.
    /// The array keeps what was copied into it as it grows.
    pub fn take(&self) -> (u32, Vec<PhotoTileUpload>) {
        let mut state = self.state.lock();
        (state.layers, std::mem::take(&mut state.tiles))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TileKey {
    photo: PhotoId,
    level: u32,
    x: u32,
    y: u32,
}

struct ResidentTile {
    layer: u32,
    allocation: AllocId,
    /// Where its pixels start in its layer, inside its gutter.
    origin: Point<u32>,
    /// The frame that last drew from it.
    last_used: u64,
}

type DecodedTile = (TileKey, Result<(Size<u32>, Vec<u8>)>);

/// A window's photo tiles: where each resident one is, which are being
/// decoded, and what its renderer has yet to copy.
pub(crate) struct PhotoResidency {
    layers: Vec<AtlasAllocator>,
    max_layers: u32,
    tiles: FxHashMap<TileKey, ResidentTile>,
    pending: FxHashSet<TileKey>,
    failed: FxHashSet<TileKey>,
    decoded: Arc<Mutex<Vec<DecodedTile>>>,
    uploads: Arc<PhotoUploads>,
    frame: u64,
}

impl Default for PhotoResidency {
    fn default() -> Self {
        let layer_bytes = (PHOTO_LAYER_SIZE as usize).pow(2) * 4;
        Self {
            layers: Vec::new(),
            max_layers: (PHOTO_MEMORY_BUDGET / layer_bytes).max(1) as u32,
            tiles: FxHashMap::default(),
            pending: FxHashSet::default(),
            failed: FxHashSet::default(),
            decoded: Arc::default(),
            uploads: Arc::default(),
            frame: 0,
        }
    }
}

/// A tile that has been asked for, and the photo it is cut from.
struct Request {
    key: TileKey,
    photo: Photo,
}

impl PhotoResidency {
    /// Chooses the level each of `scene`'s photo paints draws from, for the
    /// size it appears at in a viewport `viewport_size`, and fills in the
    /// tiles that cover what is in view. Tiles not yet resident are drawn
    /// from coarser ones that are, and decoded in the background; `window`
    /// is drawn again as each arrives.
    pub(crate) fn prepare(
        &mut self,
        scene: &mut Scene,
        viewport_size: Size<DevicePixels>,
        window: AnyWindowHandle,
        cx: &App,
    ) {
        self.frame += 1;
        self.place_decoded();
        scene.photo_uploads = Some(self.uploads.clone());
        if scene.photo_paints().is_empty() {
            return;
        }
        let mut requests = Vec::new();
        for paint in scene.photo_paints().to_vec() {
            self.resolve(scene, &paint, viewport_size, &mut requests);
        }
        self.request(requests, window, cx);
    }

    /// Places the tiles decoded since the last frame, to be uploaded before
    /// the next draw.
    fn place_decoded(&mut self) {
        let decoded = std::mem::take(&mut *self.decoded.lock());
        for (key, result) in decoded {
            self.pending.remove(&key);
            let (size, pixels) = match result {
                Ok(tile) => tile,
                Err(error) => {
                    log::error!("failed to decode a photo tile: {error:#}");
                    self.failed.insert(key);
                    continue;
                }
            };
            if self.tiles.contains_key(&key) {
                continue;
            }
            let Some((layer, allocation)) = self.allocate(size) else {
                continue;
            };
            let origin = point(
                allocation.rectangle.min.x as u32,
                allocation.rectangle.min.y as u32,
            );
            self.tiles.insert(
                key,
                ResidentTile {
                    layer,
                    allocation: allocation.id,
                    origin: point(origin.x + PHOTO_TILE_GUTTER, origin.y + PHOTO_TILE_GUTTER),
                    last_used: self.frame,
                },
            );
            let mut uploads = self.uploads.state.lock();
            uploads.layers = self.layers.len() as u32;
            uploads.tiles.push(PhotoTileUpload {
                layer,
                origin,
                size,
                pixels,
            });
        }
    }

    /// Space for a tile `size`: in a layer with room, a new layer within the
    /// budget, or space let go of by tiles the last two frames didn't draw.
    fn allocate(&mut self, size: Size<u32>) -> Option<(u32, etagere::Allocation)> {
        let size = etagere::size2(size.width as i32, size.height as i32);
        for (layer, allocator) in self.layers.iter_mut().enumerate() {
            if let Some(allocation) = allocator.allocate(size) {
                return Some((layer as u32, allocation));
            }
        }
        if (self.layers.len() as u32) < self.max_layers {
            let side = PHOTO_LAYER_SIZE as i32;
            let mut allocator = AtlasAllocator::new(etagere::size2(side, side));
            let allocation = allocator.allocate(size)?;
            self.layers.push(allocator);
            return Some((self.layers.len() as u32 - 1, allocation));
        }

        let recent = self.frame.saturating_sub(1);
        let mut idle: Vec<(u64, TileKey)> = self
            .tiles
            .iter()
            .filter(|(_, tile)| tile.last_used < recent)
            .map(|(key, tile)| (tile.last_used, *key))
            .collect();
        idle.sort_unstable_by_key(|(last_used, _)| *last_used);
        for batch in idle.chunks(16) {
            for (_, key) in batch {
                if let Some(tile) = self.tiles.remove(key) {
                    self.layers[tile.layer as usize].deallocate(tile.allocation);
                }
            }
            for (layer, allocator) in self.layers.iter_mut().enumerate() {
                if let Some(allocation) = allocator.allocate(size) {
                    return Some((layer as u32, allocation));
                }
            }
        }
        None
    }

    /// Chooses `paint`'s level, fills in its tile words, and asks for the
    /// tiles it lacks.
    fn resolve(
        &mut self,
        scene: &mut Scene,
        paint: &PhotoPaint,
        viewport_size: Size<DevicePixels>,
        requests: &mut Vec<Request>,
    ) {
        let entry = ScenePaint::from_words(&scene.paint_table()[paint.index as usize..]);
        let photo_size = paint.photo.size();
        let last_level = paint.photo.level_count() - 1;
        let id = paint.photo.id();

        // Photo pixels per viewport pixel: the level where they are about one
        // to one, or a little more.
        let [[a, b], [c, d]] = entry.transformation.rotation_scale;
        let texels_per_pixel = (a * d - b * c).abs().sqrt();
        let mut level = if texels_per_pixel > 1. {
            (texels_per_pixel.log2().floor() as u32).min(last_level)
        } else {
            0
        };

        // What of level 0 is in view.
        let corners = [
            (0., 0.),
            (viewport_size.width.0 as f32, 0.),
            (0., viewport_size.height.0 as f32),
            (viewport_size.width.0 as f32, viewport_size.height.0 as f32),
        ]
        .map(|(x, y)| {
            let corner = entry
                .transformation
                .apply(point(crate::px(x), crate::px(y)));
            (corner.x.0, corner.y.0)
        });
        let range = |extend: PaintExtend, length: u32, coordinate: fn(&(f32, f32)) -> f32| {
            if extend != PaintExtend::Pad {
                return (0., length as f32);
            }
            let low = corners.iter().map(coordinate).fold(f32::INFINITY, f32::min);
            let high = corners
                .iter()
                .map(coordinate)
                .fold(f32::NEG_INFINITY, f32::max);
            (low.max(0.), high.min(length as f32))
        };
        let (mut left, mut right) = range(entry.extend, photo_size.width, |corner| corner.0);
        let (mut top, mut bottom) = range(entry.y_extend, photo_size.height, |corner| corner.1);
        if let Some(region) = paint.region {
            left = left.max(region.x0 as f32);
            top = top.max(region.y0 as f32);
            right = right.min(region.x1 as f32);
            bottom = bottom.min(region.y1 as f32);
        }
        if right <= left || bottom <= top {
            scene.place_photo_tiles(paint.index, level, (0, 0), (0, 0), &[]);
            return;
        }

        let (first, count) = loop {
            let scale = (1u32 << level) as f32 * PHOTO_TILE_SIZE as f32;
            let tiles =
                photo_level_size(photo_size, level).map(|length| length.div_ceil(PHOTO_TILE_SIZE));
            let first_x = ((left / scale).floor() as u32).min(tiles.width - 1);
            let first_y = ((top / scale).floor() as u32).min(tiles.height - 1);
            let end_x = ((right / scale).ceil() as u32).clamp(first_x + 1, tiles.width);
            let end_y = ((bottom / scale).ceil() as u32).clamp(first_y + 1, tiles.height);
            let count = (end_x - first_x, end_y - first_y);
            if count.0 * count.1 <= MAX_TILES_PER_PAINT || level == last_level {
                break ((first_x, first_y), count);
            }
            level += 1;
        };

        let mut words: Vec<PaintWord> = Vec::with_capacity((count.0 * count.1) as usize);
        for y in first.1..first.1 + count.1 {
            for x in first.0..first.0 + count.0 {
                let key = TileKey {
                    photo: id,
                    level,
                    x,
                    y,
                };
                let mut word = [-1., 0., 0., 0.];
                for offset in 0..=last_level - level {
                    let covering = TileKey {
                        photo: id,
                        level: level + offset,
                        x: x >> offset,
                        y: y >> offset,
                    };
                    if let Some(tile) = self.tiles.get_mut(&covering) {
                        tile.last_used = self.frame;
                        word = [
                            tile.layer as f32,
                            tile.origin.x as f32,
                            tile.origin.y as f32,
                            offset as f32,
                        ];
                        if offset > 0 {
                            self.want(key, &paint.photo, requests);
                        }
                        break;
                    }
                }
                if word[0] < 0. {
                    self.want(key, &paint.photo, requests);
                }
                words.push(word);
            }
        }

        // The coarsest level that is a single tile, to draw from until finer
        // tiles arrive.
        let single_tile = (0..=last_level)
            .find(|&level| {
                let size = photo_level_size(photo_size, level);
                size.width <= PHOTO_TILE_SIZE && size.height <= PHOTO_TILE_SIZE
            })
            .unwrap_or(last_level);
        if single_tile > level {
            let key = TileKey {
                photo: id,
                level: single_tile,
                x: 0,
                y: 0,
            };
            if let Some(tile) = self.tiles.get_mut(&key) {
                tile.last_used = self.frame;
            } else {
                self.want(key, &paint.photo, requests);
            }
        }

        scene.place_photo_tiles(paint.index, level, first, count, &words);
    }

    fn want(&self, key: TileKey, photo: &Photo, requests: &mut Vec<Request>) {
        if !self.pending.contains(&key) && !self.failed.contains(&key) {
            requests.push(Request {
                key,
                photo: photo.clone(),
            });
        }
    }

    /// Starts decoding the tiles asked for, coarsest first, as many at a
    /// time as there are threads to decode them.
    fn request(&mut self, mut requests: Vec<Request>, window: AnyWindowHandle, cx: &App) {
        let max_pending = cx.background_executor().num_cpus().max(1) * 2;
        requests.sort_by_key(|request| std::cmp::Reverse(request.key.level));
        for Request { key, photo } in requests {
            if self.pending.len() >= max_pending {
                break;
            }
            if !self.pending.insert(key) {
                continue;
            }
            let decoded = self.decoded.clone();
            let decode = cx
                .background_executor()
                .spawn(async move { photo.decode_tile(key.level, key.x, key.y) });
            cx.spawn(async move |cx| {
                let tile = decode.await;
                decoded.lock().push((key, tile));
                cx.update_window(window, |_, window, _| window.refresh())
                    .ok();
            })
            .detach();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photo_level_count;

    #[test]
    fn a_full_layer_lets_go_of_tiles_not_drawn_lately() {
        let mut residency = PhotoResidency {
            max_layers: 1,
            ..PhotoResidency::default()
        };
        let tile = Size {
            width: PHOTO_TILE_SIZE + 2,
            height: PHOTO_TILE_SIZE + 2,
        };
        let mut placed = 0;
        while let Some((layer, allocation)) = residency.allocate(tile) {
            residency.tiles.insert(
                TileKey {
                    photo: PhotoId(0),
                    level: 0,
                    x: placed,
                    y: 0,
                },
                ResidentTile {
                    layer,
                    allocation: allocation.id,
                    origin: point(0, 0),
                    last_used: residency.frame,
                },
            );
            placed += 1;
        }
        assert!(
            placed >= 200,
            "a layer holds 200 full tiles or more, not {placed}"
        );

        // Two frames later, the tiles are idle, and make room.
        residency.frame += 2;
        assert!(residency.allocate(tile).is_some());
        assert!(residency.tiles.len() < placed as usize);
    }

    #[test]
    fn levels_count_down_to_a_single_pixel() {
        assert_eq!(
            photo_level_count(Size {
                width: 4096,
                height: 2048
            }),
            13
        );
    }
}
