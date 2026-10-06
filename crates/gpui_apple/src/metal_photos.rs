//! The texture array a window's resident photo tiles are kept in.

use gpui::{PHOTO_LAYER_SIZE, Scene};
use metal::{MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize};

/// The photo tile array, which grows a layer at a time as tiles are placed in
/// new layers, keeping what it holds.
///
/// Tiles are written into it from the CPU before the frame that first draws
/// them is committed, so no GPU work waits for them; writing with the GPU
/// would make each frame that adds tiles wait for the frames before it to
/// stop reading the array. Space is written only once no frame in flight
/// reads it (see `PhotoResidency`).
pub(crate) struct PhotoTiles {
    texture: metal::Texture,
    /// How many layers of tiles it has: 0 while it is a placeholder.
    layers: u32,
    storage_mode: metal::MTLStorageMode,
}

impl PhotoTiles {
    /// The array for `device`: in memory the CPU and GPU share, or, where
    /// they have their own, in memory Metal keeps in step for both.
    pub(crate) fn new(device: &metal::DeviceRef, is_unified_memory: bool) -> Self {
        let storage_mode = if is_unified_memory {
            metal::MTLStorageMode::Shared
        } else {
            metal::MTLStorageMode::Managed
        };
        Self {
            texture: new_array(device, 1, 1, storage_mode),
            layers: 0,
            storage_mode,
        }
    }

    pub(crate) fn texture(&self) -> &metal::Texture {
        &self.texture
    }

    /// Writes the tiles `scene`'s photos have placed since the last frame into
    /// the array, growing it first if they need more layers.
    pub(crate) fn upload(&mut self, device: &metal::DeviceRef, scene: &Scene) {
        let Some(uploads) = &scene.photo_uploads else {
            return;
        };
        let (layers, tiles) = uploads.take();
        if layers > self.layers {
            let texture = new_array(device, PHOTO_LAYER_SIZE, layers, self.storage_mode);
            let side = PHOTO_LAYER_SIZE as u64;
            let region = MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width: side,
                    height: side,
                    depth: 1,
                },
            };
            let row_bytes = side * 4;
            let mut pixels = vec![0u8; (row_bytes * side) as usize];
            for layer in 0..self.layers as u64 {
                self.texture.get_bytes_in_slice(
                    pixels.as_mut_ptr().cast(),
                    row_bytes,
                    row_bytes * side,
                    region,
                    0,
                    layer,
                );
                texture.replace_region_in_slice(
                    region,
                    0,
                    layer,
                    pixels.as_ptr().cast(),
                    row_bytes,
                    row_bytes * side,
                );
            }
            self.texture = texture;
            self.layers = layers;
        }
        for tile in &tiles {
            let row_bytes = tile.size.width as u64 * 4;
            self.texture.replace_region_in_slice(
                MTLRegion {
                    origin: MTLOrigin {
                        x: tile.origin.x as u64,
                        y: tile.origin.y as u64,
                        z: 0,
                    },
                    size: MTLSize {
                        width: tile.size.width as u64,
                        height: tile.size.height as u64,
                        depth: 1,
                    },
                },
                0,
                tile.layer as u64,
                tile.pixels.as_ptr().cast(),
                row_bytes,
                row_bytes * tile.size.height as u64,
            );
        }
    }
}

fn new_array(
    device: &metal::DeviceRef,
    side: u32,
    layers: u32,
    storage_mode: metal::MTLStorageMode,
) -> metal::Texture {
    let descriptor = metal::TextureDescriptor::new();
    descriptor.set_texture_type(metal::MTLTextureType::D2Array);
    descriptor.set_width(side as u64);
    descriptor.set_height(side as u64);
    descriptor.set_array_length(layers.max(1) as u64);
    descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    descriptor.set_storage_mode(storage_mode);
    descriptor.set_usage(metal::MTLTextureUsage::ShaderRead);
    device.new_texture(&descriptor)
}
