//! The texture array a window's resident photo tiles are kept in.

use gpui::{PHOTO_LAYER_SIZE, Scene};
use metal::{MTLOrigin, MTLPixelFormat, MTLSize};

/// The photo tile array, which grows a layer at a time as tiles are placed in
/// new layers, keeping what it holds.
pub(crate) struct PhotoTiles {
    texture: metal::Texture,
    /// How many layers of tiles it has: 0 while it is a placeholder.
    layers: u32,
}

impl PhotoTiles {
    pub(crate) fn new(device: &metal::DeviceRef) -> Self {
        Self {
            texture: new_array(device, 1, 1),
            layers: 0,
        }
    }

    pub(crate) fn texture(&self) -> &metal::Texture {
        &self.texture
    }

    /// Copies the tiles `scene`'s photos have placed since the last frame into
    /// the array, growing it first if they need more layers. The copies are
    /// encoded in `command_buffer`, ahead of its draws and after earlier
    /// frames', which may still be reading the space they reuse.
    pub(crate) fn upload(
        &mut self,
        device: &metal::DeviceRef,
        scene: &Scene,
        command_buffer: &metal::CommandBufferRef,
    ) {
        let Some(uploads) = &scene.photo_uploads else {
            return;
        };
        let (layers, tiles) = uploads.take();
        if layers <= self.layers && tiles.is_empty() {
            return;
        }
        let blit = command_buffer.new_blit_command_encoder();
        if layers > self.layers {
            let texture = new_array(device, PHOTO_LAYER_SIZE, layers);
            let side = PHOTO_LAYER_SIZE as u64;
            for layer in 0..self.layers as u64 {
                blit.copy_from_texture(
                    &self.texture,
                    layer,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                    MTLSize {
                        width: side,
                        height: side,
                        depth: 1,
                    },
                    &texture,
                    layer,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                );
            }
            self.texture = texture;
            self.layers = layers;
        }
        if !tiles.is_empty() {
            let bytes: usize = tiles.iter().map(|tile| tile.pixels.len()).sum();
            let staging = device.new_buffer(
                bytes as u64,
                metal::MTLResourceOptions::StorageModeShared
                    | metal::MTLResourceOptions::CPUCacheModeWriteCombined,
            );
            let mut offset = 0usize;
            for tile in &tiles {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        tile.pixels.as_ptr(),
                        (staging.contents() as *mut u8).add(offset),
                        tile.pixels.len(),
                    );
                }
                let row_bytes = tile.size.width as u64 * 4;
                blit.copy_from_buffer_to_texture(
                    &staging,
                    offset as u64,
                    row_bytes,
                    row_bytes * tile.size.height as u64,
                    MTLSize {
                        width: tile.size.width as u64,
                        height: tile.size.height as u64,
                        depth: 1,
                    },
                    &self.texture,
                    tile.layer as u64,
                    0,
                    MTLOrigin {
                        x: tile.origin.x as u64,
                        y: tile.origin.y as u64,
                        z: 0,
                    },
                    metal::MTLBlitOption::empty(),
                );
                offset += tile.pixels.len();
            }
        }
        blit.end_encoding();
    }
}

fn new_array(device: &metal::DeviceRef, side: u32, layers: u32) -> metal::Texture {
    let descriptor = metal::TextureDescriptor::new();
    descriptor.set_texture_type(metal::MTLTextureType::D2Array);
    descriptor.set_width(side as u64);
    descriptor.set_height(side as u64);
    descriptor.set_array_length(layers.max(1) as u64);
    descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    descriptor.set_storage_mode(metal::MTLStorageMode::Private);
    descriptor.set_usage(metal::MTLTextureUsage::ShaderRead);
    device.new_texture(&descriptor)
}
