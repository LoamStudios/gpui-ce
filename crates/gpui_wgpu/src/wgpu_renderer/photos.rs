//! The texture array a window's resident photo tiles are kept in.

use gpui::{PHOTO_LAYER_SIZE, Scene};

/// Photo tiles are premultiplied RGBA, eight bits a channel, sampled as they
/// are stored: not sRGB-decoded.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// The photo tile array, which grows a layer at a time as tiles are placed in
/// new layers, keeping what it holds, and the sampler that filters it.
pub(super) struct PhotoTiles {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    /// How many layers of tiles it has: 0 while it is a placeholder.
    layers: u32,
    /// The fewest layers it is made with. OpenGL makes a texture of one layer
    /// a plain 2D texture, which cannot be bound as an array, so the downlevel
    /// tier makes at least two.
    min_layers: u32,
}

impl PhotoTiles {
    pub(super) fn new(device: &wgpu::Device, tier: crate::RendererTier) -> Self {
        let min_layers = match tier {
            crate::RendererTier::Modern => 1,
            crate::RendererTier::WebGl2 => 2,
        };
        let (texture, view) = new_array(device, 1, min_layers);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("photo_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self {
            texture,
            view,
            sampler,
            layers: 0,
            min_layers,
        }
    }

    /// The whole array, viewed as one, as group 0 binds it.
    pub(super) fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub(super) fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// Copies the tiles `scene`'s photos have placed since the last frame into
    /// the array, growing it first if they need more layers. Returns whether
    /// the array was replaced, so the bind groups holding it must be made
    /// again.
    ///
    /// Growing copies the layers it has into the new array in a submission of
    /// its own; the tiles are written by the queue, which does so ahead of the
    /// next submission, the frame's, and after earlier ones, the growth's and
    /// earlier frames', which may still be reading the space they reuse.
    pub(super) fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
    ) -> bool {
        let Some(uploads) = &scene.photo_uploads else {
            return false;
        };
        let (layers, tiles) = uploads.take(self.layers);
        let mut replaced = false;
        if layers > self.layers {
            let limits = device.limits();
            if PHOTO_LAYER_SIZE > limits.max_texture_dimension_2d
                || layers.max(self.min_layers) > limits.max_texture_array_layers
            {
                log::error!(
                    "photo tiles need {layers} layers of {PHOTO_LAYER_SIZE} pixels square, \
                     more than the GPU allows: {} layers of {} pixels",
                    limits.max_texture_array_layers,
                    limits.max_texture_dimension_2d,
                );
                return false;
            }
            let (texture, view) = new_array(device, PHOTO_LAYER_SIZE, layers.max(self.min_layers));
            if self.layers > 0 {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("photo_tiles_growth"),
                });
                encoder.copy_texture_to_texture(
                    self.texture.as_image_copy(),
                    texture.as_image_copy(),
                    wgpu::Extent3d {
                        width: PHOTO_LAYER_SIZE,
                        height: PHOTO_LAYER_SIZE,
                        depth_or_array_layers: self.layers,
                    },
                );
                queue.submit([encoder.finish()]);
            }
            self.texture = texture;
            self.view = view;
            self.layers = layers;
            replaced = true;
        }
        for tile in &tiles {
            if tile.layer >= self.layers {
                continue;
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: tile.origin.x,
                        y: tile.origin.y,
                        z: tile.layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &tile.pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(tile.size.width * 4),
                    rows_per_image: Some(tile.size.height),
                },
                wgpu::Extent3d {
                    width: tile.size.width,
                    height: tile.size.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        replaced
    }
}

fn new_array(device: &wgpu::Device, side: u32, layers: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("photo_tiles"),
        size: wgpu::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    // A single layer would otherwise be viewed as a 2D texture.
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("photo_tiles_view"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}
