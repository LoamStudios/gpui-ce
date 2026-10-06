//! The textures a frame renders groups and blurs into, pooled across frames.

use gpui::{DevicePixels, Size};

/// A colour texture a frame draws into and samples from.
#[derive(Clone)]
pub(super) struct PooledTexture {
    pub(super) texture: wgpu::Texture,
    pub(super) view: wgpu::TextureView,
}

impl PooledTexture {
    fn bytes(&self) -> usize {
        self.texture.width() as usize * self.texture.height() as usize * BYTES_PER_TEXEL
    }
}

/// The renderer's colour formats are 8-bit RGBA or BGRA.
const BYTES_PER_TEXEL: usize = 4;

/// Render targets for isolated groups, the blurs of groups and backdrops, and
/// copies of what is beneath a blended group, reused across frames. A
/// texture is taken while it is drawn into and sampled, and given back once
/// the passes that use it are encoded: later passes in the same command
/// encoder may draw into it again, since wgpu orders them.
pub(super) struct TexturePool {
    free: Vec<(PooledTexture, u64)>,
    /// The bytes of the textures taken and not yet given back.
    taken_bytes: usize,
    frame: u64,
}

impl TexturePool {
    /// The most texture memory the pool lends at once: a group that would
    /// take more is drawn in place instead, and a blur that would is skipped.
    const BUDGET: usize = 512 << 20;
    /// How many frames a free texture is kept for unused.
    const KEPT_FRAMES: u64 = 60;

    pub(super) fn new() -> Self {
        Self {
            free: Vec::new(),
            taken_bytes: 0,
            frame: 0,
        }
    }

    /// A texture of exactly `size` in `format`, or `None` if lending one
    /// would pass the budget.
    pub(super) fn take(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        size: Size<DevicePixels>,
    ) -> Option<PooledTexture> {
        let width = size.width.0.max(1) as u32;
        let height = size.height.0.max(1) as u32;
        let bytes = width as usize * height as usize * BYTES_PER_TEXEL;
        if self.taken_bytes + bytes > Self::BUDGET {
            return None;
        }
        self.taken_bytes += bytes;
        if let Some(index) = self.free.iter().position(|(pooled, _)| {
            pooled.texture.width() == width
                && pooled.texture.height() == height
                && pooled.texture.format() == format
        }) {
            return Some(self.free.swap_remove(index).0);
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pooled_render_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // Group targets are copied from when a blended child reads what is beneath it,
            // and backdrop copies are copied into.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Some(PooledTexture { texture, view })
    }

    pub(super) fn give_back(&mut self, pooled: PooledTexture) {
        self.taken_bytes = self.taken_bytes.saturating_sub(pooled.bytes());
        self.free.push((pooled, self.frame));
    }

    /// Ends a frame, letting go of textures no frame has used for a while.
    /// Returns whether any were let go, so that bind groups which hold them
    /// can be let go too.
    pub(super) fn end_frame(&mut self) -> bool {
        // A frame abandoned midway, on an error, drops the textures it had taken
        // rather than giving them back; they no longer count against the budget.
        self.taken_bytes = 0;
        self.frame += 1;
        let frame = self.frame;
        let count = self.free.len();
        self.free
            .retain(|(_, used)| frame.saturating_sub(*used) <= Self::KEPT_FRAMES);
        self.free.len() != count
    }

    /// Lets go of every free texture, as when the device's targets are rebuilt.
    pub(super) fn clear(&mut self) {
        self.free.clear();
        self.taken_bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn size(width: i32, height: i32) -> Size<DevicePixels> {
        Size {
            width: DevicePixels(width),
            height: DevicePixels(height),
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn the_pool_reuses_textures_of_the_same_size_and_keeps_to_its_budget() -> anyhow::Result<()> {
        let context = crate::WgpuContext::new_headless(None)?;
        let format = wgpu::TextureFormat::Bgra8Unorm;
        let mut pool = TexturePool::new();
        let first = pool
            .take(&context.device, format, size(64, 128))
            .expect("a small texture fits the budget");
        pool.give_back(first.clone());
        let second = pool
            .take(&context.device, format, size(64, 128))
            .expect("a small texture fits the budget");
        assert!(second.texture == first.texture, "a free texture is reused");
        let other = pool
            .take(&context.device, format, size(128, 64))
            .expect("a small texture fits the budget");
        assert!(other.texture != first.texture, "sizes must match exactly");
        assert!(
            pool.take(&context.device, format, size(16384, 16384))
                .is_none(),
            "a texture past the budget is refused"
        );
        pool.give_back(second);
        pool.give_back(other);
        for _ in 0..TexturePool::KEPT_FRAMES {
            assert!(!pool.end_frame(), "recently used textures are kept");
        }
        assert!(pool.end_frame(), "unused textures are let go");
        assert!(pool.free.is_empty());
        Ok(())
    }
}
