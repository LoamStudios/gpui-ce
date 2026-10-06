//! The textures a frame renders groups into, pooled across frames.

use gpui::{DevicePixels, Size};

/// Render targets for isolated groups and their blurs, reused across frames.
/// A texture is taken while a group is rendered and composited, and given
/// back once its composite is encoded: later work in the same command
/// buffer may draw into it again, since Metal orders the passes.
pub(crate) struct TexturePool {
    free: Vec<(metal::Texture, u64)>,
    /// The bytes of the textures taken and not yet given back.
    taken_bytes: usize,
    frame: u64,
}

impl TexturePool {
    /// The most texture memory groups may hold at once: a group that would
    /// take more is drawn in place instead.
    pub(crate) const BUDGET: usize = 512 << 20;
    /// How many frames a free texture is kept for unused.
    const KEPT_FRAMES: u64 = 60;

    pub(crate) fn new() -> Self {
        Self {
            free: Vec::new(),
            taken_bytes: 0,
            frame: 0,
        }
    }

    /// A colour target of `size`, or `None` if taking one would pass the
    /// budget.
    pub(crate) fn take(
        &mut self,
        device: &metal::DeviceRef,
        size: Size<DevicePixels>,
    ) -> Option<metal::Texture> {
        let (width, height) = (size.width.0.max(1) as u64, size.height.0.max(1) as u64);
        let bytes = (width * height * 4) as usize;
        if self.taken_bytes + bytes > Self::BUDGET {
            return None;
        }
        self.taken_bytes += bytes;
        if let Some(ix) = self
            .free
            .iter()
            .position(|(texture, _)| texture.width() == width && texture.height() == height)
        {
            return Some(self.free.swap_remove(ix).0);
        }
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(width);
        descriptor.set_height(height);
        descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        Some(device.new_texture(&descriptor))
    }

    pub(crate) fn give_back(&mut self, texture: metal::Texture) {
        self.taken_bytes = self
            .taken_bytes
            .saturating_sub((texture.width() * texture.height() * 4) as usize);
        self.free.push((texture, self.frame));
    }

    /// Ends a frame, letting go of textures no frame has used for a while.
    pub(crate) fn end_frame(&mut self) {
        self.frame += 1;
        let frame = self.frame;
        self.free
            .retain(|(_, used)| frame.saturating_sub(*used) <= Self::KEPT_FRAMES);
    }
}
