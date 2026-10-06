//! The textures a frame renders groups into, pooled across frames.

use anyhow::Result;
use gpui::{DevicePixels, Size};
use windows::Win32::Graphics::{
    Direct3D11::*,
    Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC},
};

/// A colour texture that can be drawn into and sampled.
#[derive(Clone)]
pub(crate) struct ColorTarget {
    pub(crate) texture: ID3D11Texture2D,
    pub(crate) rtv: Option<ID3D11RenderTargetView>,
    pub(crate) srv: Option<ID3D11ShaderResourceView>,
    pub(crate) size: Size<DevicePixels>,
}

impl ColorTarget {
    pub(crate) fn new(
        device: &ID3D11Device,
        format: DXGI_FORMAT,
        size: Size<DevicePixels>,
    ) -> Result<Self> {
        let size = Size {
            width: DevicePixels(size.width.0.max(1)),
            height: DevicePixels(size.height.0.max(1)),
        };
        let desc = D3D11_TEXTURE2D_DESC {
            Width: size.width.0 as u32,
            Height: size.height.0 as u32,
            MipLevels: 1,
            ArraySize: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut texture = None;
        unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture))? };
        let texture =
            texture.ok_or_else(|| anyhow::anyhow!("CreateTexture2D returned no texture"))?;
        let mut rtv = None;
        unsafe { device.CreateRenderTargetView(&texture, None, Some(&mut rtv))? };
        let mut srv = None;
        unsafe { device.CreateShaderResourceView(&texture, None, Some(&mut srv))? };
        Ok(Self {
            texture,
            rtv,
            srv,
            size,
        })
    }

    fn bytes(&self) -> usize {
        self.size.width.0 as usize * self.size.height.0 as usize * 4
    }
}

/// Render targets for isolated groups, their blurs and the backdrops blend modes read, reused
/// across frames. A texture is taken while a group is rendered and composited, and given back
/// once its composite is issued: the immediate context orders later draws into it after the
/// reads already issued.
pub(crate) struct TexturePool {
    format: DXGI_FORMAT,
    free: Vec<(ColorTarget, u64)>,
    /// The bytes of the textures taken this frame and not yet given back.
    taken_bytes: usize,
    frame: u64,
}

impl TexturePool {
    /// The most texture memory groups may hold at once: a group that would take more is drawn
    /// in place instead.
    pub(crate) const BUDGET: usize = 512 << 20;
    /// How many frames a free texture is kept for unused.
    const KEPT_FRAMES: u64 = 60;

    pub(crate) fn new(format: DXGI_FORMAT) -> Self {
        Self {
            format,
            free: Vec::new(),
            taken_bytes: 0,
            frame: 0,
        }
    }

    /// A colour target of exactly `size`, or `None` if taking one would pass the budget or the
    /// device cannot create one.
    pub(crate) fn take(
        &mut self,
        device: &ID3D11Device,
        size: Size<DevicePixels>,
    ) -> Option<ColorTarget> {
        let size = Size {
            width: DevicePixels(size.width.0.max(1)),
            height: DevicePixels(size.height.0.max(1)),
        };
        let bytes = size.width.0 as usize * size.height.0 as usize * 4;
        if self.taken_bytes + bytes > Self::BUDGET {
            return None;
        }
        let target = match self.free.iter().position(|(target, _)| target.size == size) {
            Some(ix) => self.free.swap_remove(ix).0,
            None => match ColorTarget::new(device, self.format, size) {
                Ok(target) => target,
                Err(error) => {
                    log::error!("creating a {size:?} group target: {error:#}");
                    return None;
                }
            },
        };
        self.taken_bytes += bytes;
        Some(target)
    }

    pub(crate) fn give_back(&mut self, target: ColorTarget) {
        self.taken_bytes = self.taken_bytes.saturating_sub(target.bytes());
        self.free.push((target, self.frame));
    }

    /// Ends a frame, letting go of textures no frame has used for a while. Whatever a frame
    /// failed to give back (when encoding stopped on an error) has been dropped by now, so the
    /// budget starts the next frame empty.
    pub(crate) fn end_frame(&mut self) {
        self.frame += 1;
        self.taken_bytes = 0;
        let frame = self.frame;
        self.free
            .retain(|(_, used)| frame.saturating_sub(*used) <= Self::KEPT_FRAMES);
    }
}
