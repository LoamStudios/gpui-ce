//! The texture array a window's resident photo tiles are kept in.

use anyhow::{Context, Result};
use gpui::{PHOTO_LAYER_SIZE, Scene};
use windows::Win32::Graphics::{
    Direct3D::D3D11_SRV_DIMENSION_TEXTURE2DARRAY,
    Direct3D11::*,
    Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC},
};

// Feature level 11.0, the lowest the renderer creates a device at, allows 2D textures
// 16384 texels square and arrays of 2048 of them, far beyond what the photo budget asks for.
const _: () = assert!(
    PHOTO_LAYER_SIZE <= D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION,
    "a photo layer must fit a feature level 11.0 texture"
);
const _: () = assert!(
    (gpui::PHOTO_MEMORY_BUDGET / (PHOTO_LAYER_SIZE as usize * PHOTO_LAYER_SIZE as usize * 4))
        <= D3D11_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION as usize,
    "the photo budget's layers must fit a feature level 11.0 texture array"
);

/// The photo tile array, which grows as tiles are placed in new layers, keeping what it
/// holds, and the sampler that filters it.
pub(crate) struct PhotoTiles {
    texture: ID3D11Texture2D,
    view: Option<ID3D11ShaderResourceView>,
    sampler: Option<ID3D11SamplerState>,
    /// How many layers of tiles it has: 0 while it is a placeholder.
    layers: u32,
}

impl PhotoTiles {
    pub(crate) fn new(device: &ID3D11Device) -> Result<Self> {
        let (texture, view) = new_array(device, 1, 1)?;
        let sampler = unsafe {
            let desc = D3D11_SAMPLER_DESC {
                // Tiles carry no mips; the level a photo draws from is chosen on the CPU.
                Filter: D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT,
                AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D11_COMPARISON_NEVER,
                BorderColor: [0.0; 4],
                MinLOD: 0.0,
                MaxLOD: 0.0,
            };
            let mut output = None;
            device
                .CreateSamplerState(&desc, Some(&mut output))
                .context("Creating the photo sampler")?;
            output
        };
        Ok(Self {
            texture,
            view,
            sampler,
            layers: 0,
        })
    }

    /// The array's view, for the photo tiles register.
    pub(crate) fn view(&self) -> &Option<ID3D11ShaderResourceView> {
        &self.view
    }

    /// The sampler photo tiles are filtered with, for the photo sampler register.
    pub(crate) fn sampler(&self) -> &Option<ID3D11SamplerState> {
        &self.sampler
    }

    /// Copies the tiles `scene`'s photos have placed since the last frame into the array,
    /// growing it first if they need more layers. The copies are queued on `device_context`
    /// ahead of the frame's draws.
    pub(crate) fn upload(
        &mut self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
        scene: &Scene,
    ) -> Result<()> {
        let Some(uploads) = &scene.photo_uploads else {
            return Ok(());
        };
        let (layers, tiles) = uploads.take(self.layers);
        if layers > self.layers {
            let (texture, view) = new_array(device, PHOTO_LAYER_SIZE, layers)?;
            for layer in 0..self.layers {
                // One mip a layer, so a layer's subresource is its index.
                unsafe {
                    device_context.CopySubresourceRegion(
                        &texture,
                        layer,
                        0,
                        0,
                        0,
                        &self.texture,
                        layer,
                        None,
                    );
                }
            }
            self.texture = texture;
            self.view = view;
            self.layers = layers;
        }
        for tile in &tiles {
            if tile.layer >= self.layers {
                log::error!(
                    "photo tile placed in layer {} of a {}-layer array",
                    tile.layer,
                    self.layers
                );
                continue;
            }
            let row_bytes = tile.size.width * 4;
            debug_assert_eq!(tile.pixels.len(), (row_bytes * tile.size.height) as usize);
            unsafe {
                device_context.UpdateSubresource(
                    &self.texture,
                    tile.layer,
                    Some(&D3D11_BOX {
                        left: tile.origin.x,
                        top: tile.origin.y,
                        front: 0,
                        right: tile.origin.x + tile.size.width,
                        bottom: tile.origin.y + tile.size.height,
                        back: 1,
                    }),
                    tile.pixels.as_ptr().cast(),
                    row_bytes,
                    0,
                );
            }
        }
        Ok(())
    }
}

fn new_array(
    device: &ID3D11Device,
    side: u32,
    layers: u32,
) -> Result<(ID3D11Texture2D, Option<ID3D11ShaderResourceView>)> {
    let layers = layers.max(1);
    let desc = D3D11_TEXTURE2D_DESC {
        Width: side,
        Height: side,
        MipLevels: 1,
        ArraySize: layers,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let mut texture = None;
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture)) }
        .with_context(|| format!("Creating a {layers}-layer photo tile array"))?;
    let texture = texture.context("photo tile array missing")?;
    let view_desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D11_SRV_DIMENSION_TEXTURE2DARRAY,
        Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2DArray: D3D11_TEX2D_ARRAY_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                FirstArraySlice: 0,
                ArraySize: layers,
            },
        },
    };
    let mut view = None;
    unsafe { device.CreateShaderResourceView(&texture, Some(&view_desc), Some(&mut view)) }
        .context("Creating the photo tile array's view")?;
    Ok((texture, view))
}
