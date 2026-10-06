//! What the renderers share to render a group into a target of its own and
//! composite it into its parent.

use crate::shaders::common::Bounds as ShaderBounds;
pub use crate::shaders::{
    common::{GroupBlendMode, GroupMask},
    group::GroupUniforms,
};
use gpui::{BlendMode, Bounds, DevicePixels, MaskMode, ScaledPixels, Size, point, size};
use wgsl_rs::std::vec2f;

/// Group targets are sized in steps of this many device pixels, so that a
/// pool of them is reused as groups change size, as they do while animating.
pub const GROUP_TARGET_QUANTUM: i32 = 64;

/// Where a group whose contents cover `region` of the viewport is rendered:
/// the region rounded out to whole pixels, clipped to the viewport, and
/// grown to a whole number of quanta. `None` when none of it is in view.
pub fn group_target_bounds(
    region: Bounds<ScaledPixels>,
    viewport_size: Size<DevicePixels>,
) -> Option<Bounds<DevicePixels>> {
    let left = region.origin.x.0.floor().max(0.0) as i32;
    let top = region.origin.y.0.floor().max(0.0) as i32;
    let right = (region.origin.x.0 + region.size.width.0)
        .ceil()
        .min(viewport_size.width.0 as f32) as i32;
    let bottom = (region.origin.y.0 + region.size.height.0)
        .ceil()
        .min(viewport_size.height.0 as f32) as i32;
    if right <= left || bottom <= top {
        return None;
    }
    let quantized = |length: i32| {
        (length + GROUP_TARGET_QUANTUM - 1) / GROUP_TARGET_QUANTUM * GROUP_TARGET_QUANTUM
    };
    Some(Bounds {
        origin: point(DevicePixels(left), DevicePixels(top)),
        size: size(
            DevicePixels(quantized(right - left)),
            DevicePixels(quantized(bottom - top)),
        ),
    })
}

/// The shader's blend mode for `mode`.
pub fn shader_blend_mode(mode: BlendMode) -> GroupBlendMode {
    match mode {
        BlendMode::Normal => GroupBlendMode::Normal,
        BlendMode::Multiply => GroupBlendMode::Multiply,
        BlendMode::Screen => GroupBlendMode::Screen,
        BlendMode::Overlay => GroupBlendMode::Overlay,
        BlendMode::Darken => GroupBlendMode::Darken,
        BlendMode::Lighten => GroupBlendMode::Lighten,
        BlendMode::ColorDodge => GroupBlendMode::ColorDodge,
        BlendMode::ColorBurn => GroupBlendMode::ColorBurn,
        BlendMode::HardLight => GroupBlendMode::HardLight,
        BlendMode::SoftLight => GroupBlendMode::SoftLight,
        BlendMode::Difference => GroupBlendMode::Difference,
        BlendMode::Exclusion => GroupBlendMode::Exclusion,
        BlendMode::Hue => GroupBlendMode::Hue,
        BlendMode::Saturation => GroupBlendMode::Saturation,
        BlendMode::Color => GroupBlendMode::Color,
        BlendMode::Luminosity => GroupBlendMode::Luminosity,
    }
}

fn shader_bounds(bounds: Bounds<DevicePixels>) -> ShaderBounds {
    ShaderBounds {
        origin: vec2f(bounds.origin.x.0 as f32, bounds.origin.y.0 as f32),
        size: vec2f(bounds.size.width.0 as f32, bounds.size.height.0 as f32),
    }
}

impl GroupUniforms {
    /// Composites a group rendered into `target`, from `source` — its target,
    /// or a blurred copy of it, which covers the same viewport rectangle —
    /// clipped to `content_mask`, faded to `opacity`, and mixed by
    /// `blend_mode` with `backdrop`, a copy of the parent's pixels under
    /// `target`, which blend modes other than normal need. A masked group
    /// passes its mask's target and mode in `mask`, and shows only within
    /// that target, by its mask.
    pub fn composite(
        target: Bounds<DevicePixels>,
        content_mask: Bounds<ScaledPixels>,
        opacity: f32,
        blend_mode: BlendMode,
        backdrop: Option<Bounds<DevicePixels>>,
        mask: Option<(Bounds<DevicePixels>, MaskMode)>,
    ) -> Self {
        let target_bounds = shader_bounds(target);
        let backdrop = backdrop.map_or(target_bounds, shader_bounds);
        let mask_bounds = mask.map_or(target_bounds, |(bounds, _)| shader_bounds(bounds));
        Self {
            bounds: target_bounds,
            content_mask: content_mask.into(),
            source_origin: target_bounds.origin,
            source_size: target_bounds.size,
            backdrop_origin: backdrop.origin,
            backdrop_size: backdrop.size,
            mask_origin: mask_bounds.origin,
            mask_size: mask_bounds.size,
            opacity,
            blend_mode: shader_blend_mode(blend_mode),
            mask: match mask {
                None => GroupMask::None,
                Some((_, MaskMode::Alpha)) => GroupMask::Alpha,
                Some((_, MaskMode::Luminance)) => GroupMask::Luminance,
            },
            padding: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scaled(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(width), ScaledPixels(height)),
        }
    }

    #[test]
    fn a_group_target_covers_its_region_in_view_in_whole_quanta() {
        let viewport = size(DevicePixels(1000), DevicePixels(800));
        let target = group_target_bounds(scaled(10.5, -20., 100., 50.), viewport).unwrap();
        assert_eq!(target.origin, point(DevicePixels(10), DevicePixels(0)));
        assert_eq!(target.size, size(DevicePixels(128), DevicePixels(64)));
        assert!(group_target_bounds(scaled(-200., 0., 100., 50.), viewport).is_none());
    }
}
