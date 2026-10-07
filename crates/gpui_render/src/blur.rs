pub use crate::shaders::{
    blur::BlurUniforms,
    common::{
        BlurCompositeClip as ShaderBlurCompositeClip, Bounds as ShaderBounds,
        Corners as ShaderCorners, DownsampleMode,
    },
};
use gpui::{Bounds, Corners, ScaledPixels};
use wgsl_rs::std::{vec2f, vec4f};

pub const DOWNSAMPLE_FACTOR: u32 = 2;

/// Texture dimension for the downsampled blur passes; allocations must agree.
pub fn downsampled_dimension(dimension: u32) -> u32 {
    dimension.div_ceil(DOWNSAMPLE_FACTOR).max(1)
}
const RADIUS_TO_STANDARD_DEVIATION: f32 = 0.5;
pub const GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS: f32 = 3.0;
pub const MAX_GAUSSIAN_SAMPLES_PER_SIDE: u32 = 32;

/// Group filters blur at full resolution below this standard deviation, in
/// device pixels, and at a lower one from it, where the difference does not
/// show.
pub const FULL_RESOLUTION_MAX_DEVIATION: f32 = 4.0;

/// Past half resolution, a group's blur runs at the lowest resolution at
/// which its kernel's standard deviation is still this many texels: enough
/// for the bilinear upsampling of the result to stay within a level of the
/// Gaussian.
pub const MIN_DOWNSAMPLED_DEVIATION: f32 = 3.0;

/// The most a group's blur is downsampled, along each axis. Group targets
/// are whole multiples of it.
pub const MAX_GROUP_DOWNSAMPLE_FACTOR: u32 = 16;

/// How a group filter blurs a picture: its kernel, in the texels of the
/// resolution it runs at, and how many of the picture's pixels make one of
/// those texels along each axis.
#[derive(Clone, Copy)]
pub struct GroupBlur {
    pub kernel: BlurKernel,
    /// 1 at full resolution, else 2, 4, 8 or 16.
    pub factor: u32,
}

/// What a blur's first pass reads in place of its source's colours: its
/// coverage in `color`, moved by `offset` device pixels. A drop shadow.
#[derive(Clone, Copy)]
pub struct BlurTint {
    pub color: [f32; 4],
    pub offset: [f32; 2],
}

/// The passes of a group's blur, over a source of a given size: a box
/// downsample unless it runs at full resolution, then the two separable
/// passes, each into a texture of `size`.
pub struct GroupBlurPasses {
    pub size: [u32; 2],
    pub downsample: Option<BlurUniforms>,
    pub horizontal: BlurUniforms,
    pub vertical: BlurUniforms,
}

impl GroupBlur {
    /// The blur of standard deviation `std_deviation`, in device pixels:
    /// `None` for no blur. Downsampled by a factor `f`, a kernel of
    /// `std_deviation / f` texels is `std_deviation` pixels.
    pub fn new(std_deviation: f32) -> Option<Self> {
        let factor = if std_deviation < FULL_RESOLUTION_MAX_DEVIATION {
            1
        } else {
            let mut factor = DOWNSAMPLE_FACTOR;
            while factor < MAX_GROUP_DOWNSAMPLE_FACTOR
                && std_deviation / (2 * factor) as f32 >= MIN_DOWNSAMPLED_DEVIATION
            {
                factor *= 2;
            }
            factor
        };
        Some(Self {
            kernel: BlurKernel::for_radius(2. * std_deviation / factor as f32)?,
            factor,
        })
    }

    /// The passes that blur a source of `source_size` pixels, reading it
    /// through `tint` if given.
    pub fn passes(&self, source_size: [u32; 2], tint: Option<BlurTint>) -> GroupBlurPasses {
        let size = source_size.map(|length| length.div_ceil(self.factor).max(1));
        let source = source_size.map(|length| length.max(1) as f32);
        let blurred = size.map(|length| length as f32);
        let tinted = |uniforms: BlurUniforms| match tint {
            Some(tint) => uniforms.tinted(tint, source),
            None => uniforms,
        };
        let horizontal = BlurUniforms::gaussian(BlurAxis::Horizontal, blurred, self.kernel);
        let vertical = BlurUniforms::gaussian(BlurAxis::Vertical, blurred, self.kernel);
        if self.factor == 1 {
            return GroupBlurPasses {
                size,
                downsample: None,
                horizontal: tinted(horizontal),
                vertical,
            };
        }
        let mut downsample = BlurUniforms::downsample(source, blurred);
        downsample.downsample_factor = self.factor;
        GroupBlurPasses {
            size,
            downsample: Some(tinted(downsample)),
            horizontal,
            vertical,
        }
    }
}

#[derive(Clone, Copy)]
pub enum BlurAxis {
    Horizontal,
    Vertical,
}

#[derive(Clone, Copy)]
pub enum FilterCompositeClip {
    RoundedBounds,
    ContentShape,
}

#[derive(Clone, Copy)]
pub struct FilterCompositeParameters {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub corner_smoothing: f32,
    pub blur_radius: f32,
    pub opacity: f32,
    pub clip: FilterCompositeClip,
}

#[derive(Clone, Copy)]
pub struct BlurKernel {
    pub standard_deviation: f32,
    pub sample_count: u32,
    pub sample_step: f32,
}

impl BlurKernel {
    pub fn for_radius(blur_radius: f32) -> Option<Self> {
        let standard_deviation = (blur_radius * RADIUS_TO_STANDARD_DEVIATION).max(0.0);
        if standard_deviation <= 0.0 {
            return None;
        }

        let ideal_sample_count = (GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * standard_deviation).ceil();
        let sample_count = (ideal_sample_count as u32).clamp(1, MAX_GAUSSIAN_SAMPLES_PER_SIDE);
        Some(Self {
            standard_deviation,
            sample_count,
            sample_step: (ideal_sample_count / sample_count as f32).max(1.0),
        })
    }
}

impl BlurUniforms {
    pub fn copy(size: [f32; 2]) -> Self {
        with_texture_sizes(empty_uniforms(DownsampleMode::Copy), size, size)
    }

    pub fn downsample(source_size: [f32; 2], target_size: [f32; 2]) -> Self {
        Self {
            downsample_factor: DOWNSAMPLE_FACTOR,
            ..with_texture_sizes(
                empty_uniforms(DownsampleMode::HalfResolution),
                source_size,
                target_size,
            )
        }
    }

    /// This pass reading its source's coverage in `tint`'s colour, moved,
    /// from a source of `source_size` pixels.
    pub fn tinted(self, tint: BlurTint, source_size: [f32; 2]) -> Self {
        let [red, green, blue, alpha] = tint.color;
        Self {
            tint: vec4f(red, green, blue, alpha),
            tint_offset: vec2f(
                tint.offset[0] / source_size[0],
                tint.offset[1] / source_size[1],
            ),
            tinted: 1,
            ..self
        }
    }

    pub fn gaussian(axis: BlurAxis, texture_size: [f32; 2], kernel: BlurKernel) -> Self {
        let direction = match axis {
            BlurAxis::Horizontal => vec2f(1.0 / texture_size[0], 0.0),
            BlurAxis::Vertical => vec2f(0.0, 1.0 / texture_size[1]),
        };
        Self {
            direction,
            standard_deviation: kernel.standard_deviation,
            sample_count: kernel.sample_count,
            sample_step: kernel.sample_step,
            ..with_texture_sizes(
                empty_uniforms(DownsampleMode::Copy),
                texture_size,
                texture_size,
            )
        }
    }

    pub fn composite(
        bounds: Bounds<ScaledPixels>,
        content_mask: Bounds<ScaledPixels>,
        corner_radii: Corners<ScaledPixels>,
        corner_smoothing: f32,
        opacity: f32,
        clip: FilterCompositeClip,
        source_size: [f32; 2],
        target_size: [f32; 2],
        source_origin: [f32; 2],
    ) -> Self {
        Self {
            source_origin: vec2f(source_origin[0], source_origin[1]),
            bounds: bounds.into(),
            content_mask: content_mask.into(),
            corner_radii: corner_radii.into(),
            corner_smoothing,
            opacity,
            composite_clip: match clip {
                FilterCompositeClip::RoundedBounds => ShaderBlurCompositeClip::RoundedBounds,
                FilterCompositeClip::ContentShape => ShaderBlurCompositeClip::None,
            },
            ..with_texture_sizes(
                empty_uniforms(DownsampleMode::Copy),
                source_size,
                target_size,
            )
        }
    }
}

#[derive(Clone, Copy)]
pub struct ScissorRectangle {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl ScissorRectangle {
    pub fn for_blurred_bounds(
        bounds: Bounds<ScaledPixels>,
        dilation: f32,
        full_width: u32,
        full_height: u32,
    ) -> Self {
        let downsampled_width = full_width.div_ceil(DOWNSAMPLE_FACTOR).max(1);
        let downsampled_height = full_height.div_ceil(DOWNSAMPLE_FACTOR).max(1);
        let minimum_x = downsampled_coordinate(
            bounds.origin.x.0 - dilation,
            downsampled_width,
            EdgeRounding::OutwardMinimum,
        );
        let minimum_y = downsampled_coordinate(
            bounds.origin.y.0 - dilation,
            downsampled_height,
            EdgeRounding::OutwardMinimum,
        );
        let maximum_x = downsampled_coordinate(
            bounds.origin.x.0 + bounds.size.width.0 + dilation,
            downsampled_width,
            EdgeRounding::OutwardMaximum,
        )
        .max(minimum_x);
        let maximum_y = downsampled_coordinate(
            bounds.origin.y.0 + bounds.size.height.0 + dilation,
            downsampled_height,
            EdgeRounding::OutwardMaximum,
        )
        .max(minimum_y);
        Self {
            x: minimum_x,
            y: minimum_y,
            width: maximum_x - minimum_x,
            height: maximum_y - minimum_y,
        }
    }

    pub fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

#[derive(Clone, Copy)]
enum EdgeRounding {
    OutwardMinimum,
    OutwardMaximum,
}

fn downsampled_coordinate(value: f32, maximum: u32, rounding: EdgeRounding) -> u32 {
    let downsampled = value / DOWNSAMPLE_FACTOR as f32;
    let rounded = match rounding {
        EdgeRounding::OutwardMinimum => downsampled.floor(),
        EdgeRounding::OutwardMaximum => downsampled.ceil(),
    };
    (rounded.max(0.0) as u32).min(maximum)
}

fn with_texture_sizes(
    mut uniforms: BlurUniforms,
    source_size: [f32; 2],
    target_size: [f32; 2],
) -> BlurUniforms {
    uniforms.source_size = vec2f(source_size[0], source_size[1]);
    uniforms.target_size = vec2f(target_size[0], target_size[1]);
    uniforms
}

fn empty_uniforms(downsample_mode: DownsampleMode) -> BlurUniforms {
    let bounds = ShaderBounds {
        origin: vec2f(0.0, 0.0),
        size: vec2f(0.0, 0.0),
    };
    BlurUniforms {
        bounds,
        content_mask: bounds,
        corner_radii: ShaderCorners {
            top_left: 0.0,
            top_right: 0.0,
            bottom_right: 0.0,
            bottom_left: 0.0,
        },
        direction: vec2f(0.0, 0.0),
        standard_deviation: 0.0,
        opacity: 0.0,
        sample_count: 0,
        sample_step: 0.0,
        composite_clip: ShaderBlurCompositeClip::None,
        downsample_mode,
        source_size: vec2f(1.0, 1.0),
        target_size: vec2f(1.0, 1.0),
        corner_smoothing: 0.0,
        downsample_factor: 0,
        source_origin: vec2f(0.0, 0.0),
        tint: vec4f(0.0, 0.0, 0.0, 0.0),
        tint_offset: vec2f(0.0, 0.0),
        tinted: 0,
        padding1: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::downsampled_dimension;

    #[test]
    fn group_blurs_downsample_as_far_as_their_deviation_allows() {
        let factor = |std_deviation| super::GroupBlur::new(std_deviation).unwrap().factor;
        assert!(super::GroupBlur::new(0.).is_none());
        assert_eq!(factor(1.5), 1);
        assert_eq!(factor(4.), 2);
        assert_eq!(factor(11.9), 2);
        assert_eq!(factor(12.), 4);
        assert_eq!(factor(16.), 4);
        assert_eq!(factor(24.), 8);
        assert_eq!(factor(1000.), super::MAX_GROUP_DOWNSAMPLE_FACTOR);
        // The kernel keeps the deviation, in texels of its resolution.
        let blur = super::GroupBlur::new(16.).unwrap();
        assert_eq!(blur.kernel.standard_deviation, 4.);
        // Its taps are a texel apart, to be paired.
        assert_eq!(blur.kernel.sample_step, 1.);
        let passes = blur.passes([2048, 1600], None);
        assert_eq!(passes.size, [512, 400]);
        assert_eq!(passes.downsample.unwrap().downsample_factor, 4);
    }

    #[test]
    fn downsampled_dimensions_cover_odd_edges() {
        assert_eq!(downsampled_dimension(0), 1);
        assert_eq!(downsampled_dimension(1), 1);
        assert_eq!(downsampled_dimension(2), 1);
        assert_eq!(downsampled_dimension(3), 2);
        assert_eq!(downsampled_dimension(33), 17);
    }
}
