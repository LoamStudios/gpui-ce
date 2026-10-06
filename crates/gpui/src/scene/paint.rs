//! The scene's paint table: gradients that primitives refer to by index,
//! with their colour stops in a table of their own.

use super::TransformationMatrix;
use peniko::color::{ColorSpaceTag, DynamicColor, HueDirection};
use peniko::{Extend, Gradient, GradientKind, InterpolationAlphaSpace};

/// The shape of a paint-table gradient.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
#[expect(missing_docs)]
pub enum PaintKind {
    #[default]
    Linear = 0,
    Radial = 1,
    Sweep = 2,
}

/// How a paint-table gradient continues past its ends.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
#[expect(missing_docs)]
pub enum PaintExtend {
    #[default]
    Pad = 0,
    Repeat = 1,
    Reflect = 2,
}

/// The colour space a paint-table gradient's stops are interpolated in: those
/// the shaders convert from. Gradients in other spaces are sampled into
/// Oklab.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
#[expect(missing_docs)]
pub enum PaintColorSpace {
    #[default]
    Srgb = 0,
    LinearSrgb = 1,
    Oklab = 2,
    Oklch = 3,
}

/// An entry of a scene's paint table: a gradient, placed by the
/// transformation from viewport positions, in device pixels, to its own
/// space, with its stops at `first_stop..first_stop + stop_count` of the
/// stop table.
///
/// Its geometry is, by kind: for linear, the start and end points; for
/// radial, the start and end centres, and in `radii` their radii; for
/// sweep, the centre, then the start and end angles in radians.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct ScenePaint {
    pub transformation: TransformationMatrix,
    pub kind: PaintKind,
    pub extend: PaintExtend,
    pub color_space: PaintColorSpace,
    pub first_stop: u32,
    pub stop_count: u32,
    pub padding: u32,
    pub geometry: [f32; 4],
    pub radii: [f32; 4],
}

/// An entry of a scene's stop table: a colour in its gradient's
/// interpolation space, premultiplied except for hue, and its offset.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SceneColorStop {
    pub color: [f32; 4],
    pub offset: f32,
    pub padding0: u32,
    pub padding1: u32,
    pub padding2: u32,
}

/// How many stops each segment of a gradient in a colour space the shaders
/// don't convert from is sampled into.
const SAMPLES_PER_SEGMENT: usize = 16;

impl ScenePaint {
    /// The entry for `gradient`, placed by `to_gradient`, from viewport
    /// positions to the gradient's space, with its stops added to `stops`.
    /// `None` for a gradient with no stops.
    pub(crate) fn gradient(
        gradient: &Gradient,
        to_gradient: TransformationMatrix,
        stops: &mut Vec<SceneColorStop>,
    ) -> Option<Self> {
        let (first, rest) = gradient.stops.split_first()?;
        let first_stop = stops.len() as u32;
        let shader_space = match gradient.interpolation_cs {
            ColorSpaceTag::Srgb => Some(PaintColorSpace::Srgb),
            ColorSpaceTag::LinearSrgb => Some(PaintColorSpace::LinearSrgb),
            ColorSpaceTag::Oklab => Some(PaintColorSpace::Oklab),
            ColorSpaceTag::Oklch => Some(PaintColorSpace::Oklch),
            _ => None,
        };
        let premultiplied =
            gradient.interpolation_alpha_space == InterpolationAlphaSpace::Premultiplied;
        let color_space = match (shader_space, premultiplied) {
            (Some(space), true) => space,
            _ => PaintColorSpace::Oklab,
        };
        let push = |stops: &mut Vec<SceneColorStop>, offset: f32, color: DynamicColor| {
            stops.push(SceneColorStop {
                color: premultiplied_components(color, color_space),
                offset,
                padding0: 0,
                padding1: 0,
                padding2: 0,
            });
        };
        if rest.is_empty() {
            push(
                stops,
                first.offset,
                first.color.convert(space_tag(color_space)),
            );
        }
        let mut previous = first;
        for stop in rest {
            let (start, end) = (previous.offset, stop.offset);
            if shader_space.is_some() && premultiplied {
                // Each segment's ends, interpolated as the shaders do: a hue
                // is unwrapped for its segment, so each segment has both.
                let segment = previous.color.interpolate(
                    stop.color,
                    gradient.interpolation_cs,
                    gradient.hue_direction,
                );
                push(stops, start, segment.eval(0.0));
                push(stops, end, segment.eval(1.0));
            } else {
                for sample in 0..=SAMPLES_PER_SEGMENT {
                    let t = sample as f32 / SAMPLES_PER_SEGMENT as f32;
                    let color = sample_segment(
                        previous.color,
                        stop.color,
                        gradient.interpolation_cs,
                        gradient.hue_direction,
                        premultiplied,
                        t,
                    );
                    push(
                        stops,
                        start + (end - start) * t,
                        color.convert(ColorSpaceTag::Oklab),
                    );
                }
            }
            previous = stop;
        }

        let (kind, geometry, radii) = match gradient.kind {
            GradientKind::Linear(line) => (
                PaintKind::Linear,
                [
                    line.start.x as f32,
                    line.start.y as f32,
                    line.end.x as f32,
                    line.end.y as f32,
                ],
                [0.; 4],
            ),
            GradientKind::Radial(radial) => (
                PaintKind::Radial,
                [
                    radial.start_center.x as f32,
                    radial.start_center.y as f32,
                    radial.end_center.x as f32,
                    radial.end_center.y as f32,
                ],
                [radial.start_radius, radial.end_radius, 0., 0.],
            ),
            GradientKind::Sweep(sweep) => (
                PaintKind::Sweep,
                [
                    sweep.center.x as f32,
                    sweep.center.y as f32,
                    sweep.start_angle,
                    sweep.end_angle,
                ],
                [0.; 4],
            ),
        };
        Some(Self {
            transformation: to_gradient,
            kind,
            extend: match gradient.extend {
                Extend::Pad => PaintExtend::Pad,
                Extend::Repeat => PaintExtend::Repeat,
                Extend::Reflect => PaintExtend::Reflect,
            },
            color_space,
            first_stop,
            stop_count: stops.len() as u32 - first_stop,
            padding: 0,
            geometry,
            radii,
        })
    }
}

fn space_tag(space: PaintColorSpace) -> ColorSpaceTag {
    match space {
        PaintColorSpace::Srgb => ColorSpaceTag::Srgb,
        PaintColorSpace::LinearSrgb => ColorSpaceTag::LinearSrgb,
        PaintColorSpace::Oklab => ColorSpaceTag::Oklab,
        PaintColorSpace::Oklch => ColorSpaceTag::Oklch,
    }
}

/// The colour `t` of the way from `start` to `end`, interpolated in `space`.
fn sample_segment(
    start: DynamicColor,
    end: DynamicColor,
    space: ColorSpaceTag,
    direction: HueDirection,
    premultiplied: bool,
    t: f32,
) -> DynamicColor {
    if premultiplied {
        start.interpolate(end, space, direction).eval(t)
    } else {
        start
            .interpolate_unpremultiplied(end, space, direction)
            .eval(t)
    }
}

/// `color`'s components in `space`, which it is in, premultiplied by its
/// alpha except for a hue, with the alpha last.
fn premultiplied_components(color: DynamicColor, space: PaintColorSpace) -> [f32; 4] {
    let [first, second, third, alpha] = color.components;
    match space {
        PaintColorSpace::Oklch => [first * alpha, second * alpha, third, alpha],
        _ => [first * alpha, second * alpha, third * alpha, alpha],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use peniko::color::{AlphaColor, Srgb, palette};

    fn stops_of(gradient: &Gradient) -> (ScenePaint, Vec<SceneColorStop>) {
        let mut stops = Vec::new();
        let paint = ScenePaint::gradient(gradient, TransformationMatrix::UNIT, &mut stops).unwrap();
        (paint, stops)
    }

    #[test]
    fn each_segment_keeps_both_ends() {
        let gradient = Gradient::new_linear((0., 0.), (100., 0.)).with_stops([
            palette::css::RED,
            palette::css::LIME,
            palette::css::BLUE,
        ]);
        let (paint, stops) = stops_of(&gradient);
        assert_eq!(paint.color_space, PaintColorSpace::Srgb);
        assert_eq!(paint.stop_count, 4);
        let offsets: Vec<f32> = stops.iter().map(|stop| stop.offset).collect();
        assert_eq!(offsets, [0., 0.5, 0.5, 1.]);
        assert_eq!(stops[0].color, [1., 0., 0., 1.]);
        assert_eq!(stops[3].color, [0., 0., 1., 1.]);
    }

    #[test]
    fn stops_are_premultiplied_except_for_hue() {
        let half_red = AlphaColor::<Srgb>::new([1., 0., 0., 0.5]);
        let gradient = Gradient::new_linear((0., 0.), (1., 0.))
            .with_interpolation_cs(ColorSpaceTag::Oklch)
            .with_stops([half_red, half_red]);
        let (_, stops) = stops_of(&gradient);
        let lch = DynamicColor::from_alpha_color(half_red).convert(ColorSpaceTag::Oklch);
        let [lightness, chroma, hue, _] = lch.components;
        let [l, c, h, a] = stops[0].color;
        assert!((l - lightness * 0.5).abs() < 1e-5);
        assert!((c - chroma * 0.5).abs() < 1e-5);
        assert!((h - hue).abs() < 1e-3, "hue {h} is not premultiplied");
        assert_eq!(a, 0.5);
    }

    #[test]
    fn hues_take_the_shorter_way_round_within_a_segment() {
        let gradient = Gradient::new_linear((0., 0.), (1., 0.))
            .with_interpolation_cs(ColorSpaceTag::Oklch)
            .with_stops([
                DynamicColor::from_alpha_color(AlphaColor::<peniko::color::Oklch>::new([
                    0.7, 0.1, 350., 1.,
                ])),
                DynamicColor::from_alpha_color(AlphaColor::<peniko::color::Oklch>::new([
                    0.7, 0.1, 10., 1.,
                ])),
            ]);
        let (_, stops) = stops_of(&gradient);
        let span = (stops[1].color[2] - stops[0].color[2]).abs();
        assert!(span <= 180., "the hue turns {span} degrees");
    }

    #[test]
    fn other_spaces_are_sampled_into_oklab() {
        let gradient = Gradient::new_linear((0., 0.), (1., 0.))
            .with_interpolation_cs(ColorSpaceTag::Hsl)
            .with_stops([palette::css::RED, palette::css::BLUE]);
        let (paint, stops) = stops_of(&gradient);
        assert_eq!(paint.color_space, PaintColorSpace::Oklab);
        assert_eq!(stops.len(), SAMPLES_PER_SEGMENT + 1);
        assert_eq!(stops.last().unwrap().offset, 1.);
    }
}
