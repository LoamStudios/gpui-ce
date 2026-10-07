//! The scene's paint table: gradients and patterns that primitives refer
//! to by index, each followed by its colour stops, in one table of
//! four-float words. Its few integers are kept as floats, which hold them
//! exactly, so every backend reads the table as plain `vec4<f32>`s.

use super::{SceneHsla, TransformationMatrix};
use crate::{Background, BackgroundKind, Bounds, ColorSpace, ScaledPixels, Size};
use peniko::color::{ColorSpaceTag, DynamicColor, HueDirection};
use peniko::{Extend, Gradient, GradientKind, ImageQuality, ImageSampler, InterpolationAlphaSpace};

/// The shape of a paint-table gradient.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
#[expect(missing_docs)]
pub enum PaintKind {
    #[default]
    Linear = 0,
    Radial = 1,
    Sweep = 2,
    /// Diagonal stripes of the first stop's colour, `geometry[0]` wide and
    /// `geometry[1]` apart.
    Stripes = 3,
    /// Squares of the first stop's colour, `geometry[0]` wide, alternating
    /// with transparent ones.
    Checkerboard = 4,
    /// A photo, placed by the transformation into pixels of its level 0. Its
    /// tile words take the place of stops; see [`ScenePaint::photo`].
    Image = 5,
    /// A compiled shader program, placed by the transformation into the
    /// element's own logical pixels. Its parameter words take the place of
    /// stops; see [`ScenePaint::program`].
    Program = 6,
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
    /// [`Background`]'s two-stop sRGB gradients, interpolated as they always
    /// have been: unpremultiplied, between the stops' colours encoded again.
    LegacySrgb = 4,
    /// [`Background`]'s two-stop Oklab gradients, interpolated as they always
    /// have been: unpremultiplied, with a 2.2 transfer curve.
    LegacyOklab = 5,
}

/// One word of a scene's paint table.
pub type PaintWord = [f32; 4];

/// An entry of a scene's paint table: a gradient, placed by the
/// transformation from viewport positions, in device pixels, to its own
/// space, with its stops at word `first_stop` of the table, two words
/// each. It takes [`Self::WORDS`] words.
///
/// Its geometry is, by kind: for linear, the start and end points; for
/// radial, the start and end centres, and in `radii` their radii; for
/// sweep, the centre, then the start and end angles in radians.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[expect(missing_docs)]
pub struct ScenePaint {
    pub transformation: TransformationMatrix,
    pub kind: PaintKind,
    pub extend: PaintExtend,
    pub color_space: PaintColorSpace,
    pub first_stop: u32,
    pub stop_count: u32,
    /// How an image continues past its top and bottom; `extend` is how it
    /// continues past its sides, and how a gradient continues.
    pub y_extend: PaintExtend,
    pub geometry: [f32; 4],
    pub radii: [f32; 4],
}

/// A colour stop of a paint-table entry: a colour in its gradient's
/// interpolation space, premultiplied except for hue, and its offset. It
/// takes [`Self::WORDS`] words.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[expect(missing_docs)]
pub struct SceneColorStop {
    pub color: [f32; 4],
    pub offset: f32,
}

impl ScenePaint {
    /// How many words an entry takes.
    pub const WORDS: usize = 5;

    /// The entry as table words: the transformation's matrix, then its
    /// translation, kind and extend, then its colour space and stops, then
    /// its geometry and radii.
    pub fn words(&self) -> [PaintWord; Self::WORDS] {
        let [[a, b], [c, d]] = self.transformation.rotation_scale;
        let [x, y] = self.transformation.translation;
        [
            [a, b, c, d],
            [x, y, self.kind as u32 as f32, self.extend as u32 as f32],
            [
                self.color_space as u32 as f32,
                self.first_stop as f32,
                self.stop_count as f32,
                self.y_extend as u32 as f32,
            ],
            self.geometry,
            self.radii,
        ]
    }

    /// The entry whose table words begin `words`.
    pub fn from_words(words: &[PaintWord]) -> Self {
        let [a, b, c, d] = words[0];
        let [x, y, kind, extend] = words[1];
        let [color_space, first_stop, stop_count, y_extend] = words[2];
        let extend_of = |value: f32| match value as u32 {
            1 => PaintExtend::Repeat,
            2 => PaintExtend::Reflect,
            _ => PaintExtend::Pad,
        };
        Self {
            transformation: TransformationMatrix {
                rotation_scale: [[a, b], [c, d]],
                translation: [x, y],
            },
            kind: match kind as u32 {
                1 => PaintKind::Radial,
                2 => PaintKind::Sweep,
                3 => PaintKind::Stripes,
                4 => PaintKind::Checkerboard,
                5 => PaintKind::Image,
                6 => PaintKind::Program,
                _ => PaintKind::Linear,
            },
            extend: extend_of(extend),
            y_extend: extend_of(y_extend),
            color_space: match color_space as u32 {
                1 => PaintColorSpace::LinearSrgb,
                2 => PaintColorSpace::Oklab,
                3 => PaintColorSpace::Oklch,
                4 => PaintColorSpace::LegacySrgb,
                5 => PaintColorSpace::LegacyOklab,
                _ => PaintColorSpace::Srgb,
            },
            first_stop: first_stop as u32,
            stop_count: stop_count as u32,
            geometry: words[3],
            radii: words[4],
        }
    }
}

impl SceneColorStop {
    /// How many words a stop takes.
    pub const WORDS: usize = 2;

    /// The stop as table words: its colour, then its offset.
    pub fn words(&self) -> [PaintWord; Self::WORDS] {
        [self.color, [self.offset, 0., 0., 0.]]
    }

    /// The stop whose table words begin `words`.
    pub fn from_words(words: &[PaintWord]) -> Self {
        Self {
            color: words[0],
            offset: words[1][0],
        }
    }
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
            stops.push(stop(premultiplied_components(color, color_space), offset));
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
            extend: gradient.extend.into(),
            color_space,
            first_stop,
            stop_count: stops.len() as u32 - first_stop,
            y_extend: PaintExtend::Pad,
            geometry,
            radii,
        })
    }

    /// The entry for a photo `size` at level 0, placed by `to_photo`, from
    /// viewport positions to pixels of its level 0, and sampled by `sampler`.
    /// Its level and tile words are filled in when the frame is prepared:
    /// until then it draws nothing.
    pub(crate) fn photo(
        size: Size<u32>,
        to_photo: TransformationMatrix,
        sampler: &ImageSampler,
    ) -> Self {
        let filtered = !matches!(sampler.quality, ImageQuality::Low);
        Self {
            transformation: to_photo,
            kind: PaintKind::Image,
            extend: sampler.x_extend.into(),
            y_extend: sampler.y_extend.into(),
            geometry: [
                size.width as f32,
                size.height as f32,
                0.,
                if filtered { 1. } else { 0. },
            ],
            ..Self::default()
        }
    }

    /// The entry for shader program `program_id` filling a box of `size`
    /// logical pixels, placed by `to_local`, from viewport positions to the
    /// box's own logical pixels, measured from its top left. Its parameter
    /// words are added after it, one word each, in place of stops; until
    /// the renderer has linked the program, it draws `fallback`, an
    /// unpremultiplied sRGB-encoded colour.
    pub(crate) fn program(
        program_id: u32,
        size: [f32; 2],
        to_local: TransformationMatrix,
        fallback: [f32; 4],
    ) -> Self {
        assert!(
            program_id < crate::shader::MAX_PROGRAM_ID,
            "program ids must be held exactly by a float"
        );
        Self {
            transformation: to_local,
            kind: PaintKind::Program,
            geometry: [size[0], size[1], program_id as f32, 0.],
            radii: fallback,
            ..Self::default()
        }
    }

    /// The program id of a [`PaintKind::Program`] entry.
    pub fn program_id(&self) -> u32 {
        self.geometry[2] as u32
    }
}

impl From<Extend> for PaintExtend {
    fn from(extend: Extend) -> Self {
        match extend {
            Extend::Pad => Self::Pad,
            Extend::Repeat => Self::Repeat,
            Extend::Reflect => Self::Reflect,
        }
    }
}

impl ScenePaint {
    /// The entry that draws `background` as a primitive with `bounds`, in
    /// the space `to_viewport` maps to the viewport, with its stops added to
    /// `stops`: `None` for a solid colour, or a background that is already
    /// an entry.
    pub(crate) fn background(
        background: &Background,
        bounds: Bounds<ScaledPixels>,
        to_viewport: TransformationMatrix,
        stops: &mut Vec<SceneColorStop>,
    ) -> Option<Self> {
        let to_local = to_viewport.inverse().unwrap_or(TransformationMatrix::UNIT);
        let first_stop = stops.len() as u32;
        let mut entry = Self {
            transformation: to_local,
            first_stop,
            ..Self::default()
        };
        let solid_stop = |stops: &mut Vec<SceneColorStop>, color: SceneHsla| {
            let [red, green, blue] = hsl_to_rgb(color);
            let alpha = color.a;
            stops.push(stop([red * alpha, green * alpha, blue * alpha, alpha], 0.));
        };
        // Patterns are measured from the primitive's origin.
        let from_origin = TransformationMatrix {
            rotation_scale: TransformationMatrix::UNIT.rotation_scale,
            translation: [-bounds.origin.x.0, -bounds.origin.y.0],
        }
        .compose(to_local);
        match background.kind() {
            BackgroundKind::Solid(_) | BackgroundKind::Paint { .. } => return None,
            BackgroundKind::LinearGradient { angle, stops: ends } => {
                // The gradient line runs through the centre at `angle`, its
                // direction squashed to the bounds' aspect, and spans the
                // bounds' width or height, whichever it runs more along.
                let radians = (angle % 360. - 90.).to_radians();
                let (width, height) = (bounds.size.width.0, bounds.size.height.0);
                let mut direction = [radians.cos(), radians.sin()];
                if width > height {
                    direction[1] *= height / width;
                } else {
                    direction[0] *= width / height;
                }
                let length = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
                let unit = [direction[0] / length, direction[1] / length];
                let span = if unit[0].abs() > unit[1].abs() {
                    width
                } else {
                    height
                };
                let center = [
                    bounds.origin.x.0 + width / 2.,
                    bounds.origin.y.0 + height / 2.,
                ];
                let half = [unit[0] * span / 2., unit[1] * span / 2.];
                entry.geometry = [
                    center[0] - half[0],
                    center[1] - half[1],
                    center[0] + half[0],
                    center[1] + half[1],
                ];
                entry.color_space = match background.interpolation_space() {
                    ColorSpace::Srgb => PaintColorSpace::LegacySrgb,
                    ColorSpace::Oklab => PaintColorSpace::LegacyOklab,
                };
                for end in ends {
                    let color = legacy_components(end.color, entry.color_space);
                    stops.push(stop(color, end.percentage));
                }
            }
            BackgroundKind::PatternSlash {
                width, interval, ..
            } => {
                entry.kind = PaintKind::Stripes;
                entry.transformation = from_origin;
                entry.geometry = [width, interval, 0., 0.];
                solid_stop(stops, background.solid);
            }
            BackgroundKind::Checkerboard { size, .. } => {
                entry.kind = PaintKind::Checkerboard;
                entry.transformation = from_origin;
                entry.geometry = [size, 0., 0., 0.];
                solid_stop(stops, background.solid);
            }
        }
        entry.stop_count = stops.len() as u32 - first_stop;
        Some(entry)
    }
}

fn stop(color: [f32; 4], offset: f32) -> SceneColorStop {
    SceneColorStop { color, offset }
}

/// The sRGB-encoded colour of an HSL colour, as the shaders compute it.
fn hsl_to_rgb(color: SceneHsla) -> [f32; 3] {
    let chroma = color.s * (1. - (2. * color.l - 1.).abs());
    [0., 2. / 3., 1. / 3.].map(|phase| {
        let phase = (color.h + phase).fract();
        let unit = ((phase * 6. - 3.).abs() - 1.).clamp(0., 1.);
        (unit - 0.5) * chroma + color.l
    })
}

/// A legacy gradient stop's colour in `space`, unpremultiplied, as the
/// shaders interpolate it.
fn legacy_components(color: SceneHsla, space: PaintColorSpace) -> [f32; 4] {
    let rgb = hsl_to_rgb(color);
    let [first, second, third] = match space {
        PaintColorSpace::LegacyOklab => {
            // The 2.2 transfer curve, then linear sRGB to Oklab.
            let [red, green, blue] = rgb.map(|channel| channel.max(0.).powf(2.2));
            let cone = [
                0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue,
                0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue,
                0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue,
            ]
            .map(f32::cbrt);
            [
                0.2104542553 * cone[0] + 0.7936177850 * cone[1] - 0.0040720468 * cone[2],
                1.9779984951 * cone[0] - 2.4285922050 * cone[1] + 0.4505938995 * cone[2],
                0.0259040371 * cone[0] + 0.7827717662 * cone[1] - 0.8086757660 * cone[2],
            ]
        }
        // The colour encoded again, as the shaders' sRGB gradients are.
        _ => rgb.map(|channel| {
            if channel < 0.0031308 {
                channel * 12.92
            } else {
                1.055 * channel.powf(1. / 2.4) - 0.055
            }
        }),
    };
    [first, second, third, color.a]
}

fn space_tag(space: PaintColorSpace) -> ColorSpaceTag {
    match space {
        PaintColorSpace::Srgb | PaintColorSpace::LegacySrgb => ColorSpaceTag::Srgb,
        PaintColorSpace::LinearSrgb => ColorSpaceTag::LinearSrgb,
        PaintColorSpace::Oklab | PaintColorSpace::LegacyOklab => ColorSpaceTag::Oklab,
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
    fn program_entries_round_trip_through_words() {
        let to_local = TransformationMatrix {
            rotation_scale: [[0.5, -0.25], [0.25, 0.5]],
            translation: [-10., 20.],
        };
        let mut entry =
            ScenePaint::program((1 << 24) - 1, [120., 72.], to_local, [0.2, 0.4, 0.6, 1.]);
        entry.first_stop = 42;
        entry.stop_count = 3;
        let decoded = ScenePaint::from_words(&entry.words());
        assert_eq!(decoded, entry);
        assert_eq!(decoded.kind, PaintKind::Program);
        assert_eq!(decoded.program_id(), (1 << 24) - 1);
        assert_eq!(decoded.geometry[..2], [120., 72.]);
        assert_eq!(decoded.radii, [0.2, 0.4, 0.6, 1.]);
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
