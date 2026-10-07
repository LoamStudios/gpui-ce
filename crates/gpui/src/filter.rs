//! Filters: what a group's picture goes through before it is composited.
//!
//! A [`Filter`] is one operation of a chain applied to an element and its
//! children rendered as one picture, as CSS `filter` applies its functions:
//! blurs, colour matrices (and the CSS functions built on them), drop
//! shadows, and shader programs that read the picture. Each is plain data
//! that maps onto a CSS filter function or an SVG filter primitive where one
//! exists, so a chain can be described, compared and exported.
//!
//! [`Window::with_compositing`](crate::Window::with_compositing) lowers a
//! chain to [`ScaledFilter`]s in the viewport's device pixels, and the
//! render plan lowers those to [`FilterPass`]es, which every renderer runs
//! the same way on the group's target before compositing it.
//!
//! Colour. Group targets hold premultiplied, sRGB-encoded colour, as
//! everything GPUI draws does. A blur, a drop shadow and the merge of a
//! shadow under its picture work on those values directly, as browsers do.
//! A colour matrix works on straight (unpremultiplied) colour, in the space
//! it names: the CSS functions in sRGB, as browsers apply them, and the
//! photographic adjustments ([`Filter::exposure`], [`Filter::temperature`],
//! [`Filter::tint`]) in linear light, as SVG's `linearRGB` would.

use crate::{Edges, Hsla, Pixels, Point, ScaledPixels, TransformationMatrix, point, px};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// The colour space a [`ColorMatrix`] works in, as SVG's
/// `color-interpolation-filters` names them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum FilterColorSpace {
    /// sRGB-encoded values, as stored: what CSS's filter functions use.
    #[default]
    Srgb,
    /// Linear light: sRGB decoded before the matrix and encoded after.
    LinearRgb,
}

/// A 4×5 matrix applied to straight-alpha colour, as SVG's `feColorMatrix`
/// applies its `values`: row by row, each output channel of R, G, B and A
/// is the dot product of its row with `(R, G, B, A, 1)`. Results are
/// clamped to `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ColorMatrix {
    /// The matrix, row-major: R′, G′, B′ and A′, five values each.
    pub values: [f32; 20],
    /// The space the matrix works in.
    pub color_space: FilterColorSpace,
}

impl Default for ColorMatrix {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// The luma coefficients SVG's `saturate` and `hueRotate` matrices are
/// built on, rounded as the specification rounds them.
const LUMA: [f32; 3] = [0.213, 0.715, 0.072];

/// Rec. 709 luma coefficients, which CSS's `grayscale` is built on.
const REC_709_LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

impl ColorMatrix {
    /// The matrix that changes nothing.
    pub const IDENTITY: Self = Self {
        values: [
            1., 0., 0., 0., 0., //
            0., 1., 0., 0., 0., //
            0., 0., 1., 0., 0., //
            0., 0., 0., 1., 0., //
        ],
        color_space: FilterColorSpace::Srgb,
    };

    /// A matrix of `values`, row-major, in sRGB.
    pub fn new(values: [f32; 20]) -> Self {
        Self {
            values,
            color_space: FilterColorSpace::Srgb,
        }
    }

    /// The same matrix, working in `color_space`.
    pub fn in_space(mut self, color_space: FilterColorSpace) -> Self {
        self.color_space = color_space;
        self
    }

    /// Scales R, G and B by `gains`, leaving alpha.
    pub fn gains(red: f32, green: f32, blue: f32) -> Self {
        Self::new([
            red, 0., 0., 0., 0., //
            0., green, 0., 0., 0., //
            0., 0., blue, 0., 0., //
            0., 0., 0., 1., 0., //
        ])
    }

    /// `slope * channel + intercept` on R, G and B, as SVG's
    /// `feComponentTransfer` of type `linear`.
    pub fn linear(slope: f32, intercept: f32) -> Self {
        Self::new([
            slope, 0., 0., 0., intercept, //
            0., slope, 0., 0., intercept, //
            0., 0., slope, 0., intercept, //
            0., 0., 0., 1., 0., //
        ])
    }

    /// SVG's `saturate` matrix: 0 is grey, 1 unchanged, more oversaturates.
    pub fn saturate(amount: f32) -> Self {
        let s = amount;
        let [r, g, b] = LUMA;
        Self::new([
            r + (1. - r) * s,
            g - g * s,
            b - b * s,
            0.,
            0., //
            r - r * s,
            g + (1. - g) * s,
            b - b * s,
            0.,
            0., //
            r - r * s,
            g - g * s,
            b + (1. - b) * s,
            0.,
            0., //
            0.,
            0.,
            0.,
            1.,
            0., //
        ])
    }

    /// SVG's `hueRotate` matrix, turning hues by `degrees`.
    pub fn hue_rotate(degrees: f32) -> Self {
        let (sin, cos) = degrees.to_radians().sin_cos();
        let [r, g, b] = LUMA;
        Self::new([
            r + cos * (1. - r) - sin * r,
            g - cos * g - sin * g,
            b - cos * b + sin * (1. - b),
            0.,
            0., //
            r - cos * r + sin * 0.143,
            g + cos * (1. - g) + sin * 0.140,
            b - cos * b - sin * 0.283,
            0.,
            0., //
            r - cos * r - sin * (1. - r),
            g - cos * g + sin * g,
            b + cos * (1. - b) + sin * b,
            0.,
            0., //
            0.,
            0.,
            0.,
            1.,
            0., //
        ])
    }

    /// CSS's `grayscale` matrix: 0 unchanged, 1 fully grey.
    pub fn grayscale(amount: f32) -> Self {
        let a = 1. - amount.clamp(0., 1.);
        let [r, g, b] = REC_709_LUMA;
        Self::new([
            r + (1. - r) * a,
            g - g * a,
            b - b * a,
            0.,
            0., //
            r - r * a,
            g + (1. - g) * a,
            b - b * a,
            0.,
            0., //
            r - r * a,
            g - g * a,
            b + (1. - b) * a,
            0.,
            0., //
            0.,
            0.,
            0.,
            1.,
            0., //
        ])
    }

    /// CSS's `sepia` matrix: 0 unchanged, 1 fully sepia.
    pub fn sepia(amount: f32) -> Self {
        let a = 1. - amount.clamp(0., 1.);
        Self::new([
            0.393 + 0.607 * a,
            0.769 - 0.769 * a,
            0.189 - 0.189 * a,
            0.,
            0., //
            0.349 - 0.349 * a,
            0.686 + 0.314 * a,
            0.168 - 0.168 * a,
            0.,
            0., //
            0.272 - 0.272 * a,
            0.534 - 0.534 * a,
            0.131 + 0.869 * a,
            0.,
            0., //
            0.,
            0.,
            0.,
            1.,
            0., //
        ])
    }

    /// CSS's `invert`: 0 unchanged, 1 fully inverted.
    pub fn invert(amount: f32) -> Self {
        let amount = amount.clamp(0., 1.);
        Self::linear(1. - 2. * amount, amount)
    }

    /// CSS's `opacity`: alpha scaled by `amount`.
    pub fn opacity(amount: f32) -> Self {
        let mut matrix = Self::IDENTITY;
        matrix.values[18] = amount.clamp(0., 1.);
        matrix
    }

    /// A drop shadow's matrix: any colour's coverage in `color`,
    /// premultiplied.
    pub fn shadow(color: [f32; 4]) -> Self {
        let straight = color.map(|channel| channel / color[3].max(f32::MIN_POSITIVE));
        #[rustfmt::skip]
        let values = [
            0., 0., 0., 0., straight[0],
            0., 0., 0., 0., straight[1],
            0., 0., 0., 0., straight[2],
            0., 0., 0., color[3], 0.,
        ];
        Self::new(values)
    }

    /// The matrix that applies `self`, then `next`, when both work in the
    /// same space: their product, with no clamping between them.
    pub fn then(&self, next: &Self) -> Self {
        let a = &self.values;
        let b = &next.values;
        let mut values = [0.; 20];
        for row in 0..4 {
            for column in 0..5 {
                let mut value = (0..4)
                    .map(|k| b[row * 5 + k] * a[k * 5 + column])
                    .sum::<f32>();
                if column == 4 {
                    value += b[row * 5 + 4];
                }
                values[row * 5 + column] = value;
            }
        }
        Self {
            values,
            color_space: next.color_space,
        }
    }

    /// Whether the matrix changes nothing.
    pub fn is_identity(&self) -> bool {
        self.values == Self::IDENTITY.values
    }

    /// Whether every colour in `[0, 1]⁴` stays in it, so nothing is clamped
    /// and the matrix can be folded into the next one exactly. The matrix is
    /// affine, so it is enough to check the corners of the cube.
    pub fn keeps_unit_cube(&self) -> bool {
        (0..16u32).all(|corner| {
            let input = [0, 1, 2, 3].map(|bit| ((corner >> bit) & 1) as f32);
            self.apply_straight(input)
                .iter()
                .all(|value| (-1e-6..=1. + 1e-6).contains(value))
        })
    }

    /// The matrix applied to straight-alpha `rgba`, unclamped, in its own
    /// space.
    pub fn apply_straight(&self, rgba: [f32; 4]) -> [f32; 4] {
        let v = &self.values;
        [0, 1, 2, 3].map(|row| {
            v[row * 5] * rgba[0]
                + v[row * 5 + 1] * rgba[1]
                + v[row * 5 + 2] * rgba[2]
                + v[row * 5 + 3] * rgba[3]
                + v[row * 5 + 4]
        })
    }

    /// What a renderer computes for premultiplied, sRGB-encoded `rgba`:
    /// unpremultiplied, decoded to linear light if the matrix works there,
    /// the matrix applied and clamped, encoded again and premultiplied. The
    /// reference the renderers are tested against.
    pub fn apply_premultiplied(&self, rgba: [f32; 4]) -> [f32; 4] {
        let alpha = rgba[3];
        let mut straight = if alpha > 0. {
            [rgba[0] / alpha, rgba[1] / alpha, rgba[2] / alpha, alpha]
        } else {
            [0.; 4]
        };
        let linear = self.color_space == FilterColorSpace::LinearRgb;
        if linear {
            for channel in &mut straight[..3] {
                *channel = srgb_to_linear(*channel);
            }
        }
        let mut out = self
            .apply_straight(straight)
            .map(|value| value.clamp(0., 1.));
        if linear {
            for channel in &mut out[..3] {
                *channel = linear_to_srgb(*channel);
            }
        }
        [out[0] * out[3], out[1] * out[3], out[2] * out[3], out[3]]
    }

    /// The rows as the shaders take them: R′, G′, B′ and A′ coefficients of
    /// R, G, B and A, then the offsets.
    pub fn shader_rows(&self) -> [[f32; 4]; 5] {
        let v = &self.values;
        [
            [v[0], v[1], v[2], v[3]],
            [v[5], v[6], v[7], v[8]],
            [v[10], v[11], v[12], v[13]],
            [v[15], v[16], v[17], v[18]],
            [v[4], v[9], v[14], v[19]],
        ]
    }
}

/// An sRGB-encoded channel in linear light.
pub fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// A linear-light channel, sRGB-encoded.
pub fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1. / 2.4) - 0.055
    }
}

/// A shadow cast by a group's picture, composited beneath it: CSS's
/// `drop-shadow()`, SVG's `feDropShadow`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DropShadow {
    /// How far the shadow is moved from the picture, in the element's
    /// logical pixels.
    pub offset: Point<Pixels>,
    /// How far it is blurred, as CSS measures a shadow's blur: twice the
    /// Gaussian's standard deviation.
    pub blur_radius: Pixels,
    /// Its colour, scaled by the picture's coverage.
    pub color: Hsla,
}

/// A shader program applied to a group's picture: a
/// [`Paint`](crate::shader::Paint) that reads the picture with
/// [`Pixel::input`](crate::shader::Pixel::input) and
/// [`Pixel::input_at`](crate::shader::Pixel::input_at), evaluated over the
/// group's element, in its own logical pixels, as a fill would be. What it
/// returns replaces the picture.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramFilter {
    /// The program.
    pub paint: crate::shader::Paint,
    /// How far, in the element's logical pixels, what it returns may reach
    /// beyond what the group draws: the group's target is grown by this
    /// much on every side. Programs that only recolour need none; one that
    /// reads its neighbours `n` pixels away and draws where the picture is
    /// clear, as a glow does, needs `n`.
    pub extent: Pixels,
}

/// A graphical filter that can be applied either to an element's own content
/// (via [`Styled::filter`](crate::Styled::filter), like CSS `filter`) or to the
/// content rendered behind it (via
/// [`Styled::backdrop_filter`](crate::Styled::backdrop_filter), like CSS
/// `backdrop-filter`, which applies only blurs).
///
/// Build chains with [`Filter::then`]:
///
/// ```ignore
/// div().filter(Filter::blur(px(4.)).then(Filter::saturate(1.2)))
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Filter {
    /// A Gaussian blur of this standard deviation, in logical pixels: CSS
    /// `blur()`, SVG `feGaussianBlur`'s `stdDeviation`.
    Blur(Pixels),
    /// A colour matrix: SVG `feColorMatrix`, and CSS's `brightness`,
    /// `contrast`, `saturate`, `hue-rotate`, `grayscale`, `sepia`,
    /// `invert` and `opacity`.
    ColorMatrix(ColorMatrix),
    /// A shadow beneath the picture: CSS `drop-shadow()`.
    DropShadow(DropShadow),
    /// A shader program over the picture. Not serialized: export rasterizes
    /// it.
    #[serde(skip)]
    Program(ProgramFilter),
}

impl Filter {
    /// CSS `blur(<std_deviation>)`: a Gaussian blur of that standard
    /// deviation, in logical pixels. The group's target grows to hold it,
    /// so it is not cut off at the element's edges.
    pub fn blur(std_deviation: impl Into<Pixels>) -> Self {
        Self::Blur(std_deviation.into())
    }

    /// A colour matrix.
    pub fn color_matrix(matrix: ColorMatrix) -> Self {
        Self::ColorMatrix(matrix)
    }

    /// CSS `brightness()`: R, G and B scaled by `amount`.
    pub fn brightness(amount: f32) -> Self {
        Self::ColorMatrix(ColorMatrix::gains(amount, amount, amount))
    }

    /// CSS `contrast()`: R, G and B moved away from mid-grey by `amount`.
    pub fn contrast(amount: f32) -> Self {
        Self::ColorMatrix(ColorMatrix::linear(amount, 0.5 - 0.5 * amount))
    }

    /// CSS `saturate()`.
    pub fn saturate(amount: f32) -> Self {
        Self::ColorMatrix(ColorMatrix::saturate(amount))
    }

    /// CSS `hue-rotate()`, in degrees.
    pub fn hue_rotate(degrees: f32) -> Self {
        Self::ColorMatrix(ColorMatrix::hue_rotate(degrees))
    }

    /// CSS `grayscale()`.
    pub fn grayscale(amount: f32) -> Self {
        Self::ColorMatrix(ColorMatrix::grayscale(amount))
    }

    /// CSS `sepia()`.
    pub fn sepia(amount: f32) -> Self {
        Self::ColorMatrix(ColorMatrix::sepia(amount))
    }

    /// CSS `invert()`.
    pub fn invert(amount: f32) -> Self {
        Self::ColorMatrix(ColorMatrix::invert(amount))
    }

    /// CSS `opacity()`, as a filter in the chain.
    pub fn opacity(amount: f32) -> Self {
        Self::ColorMatrix(ColorMatrix::opacity(amount))
    }

    /// A photographic exposure change of `stops`: linear light scaled by
    /// 2^`stops`.
    pub fn exposure(stops: f32) -> Self {
        let gain = stops.exp2();
        Self::ColorMatrix(
            ColorMatrix::gains(gain, gain, gain).in_space(FilterColorSpace::LinearRgb),
        )
    }

    /// A white-balance shift, from -1 (cooler, bluer) to 1 (warmer,
    /// yellower): linear-light red scaled by `1 + 0.3·amount` and blue by
    /// `1 − 0.3·amount`.
    pub fn temperature(amount: f32) -> Self {
        Self::ColorMatrix(
            ColorMatrix::gains(1. + 0.3 * amount, 1., 1. - 0.3 * amount)
                .in_space(FilterColorSpace::LinearRgb),
        )
    }

    /// A tint shift, from -1 (greener) to 1 (more magenta): linear-light
    /// green scaled by `1 − 0.3·amount`.
    pub fn tint(amount: f32) -> Self {
        Self::ColorMatrix(
            ColorMatrix::gains(1., 1. - 0.3 * amount, 1.).in_space(FilterColorSpace::LinearRgb),
        )
    }

    /// CSS `drop-shadow(<offset_x> <offset_y> <blur_radius> <color>)`.
    pub fn drop_shadow(
        offset_x: impl Into<Pixels>,
        offset_y: impl Into<Pixels>,
        blur_radius: impl Into<Pixels>,
        color: impl palette::IntoColor<Hsla>,
    ) -> Self {
        Self::DropShadow(DropShadow {
            offset: point(offset_x.into(), offset_y.into()),
            blur_radius: blur_radius.into(),
            color: color.into_color(),
        })
    }

    /// A shader program over the picture; see [`ProgramFilter`].
    pub fn program(paint: crate::shader::Paint) -> Self {
        Self::Program(ProgramFilter {
            paint,
            extent: px(0.),
        })
    }

    /// A shader program over the picture whose result may reach `extent`
    /// logical pixels beyond what the group draws; see [`ProgramFilter`].
    pub fn program_with_extent(paint: crate::shader::Paint, extent: impl Into<Pixels>) -> Self {
        Self::Program(ProgramFilter {
            paint,
            extent: extent.into(),
        })
    }

    /// This filter, then `next`.
    pub fn then(self, next: Filter) -> Filters {
        Filters(vec![self, next])
    }

    /// Whether this filter has no visible effect, so painting can skip it entirely (and the
    /// element can avoid the offscreen isolation pass when *all* of its filters are identities).
    pub fn is_identity(&self) -> bool {
        match self {
            Filter::Blur(std_deviation) => *std_deviation <= Pixels::ZERO,
            Filter::ColorMatrix(matrix) => matrix.is_identity(),
            Filter::DropShadow(shadow) => shadow.color.alpha <= 0.,
            Filter::Program(_) => false,
        }
    }

    /// Whether this filter blurs, which is all a backdrop filter applies.
    pub fn is_blur(&self) -> bool {
        matches!(self, Filter::Blur(_))
    }
}

/// An ordered chain of [`Filter`]s, built with [`Filter::then`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Filters(pub Vec<Filter>);

impl Filters {
    /// This chain, then `next`.
    pub fn then(mut self, next: Filter) -> Self {
        self.0.push(next);
        self
    }
}

impl From<Filter> for Vec<Filter> {
    fn from(filter: Filter) -> Self {
        vec![filter]
    }
}

impl From<Filters> for Vec<Filter> {
    fn from(filters: Filters) -> Self {
        filters.0
    }
}

/// The scene-space (device-pixel) form of a [`Filter`], carried on the scene primitives that the
/// renderers consume, in the viewport's device pixels.
///
/// This is intentionally a separate enum from [`Filter`] (rather than reusing it) so the scene
/// stays in device space like every other primitive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScaledFilter {
    /// A gaussian blur of this standard deviation, in device pixels.
    Blur(ScaledPixels),
    /// A colour matrix.
    ColorMatrix(ColorMatrix),
    /// A shadow beneath the picture.
    DropShadow {
        /// How far it is moved, in the viewport's device pixels.
        offset: Point<ScaledPixels>,
        /// Its blur's standard deviation, in device pixels.
        std_deviation: ScaledPixels,
        /// Its colour, premultiplied and sRGB-encoded.
        color: [f32; 4],
    },
    /// A shader program over the picture.
    Program {
        /// Its paint-table entry, which places it over the element.
        paint: u32,
        /// Its program's id.
        program: u32,
        /// From the element's logical pixels to the viewport's device
        /// pixels: where `input_at` offsets go.
        to_viewport: TransformationMatrix,
        /// How far its result may reach beyond the picture, in device pixels.
        extent: ScaledPixels,
    },
}

/// How far a gaussian blur spreads, per unit of its standard deviation: the
/// renderers cut its kernel off at three.
pub const GAUSSIAN_EXTENT_PER_RADIUS: f32 = 3.0;

impl ScaledFilter {
    /// The filter as it is where `placement` moves what it filters, from
    /// one viewport position to another: offsets turn and scale, blurs
    /// scale, and a program's placement follows.
    pub fn placed(&self, placement: &TransformationMatrix) -> Self {
        let [[a, b], [c, d]] = placement.rotation_scale;
        let scale = (a * d - b * c).abs().sqrt();
        let length = |value: ScaledPixels| ScaledPixels(value.0 * scale);
        match *self {
            ScaledFilter::Blur(std_deviation) => ScaledFilter::Blur(length(std_deviation)),
            ScaledFilter::ColorMatrix(matrix) => ScaledFilter::ColorMatrix(matrix),
            ScaledFilter::DropShadow {
                offset,
                std_deviation,
                color,
            } => ScaledFilter::DropShadow {
                offset: point(
                    ScaledPixels(a * offset.x.0 + b * offset.y.0),
                    ScaledPixels(c * offset.x.0 + d * offset.y.0),
                ),
                std_deviation: length(std_deviation),
                color,
            },
            ScaledFilter::Program {
                paint,
                program,
                to_viewport,
                extent,
            } => ScaledFilter::Program {
                paint,
                program,
                to_viewport: placement.compose(to_viewport),
                extent: length(extent),
            },
        }
    }
}

/// How far a chain of filters spreads what it filters beyond what it was
/// drawn on, on each side, in device pixels.
pub fn filter_outsets(filters: &[ScaledFilter]) -> Edges<f32> {
    let mut outsets = Edges::<f32>::default();
    for filter in filters {
        match filter {
            ScaledFilter::Blur(std_deviation) => {
                let spread = GAUSSIAN_EXTENT_PER_RADIUS * std_deviation.0.max(0.);
                outsets = outsets.map(|side| side + spread);
            }
            ScaledFilter::ColorMatrix(_) => {}
            ScaledFilter::DropShadow {
                offset,
                std_deviation,
                ..
            } => {
                let spread = GAUSSIAN_EXTENT_PER_RADIUS * std_deviation.0.max(0.);
                // The shadow is the picture so far, moved and spread; the
                // result holds both.
                outsets = Edges {
                    top: outsets.top.max(outsets.top + spread - offset.y.0),
                    right: outsets.right.max(outsets.right + spread + offset.x.0),
                    bottom: outsets.bottom.max(outsets.bottom + spread + offset.y.0),
                    left: outsets.left.max(outsets.left + spread - offset.x.0),
                };
            }
            ScaledFilter::Program { extent, .. } => {
                let extent = extent.0.max(0.);
                outsets = outsets.map(|side| side + extent);
            }
        }
    }
    outsets
}

/// The largest blur in a chain, its standard deviation in device pixels:
/// what a backdrop filter applies.
pub fn max_blur_radius(filters: &[ScaledFilter]) -> f32 {
    filters.iter().fold(0.0, |radius, filter| match filter {
        ScaledFilter::Blur(filter_radius) => radius.max(filter_radius.0),
        _ => radius,
    })
}

/// A picture a [`FilterPass`] reads: the group's own, or what an earlier
/// pass made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterImage {
    /// What the group drew.
    Content,
    /// The result of the pass at this index.
    Pass(usize),
}

/// One pass of a group's filter chain, as a renderer runs it: it reads one
/// or two pictures that cover the group's target and writes a new one that
/// covers it too.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FilterPass {
    /// A colour matrix applied to `input`, moved by `offset`.
    ColorMatrix {
        /// What it reads.
        input: FilterImage,
        /// The matrix.
        matrix: ColorMatrix,
        /// How far `input` is moved first, in device pixels: a drop
        /// shadow's offset, so that it is blurred where it is shown.
        offset: Point<ScaledPixels>,
    },
    /// A separable Gaussian blur of `input`.
    Blur {
        /// What it reads.
        input: FilterImage,
        /// Its standard deviation, in device pixels.
        std_deviation: f32,
        /// For a drop shadow, what it blurs is `input`'s coverage in this
        /// colour, moved: the shadow is cast as it is blurred, not in a
        /// pass of its own.
        tint: Option<ShadowTint>,
    },
    /// `top` composited over `bottom`.
    Merge {
        /// What is on top.
        top: FilterImage,
        /// What is beneath it.
        bottom: FilterImage,
    },
    /// A shader program, reading `input`.
    Program {
        /// What it reads.
        input: FilterImage,
        /// Its paint-table entry.
        paint: u32,
        /// Its program's id.
        program: u32,
        /// From the element's logical pixels to the viewport's device pixels.
        to_viewport: TransformationMatrix,
    },
}

/// A drop shadow's colour and offset, as a [`FilterPass::Blur`] casts it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowTint {
    /// Its colour, premultiplied and sRGB-encoded.
    pub color: [f32; 4],
    /// How far it is moved, in device pixels.
    pub offset: Point<ScaledPixels>,
}

/// What a group's composite does to the picture it draws, in place of the
/// plan's last pass.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum CompositeFilter {
    /// Nothing: the picture is drawn as it is.
    #[default]
    None,
    /// The picture is recoloured by this matrix as it is drawn.
    ColorMatrix(ColorMatrix),
    /// The picture is merged over this one as it is drawn.
    Merge {
        /// What is beneath it.
        bottom: FilterImage,
    },
}

/// The passes a group's filters run, in order, the picture that is
/// composited (the group's own when `output` is `None`), and what the
/// composite does to it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FilterPlan {
    /// The passes, in order.
    pub passes: SmallVec<[FilterPass; 4]>,
    /// What is composited.
    pub output: Option<usize>,
    /// What the composite does to it.
    pub composite: CompositeFilter,
}

impl FilterPlan {
    /// The passes that run `filters`.
    ///
    /// Adjacent colour matrices in the same space are folded into one pass
    /// when the first keeps every colour in range, so nothing it would have
    /// clamped is lost. A drop shadow is two passes: its blur, of the
    /// picture's coverage in its colour, moved (or, unblurred, a colour
    /// matrix that makes that); and the picture merged over it.
    pub fn new(filters: &[ScaledFilter]) -> Self {
        let mut plan = Self::default();
        for filter in filters {
            let input = plan.head();
            match *filter {
                ScaledFilter::ColorMatrix(matrix) => {
                    if matrix.is_identity() {
                        continue;
                    }
                    if let Some(last) = plan.output
                        && let FilterPass::ColorMatrix {
                            matrix: previous,
                            offset,
                            ..
                        } = &mut plan.passes[last]
                        && offset.x.0 == 0.
                        && offset.y.0 == 0.
                        && previous.color_space == matrix.color_space
                        && previous.keeps_unit_cube()
                    {
                        *previous = previous.then(&matrix);
                        continue;
                    }
                    plan.push(FilterPass::ColorMatrix {
                        input,
                        matrix,
                        offset: Point::default(),
                    });
                }
                ScaledFilter::Blur(std_deviation) => {
                    if std_deviation.0 > 0. {
                        plan.push(FilterPass::Blur {
                            input,
                            std_deviation: std_deviation.0,
                            tint: None,
                        });
                    }
                }
                ScaledFilter::DropShadow {
                    offset,
                    std_deviation,
                    color,
                } => {
                    if color[3] <= 0. {
                        continue;
                    }
                    let shadow = if std_deviation.0 > 0. {
                        plan.push(FilterPass::Blur {
                            input,
                            std_deviation: std_deviation.0,
                            tint: Some(ShadowTint { color, offset }),
                        })
                    } else {
                        plan.push(FilterPass::ColorMatrix {
                            input,
                            matrix: ColorMatrix::shadow(color),
                            offset,
                        })
                    };
                    plan.push(FilterPass::Merge {
                        top: input,
                        bottom: shadow,
                    });
                }
                ScaledFilter::Program {
                    paint,
                    program,
                    to_viewport,
                    ..
                } => {
                    plan.push(FilterPass::Program {
                        input,
                        paint,
                        program,
                        to_viewport,
                    });
                }
            }
        }
        plan
    }

    /// This plan with its last pass done by the composite instead, where it
    /// can be: a colour matrix that does not move the picture, or a merge,
    /// when the composite's `blend_mode` leaves it free to read a second
    /// picture. Each spares a pass over the whole target.
    pub fn fused_into_composite(mut self, blend_mode: crate::BlendMode) -> Self {
        let Some(last) = self.output else {
            return self;
        };
        if last + 1 != self.passes.len() || self.composite != CompositeFilter::None {
            return self;
        }
        let (composite, output) = match self.passes[last] {
            FilterPass::ColorMatrix {
                input,
                matrix,
                offset,
            } if offset.x.0 == 0. && offset.y.0 == 0. => {
                (CompositeFilter::ColorMatrix(matrix), input)
            }
            FilterPass::Merge { top, bottom } if blend_mode == crate::BlendMode::Normal => {
                (CompositeFilter::Merge { bottom }, top)
            }
            _ => return self,
        };
        self.passes.pop();
        self.composite = composite;
        self.output = match output {
            FilterImage::Content => None,
            FilterImage::Pass(pass) => Some(pass),
        };
        self
    }

    /// The pictures the composite reads besides the output: a merge's
    /// bottom.
    pub fn composite_inputs(&self) -> Option<FilterImage> {
        match self.composite {
            CompositeFilter::Merge { bottom } => Some(bottom),
            _ => None,
        }
    }

    /// The picture the next pass reads.
    fn head(&self) -> FilterImage {
        self.output.map_or(FilterImage::Content, FilterImage::Pass)
    }

    fn push(&mut self, pass: FilterPass) -> FilterImage {
        self.passes.push(pass);
        let index = self.passes.len() - 1;
        self.output = Some(index);
        FilterImage::Pass(index)
    }

    /// The pictures `pass` reads.
    pub fn inputs(pass: &FilterPass) -> SmallVec<[FilterImage; 2]> {
        match *pass {
            FilterPass::ColorMatrix { input, .. }
            | FilterPass::Blur { input, .. }
            | FilterPass::Program { input, .. } => smallvec::smallvec![input],
            FilterPass::Merge { top, bottom, .. } => smallvec::smallvec![top, bottom],
        }
    }

    /// For each pass's result, the index of the last pass that reads it, or
    /// `None` if it is only composited: after that pass, a renderer can let
    /// go of it.
    pub fn last_reads(&self) -> SmallVec<[Option<usize>; 4]> {
        let mut last = SmallVec::from_elem(None, self.passes.len());
        for (index, pass) in self.passes.iter().enumerate() {
            for input in Self::inputs(pass) {
                if let FilterImage::Pass(read) = input {
                    last[read] = Some(index);
                }
            }
        }
        last
    }
}

/// A Gaussian of standard deviation `sigma` at `x`, unnormalized.
fn gaussian(x: f32, sigma: f32) -> f32 {
    (-(x * x) / (2. * sigma * sigma)).exp()
}

/// The coverage a hard edge at 0, opaque for negative `x`, has at `x` once
/// blurred with standard deviation `sigma`: the reference the renderers'
/// blurs are tested against.
pub fn blurred_edge_coverage(x: f32, sigma: f32) -> f32 {
    // A numerical integral of the normalized Gaussian, fine enough for tests.
    let steps = 2000;
    let reach = 6. * sigma;
    let step = 2. * reach / steps as f32;
    let (mut inside, mut total) = (0., 0.);
    for index in 0..steps {
        let at = -reach + (index as f32 + 0.5) * step;
        let weight = gaussian(at, sigma);
        total += weight;
        if x + at < 0. {
            inside += weight;
        }
    }
    inside / total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-4)
    }

    #[test]
    fn css_matrices_match_their_definitions() {
        // saturate(0) is grey by SVG's luma; grayscale(1) by Rec. 709's.
        let grey = ColorMatrix::saturate(0.).apply_straight([1., 0., 0., 1.]);
        assert!(close(grey, [0.213, 0.213, 0.213, 1.]));
        let grey = ColorMatrix::grayscale(1.).apply_straight([0., 1., 0., 1.]);
        assert!(close(grey, [0.7152, 0.7152, 0.7152, 1.]));
        // hue-rotate(0) and saturate(1) change nothing.
        for matrix in [ColorMatrix::hue_rotate(0.), ColorMatrix::saturate(1.)] {
            assert!(close(
                matrix.apply_straight([0.2, 0.4, 0.6, 1.]),
                [0.2, 0.4, 0.6, 1.]
            ));
        }
        // hue-rotate(180) of grey is grey.
        let turned = ColorMatrix::hue_rotate(180.).apply_straight([0.5, 0.5, 0.5, 1.]);
        assert!(close(turned, [0.5, 0.5, 0.5, 1.]), "{turned:?}");
        // invert(1) and contrast(0).
        assert!(close(
            ColorMatrix::invert(1.).apply_straight([0.25, 0.5, 1., 1.]),
            [0.75, 0.5, 0., 1.]
        ));
        let Filter::ColorMatrix(contrast) = Filter::contrast(0.) else {
            unreachable!()
        };
        assert!(close(
            contrast.apply_straight([0.1, 0.9, 0.3, 1.]),
            [0.5, 0.5, 0.5, 1.]
        ));
        // sepia(1) of white is the sum of each row.
        let sepia = ColorMatrix::sepia(1.).apply_straight([1., 1., 1., 1.]);
        assert!(close(sepia, [1.351, 1.203, 0.937, 1.]));
    }

    #[test]
    fn exposure_works_in_linear_light() {
        let Filter::ColorMatrix(exposure) = Filter::exposure(1.) else {
            unreachable!()
        };
        // Mid-grey in sRGB, 0.214 linear, doubled is 0.428 linear: 0.686.
        let brighter = exposure.apply_premultiplied([0.5, 0.5, 0.5, 1.]);
        assert!((brighter[0] - 0.6858).abs() < 1e-3, "{brighter:?}");
        // Premultiplied colour keeps its alpha and is unpremultiplied first.
        let half = exposure.apply_premultiplied([0.25, 0.25, 0.25, 0.5]);
        assert!(close(
            half,
            [brighter[0] / 2., brighter[0] / 2., brighter[0] / 2., 0.5]
        ));
    }

    #[test]
    fn composed_matrices_apply_one_then_the_other() {
        let first = ColorMatrix::saturate(0.6);
        let second = ColorMatrix::linear(0.8, 0.1);
        let color = [0.3, 0.7, 0.2, 1.];
        let one_then_other = second.apply_straight(first.apply_straight(color));
        assert!(close(
            first.then(&second).apply_straight(color),
            one_then_other
        ));
    }

    #[test]
    fn adjacent_matrices_fold_only_when_nothing_would_be_clamped() {
        let matrix = |matrix: ColorMatrix| ScaledFilter::ColorMatrix(matrix);
        let plan = FilterPlan::new(&[
            matrix(ColorMatrix::saturate(0.5)),
            matrix(ColorMatrix::sepia(0.3)),
        ]);
        assert_eq!(plan.passes.len(), 1);
        // brightness(2) clamps; its result is not folded into the next.
        let plan = FilterPlan::new(&[
            matrix(ColorMatrix::gains(2., 2., 2.)),
            matrix(ColorMatrix::gains(0.5, 0.5, 0.5)),
        ]);
        assert_eq!(plan.passes.len(), 2);
        // Nor are matrices in different spaces.
        let plan = FilterPlan::new(&[
            matrix(ColorMatrix::saturate(0.5)),
            matrix(ColorMatrix::IDENTITY.in_space(FilterColorSpace::LinearRgb)),
            matrix(ColorMatrix::gains(0.5, 1., 1.).in_space(FilterColorSpace::LinearRgb)),
        ]);
        assert_eq!(plan.passes.len(), 2);
        assert_eq!(plan.output, Some(1));
    }

    #[test]
    fn a_drop_shadow_is_cast_as_it_is_blurred_and_merged_beneath() {
        let offset = point(ScaledPixels(4.), ScaledPixels(-2.));
        let color = [0., 0., 0.5, 0.5];
        let plan = FilterPlan::new(&[
            ScaledFilter::Blur(ScaledPixels(2.)),
            ScaledFilter::DropShadow {
                offset,
                std_deviation: ScaledPixels(6.),
                color,
            },
        ]);
        // The shadow's blur reads the blurred picture's coverage, moved
        // before it is blurred, so its blur is not cut off.
        assert_eq!(
            plan.passes.as_slice(),
            [
                FilterPass::Blur {
                    input: FilterImage::Content,
                    std_deviation: 2.,
                    tint: None,
                },
                FilterPass::Blur {
                    input: FilterImage::Pass(0),
                    std_deviation: 6.,
                    tint: Some(ShadowTint { color, offset }),
                },
                FilterPass::Merge {
                    top: FilterImage::Pass(0),
                    bottom: FilterImage::Pass(1),
                },
            ]
        );
        assert_eq!(plan.output, Some(2));
        assert_eq!(plan.last_reads().as_slice(), [Some(2), Some(2), None]);

        // Unblurred, it is a colour matrix, moved: any opaque colour
        // becomes the shadow's.
        let plan = FilterPlan::new(&[ScaledFilter::DropShadow {
            offset,
            std_deviation: ScaledPixels(0.),
            color,
        }]);
        let FilterPass::ColorMatrix {
            input,
            matrix,
            offset: moved,
        } = plan.passes[0]
        else {
            panic!("{:?}", plan.passes[0]);
        };
        assert_eq!(input, FilterImage::Content);
        assert_eq!(moved, offset);
        assert!(close(
            matrix.apply_premultiplied([0.3, 0.6, 0.9, 1.]),
            [0., 0., 0.5, 0.5]
        ));
    }

    /// How many passes over the target each kind of chain takes, once its
    /// last is done by the composite where it can be: a guard against
    /// passes creeping back.
    #[test]
    fn chains_take_the_fewest_passes() {
        use crate::BlendMode;
        let shadow = |std_deviation: f32| ScaledFilter::DropShadow {
            offset: point(ScaledPixels(8.), ScaledPixels(12.)),
            std_deviation: ScaledPixels(std_deviation),
            color: [0., 0., 0., 0.5],
        };
        let saturate = ScaledFilter::ColorMatrix(ColorMatrix::saturate(1.4));
        let blur = ScaledFilter::Blur(ScaledPixels(16.));
        let program = ScaledFilter::Program {
            paint: 0,
            program: 1,
            to_viewport: TransformationMatrix::unit(),
            extent: ScaledPixels(0.),
        };
        let passes = |filters: &[ScaledFilter], blend_mode| {
            let plan = FilterPlan::new(filters).fused_into_composite(blend_mode);
            (plan.passes.len(), plan.composite)
        };
        // A colour matrix alone is drawn by the composite.
        assert_eq!(
            passes(&[saturate], BlendMode::Normal),
            (0, CompositeFilter::ColorMatrix(ColorMatrix::saturate(1.4)))
        );
        assert_eq!(passes(&[blur], BlendMode::Normal).0, 1);
        assert_eq!(passes(&[blur, saturate], BlendMode::Normal).0, 1);
        assert_eq!(passes(&[program], BlendMode::Normal).0, 1);
        // A shadow is one blur, merged beneath by the composite; unblurred,
        // one colour matrix.
        assert_eq!(
            passes(&[shadow(16.)], BlendMode::Normal),
            (
                1,
                CompositeFilter::Merge {
                    bottom: FilterImage::Pass(0)
                }
            )
        );
        assert_eq!(passes(&[shadow(0.)], BlendMode::Normal).0, 1);
        // A blend mode reads the parent where the merge would read the
        // shadow: the merge is a pass of its own.
        assert_eq!(
            passes(&[shadow(16.)], BlendMode::Multiply),
            (2, CompositeFilter::None)
        );
        // A matrix after a shadow is the composite's; the merge is not.
        let plan = FilterPlan::new(&[shadow(4.), saturate]).fused_into_composite(BlendMode::Normal);
        assert_eq!(plan.passes.len(), 2);
        assert_eq!(plan.output, Some(1));
        assert!(matches!(plan.composite, CompositeFilter::ColorMatrix(_)));
    }

    #[test]
    fn outsets_hold_blurs_and_moved_shadows() {
        let outsets = filter_outsets(&[
            ScaledFilter::Blur(ScaledPixels(2.)),
            ScaledFilter::DropShadow {
                offset: point(ScaledPixels(10.), ScaledPixels(0.)),
                std_deviation: ScaledPixels(4.),
                color: [0., 0., 0., 1.],
            },
        ]);
        // The blur spreads 6; the shadow 12 more, moved 10 right.
        assert_eq!(outsets.left, 8.);
        assert_eq!(outsets.right, 28.);
        assert_eq!(outsets.top, 18.);
        assert_eq!(outsets.bottom, 18.);
    }

    #[test]
    fn placed_filters_turn_their_offsets_and_scale_their_blurs() {
        let shadow = ScaledFilter::DropShadow {
            offset: point(ScaledPixels(10.), ScaledPixels(0.)),
            std_deviation: ScaledPixels(4.),
            color: [0., 0., 0., 1.],
        };
        // A quarter turn, doubled.
        let placement = TransformationMatrix {
            rotation_scale: [[0., -2.], [2., 0.]],
            translation: [5., 5.],
        };
        let ScaledFilter::DropShadow {
            offset,
            std_deviation,
            ..
        } = shadow.placed(&placement)
        else {
            unreachable!()
        };
        assert_eq!(offset, point(ScaledPixels(0.), ScaledPixels(20.)));
        assert_eq!(std_deviation, ScaledPixels(8.));
    }

    #[test]
    fn a_blurred_edge_is_half_covered_at_the_edge() {
        assert!((blurred_edge_coverage(0., 3.) - 0.5).abs() < 1e-3);
        // One standard deviation out: 15.9%.
        assert!((blurred_edge_coverage(3., 3.) - 0.1587).abs() < 2e-3);
    }

    #[test]
    fn chains_are_built_in_order() {
        let chain: Vec<Filter> = Filter::blur(px(4.))
            .then(Filter::saturate(1.2))
            .then(Filter::invert(1.))
            .into();
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0], Filter::Blur(px(4.)));
    }
}
