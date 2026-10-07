//! Paints: premultiplied colors that vary per fragment.

use std::{
    fmt,
    sync::{Arc, OnceLock},
};

use wgsl_rs::std::{Vec2f, Vec4f};

use palette::IntoColor;

use crate::{Hsla, hsla_to_rgba};

use super::{
    Expr, Operand, Scalar, Vec2, Vec3, Vec4,
    compile::{CompiledPaint, ShaderError, compile, evaluate},
    expr::{Node, Op},
    library::{fade, over, premultiply, unpremultiply},
    prelude::Fragment,
    value::{Val, sealed::Sealed},
};

/// A premultiplied fragment color built from typed shader expressions.
///
/// Immutable expressions built with ordinary Rust helpers and control flow.
/// Compilation caches one fragment program per expression structure; changing
/// uniform values reuses that program.
///
/// ```ignore
/// fn waves(phase: f32) -> Paint {
///     paint(|px| {
///         let wave = (px.uv().x() * 12.0 + phase).sin() * 0.5 + 0.5;
///         color(rgb(0x315bff)).mix(color(rgb(0xf48bcb)), wave)
///     })
/// }
///
/// let compiled = waves(phase).compile()?;
/// ```
#[derive(Clone)]
pub struct Paint {
    pub(crate) node: Arc<Node>,
    compiled: Arc<OnceLock<Result<CompiledPaint, ShaderError>>>,
    fallback: Option<Hsla>,
}

impl Paint {
    pub(crate) fn from_node(node: Arc<Node>) -> Self {
        debug_assert_eq!(node.ty, Vec4f::TY);
        Self {
            node,
            compiled: Arc::default(),
            fallback: None,
        }
    }

    /// A paint from premultiplied RGBA in GPUI's sRGB channel convention.
    pub fn premultiplied(rgba: impl Operand<Value = Vec4f>) -> Self {
        Self::from_node(rgba.into_expr().node)
    }

    /// A paint from straight-alpha RGBA. Alpha is clamped to `[0, 1]`.
    pub fn straight(rgba: impl Operand<Value = Vec4f>) -> Self {
        Self::premultiplied(premultiply(rgba))
    }

    /// The premultiplied RGBA value. A paint is also an operand standing for
    /// this value, so paints pass straight to `select` and foreign functions.
    pub fn rgba(&self) -> Vec4 {
        Expr::from_node(self.node.clone())
    }

    /// The straight-alpha RGBA value.
    pub fn straight_rgba(&self) -> Vec4 {
        unpremultiply(self)
    }

    /// Straight-alpha red.
    pub fn red(&self) -> Scalar {
        self.straight_rgba().x()
    }

    /// Straight-alpha green.
    pub fn green(&self) -> Scalar {
        self.straight_rgba().y()
    }

    /// Straight-alpha blue.
    pub fn blue(&self) -> Scalar {
        self.straight_rgba().z()
    }

    /// Straight-alpha RGB.
    pub fn rgb(&self) -> Vec3 {
        self.straight_rgba().xyz()
    }

    /// Alpha.
    pub fn alpha(&self) -> Scalar {
        self.rgba().w()
    }

    /// Composite this paint over `back` (premultiplied source-over).
    pub fn over(&self, back: impl Into<Paint>) -> Paint {
        Paint::premultiplied(over(self, back.into()))
    }

    /// Interpolate towards `other` in premultiplied space. The amount is not
    /// clamped, following WGSL `mix`.
    pub fn mix(&self, other: impl Into<Paint>, amount: impl Operand<Value = f32>) -> Paint {
        Paint::premultiplied(self.rgba().mix(other.into(), amount))
    }

    /// This paint where `condition` holds, and `other` elsewhere, per fragment.
    pub fn select(&self, condition: impl Operand<Value = bool>, other: impl Into<Paint>) -> Paint {
        Paint::premultiplied(condition.into_expr().select(self, other.into()))
    }

    /// Scale color and alpha by an opacity clamped to `[0, 1]`.
    pub fn opacity(&self, amount: impl Operand<Value = f32>) -> Paint {
        Paint::premultiplied(fade(self, amount))
    }

    /// Restrict this paint to a coverage clamped to `[0, 1]`, such as a
    /// shape's antialiased edge. Equivalent to [`Paint::opacity`].
    pub fn mask(&self, coverage: impl Operand<Value = f32>) -> Paint {
        self.opacity(coverage)
    }

    /// Restrict this paint to the inside of a signed distance field, with an
    /// antialiased edge: `self.mask(distance.coverage())`. See
    /// [`shape`](super::shape).
    pub fn clip(&self, distance: impl Operand<Value = f32>) -> Paint {
        self.mask(distance.into_expr().coverage())
    }

    /// Evaluate this paint at another fragment.
    pub fn at_fragment(&self, fragment: impl Operand<Value = Fragment>) -> Paint {
        Paint::premultiplied(Expr::<Vec4f>::make(
            Op::Apply,
            [self.node.clone(), fragment.into_expr().node],
        ))
    }

    /// Evaluate this paint at another normalized coordinate of the same box.
    pub fn at(&self, uv: impl Operand<Value = Vec2f>) -> Paint {
        self.at_fragment(Pixel.fragment().at(uv))
    }

    /// Re-address this paint through a mapping of normalized coordinates.
    ///
    /// Mapping changes where the paint is sampled, never the element's
    /// geometry, hit testing, or clipping. Mapping a composite maps all of its
    /// parts; mapping one part before composing affects only that part.
    pub fn map_uv(&self, map: impl FnOnce(Vec2) -> Vec2) -> Paint {
        self.at(map(Pixel.uv()))
    }

    /// The colour painted in place of this paint while a renderer is still
    /// preparing it, and by renderers that cannot run it. Transparent unless
    /// set: estimating it, say from the paint's colour at the centre of the
    /// box, would cost the CPU an evaluation each time the paint is drawn.
    pub fn fallback(mut self, color: impl IntoColor<Hsla>) -> Self {
        self.fallback = Some(color.into_color());
        self
    }

    /// The colour set by [`Paint::fallback`], or transparent.
    pub fn fallback_color(&self) -> Hsla {
        self.fallback.unwrap_or_default()
    }

    /// Evaluate this paint on the CPU, returning premultiplied RGBA.
    /// Returns `None` for excessive depth, or an expression that cannot
    /// compile, or one that reads a filter's [input](Pixel::input).
    /// Derivatives evaluate to zero, so edges antialiased with
    /// [`Scalar::coverage`] are hard on the CPU.
    pub fn evaluate(&self, fragment: Fragment) -> Option<Vec4f> {
        match evaluate(&self.node, Some(fragment), None)? {
            Val::Vec4(rgba) => Some(rgba),
            _ => None,
        }
    }

    /// Evaluate this paint on the CPU as a filter, whose
    /// [input](Pixel::input) at a logical-pixel position in the element is
    /// `input(position)`, premultiplied RGBA. Otherwise as
    /// [`Self::evaluate`].
    pub fn evaluate_filter(
        &self,
        fragment: Fragment,
        input: &dyn Fn(Vec2f) -> Vec4f,
    ) -> Option<Vec4f> {
        match evaluate(&self.node, Some(fragment), Some(input))? {
            Val::Vec4(rgba) => Some(rgba),
            _ => None,
        }
    }

    /// Whether this paint reads a filter's [input](Pixel::input).
    pub fn reads_input(&self) -> bool {
        fn reads(node: &Node, seen: &mut collections::FxHashSet<*const Node>) -> bool {
            if !seen.insert(std::ptr::from_ref(node)) {
                return false;
            }
            matches!(node.op, Op::Input) || node.args.iter().any(|arg| reads(arg, seen))
        }
        reads(&self.node, &mut collections::FxHashSet::default())
    }

    /// Compile and validate this paint. Clones share the result.
    pub fn compile(&self) -> Result<CompiledPaint, ShaderError> {
        self.compiled.get_or_init(|| compile(&self.node)).clone()
    }
}

impl fmt::Debug for Paint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Paint").finish_non_exhaustive()
    }
}

impl PartialEq for Paint {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.node, &other.node)
            || matches!((self.compile(), other.compile()), (Ok(a), Ok(b)) if a == b)
    }
}

impl From<&Paint> for Paint {
    fn from(paint: &Paint) -> Self {
        paint.clone()
    }
}

impl Operand for Paint {
    type Value = Vec4f;
    fn into_expr(self) -> Vec4 {
        Expr::from_node(self.node)
    }
}

impl Operand for &Paint {
    type Value = Vec4f;
    fn into_expr(self) -> Vec4 {
        self.rgba()
    }
}

impl<C: IntoColor<Hsla>> From<C> for Paint {
    fn from(color: C) -> Self {
        let rgba = hsla_to_rgba(color.into_color());
        color_value([
            rgba.color.red,
            rgba.color.green,
            rgba.color.blue,
            rgba.alpha,
        ])
    }
}

/// A uniform color paint. Accepts any GPUI color.
pub fn color(color: impl Into<Paint>) -> Paint {
    color.into()
}

/// A paint from straight-alpha channels, which may vary per fragment.
pub fn rgba(
    red: impl Operand<Value = f32>,
    green: impl Operand<Value = f32>,
    blue: impl Operand<Value = f32>,
    alpha: impl Operand<Value = f32>,
) -> Paint {
    Paint::straight(super::vec4(red, green, blue, alpha))
}

fn color_value([r, g, b, a]: [f32; 4]) -> Paint {
    let a = if a.is_finite() { a.clamp(0., 1.) } else { a };
    Paint::premultiplied(Vec4f {
        x: r * a,
        y: g * a,
        z: b * a,
        w: a,
    })
}

/// The fragment a paint is evaluated at. Zero-sized and `Copy`: pass it to
/// helpers freely.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pixel;

impl Pixel {
    /// The fragment itself, for foreign shader functions.
    pub fn fragment(self) -> Expr<Fragment> {
        Expr::from_node(Node::fragment())
    }

    /// Normalized coordinate: `(0, 0)` at the box's top-left, `(1, 1)` at its
    /// bottom-right.
    pub fn uv(self) -> Vec2 {
        self.fragment().uv()
    }

    /// Logical-pixel offset from the box's top-left corner, in the
    /// element's own space: the paint moves, turns and zooms with the
    /// element.
    pub fn position(self) -> Vec2 {
        self.fragment().position()
    }

    /// Logical-pixel size of the box.
    pub fn size(self) -> Vec2 {
        self.fragment().size()
    }

    /// Logical-pixel offset from the box's center: the natural frame for
    /// [`shape`](super::shape) distances.
    pub fn centered(self) -> Vec2 {
        self.fragment().centered()
    }

    /// Device pixels per logical pixel.
    pub fn scale(self) -> Scalar {
        self.fragment().scale()
    }

    /// Stroke coordinates where a [`Mesh`](crate::Mesh) is painted:
    /// [`Self::along`] and [`Self::across`]. Zero for anything else.
    pub fn stroke(self) -> Vec2 {
        self.fragment().stroke()
    }

    /// The picture a filter program reads, at this fragment: premultiplied
    /// colour. Only a paint run as a filter (see
    /// [`Filter::program`](crate::Filter::program)) has one; as a fill it
    /// does not compile into the renderer, and draws its fallback colour.
    pub fn input(self) -> Paint {
        self.input_at(super::vec2(0.0, 0.0))
    }

    /// The picture a filter program reads, `offset` logical pixels from
    /// this fragment in the element's own space: premultiplied colour,
    /// sampled between pixels linearly. See [`Self::input`].
    pub fn input_at(self, offset: impl Operand<Value = Vec2f>) -> Paint {
        Paint::premultiplied(Expr::<Vec4f>::make(
            Op::Input,
            [self.fragment().node, offset.into_expr().node],
        ))
    }

    /// Distance along the stroke a mesh strip draws, in logical pixels,
    /// from its start (see [`Mesh::strip`](crate::Mesh::strip)).
    pub fn along(self) -> Scalar {
        self.stroke().x()
    }

    /// Where across the stroke a mesh strip draws the fragment is: -1 on
    /// its left edge, 0 at its middle, 1 on its right.
    pub fn across(self) -> Scalar {
        self.stroke().y()
    }
}

/// Build a paint from a closure over the fragment. The closure runs once, on
/// the CPU, to describe the paint; it is not transpiled.
pub fn paint<P: Into<Paint>>(build: impl FnOnce(Pixel) -> P) -> Paint {
    build(Pixel).into()
}
