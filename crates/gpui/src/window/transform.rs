//! Transforms on elements.
//!
//! [`Window::with_transform`] draws a subtree under an affine transform:
//! moved, rotated, scaled or skewed. Elements inside keep laying out and
//! painting in their own coordinates, and the window maps what they paint,
//! the hitboxes they insert, and the mouse events they receive.
//!
//! A transform that only translates and scales uniformly is folded into the
//! geometry as it is painted, so it stays on the device-pixel grid and text
//! is rasterized at the size it appears. Any other transform gets an entry in
//! the scene's transform table, and its subtree is painted at the transform's
//! scale and placed by that entry on the GPU. Clips set inside such a subtree
//! go into the scene's clip table, since they are no longer aligned with the
//! viewport.

use super::*;

/// How the elements being drawn map to the window: the top of the window's
/// transform stack.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ElementSpace {
    /// From element coordinates to window coordinates, in logical pixels.
    pub(crate) to_window: TransformationMatrix,
    /// From window coordinates back to element coordinates.
    pub(crate) to_element: TransformationMatrix,
    /// Paint geometry is element geometry scaled by `scale` and moved by
    /// `offset`, in logical pixels; the window's scale factor then makes it
    /// device pixels.
    pub(crate) scale: f32,
    pub(crate) offset: Point<Pixels>,
    /// For a space not aligned with the window: from paint geometry, in device
    /// pixels, to the viewport. `None` where paint geometry is already in
    /// window coordinates.
    pub(crate) scene_transformation: Option<TransformationMatrix>,
    /// The space's entry in the scene's transform table, once a primitive has
    /// used it.
    pub(crate) scene_transform: Option<u32>,
}

impl ElementSpace {
    /// The window's own space.
    pub(crate) const WINDOW: Self = Self {
        to_window: TransformationMatrix::UNIT,
        to_element: TransformationMatrix::UNIT,
        scale: 1.,
        offset: Point {
            x: Pixels::ZERO,
            y: Pixels::ZERO,
        },
        scene_transformation: None,
        scene_transform: None,
    };

    /// The space that `to_window` maps into the window, painted at
    /// `scale_factor` device pixels per logical pixel.
    fn new(to_window: TransformationMatrix, scale_factor: f32) -> Self {
        let to_element = to_window.inverse().unwrap_or(TransformationMatrix::UNIT);
        let [[a, b], [c, d]] = to_window.rotation_scale;
        let [x, y] = to_window.translation;
        if b == 0. && c == 0. && a == d && a > 0. {
            return Self {
                to_window,
                to_element,
                scale: a,
                offset: point(px(x), px(y)),
                scene_transformation: None,
                scene_transform: None,
            };
        }
        // Paint at the transform's scale, so text and edges are rasterized
        // at the size they appear, and let the scene transform place it.
        let mut scale = (a * d - b * c).abs().sqrt().max(f32::MIN_POSITIVE);
        if (scale - 1.).abs() < 1e-5 {
            // A rotation, up to rounding: paint at the element's own size.
            scale = 1.;
        }
        Self {
            to_window,
            to_element,
            scale,
            offset: Point::default(),
            scene_transformation: Some(TransformationMatrix {
                rotation_scale: [[a / scale, b / scale], [c / scale, d / scale]],
                translation: [x * scale_factor, y * scale_factor],
            }),
            scene_transform: None,
        }
    }

    /// Whether this is the window's own space.
    pub(crate) fn is_window(&self) -> bool {
        self.to_window == TransformationMatrix::UNIT
    }

    /// Whether paint geometry in this space is in window coordinates.
    pub(crate) fn is_aligned(&self) -> bool {
        self.scene_transformation.is_none()
    }

    /// Element bounds as paint geometry.
    pub(crate) fn paint_bounds(&self, bounds: Bounds<Pixels>) -> Bounds<Pixels> {
        if self.is_window() {
            return bounds;
        }
        Bounds {
            origin: self.paint_point(bounds.origin),
            size: bounds.size.map(|length| length * self.scale),
        }
    }

    /// An element point as paint geometry.
    pub(crate) fn paint_point(&self, point: Point<Pixels>) -> Point<Pixels> {
        if self.is_window() {
            return point;
        }
        Point {
            x: point.x * self.scale + self.offset.x,
            y: point.y * self.scale + self.offset.y,
        }
    }

    /// The window-coordinate bounds that contain `bounds` in this space.
    pub(crate) fn window_bounds(&self, bounds: Bounds<Pixels>) -> Bounds<Pixels> {
        if self.is_window() {
            return bounds;
        }
        transformed_bounds(&self.to_window, bounds)
    }

    /// The element-coordinate bounds that contain window `bounds`.
    pub(crate) fn element_bounds(&self, bounds: Bounds<Pixels>) -> Bounds<Pixels> {
        if self.is_window() {
            return bounds;
        }
        transformed_bounds(&self.to_element, bounds)
    }

    /// A window point in element coordinates.
    pub(crate) fn element_point(&self, point: Point<Pixels>) -> Point<Pixels> {
        if self.is_window() {
            return point;
        }
        self.to_element.apply(point)
    }
}

/// The axis-aligned bounds of `bounds` under `transformation`.
pub(crate) fn transformed_bounds(
    transformation: &TransformationMatrix,
    bounds: Bounds<Pixels>,
) -> Bounds<Pixels> {
    let corners = [
        bounds.origin,
        bounds.top_right(),
        bounds.bottom_right(),
        bounds.bottom_left(),
    ]
    .map(|corner| transformation.apply(corner));
    let (mut min, mut max) = (corners[0], corners[0]);
    for corner in &corners[1..] {
        min = min.min(corner);
        max = max.max(corner);
    }
    Bounds::from_corners(min, max)
}

/// Where records reused from the previous frame go.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Placement {
    /// Moved by an offset, in the coordinates of the element they were
    /// recorded in.
    Offset(Point<Pixels>),
    /// Placed by a transform of window coordinates, in logical pixels: for
    /// records whose element is now drawn under another transform.
    Transform(TransformationMatrix),
}

impl Default for Placement {
    fn default() -> Self {
        Self::Offset(Point::default())
    }
}

impl Placement {
    /// Whether the records stay where they were.
    pub(crate) fn is_zero(&self) -> bool {
        match self {
            Self::Offset(offset) => offset.is_zero(),
            Self::Transform(transformation) => *transformation == TransformationMatrix::UNIT,
        }
    }

    /// This placement, then `next`, for records in window coordinates (as
    /// deferred draws are), where an offset is a window offset.
    pub(crate) fn then(self, next: Self) -> Self {
        match (self, next) {
            (Self::Offset(first), Self::Offset(second)) => Self::Offset(first + second),
            (first, second) => Self::Transform(second.matrix().compose(first.matrix())),
        }
    }

    fn matrix(self) -> TransformationMatrix {
        match self {
            Self::Offset(offset) => TransformationMatrix {
                rotation_scale: TransformationMatrix::UNIT.rotation_scale,
                translation: [offset.x.0, offset.y.0],
            },
            Self::Transform(transformation) => transformation,
        }
    }
}

/// A clip set in a space not aligned with the window: an entry of the
/// window's transformed-clip stack.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TransformedClip {
    /// The clip in its element space.
    pub(crate) bounds: Bounds<Pixels>,
    /// That element space's position in the window's transform stack.
    pub(crate) space: usize,
    /// The clip this one is nested in, by position in the stack.
    pub(crate) parent: Option<usize>,
    /// The clip's entry in the scene's clip table, once a primitive has used it.
    pub(crate) scene_clip: Option<u32>,
}

/// A transformed clip as a hitbox tests it: element bounds, and the transform
/// from window coordinates into that element space.
pub(crate) type HitboxClip = (Bounds<Pixels>, TransformationMatrix);

impl From<kurbo::Affine> for TransformationMatrix {
    fn from(affine: kurbo::Affine) -> Self {
        let [a, b, c, d, e, f] = affine.as_coeffs();
        Self {
            rotation_scale: [[a as f32, c as f32], [b as f32, d as f32]],
            translation: [e as f32, f as f32],
        }
    }
}

impl From<TransformationMatrix> for kurbo::Affine {
    fn from(matrix: TransformationMatrix) -> Self {
        let [[a, c], [b, d]] = matrix.rotation_scale;
        let [e, f] = matrix.translation;
        kurbo::Affine::new([a, b, c, d, e, f].map(f64::from))
    }
}

impl Window {
    /// Draws the elements in `f` under `transform`, an affine map from their
    /// coordinates to the coordinates of the element drawing them, in logical
    /// pixels. Use it in both prepaint and paint, so hitboxes and painting
    /// agree.
    ///
    /// Layout is unaffected: the elements inside lay out, paint and receive
    /// mouse events in their own coordinates, and the window maps what they
    /// paint, their hitboxes and their clips. A transform about a point other
    /// than the origin is built by translating there and back, as in
    /// `Affine::translate(center) * Affine::rotate(angle) * Affine::translate(-center)`.
    pub fn with_transform<R>(
        &mut self,
        transform: impl Into<kurbo::Affine>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.invalidator.debug_assert_paint_or_prepaint();
        let transform: kurbo::Affine = transform.into();
        if transform == kurbo::Affine::IDENTITY {
            return f(self);
        }
        let to_window = self
            .element_space()
            .to_window
            .compose(TransformationMatrix::from(transform));
        self.element_spaces
            .push(ElementSpace::new(to_window, self.scale_factor()));
        let result = f(self);
        self.element_spaces.pop();
        result
    }

    /// The transform from the current element's coordinates to the window's,
    /// in logical pixels: the identity outside [`Self::with_transform`].
    pub fn element_to_window(&self) -> kurbo::Affine {
        self.element_space().to_window.into()
    }

    /// How far `offset`, in the current element's coordinates, moves things
    /// in the window's.
    pub(crate) fn element_offset_in_window(&self, offset: Point<Pixels>) -> Point<Pixels> {
        let space = self.element_space();
        if space.is_window() {
            return offset;
        }
        space.to_window.apply(offset) - space.to_window.apply(Point::default())
    }

    /// The current element space: the window's own outside any transform.
    pub(crate) fn element_space(&self) -> ElementSpace {
        self.element_spaces
            .last()
            .copied()
            .unwrap_or(ElementSpace::WINDOW)
    }

    /// Device pixels per logical pixel of paint geometry in the current space.
    pub(crate) fn paint_scale(&self) -> f32 {
        self.scale_factor() * self.element_space().scale
    }

    /// The current space's entry in the scene's transform table, made the
    /// first time a primitive needs it: 0 where paint geometry is already in
    /// window coordinates.
    pub(crate) fn scene_transform(&mut self) -> u32 {
        let Some(space) = self.element_spaces.last_mut() else {
            return 0;
        };
        Self::space_scene_transform(space, &mut self.next_frame.scene)
    }

    fn space_scene_transform(space: &mut ElementSpace, scene: &mut Scene) -> u32 {
        let Some(transformation) = space.scene_transformation else {
            return 0;
        };
        *space.scene_transform.get_or_insert_with(|| {
            scene.push_transform(SceneTransform {
                transformation,
                inverse: transformation
                    .inverse()
                    .unwrap_or(TransformationMatrix::UNIT),
            })
        })
    }

    /// The innermost transformed clip's entry in the scene's clip table, made
    /// with those it is nested in the first time a primitive needs it: 0 where
    /// no transformed clip applies.
    pub(crate) fn scene_clip(&mut self) -> u32 {
        match self.transformed_clips.len() {
            0 => 0,
            len => self.ensure_scene_clip(len - 1),
        }
    }

    fn ensure_scene_clip(&mut self, index: usize) -> u32 {
        if let Some(scene_clip) = self.transformed_clips[index].scene_clip {
            return scene_clip;
        }
        let clip = self.transformed_clips[index];
        let parent = clip
            .parent
            .map_or(0, |parent| self.ensure_scene_clip(parent));
        let space = &mut self.element_spaces[clip.space];
        let paint_bounds = space.paint_bounds(clip.bounds);
        let transform = Self::space_scene_transform(space, &mut self.next_frame.scene);
        let scale_factor = self.scale_factor;
        let scene_clip = self.next_frame.scene.push_clip(SceneClip {
            bounds: paint_bounds.scale(scale_factor),
            corner_radii: Corners::default(),
            transform,
            parent,
        });
        self.transformed_clips[index].scene_clip = Some(scene_clip);
        scene_clip
    }

    /// The transformed clips in effect, as a hitbox tests them.
    pub(crate) fn hitbox_clips(&self) -> Option<Arc<[HitboxClip]>> {
        if self.transformed_clips.is_empty() {
            return None;
        }
        Some(
            self.transformed_clips
                .iter()
                .map(|clip| (clip.bounds, self.element_spaces[clip.space].to_element))
                .collect(),
        )
    }

    /// The viewport bounds that contain paint `bounds`, in device pixels, in
    /// the current space.
    pub(crate) fn viewport_bounds(&self, bounds: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
        match self.element_space().scene_transformation {
            None => bounds,
            Some(transformation) => {
                let bounds = transformed_bounds(&transformation, bounds.map(|value| px(value.0)));
                bounds.map(|value| ScaledPixels(value.0))
            }
        }
    }
}

#[cfg(test)]
mod tests;
