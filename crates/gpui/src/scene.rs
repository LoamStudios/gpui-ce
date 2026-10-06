// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, Pixels, Point,
    Radians, ScaledFilter, ScaledPixels, Size, bounds_tree::BoundsTree, point, px,
};
use smallvec::SmallVec;
use std::{
    fmt::Debug,
    iter::Peekable,
    mem,
    ops::{Add, Range, Sub},
    slice,
};

mod plan;
pub use plan::*;
mod paint;
pub use paint::*;
mod abi;
#[doc(hidden)]
pub use abi::{SCENE_BUFFER_LAYOUTS, SceneBufferLayout};

#[allow(non_camel_case_types, unused)]
#[expect(missing_docs)]
pub type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

#[expect(missing_docs)]
pub type DrawOrder = u32;

pub(crate) const DEFAULT_BORDER_DASHED_LENGTH: f32 = 2.0;
pub(crate) const DEFAULT_BORDER_DASHED_GAP: f32 = 1.0;

/// A boolean with the same four-byte representation in Rust and WGSL.
/// Scene structs use it over one-byte [`bool`] to keep the storage-buffer ABI explicit.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum ShaderBool {
    /// The flag is disabled.
    #[default]
    Disabled = 0,
    /// The flag is enabled.
    Enabled = 1,
}

impl ShaderBool {
    /// Returns this flag as a regular Rust boolean.
    pub fn is_enabled(self) -> bool {
        self == Self::Enabled
    }
}

impl From<bool> for ShaderBool {
    fn from(value: bool) -> Self {
        if value { Self::Enabled } else { Self::Disabled }
    }
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<DrawOrder>,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub paths: Vec<Path<ScaledPixels>>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    surface_opacities: Vec<f32>,
    pub backdrop_filters: Vec<BackdropFilter>,
    pub group_boundaries: Vec<GroupBoundary>,
    render_plan: ScenePlan,
    is_finished: bool,
    /// Clips what is drawn, but not what is recorded; see [`Self::set_clip`].
    clip: Option<ContentMask<ScaledPixels>>,
    /// The transform table primitives refer to by index; see
    /// [`Self::transforms`].
    transforms: Vec<SceneTransform>,
    /// The clip table primitives refer to by index; see [`Self::clips`].
    clips: Vec<SceneClip>,
    /// The paint table paint references point into; see
    /// [`Self::paint_table`].
    paint_table: Vec<PaintWord>,
    /// Scratch space for [`Self::replay_run_at`], kept to reuse its allocation.
    replay_run: Vec<ReplayOperation>,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.surface_opacities.clear();
        self.backdrop_filters.clear();
        self.group_boundaries.clear();
        self.render_plan.clear();
        self.clip = None;
        self.transforms.clear();
        self.clips.clear();
        self.paint_table.clear();
        self.is_finished = false;
    }

    /// The viewport bounds that contain `primitive`: its bounds, moved by its
    /// transform-table entry if it has one.
    fn viewport_bounds(&self, primitive: &Primitive) -> Bounds<ScaledPixels> {
        let bounds = *primitive.bounds();
        match primitive.transform() {
            0 => bounds,
            transform => {
                let transformation = self.transforms()[transform as usize].transformation;
                crate::window::transformed_bounds(&transformation, bounds.map(|value| px(value.0)))
                    .map(|value| ScaledPixels(value.0))
            }
        }
    }

    /// The viewport bounds of what `batch` draws: the union of its
    /// primitives' bounds, transformed into the viewport and clipped to their
    /// masks. `None` for a batch that draws nothing.
    pub(crate) fn batch_region(&self, batch: &PrimitiveBatch) -> Option<Bounds<ScaledPixels>> {
        let mut union: Option<Bounds<ScaledPixels>> = None;
        self.for_each_primitive_region(batch, &mut |bounds| {
            union = Some(union.map_or(bounds, |union| union.union(&bounds)));
        });
        union
    }

    /// Calls `f` with the viewport bounds of each primitive `batch` draws,
    /// transformed into the viewport and clipped to its mask, skipping those
    /// that are empty.
    pub(crate) fn for_each_primitive_region(
        &self,
        batch: &PrimitiveBatch,
        f: &mut dyn FnMut(Bounds<ScaledPixels>),
    ) {
        let mut region =
            |bounds: Bounds<ScaledPixels>, mask: &ContentMask<ScaledPixels>, transform: u32| {
                let bounds = match transform {
                    0 => bounds,
                    transform => {
                        let transformation = self.transforms()[transform as usize].transformation;
                        crate::window::transformed_bounds(
                            &transformation,
                            bounds.map(|value| px(value.0)),
                        )
                        .map(|value| ScaledPixels(value.0))
                    }
                };
                let bounds = bounds.intersect(&mask.bounds);
                if !bounds.is_empty() {
                    f(bounds);
                }
            };
        match batch {
            PrimitiveBatch::Shadows { range, .. } => {
                for shadow in &self.shadows[range.clone()] {
                    region(shadow.bounds, &shadow.content_mask, shadow.transform);
                }
            }
            PrimitiveBatch::Quads { range, .. } => {
                for quad in &self.quads[range.clone()] {
                    region(quad.bounds, &quad.content_mask, quad.transform);
                }
            }
            PrimitiveBatch::Paths { range, .. } => {
                for path in &self.paths[range.clone()] {
                    region(path.bounds, &path.content_mask, 0);
                }
            }
            PrimitiveBatch::Underlines(range) => {
                for underline in &self.underlines[range.clone()] {
                    region(
                        underline.bounds,
                        &underline.content_mask,
                        underline.transform,
                    );
                }
            }
            PrimitiveBatch::MonochromeSprites { range, .. } => {
                for sprite in &self.monochrome_sprites[range.clone()] {
                    region(sprite.bounds, &sprite.content_mask, sprite.transform);
                }
            }
            PrimitiveBatch::SubpixelSprites { range, .. } => {
                for sprite in &self.subpixel_sprites[range.clone()] {
                    region(sprite.bounds, &sprite.content_mask, sprite.transform);
                }
            }
            PrimitiveBatch::PolychromeSprites { range, .. } => {
                for sprite in &self.polychrome_sprites[range.clone()] {
                    region(sprite.bounds, &sprite.content_mask, sprite.transform);
                }
            }
            PrimitiveBatch::Surfaces(range) => {
                for surface in &self.surfaces[range.clone()] {
                    region(surface.bounds, &surface.content_mask, 0);
                }
            }
            PrimitiveBatch::BackdropFilters(range) => {
                for filter in &self.backdrop_filters[range.clone()] {
                    region(filter.bounds, &filter.content_mask, 0);
                }
            }
            PrimitiveBatch::GroupBoundary(_) => {}
        }
    }

    /// Fades everything `batch` draws by `opacity`, as a group's opacity
    /// folded into its primitives.
    pub(crate) fn fade_batch(&mut self, batch: &PrimitiveBatch, opacity: f32) {
        match batch {
            PrimitiveBatch::Shadows { range, .. } => {
                for shadow in &mut self.shadows[range.clone()] {
                    shadow.color = shadow.color.opacity(opacity);
                }
            }
            PrimitiveBatch::Quads { range, .. } => {
                for quad in &mut self.quads[range.clone()] {
                    quad.background = quad.background.opacity(opacity);
                    quad.border_color = quad.border_color.opacity(opacity);
                }
            }
            PrimitiveBatch::Paths { range, .. } => {
                for path in &mut self.paths[range.clone()] {
                    path.color = path.color.opacity(opacity);
                }
            }
            PrimitiveBatch::Underlines(range) => {
                for underline in &mut self.underlines[range.clone()] {
                    underline.color = underline.color.opacity(opacity);
                }
            }
            PrimitiveBatch::MonochromeSprites { range, .. } => {
                for sprite in &mut self.monochrome_sprites[range.clone()] {
                    sprite.color = sprite.color.opacity(opacity);
                }
            }
            PrimitiveBatch::SubpixelSprites { range, .. } => {
                for sprite in &mut self.subpixel_sprites[range.clone()] {
                    sprite.color = sprite.color.opacity(opacity);
                }
            }
            PrimitiveBatch::PolychromeSprites { range, .. } => {
                for sprite in &mut self.polychrome_sprites[range.clone()] {
                    sprite.opacity *= opacity;
                }
            }
            PrimitiveBatch::Surfaces(range) => {
                for surface_opacity in &mut self.surface_opacities[range.clone()] {
                    *surface_opacity *= opacity;
                }
            }
            PrimitiveBatch::BackdropFilters(range) => {
                for filter in &mut self.backdrop_filters[range.clone()] {
                    filter.opacity *= opacity;
                }
            }
            PrimitiveBatch::GroupBoundary(_) => {}
        }
    }

    /// Rewrites `primitive`'s transform- and clip-table entries, which index
    /// `prev_scene`'s tables, as entries of this scene's, moved by `offset`.
    /// Each entry is copied once per replay, however many primitives use it.
    fn adopt_table_entries(
        &mut self,
        primitive: &mut Primitive,
        prev_scene: &Scene,
        adopted: &mut AdoptedEntries,
        offset: Point<ScaledPixels>,
    ) {
        let placement = TransformationMatrix {
            rotation_scale: TransformationMatrix::UNIT.rotation_scale,
            translation: [offset.x.0, offset.y.0],
        };
        self.adopt_paints(primitive, prev_scene, adopted, &placement);
        self.adopt_placed_entries(primitive, prev_scene, adopted, &placement);
    }

    /// Rewrites the paint-table entries `primitive`'s backgrounds draw, which
    /// index `prev_scene`'s table, as entries of this scene's placed by
    /// `placement`, with their stops.
    fn adopt_paints(
        &mut self,
        primitive: &mut Primitive,
        prev_scene: &Scene,
        adopted: &mut AdoptedEntries,
        placement: &TransformationMatrix,
    ) {
        if let Primitive::Path(path) = primitive
            && path.color.is_paint()
        {
            let index = self.adopt_paint(path.color.paint_index(), prev_scene, adopted, placement);
            path.color.set_paint_index(index);
        }
        for paint_ref in primitive.paint_refs_mut() {
            paint_ref.paint = self.adopt_paint(paint_ref.paint, prev_scene, adopted, placement);
        }
    }

    /// Entry `paint` of `prev_scene`'s paint table as an entry of this
    /// scene's, placed by `placement`, with its stops: 0 for 0.
    fn adopt_paint(
        &mut self,
        paint: u32,
        prev_scene: &Scene,
        adopted: &mut AdoptedEntries,
        placement: &TransformationMatrix,
    ) -> u32 {
        if paint == 0 {
            return 0;
        }
        if let Some(&index) = adopted.paints.get(&paint) {
            return index;
        }
        let table = prev_scene.paint_table();
        let mut entry = ScenePaint::from_words(&table[paint as usize..]);
        let stops: SmallVec<[SceneColorStop; 8]> = (0..entry.stop_count as usize)
            .map(|stop| {
                let word = entry.first_stop as usize + stop * SceneColorStop::WORDS;
                SceneColorStop::from_words(&table[word..])
            })
            .collect();
        entry.transformation = entry
            .transformation
            .compose(placement.inverse().unwrap_or(TransformationMatrix::UNIT));
        let index = self.push_paint(entry, &stops);
        adopted.paints.insert(paint, index);
        index
    }

    /// Rewrites `primitive`'s table entries, which index `prev_scene`'s
    /// tables, as entries of this scene's placed by `placement`.
    fn adopt_placed_entries(
        &mut self,
        primitive: &mut Primitive,
        prev_scene: &Scene,
        adopted: &mut AdoptedEntries,
        placement: &TransformationMatrix,
    ) {
        let Some((transform, clip)) = primitive.table_entries() else {
            return;
        };
        *transform = self.adopt_transform(*transform, prev_scene, adopted, placement);
        *clip = self.adopt_clip(*clip, prev_scene, adopted, placement);
    }

    /// Replays `range` of `prev_scene` placed by `placement`, a transformation
    /// of the viewport positions it was recorded at, and clipped to
    /// `content_mask`.
    ///
    /// A placement that keeps the recording aligned with the viewport, as a
    /// zoom does, maps its geometry exactly, and `rasterize` rasterizes its
    /// glyphs and SVGs again, from their sources placed where they now are, so
    /// the replay is as sharp as painting it again. Any other placement is
    /// applied on the GPU through the transform table, with sprites as they
    /// were rasterized.
    pub(crate) fn replay_placed(
        &mut self,
        range: Range<usize>,
        prev_scene: &Scene,
        placement: &TransformationMatrix,
        content_mask: &ContentMask<ScaledPixels>,
        rasterize: &mut dyn FnMut(RasterSource) -> Option<PlacedRaster>,
    ) {
        let uniform = UniformPlacement::of(placement);
        let mut adopted = AdoptedEntries::default();
        let mut placed = PlacedEntry::default();
        let mut place = |scene: &mut Scene, operation: &PaintOperation| {
            scene.place_operation(
                operation,
                prev_scene,
                &mut adopted,
                &mut placed,
                placement,
                uniform.as_ref(),
                content_mask,
                rasterize,
            )
        };
        self.replay_operations(
            &prev_scene.paint_operations[range],
            content_mask,
            &mut place,
        );
    }

    /// One recorded operation placed for [`Self::replay_placed`], or `None`
    /// for a sprite that is out of view, or whose raster is now empty.
    #[allow(clippy::too_many_arguments)]
    fn place_operation(
        &mut self,
        operation: &PaintOperation,
        prev_scene: &Scene,
        adopted: &mut AdoptedEntries,
        placed: &mut PlacedEntry,
        placement: &TransformationMatrix,
        uniform: Option<&UniformPlacement>,
        content_mask: &ContentMask<ScaledPixels>,
        rasterize: &mut dyn FnMut(RasterSource) -> Option<PlacedRaster>,
    ) -> Option<ReplayOperation> {
        Some(match operation {
            PaintOperation::Primitive(primitive) => {
                let mut primitive = primitive.clone();
                self.place(
                    &mut primitive,
                    prev_scene,
                    adopted,
                    placed,
                    placement,
                    uniform,
                    content_mask,
                );
                ReplayOperation::Primitive(primitive, None, None)
            }
            PaintOperation::Raster(sprite, source) => {
                let mut sprite = sprite.clone();
                let Some(uniform) = uniform.filter(|_| sprite.transform() == 0) else {
                    self.place(
                        &mut sprite,
                        prev_scene,
                        adopted,
                        placed,
                        placement,
                        uniform,
                        content_mask,
                    );
                    return Some(ReplayOperation::Primitive(
                        sprite,
                        None,
                        Some(source.clone()),
                    ));
                };
                // Rasterized again only once it is known to be in view.
                sprite.place_uniformly(uniform, content_mask);
                if sprite
                    .bounds()
                    .intersect(&sprite.content_mask().bounds)
                    .is_empty()
                {
                    return None;
                }
                let raster = rasterize(source.placed(uniform))?;
                match &mut sprite {
                    Primitive::MonochromeSprite(sprite) => {
                        (sprite.bounds, sprite.tile) = (raster.bounds, raster.tile);
                    }
                    Primitive::SubpixelSprite(sprite) => {
                        (sprite.bounds, sprite.tile) = (raster.bounds, raster.tile);
                    }
                    Primitive::PolychromeSprite(sprite) => {
                        (sprite.bounds, sprite.tile) = (raster.bounds, raster.tile);
                    }
                    _ => {}
                }
                ReplayOperation::Primitive(sprite, None, Some(Box::new(raster.source)))
            }
            PaintOperation::Surface { surface, opacity } => {
                let mut primitive = Primitive::Surface(surface.clone());
                self.place(
                    &mut primitive,
                    prev_scene,
                    adopted,
                    placed,
                    placement,
                    uniform,
                    content_mask,
                );
                ReplayOperation::Primitive(primitive, Some(*opacity), None)
            }
            PaintOperation::StartLayer(bounds) => ReplayOperation::StartLayer(
                crate::window::transformed_bounds(placement, bounds.map(|value| px(value.0)))
                    .map(|value| ScaledPixels(value.0)),
            ),
            PaintOperation::EndLayer => ReplayOperation::EndLayer,
        })
    }

    /// Places one replayed primitive; see [`Self::replay_placed`].
    #[allow(clippy::too_many_arguments)]
    fn place(
        &mut self,
        primitive: &mut Primitive,
        prev_scene: &Scene,
        adopted: &mut AdoptedEntries,
        placed: &mut PlacedEntry,
        placement: &TransformationMatrix,
        uniform: Option<&UniformPlacement>,
        content_mask: &ContentMask<ScaledPixels>,
    ) {
        self.adopt_paints(primitive, prev_scene, adopted, placement);
        if let Some(uniform) = uniform.filter(|_| primitive.transform() == 0) {
            primitive.place_uniformly(uniform, content_mask);
            return;
        }

        // Masks are viewport-aligned: they keep the bounds that contain them.
        let mask = |mask: &mut ContentMask<ScaledPixels>| {
            *mask = ContentMask {
                bounds: crate::window::transformed_bounds(
                    placement,
                    mask.bounds.map(|value| px(value.0)),
                )
                .map(|value| ScaledPixels(value.0)),
                fade_out: Edges::default(),
            }
            .intersect(content_mask);
        };
        match primitive {
            Primitive::Path(path) => {
                // Paths are tessellated in the viewport: move their vertices.
                for vertex in &mut path.vertices {
                    let position = placement.apply(vertex.xy_position.map(|value| px(value.0)));
                    vertex.xy_position = position.map(|value| ScaledPixels(value.0));
                    mask(&mut vertex.content_mask);
                }
                path.bounds = crate::window::transformed_bounds(
                    placement,
                    path.bounds.map(|value| px(value.0)),
                )
                .map(|value| ScaledPixels(value.0));
                mask(&mut path.content_mask);
            }
            Primitive::Surface(_) | Primitive::BackdropFilter(_) | Primitive::GroupBoundary(_) => {
                // Not transformed on the GPU: they take the viewport bounds
                // that contain them.
                let bounds = match primitive {
                    Primitive::Surface(surface) => &mut surface.bounds,
                    Primitive::BackdropFilter(filter) => &mut filter.bounds,
                    Primitive::GroupBoundary(boundary) => &mut boundary.bounds,
                    _ => unreachable!(),
                };
                *bounds =
                    crate::window::transformed_bounds(placement, bounds.map(|value| px(value.0)))
                        .map(|value| ScaledPixels(value.0));
                mask(primitive.content_mask_mut());
            }
            _ => {
                mask(primitive.content_mask_mut());
                if primitive.transform() == 0 {
                    let index = *placed.transform.get_or_insert_with(|| {
                        self.push_transform(SceneTransform {
                            transformation: *placement,
                            inverse: placement.inverse().unwrap_or(TransformationMatrix::UNIT),
                        })
                    });
                    let clip = primitive
                        .table_entries()
                        .map(|(_, clip)| *clip)
                        .unwrap_or(0);
                    let clip = self.adopt_clip(clip, prev_scene, adopted, placement);
                    if let Some((transform, primitive_clip)) = primitive.table_entries() {
                        *transform = index;
                        *primitive_clip = clip;
                    }
                } else {
                    self.adopt_placed_entries(primitive, prev_scene, adopted, placement);
                }
            }
        }
    }

    fn adopt_transform(
        &mut self,
        transform: u32,
        prev_scene: &Scene,
        adopted: &mut AdoptedEntries,
        placement: &TransformationMatrix,
    ) -> u32 {
        if transform == 0 {
            return 0;
        }
        if let Some(&index) = adopted.transforms.get(&transform) {
            return index;
        }
        let entry = prev_scene.transforms()[transform as usize];
        let index = self.push_transform(entry.placed_by(placement));
        adopted.transforms.insert(transform, index);
        index
    }

    fn adopt_clip(
        &mut self,
        clip: u32,
        prev_scene: &Scene,
        adopted: &mut AdoptedEntries,
        placement: &TransformationMatrix,
    ) -> u32 {
        if clip == 0 {
            return 0;
        }
        if let Some(&index) = adopted.clips.get(&clip) {
            return index;
        }
        let entry = prev_scene.clips()[clip as usize];
        let entry = SceneClip {
            transform: self.adopt_transform(entry.transform, prev_scene, adopted, placement),
            parent: self.adopt_clip(entry.parent, prev_scene, adopted, placement),
            ..entry
        };
        let index = self.push_clip(entry);
        adopted.clips.insert(clip, index);
        index
    }

    /// Adds `transform` to the transform table and returns its index.
    pub fn push_transform(&mut self, transform: SceneTransform) -> u32 {
        if self.transforms.is_empty() {
            self.transforms.push(SceneTransform::IDENTITY);
        }
        self.transforms.push(transform);
        (self.transforms.len() - 1) as u32
    }

    /// Adds `clip` to the clip table and returns its index.
    pub fn push_clip(&mut self, clip: SceneClip) -> u32 {
        if self.clips.is_empty() {
            self.clips.push(SceneClip::default());
        }
        self.clips.push(clip);
        (self.clips.len() - 1) as u32
    }

    /// Adds `gradient`, placed by `to_gradient`, from viewport positions to
    /// its space, to the paint table, and returns its index: `None` for a
    /// gradient with no stops.
    pub fn push_gradient(
        &mut self,
        gradient: &peniko::Gradient,
        to_gradient: TransformationMatrix,
    ) -> Option<u32> {
        let mut stops = Vec::new();
        let paint = ScenePaint::gradient(gradient, to_gradient, &mut stops)?;
        Some(self.push_paint(paint, &stops))
    }

    /// What a primitive with `bounds`, in the space of transform-table entry
    /// `transform`, paints `background` with: a colour, or an entry of the
    /// paint table, made for it if it is a gradient or pattern.
    pub fn paint_ref(
        &mut self,
        background: &Background,
        bounds: Bounds<ScaledPixels>,
        transform: u32,
    ) -> ScenePaintRef {
        if background.is_paint() {
            return ScenePaintRef {
                color: background.solid,
                paint: background.paint_index(),
            };
        }
        let to_viewport = self.transforms()[transform as usize].transformation;
        let mut stops = Vec::new();
        match ScenePaint::background(background, bounds, to_viewport, &mut stops) {
            Some(paint) => ScenePaintRef {
                color: crate::white().into(),
                paint: self.push_paint(paint, &stops),
            },
            None => ScenePaintRef {
                color: background.solid,
                paint: 0,
            },
        }
    }

    /// Adds `paint` and its `stops` to the paint table, and returns its
    /// index: the word it starts at.
    fn push_paint(&mut self, mut paint: ScenePaint, stops: &[SceneColorStop]) -> u32 {
        if self.paint_table.is_empty() {
            // Index 0 is no paint.
            self.paint_table
                .extend_from_slice(&ScenePaint::default().words());
        }
        let index = self.paint_table.len();
        paint.first_stop = (index + ScenePaint::WORDS) as u32;
        paint.stop_count = stops.len() as u32;
        self.paint_table.extend_from_slice(&paint.words());
        for stop in stops {
            self.paint_table.extend_from_slice(&stop.words());
        }
        index as u32
    }

    /// The paint table, for the renderer to upload as `vec4<f32>`s: entries
    /// and their stops, from word [`ScenePaint::WORDS`] on. Never empty.
    pub fn paint_table(&self) -> &[PaintWord] {
        const NONE: [PaintWord; ScenePaint::WORDS] = [[0.; 4]; ScenePaint::WORDS];
        if self.paint_table.is_empty() {
            &NONE
        } else {
            &self.paint_table
        }
    }

    /// The transform table, for the renderer to upload: entry 0 is always the
    /// identity, which primitives that are not transformed refer to.
    pub fn transforms(&self) -> &[SceneTransform] {
        if self.transforms.is_empty() {
            std::slice::from_ref(&SceneTransform::IDENTITY)
        } else {
            &self.transforms
        }
    }

    /// The clip table, for the renderer to upload: entry 0 is always the
    /// empty clip, which primitives with no transformed clip refer to.
    pub fn clips(&self) -> &[SceneClip] {
        const NONE: SceneClip = SceneClip {
            bounds: Bounds {
                origin: Point {
                    x: ScaledPixels(0.),
                    y: ScaledPixels(0.),
                },
                size: Size {
                    width: ScaledPixels(0.),
                    height: ScaledPixels(0.),
                },
            },
            corner_radii: Corners {
                top_left: ScaledPixels(0.),
                top_right: ScaledPixels(0.),
                bottom_right: ScaledPixels(0.),
                bottom_left: ScaledPixels(0.),
            },
            transform: 0,
            parent: 0,
        };
        if self.clips.is_empty() {
            std::slice::from_ref(&NONE)
        } else {
            &self.clips
        }
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        self.is_finished = false;
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.is_finished = false;
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    /// Raise the draw-order floor so every primitive inserted afterwards sorts above everything
    /// inserted before. Called before painting deferred draws so overlays (tooltips, popovers,
    /// drag images) sort above the main scene — and a deferred backdrop's order can't fall inside
    /// a content-filter (`filter`) order range left behind by the main scene.
    pub fn raise_order_floor(&mut self) {
        self.is_finished = false;
        let floor = self.primitive_bounds.max_order() + 1;
        self.primitive_bounds.set_order_floor(floor);
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        self.insert_primitive_with_surface_opacity(primitive.into(), None, None);
    }

    pub(crate) fn insert_surface(&mut self, surface: PaintSurface, opacity: f32) {
        self.insert_primitive_with_surface_opacity(
            Primitive::Surface(surface),
            Some(opacity),
            None,
        );
    }

    /// Inserts a glyph's sprite, recording where it came from so that a
    /// replay at another scale can rasterize it again.
    pub(crate) fn insert_raster(&mut self, sprite: Primitive, source: RasterSource) {
        self.insert_primitive_with_surface_opacity(sprite, None, Some(Box::new(source)));
    }

    fn insert_primitive_with_surface_opacity(
        &mut self,
        mut primitive: Primitive,
        surface_opacity: Option<f32>,
        raster_source: Option<Box<RasterSource>>,
    ) {
        self.is_finished = false;
        let clipped_bounds = self
            .viewport_bounds(&primitive)
            .intersect(&primitive.content_mask().bounds);

        // Content-filter boundaries must always be inserted as matched pairs — dropping one
        // (e.g. for an empty clipped region) would orphan its partner and corrupt the renderer's
        // target stack. Each marker takes an order strictly above ALL prior content, so the start
        // sorts after everything painted before it and the element's own children (which overlap
        // the marker bounds) sort strictly above the start. This keeps a marker's order range from
        // colliding with unrelated non-overlapping content that reuses low orderings (e.g. a
        // background grid), which would otherwise sweep that content into the group. Content
        // painted *after* the group is held above it by raising the order floor when the end
        // marker is inserted (see below) — otherwise a later non-overlapping sibling could reuse a
        // low order that lands inside the start..end range and be swept into the group.
        let is_group_boundary = matches!(primitive, Primitive::GroupBoundary(_));

        if clipped_bounds.is_empty() && !is_group_boundary {
            return;
        }

        let order = if is_group_boundary {
            let order_bounds = if clipped_bounds.is_empty() {
                *primitive.bounds()
            } else {
                clipped_bounds
            };
            self.primitive_bounds.insert_above_all(order_bounds)
        } else {
            self.layer_stack
                .last()
                .copied()
                .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds))
        };
        self.push_primitive(primitive, order, surface_opacity, raster_source);
    }

    /// Stores `primitive` at `order`, which has been assigned already: as
    /// recorded, for a later frame to replay, and to be drawn in this one,
    /// cut down to [`Self::clip`] if one is set.
    fn push_primitive(
        &mut self,
        mut primitive: Primitive,
        order: DrawOrder,
        surface_opacity: Option<f32>,
        raster_source: Option<Box<RasterSource>>,
    ) {
        primitive.set_order(order);
        if let Primitive::GroupBoundary(_) = &primitive {
            // Group markers are draw-order barriers. Everything painted after a start
            // marker sorts above it, so the group's content, which need not overlap the
            // element that made the group, falls inside the group; everything painted after
            // an end marker sorts above that, so later content cannot fall back inside the
            // group's order range (non-overlapping content otherwise reuses a low order).
            // Mirrors the floor raised before deferred draws in `raise_order_floor`.
            self.primitive_bounds.set_order_floor(order + 1);
        }
        match self.clip {
            None => self.draw_primitive(&primitive, surface_opacity),
            Some(clip) => {
                let mut drawn = primitive.clone();
                drawn.translate(Point::default(), &clip);
                // Content-filter boundaries are drawn in pairs, visible or not.
                if matches!(drawn, Primitive::GroupBoundary(_))
                    || !drawn
                        .bounds()
                        .intersect(&drawn.content_mask().bounds)
                        .is_empty()
                {
                    self.draw_primitive(&drawn, surface_opacity);
                }
            }
        }
        if let (Primitive::Surface(surface), Some(opacity)) = (&primitive, surface_opacity) {
            self.paint_operations.push(PaintOperation::Surface {
                surface: surface.clone(),
                opacity,
            });
        } else if let Some(source) = raster_source {
            self.paint_operations
                .push(PaintOperation::Raster(primitive, source));
        } else {
            self.paint_operations
                .push(PaintOperation::Primitive(primitive));
        }
    }

    /// Adds `primitive` to what this frame draws.
    fn draw_primitive(&mut self, primitive: &Primitive, surface_opacity: Option<f32>) {
        match primitive {
            Primitive::Shadow(shadow) => self.shadows.push(*shadow),
            Primitive::Quad(quad) => self.quads.push(*quad),
            Primitive::Path(path) => {
                let mut path = path.clone();
                path.id = PathId(self.paths.len());
                self.paths.push(path);
            }
            Primitive::Underline(underline) => self.underlines.push(*underline),
            Primitive::MonochromeSprite(sprite) => self.monochrome_sprites.push(*sprite),
            Primitive::SubpixelSprite(sprite) => self.subpixel_sprites.push(*sprite),
            Primitive::PolychromeSprite(sprite) => self.polychrome_sprites.push(*sprite),
            Primitive::Surface(surface) => {
                self.surfaces.push(surface.clone());
                self.surface_opacities.push(surface_opacity.unwrap_or(1.0));
            }
            Primitive::BackdropFilter(filter) => self.backdrop_filters.push(filter.clone()),
            Primitive::GroupBoundary(boundary) => self.group_boundaries.push(boundary.clone()),
        }
    }

    /// The atlas tiles this scene records: those it draws, and those a cached
    /// view recorded whole may draw when replayed.
    pub(crate) fn atlas_tiles(
        &self,
    ) -> collections::FxHashSet<(crate::AtlasTextureId, crate::TileId)> {
        self.paint_operations
            .iter()
            .filter_map(|operation| match operation {
                PaintOperation::Primitive(primitive) | PaintOperation::Raster(primitive, _) => {
                    match primitive {
                        Primitive::MonochromeSprite(sprite) => Some(sprite.tile),
                        Primitive::SubpixelSprite(sprite) => Some(sprite.tile),
                        Primitive::PolychromeSprite(sprite) => Some(sprite.tile),
                        _ => None,
                    }
                }
                _ => None,
            })
            .map(|tile| (tile.texture_id, tile.tile_id))
            .collect()
    }

    /// Sets the clip applied to what is drawn from here on, on top of each
    /// primitive's own mask, and returns the one it replaces. Primitives are
    /// still recorded, and ordered, as if unclipped: this is how a cached view
    /// is recorded whole while only what is visible of it is drawn.
    pub(crate) fn set_clip(
        &mut self,
        clip: Option<ContentMask<ScaledPixels>>,
    ) -> Option<ContentMask<ScaledPixels>> {
        mem::replace(&mut self.clip, clip)
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                PaintOperation::Raster(primitive, source) => {
                    self.insert_raster(primitive.clone(), (**source).clone())
                }
                PaintOperation::Surface { surface, opacity } => {
                    self.insert_surface(surface.clone(), *opacity)
                }
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
            }
        }
    }

    /// Replays `range` of `prev_scene` moved by `offset`, clipping every
    /// primitive to `content_mask` on top of its own (moved) mask.
    ///
    /// This is how a cached view is painted again, at the same position or a
    /// new one: its primitives were recorded where it was, and their masks
    /// include whatever clipped them there, so the masks move with them and
    /// are then cut down to what clips them here.
    pub fn replay_at(
        &mut self,
        range: Range<usize>,
        prev_scene: &Scene,
        offset: Point<ScaledPixels>,
        content_mask: &ContentMask<ScaledPixels>,
    ) {
        let mut adopted = AdoptedEntries::default();
        let mut moved = |scene: &mut Scene, operation: &PaintOperation| {
            Some(match operation {
                PaintOperation::Primitive(primitive) => {
                    let mut primitive = primitive.clone();
                    primitive.translate(offset, content_mask);
                    scene.adopt_table_entries(&mut primitive, prev_scene, &mut adopted, offset);
                    ReplayOperation::Primitive(primitive, None, None)
                }
                PaintOperation::Raster(primitive, source) => {
                    let mut primitive = primitive.clone();
                    primitive.translate(offset, content_mask);
                    scene.adopt_table_entries(&mut primitive, prev_scene, &mut adopted, offset);
                    let source = Box::new(source.moved_by(offset, &primitive));
                    ReplayOperation::Primitive(primitive, None, Some(source))
                }
                PaintOperation::Surface { surface, opacity } => {
                    let mut primitive = Primitive::Surface(surface.clone());
                    primitive.translate(offset, content_mask);
                    ReplayOperation::Primitive(primitive, Some(*opacity), None)
                }
                PaintOperation::StartLayer(bounds) => ReplayOperation::StartLayer(*bounds + offset),
                PaintOperation::EndLayer => ReplayOperation::EndLayer,
            })
        };
        self.replay_operations(
            &prev_scene.paint_operations[range],
            content_mask,
            &mut moved,
        );
    }

    /// Replays `operations`, each moved by `replayed`, as one run if it can
    /// (see [`Self::replay_run`]) and one by one if not.
    fn replay_operations(
        &mut self,
        operations: &[PaintOperation],
        content_mask: &ContentMask<ScaledPixels>,
        replayed: &mut dyn FnMut(&mut Scene, &PaintOperation) -> Option<ReplayOperation>,
    ) {
        if self.replay_run(operations, content_mask, replayed) {
            return;
        }
        for operation in operations {
            match replayed(self, operation) {
                Some(ReplayOperation::Primitive(primitive, surface_opacity, raster_source)) => self
                    .insert_primitive_with_surface_opacity(
                        primitive,
                        surface_opacity,
                        raster_source,
                    ),
                Some(ReplayOperation::StartLayer(bounds)) => self.push_layer(bounds),
                Some(ReplayOperation::EndLayer) => self.pop_layer(),
                None => {}
            }
        }
    }

    /// Replays `operations`, each moved by `replayed`, ordering them with one
    /// search of the bounds tree instead of one per primitive: they keep the
    /// orders they had among themselves, shifted together above whatever they
    /// now overlap. A layer's primitives already carry the layer's order, so
    /// layers are replayed as they are. Returns `false`, having done nothing,
    /// when this cannot keep the stacking — inside a layer, whose order every
    /// primitive must take, or for content-filter groups, which are ordered
    /// above everything — so the caller replays one by one.
    fn replay_run(
        &mut self,
        operations: &[PaintOperation],
        content_mask: &ContentMask<ScaledPixels>,
        replayed: &mut dyn FnMut(&mut Scene, &PaintOperation) -> Option<ReplayOperation>,
    ) -> bool {
        if !self.layer_stack.is_empty()
            || operations.iter().any(|operation| {
                matches!(
                    operation,
                    PaintOperation::Primitive(Primitive::GroupBoundary(_))
                )
            })
        {
            return false;
        }

        let mut run = mem::take(&mut self.replay_run);
        run.clear();
        let mut bounds: Option<Bounds<ScaledPixels>> = None;
        let mut cover = |clipped: Bounds<ScaledPixels>| {
            bounds = Some(bounds.map_or(clipped, |bounds| bounds.union(&clipped)));
        };
        let (mut first, mut last) = (DrawOrder::MAX, DrawOrder::MIN);
        for operation in operations {
            match replayed(self, operation) {
                Some(ReplayOperation::Primitive(primitive, surface_opacity, raster_source)) => {
                    let clipped = self
                        .viewport_bounds(&primitive)
                        .intersect(&primitive.content_mask().bounds);
                    if clipped.is_empty() {
                        continue;
                    }
                    let order = primitive.order();
                    first = first.min(order);
                    last = last.max(order);
                    cover(clipped);
                    run.push(ReplayOperation::Primitive(
                        primitive,
                        surface_opacity,
                        raster_source,
                    ));
                }
                Some(ReplayOperation::StartLayer(layer)) => {
                    cover(layer.intersect(&content_mask.bounds));
                    run.push(ReplayOperation::StartLayer(layer));
                }
                Some(ReplayOperation::EndLayer) => run.push(ReplayOperation::EndLayer),
                None => {}
            }
        }

        self.is_finished = false;
        let base = match bounds {
            Some(bounds) if first <= last => self.primitive_bounds.insert_run(bounds, last - first),
            _ => 0,
        };
        for operation in run.drain(..) {
            match operation {
                ReplayOperation::Primitive(primitive, surface_opacity, raster_source) => {
                    let order = base + (primitive.order() - first);
                    self.push_primitive(primitive, order, surface_opacity, raster_source);
                }
                // Recorded so that this frame can be replayed in turn; the
                // orders inside were assigned above.
                ReplayOperation::StartLayer(layer) => self
                    .paint_operations
                    .push(PaintOperation::StartLayer(layer)),
                ReplayOperation::EndLayer => self.paint_operations.push(PaintOperation::EndLayer),
            }
        }
        self.replay_run = run;
        true
    }

    pub fn finish(&mut self) {
        self.shadows.sort_by_key(|shadow| shadow.order);
        self.quads.sort_by_key(|quad| quad.order);
        self.paths.sort_by_key(|path| path.order);
        self.underlines.sort_by_key(|underline| underline.order);
        self.monochrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.subpixel_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.polychrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        let surfaces = std::mem::take(&mut self.surfaces);
        let mut surface_opacities = std::mem::take(&mut self.surface_opacities);
        surface_opacities.resize(surfaces.len(), 1.0);
        surface_opacities.truncate(surfaces.len());
        let mut surfaces_with_opacity = surfaces
            .into_iter()
            .zip(surface_opacities)
            .collect::<Vec<_>>();
        surfaces_with_opacity.sort_by_key(|(surface, _)| surface.order);
        let (surfaces, surface_opacities): (Vec<_>, Vec<_>) =
            surfaces_with_opacity.into_iter().unzip();
        self.surfaces = surfaces;
        self.surface_opacities = surface_opacities;
        self.backdrop_filters.sort_by_key(|filter| filter.order);
        // Markers normally get distinct, monotonically-increasing orders (children overlap
        // their group bounds and so sort strictly between the start and end). The `!is_start`
        // tiebreak only matters for a degenerate empty group whose start and end tie: it keeps
        // the start (false = 0) ahead of the end (true = 1) so the pair stays well-formed.
        self.group_boundaries
            .sort_by_key(|boundary| (boundary.order, !boundary.is_start));
        let commands = std::mem::take(&mut self.render_plan.commands);
        let folds = std::mem::take(&mut self.render_plan.folds);
        self.render_plan = ScenePlan::build(self, commands, folds);
        // Fade the groups drawn in place by their opacity, once: their
        // markers lose it, so a plan built again draws them as they are.
        let folds = std::mem::take(&mut self.render_plan.folds);
        for fold in &folds {
            for batch in &fold.batches {
                self.fade_batch(batch, fold.opacity);
            }
            self.group_boundaries[fold.start].opacity = 1.0;
            self.group_boundaries[fold.end].opacity = 1.0;
        }
        self.render_plan.folds = folds;
        self.is_finished = true;
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        self.render_commands().iter().map(|command| match command {
            RenderCommand::Batch(batch) => batch.clone(),
            RenderCommand::BeginGroup { boundary_index, .. } => {
                PrimitiveBatch::GroupBoundary(*boundary_index)
            }
            RenderCommand::EndGroup {
                closing_boundary_index,
                ..
            } => PrimitiveBatch::GroupBoundary(*closing_boundary_index),
        })
    }

    /// Returns the backend-neutral sequence of work needed to render this scene.
    ///
    /// In addition to preserving primitive batch order, this pairs content-filter boundaries and
    /// assigns the bounded offscreen target used by each isolated group. GPU backends only need to
    /// manage their target handles; nesting and overflow behavior stay consistent everywhere.
    pub fn render_commands(&self) -> &[RenderCommand] {
        self.render_plan().commands()
    }

    /// Returns the compiled render plan for this finished scene.
    pub fn render_plan(&self) -> &ScenePlan {
        debug_assert!(
            self.is_finished,
            "Scene::finish must be called before rendering"
        );
        self.render_plan.assert_matches(self);
        &self.render_plan
    }

    /// Returns the opacity associated with each surface in [`Self::surfaces`].
    ///
    /// Entries created through [`Self::insert_primitive`] default to fully opaque.
    pub fn surface_opacities(&self) -> &[f32] {
        &self.surface_opacities
    }

    /// Whether rendering needs an offscreen scene target for backdrop or content filters.
    pub fn requires_offscreen_rendering(&self) -> bool {
        self.render_plan().requirements().uses_offscreen_target
    }
}

/// Internal representation of [`palette::Hsla`] which is layout sensitive, as its provided to the renderer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub struct SceneHsla {
    /// Hue, in a range from 0 to 1
    pub(crate) h: f32,
    /// Saturation, in a range from 0 to 1
    pub(crate) s: f32,
    /// Lightness, in a range from 0 to 1
    pub(crate) l: f32,
    /// Alpha, in a range from 0 to 1
    pub(crate) a: f32,
}
impl SceneHsla {
    /// This colour with its alpha multiplied by `factor`.
    pub(crate) fn opacity(self, factor: f32) -> Self {
        Self {
            a: self.a * factor,
            ..self
        }
    }
}

impl Into<palette::Hsla> for SceneHsla {
    fn into(self) -> palette::Hsla {
        palette::Hsla::new(self.h * 360.0, self.s, self.l, self.a)
    }
}
impl From<palette::Hsla> for SceneHsla {
    fn from(hsla: palette::Hsla) -> Self {
        Self {
            h: hsla.hue.into_positive_degrees() / 360.0,
            s: hsla.saturation,
            l: hsla.lightness,
            a: hsla.alpha,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    // Lowest discriminant: at an equal order, a content-filter group-start is emitted before
    // the group's own content so the renderer redirects rendering before any child draws.
    GroupStart,
    Shadow,
    #[default]
    Quad,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
    BackdropFilter,
    // Highest discriminant: at an equal order, a group-end is emitted after the group's content
    // so the renderer composites the filtered group only once every child has been drawn.
    GroupEnd,
}

/// Where a rasterized sprite came from: what a replay at another scale needs
/// to rasterize it again, at the size it then appears.
#[derive(Clone, Debug)]
pub(crate) enum RasterSource {
    Glyph(GlyphSource),
    Svg(SvgSource),
}

impl RasterSource {
    /// The source of `sprite` once moved by `offset`: a sprite placed by a
    /// transform-table entry moves with the entry, not its position.
    fn moved_by(&self, offset: Point<ScaledPixels>, sprite: &Primitive) -> Self {
        let mut source = self.clone();
        if sprite.transform() == 0 {
            match &mut source {
                RasterSource::Glyph(glyph) => glyph.origin = glyph.origin + offset,
                RasterSource::Svg(svg) => svg.bounds.origin = svg.bounds.origin + offset,
            }
        }
        source
    }

    /// The source of a sprite placed by `placement`: where it now is, at the
    /// size it now appears.
    pub(crate) fn placed(&self, placement: &UniformPlacement) -> Self {
        match self {
            RasterSource::Glyph(glyph) => {
                let mut glyph = glyph.clone();
                glyph.params.scale_factor *= placement.scale;
                glyph.origin = placement.point(glyph.origin);
                RasterSource::Glyph(glyph)
            }
            RasterSource::Svg(svg) => RasterSource::Svg(SvgSource {
                bounds: placement.bounds(svg.bounds),
                ..svg.clone()
            }),
        }
    }
}

/// Where a glyph's sprite came from.
#[derive(Clone, Debug)]
pub(crate) struct GlyphSource {
    /// The parameters it was rasterized with.
    pub(crate) params: crate::RenderGlyphParams,
    /// Its origin in device pixels, before it was snapped to the pixel grid,
    /// in the space of its sprite's transform-table entry.
    pub(crate) origin: Point<ScaledPixels>,
    /// Whether it is a color glyph, which snaps to whole pixels only.
    pub(crate) color: bool,
}

/// Where an SVG's sprite came from.
#[derive(Clone, Debug)]
pub(crate) struct SvgSource {
    /// The SVG's asset path, which also keys its rasters in the atlas.
    pub(crate) path: crate::SharedString,
    /// Its bytes, when they were given rather than loaded from `path`.
    pub(crate) data: Option<std::sync::Arc<[u8]>>,
    /// The bounds it was fitted into, in device pixels, in the space of its
    /// sprite's transform-table entry.
    pub(crate) bounds: Bounds<ScaledPixels>,
}

/// A placement that scales uniformly by a positive factor, then translates:
/// one that keeps a recording aligned with the viewport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct UniformPlacement {
    pub(crate) scale: f32,
    pub(crate) translation: Point<ScaledPixels>,
}

impl UniformPlacement {
    /// `transformation` as a uniform placement, if it is one.
    pub(crate) fn of(transformation: &TransformationMatrix) -> Option<Self> {
        let [[a, b], [c, d]] = transformation.rotation_scale;
        (b == 0. && c == 0. && a == d && a > 0.).then(|| Self {
            scale: a,
            translation: point(
                ScaledPixels(transformation.translation[0]),
                ScaledPixels(transformation.translation[1]),
            ),
        })
    }

    pub(crate) fn point(&self, point: Point<ScaledPixels>) -> Point<ScaledPixels> {
        Point {
            x: ScaledPixels(point.x.0 * self.scale) + self.translation.x,
            y: ScaledPixels(point.y.0 * self.scale) + self.translation.y,
        }
    }

    pub(crate) fn bounds(&self, bounds: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
        Bounds {
            origin: self.point(bounds.origin),
            size: bounds
                .size
                .map(|length| ScaledPixels(length.0 * self.scale)),
        }
    }

    /// `transformation`, which maps viewport positions, as it maps them once
    /// placed: about the placed points.
    fn conjugate(&self, transformation: TransformationMatrix) -> TransformationMatrix {
        if transformation == TransformationMatrix::UNIT {
            return transformation;
        }
        let placement = self.matrix();
        placement
            .compose(transformation)
            .compose(placement.inverse().unwrap_or(TransformationMatrix::UNIT))
    }

    fn matrix(&self) -> TransformationMatrix {
        TransformationMatrix {
            rotation_scale: [[self.scale, 0.], [0., self.scale]],
            translation: [self.translation.x.0, self.translation.y.0],
        }
    }
}

/// A sprite rasterized again for a replay; see [`Scene::replay_placed`].
pub(crate) struct PlacedRaster {
    pub(crate) bounds: Bounds<ScaledPixels>,
    pub(crate) tile: AtlasTile,
    pub(crate) source: RasterSource,
}

/// The transform-table entry for a replay's placement itself, made the first
/// time a primitive placed in the viewport needs it.
#[derive(Default)]
struct PlacedEntry {
    transform: Option<u32>,
}

/// Table entries copied from the scene being replayed, by their index there.
#[derive(Default)]
struct AdoptedEntries {
    transforms: collections::FxHashMap<u32, u32>,
    clips: collections::FxHashMap<u32, u32>,
    paints: collections::FxHashMap<u32, u32>,
}

/// A paint operation being replayed as part of a run, moved and clipped.
enum ReplayOperation {
    Primitive(Primitive, Option<f32>, Option<Box<RasterSource>>),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    /// A glyph's sprite, with where it came from.
    Raster(Primitive, Box<RasterSource>),
    Surface {
        surface: PaintSurface,
        opacity: f32,
    },
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
    BackdropFilter(BackdropFilter),
    GroupBoundary(GroupBoundary),
}

#[expect(missing_docs)]
impl Primitive {
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
            Primitive::BackdropFilter(filter) => &filter.bounds,
            Primitive::GroupBoundary(boundary) => &boundary.bounds,
        }
    }

    fn content_mask_mut(&mut self) -> &mut ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &mut shadow.content_mask,
            Primitive::Quad(quad) => &mut quad.content_mask,
            Primitive::Path(path) => &mut path.content_mask,
            Primitive::Underline(underline) => &mut underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &mut sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &mut sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &mut sprite.content_mask,
            Primitive::Surface(surface) => &mut surface.content_mask,
            Primitive::BackdropFilter(filter) => &mut filter.content_mask,
            Primitive::GroupBoundary(boundary) => &mut boundary.content_mask,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
            Primitive::BackdropFilter(filter) => &filter.content_mask,
            Primitive::GroupBoundary(boundary) => &boundary.content_mask,
        }
    }

    fn set_order(&mut self, order: DrawOrder) {
        match self {
            Primitive::Shadow(shadow) => shadow.order = order,
            Primitive::Quad(quad) => quad.order = order,
            Primitive::Path(path) => path.order = order,
            Primitive::Underline(underline) => underline.order = order,
            Primitive::MonochromeSprite(sprite) => sprite.order = order,
            Primitive::SubpixelSprite(sprite) => sprite.order = order,
            Primitive::PolychromeSprite(sprite) => sprite.order = order,
            Primitive::Surface(surface) => surface.order = order,
            Primitive::BackdropFilter(filter) => filter.order = order,
            Primitive::GroupBoundary(boundary) => boundary.order = order,
        }
    }

    /// Maps the primitive by `placement`, which scales uniformly and
    /// translates, as if it had been painted there: its geometry, radii,
    /// widths and blurs scale, and its masks move with it and are then cut
    /// down to `content_mask`. Only for primitives with no transform-table
    /// entry, whose geometry is in viewport space.
    pub(crate) fn place_uniformly(
        &mut self,
        placement: &UniformPlacement,
        content_mask: &ContentMask<ScaledPixels>,
    ) {
        debug_assert_eq!(self.transform(), 0);
        let length = |value: ScaledPixels| ScaledPixels(value.0 * placement.scale);
        let corners = |corners: Corners<ScaledPixels>| corners.map(|corner| length(*corner));
        let mask = |mask: &mut ContentMask<ScaledPixels>| {
            *mask = ContentMask {
                bounds: placement.bounds(mask.bounds),
                fade_out: mask.fade_out.map(|fade| length(*fade)),
            }
            .intersect(content_mask);
        };
        let filters = |filters: &mut SmallVec<[ScaledFilter; 4]>| {
            for filter in filters.iter_mut() {
                match filter {
                    ScaledFilter::Blur(radius) => *radius = length(*radius),
                }
            }
        };
        match self {
            Primitive::Shadow(shadow) => {
                shadow.bounds = placement.bounds(shadow.bounds);
                shadow.element_bounds = placement.bounds(shadow.element_bounds);
                shadow.blur_radius = length(shadow.blur_radius);
                shadow.corner_radii = corners(shadow.corner_radii);
                shadow.element_corner_radii = corners(shadow.element_corner_radii);
                mask(&mut shadow.content_mask);
            }
            Primitive::Quad(quad) => {
                quad.bounds = placement.bounds(quad.bounds);
                quad.corner_radii = corners(quad.corner_radii);
                quad.border_widths = quad.border_widths.map(|width| length(*width));
                mask(&mut quad.content_mask);
            }
            Primitive::Path(path) => {
                path.bounds = placement.bounds(path.bounds);
                mask(&mut path.content_mask);
                for vertex in &mut path.vertices {
                    vertex.xy_position = placement.point(vertex.xy_position);
                    mask(&mut vertex.content_mask);
                }
            }
            Primitive::Underline(underline) => {
                underline.bounds = placement.bounds(underline.bounds);
                underline.thickness = length(underline.thickness);
                mask(&mut underline.content_mask);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.bounds = placement.bounds(sprite.bounds);
                sprite.transformation = placement.conjugate(sprite.transformation);
                mask(&mut sprite.content_mask);
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.bounds = placement.bounds(sprite.bounds);
                sprite.transformation = placement.conjugate(sprite.transformation);
                mask(&mut sprite.content_mask);
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.bounds = placement.bounds(sprite.bounds);
                sprite.corner_radii = corners(sprite.corner_radii);
                mask(&mut sprite.content_mask);
            }
            Primitive::Surface(surface) => {
                surface.bounds = placement.bounds(surface.bounds);
                mask(&mut surface.content_mask);
            }
            Primitive::BackdropFilter(filter) => {
                filter.bounds = placement.bounds(filter.bounds);
                filter.corner_radii = corners(filter.corner_radii);
                filters(&mut filter.filters);
                mask(&mut filter.content_mask);
            }
            Primitive::GroupBoundary(boundary) => {
                boundary.bounds = placement.bounds(boundary.bounds);
                filters(&mut boundary.filters);
                mask(&mut boundary.content_mask);
            }
        }
    }

    /// The primitive's entry in the scene's transform table: 0 for the
    /// identity, and for kinds that are not transformed.
    pub(crate) fn transform(&self) -> u32 {
        match self {
            Primitive::Shadow(shadow) => shadow.transform,
            Primitive::Quad(quad) => quad.transform,
            Primitive::Underline(underline) => underline.transform,
            Primitive::MonochromeSprite(sprite) => sprite.transform,
            Primitive::SubpixelSprite(sprite) => sprite.transform,
            Primitive::PolychromeSprite(sprite) => sprite.transform,
            Primitive::Path(_)
            | Primitive::Surface(_)
            | Primitive::BackdropFilter(_)
            | Primitive::GroupBoundary(_) => 0,
        }
    }

    /// The paint references the primitive paints with.
    fn paint_refs_mut(&mut self) -> SmallVec<[&mut ScenePaintRef; 2]> {
        match self {
            Primitive::Shadow(shadow) => smallvec::smallvec![&mut shadow.color],
            Primitive::Quad(quad) => {
                smallvec::smallvec![&mut quad.background, &mut quad.border_color]
            }
            _ => SmallVec::new(),
        }
    }

    /// The primitive's transform- and clip-table entries, to rewrite when it
    /// is replayed into another scene.
    fn table_entries(&mut self) -> Option<(&mut u32, &mut u32)> {
        match self {
            Primitive::Shadow(shadow) => Some((&mut shadow.transform, &mut shadow.clip)),
            Primitive::Quad(quad) => Some((&mut quad.transform, &mut quad.clip)),
            Primitive::Underline(underline) => {
                Some((&mut underline.transform, &mut underline.clip))
            }
            Primitive::MonochromeSprite(sprite) => Some((&mut sprite.transform, &mut sprite.clip)),
            Primitive::SubpixelSprite(sprite) => Some((&mut sprite.transform, &mut sprite.clip)),
            Primitive::PolychromeSprite(sprite) => Some((&mut sprite.transform, &mut sprite.clip)),
            Primitive::Path(_)
            | Primitive::Surface(_)
            | Primitive::BackdropFilter(_)
            | Primitive::GroupBoundary(_) => None,
        }
    }

    /// The draw order the primitive was given when it was inserted.
    pub(crate) fn order(&self) -> DrawOrder {
        match self {
            Primitive::Shadow(shadow) => shadow.order,
            Primitive::Quad(quad) => quad.order,
            Primitive::Path(path) => path.order,
            Primitive::Underline(underline) => underline.order,
            Primitive::MonochromeSprite(sprite) => sprite.order,
            Primitive::SubpixelSprite(sprite) => sprite.order,
            Primitive::PolychromeSprite(sprite) => sprite.order,
            Primitive::Surface(surface) => surface.order,
            Primitive::BackdropFilter(filter) => filter.order,
            Primitive::GroupBoundary(boundary) => boundary.order,
        }
    }

    /// Moves the primitive by `offset`, moving its mask with it and clipping
    /// that mask to `content_mask`.
    pub(crate) fn translate(
        &mut self,
        offset: Point<ScaledPixels>,
        content_mask: &ContentMask<ScaledPixels>,
    ) {
        // A primitive placed by a transform-table entry is moved by moving
        // that entry (`Scene::adopt_table_entries`); its geometry stays in
        // its own space. Masks are in the viewport either way.
        let geometry_offset = if self.transform() == 0 {
            offset
        } else {
            Point::default()
        };
        let moved = |mask: &mut ContentMask<ScaledPixels>| {
            *mask = mask.translate(offset).intersect(content_mask);
        };
        match self {
            Primitive::Shadow(shadow) => {
                shadow.bounds = shadow.bounds + geometry_offset;
                shadow.element_bounds = shadow.element_bounds + geometry_offset;
                moved(&mut shadow.content_mask);
            }
            Primitive::Quad(quad) => {
                quad.bounds = quad.bounds + geometry_offset;
                moved(&mut quad.content_mask);
            }
            Primitive::Path(path) => {
                path.bounds = path.bounds + geometry_offset;
                moved(&mut path.content_mask);
                for vertex in &mut path.vertices {
                    vertex.xy_position = vertex.xy_position + geometry_offset;
                    moved(&mut vertex.content_mask);
                }
            }
            Primitive::Underline(underline) => {
                underline.bounds = underline.bounds + geometry_offset;
                moved(&mut underline.content_mask);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.bounds = sprite.bounds + geometry_offset;
                sprite.transformation = sprite.transformation.moved_by(geometry_offset);
                moved(&mut sprite.content_mask);
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.bounds = sprite.bounds + geometry_offset;
                sprite.transformation = sprite.transformation.moved_by(geometry_offset);
                moved(&mut sprite.content_mask);
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.bounds = sprite.bounds + geometry_offset;
                moved(&mut sprite.content_mask);
            }
            Primitive::Surface(surface) => {
                surface.bounds = surface.bounds + geometry_offset;
                moved(&mut surface.content_mask);
            }
            Primitive::BackdropFilter(filter) => {
                filter.bounds = filter.bounds + geometry_offset;
                moved(&mut filter.content_mask);
            }
            Primitive::GroupBoundary(boundary) => {
                boundary.bounds = boundary.bounds + geometry_offset;
                moved(&mut boundary.content_mask);
            }
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    paths_start: usize,
    paths: &'a [Path<ScaledPixels>],
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
    backdrop_filters_start: usize,
    backdrop_filters_iter: Peekable<slice::Iter<'a, BackdropFilter>>,
    group_boundaries_start: usize,
    group_boundaries_iter: Peekable<slice::Iter<'a, GroupBoundary>>,
}

impl<'a> BatchIterator<'a> {
    fn new(scene: &'a Scene) -> Self {
        Self {
            shadows_start: 0,
            shadows_iter: scene.shadows.iter().peekable(),
            quads_start: 0,
            quads_iter: scene.quads.iter().peekable(),
            paths_start: 0,
            paths: &scene.paths,
            paths_iter: scene.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: scene.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: scene.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: scene.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: scene.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: scene.surfaces.iter().peekable(),
            backdrop_filters_start: 0,
            backdrop_filters_iter: scene.backdrop_filters.iter().peekable(),
            group_boundaries_start: 0,
            group_boundaries_iter: scene.group_boundaries.iter().peekable(),
        }
    }
}

fn precedes_limit(
    order: DrawOrder,
    kind: PrimitiveKind,
    limit: Option<(DrawOrder, PrimitiveKind)>,
) -> bool {
    limit.is_none_or(|limit| (order, kind) < limit)
}

fn has_corner_smoothing(corner_smoothing: f32) -> bool {
    corner_smoothing > 0.0
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
            (
                self.backdrop_filters_iter.peek().map(|f| f.order),
                PrimitiveKind::BackdropFilter,
            ),
            (
                self.group_boundaries_iter.peek().map(|b| b.order),
                // The same vec yields both start and end markers; the discriminant decides
                // where the next marker sorts relative to draw batches at an equal order
                // (start before content, end after).
                match self.group_boundaries_iter.peek() {
                    Some(boundary) if boundary.is_start => PrimitiveKind::GroupStart,
                    _ => PrimitiveKind::GroupEnd,
                },
            ),
        ];
        // Find the two lowest live cursors in one fixed-size scan.
        let mut first = None;
        let mut second = None;
        for (order, kind) in orders_and_kinds {
            let Some(order) = order else {
                continue;
            };
            let candidate = (order, kind);
            if first.is_none_or(|current| candidate < current) {
                second = first;
                first = Some(candidate);
            } else if second.is_none_or(|current| candidate < current) {
                second = Some(candidate);
            }
        }
        let (_, batch_kind) = first?;
        let max_order_and_kind = second;

        match batch_kind {
            PrimitiveKind::Shadow => {
                let smoothed =
                    has_corner_smoothing(self.shadows_iter.peek().unwrap().corner_smoothing);
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| {
                        precedes_limit(shadow.order, batch_kind, max_order_and_kind)
                            && has_corner_smoothing(shadow.corner_smoothing) == smoothed
                    })
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows {
                    range: shadows_start..shadows_end,
                    smoothed,
                })
            }
            PrimitiveKind::Quad => {
                let smoothed =
                    has_corner_smoothing(self.quads_iter.peek().unwrap().corner_smoothing);
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| {
                        precedes_limit(quad.order, batch_kind, max_order_and_kind)
                            && has_corner_smoothing(quad.corner_smoothing) == smoothed
                    })
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads {
                    range: quads_start..quads_end,
                    smoothed,
                })
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| precedes_limit(path.order, batch_kind, max_order_and_kind))
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                let range = paths_start..paths_end;
                let paths = &self.paths[range.clone()];
                let rasterization_vertex_count = paths.iter().map(|path| path.vertices.len()).sum();
                let sprite_count = if paths
                    .last()
                    .is_some_and(|path| path.order == paths[0].order)
                {
                    paths.len()
                } else {
                    1
                };
                Some(PrimitiveBatch::Paths {
                    range,
                    rasterization_vertex_count,
                    sprite_count,
                })
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| {
                        precedes_limit(underline.order, batch_kind, max_order_and_kind)
                    })
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(underlines_start..underlines_end))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        precedes_limit(sprite.order, batch_kind, max_order_and_kind)
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        precedes_limit(sprite.order, batch_kind, max_order_and_kind)
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let first_sprite = self.polychrome_sprites_iter.peek().unwrap();
                let texture_id = first_sprite.tile.texture_id;
                let smoothed = has_corner_smoothing(first_sprite.corner_smoothing);
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        precedes_limit(sprite.order, batch_kind, max_order_and_kind)
                            && sprite.tile.texture_id == texture_id
                            && has_corner_smoothing(sprite.corner_smoothing) == smoothed
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                    smoothed,
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| {
                        precedes_limit(surface.order, batch_kind, max_order_and_kind)
                    })
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(surfaces_start..surfaces_end))
            }
            PrimitiveKind::BackdropFilter => {
                let backdrop_filters_start = self.backdrop_filters_start;
                let mut backdrop_filters_end = backdrop_filters_start + 1;
                self.backdrop_filters_iter.next();
                while self
                    .backdrop_filters_iter
                    .next_if(|filter| precedes_limit(filter.order, batch_kind, max_order_and_kind))
                    .is_some()
                {
                    backdrop_filters_end += 1;
                }
                self.backdrop_filters_start = backdrop_filters_end;
                Some(PrimitiveBatch::BackdropFilters(
                    backdrop_filters_start..backdrop_filters_end,
                ))
            }
            // Boundaries are emitted one at a time (never merged) so the renderer can switch
            // render targets at exactly the right point in the batch stream.
            PrimitiveKind::GroupStart | PrimitiveKind::GroupEnd => {
                let index = self.group_boundaries_start;
                self.group_boundaries_iter.next();
                self.group_boundaries_start = index + 1;
                Some(PrimitiveBatch::GroupBoundary(index))
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub border_dashed_length: f32,
    pub border_dashed_gap: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: ScenePaintRef,
    pub border_color: ScenePaintRef,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
    pub corner_smoothing: f32,
    pub padding: u32,
    /// The primitive's entry in the scene's transform table, from its own
    /// space, where `bounds` are, to the viewport: 0 for the identity.
    pub transform: u32,
    /// The primitive's entry in the scene's clip table, for clips not aligned
    /// with the viewport, on top of `content_mask`: 0 for none.
    pub clip: u32,
}

impl Default for Quad {
    fn default() -> Self {
        Self {
            transform: 0,
            clip: 0,
            order: Default::default(),
            border_style: Default::default(),
            border_dashed_length: DEFAULT_BORDER_DASHED_LENGTH,
            border_dashed_gap: DEFAULT_BORDER_DASHED_GAP,
            bounds: Default::default(),
            content_mask: Default::default(),
            background: Default::default(),
            border_color: Default::default(),
            corner_radii: Default::default(),
            border_widths: Default::default(),
            corner_smoothing: Default::default(),
            padding: Default::default(),
        }
    }
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub padding: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: SceneHsla,
    pub thickness: ScaledPixels,
    pub wavy: ShaderBool,
    /// The primitive's entry in the scene's transform table, from its own
    /// space, where `bounds` are, to the viewport: 0 for the identity.
    pub transform: u32,
    /// The primitive's entry in the scene's clip table, for clips not aligned
    /// with the viewport, on top of `content_mask`: 0 for none.
    pub clip: u32,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: ScenePaintRef,
    /// Aligns `element_bounds` as the shaders do.
    pub padding: u32,
    pub element_bounds: Bounds<ScaledPixels>,
    pub element_corner_radii: Corners<ScaledPixels>,
    /// Whether this shadow is rendered inside the element instead of outside it.
    pub inset: ShaderBool,
    pub corner_smoothing: f32,
    /// The primitive's entry in the scene's transform table, from its own
    /// space, where `bounds` are, to the viewport: 0 for the identity.
    pub transform: u32,
    /// The primitive's entry in the scene's clip table, for clips not aligned
    /// with the viewport, on top of `content_mask`: 0 for none.
    pub clip: u32,
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// A backdrop filter blurs (and may otherwise filter) the content already rendered behind
/// `bounds`, compositing the result into a rounded rectangle — the frosted-glass effect.
/// Emitted by [`crate::Window::paint_backdrop_filter`]; produces the CSS `backdrop-filter` effect.
#[derive(Default, Debug, Clone)]
#[expect(missing_docs)]
pub struct BackdropFilter {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub corner_smoothing: f32,
    /// The filter chain applied to the backdrop, in scene (device-pixel) space. Identity filters
    /// are dropped at paint time, so a `BackdropFilter` is only emitted when this is non-empty.
    ///
    /// Inline capacity is 4: a `SmallVec<[ScaledFilter; 4]>` is the same size as capacity 1 here
    /// (the heap repr already occupies that space), so chains up to 4 filters avoid allocating
    /// at no extra struct size.
    pub filters: SmallVec<[ScaledFilter; 4]>,
    /// Element opacity captured at paint time, multiplied into the composited result.
    pub opacity: f32,
}

impl BackdropFilter {
    /// Largest gaussian blur radius in this filter chain, in device pixels.
    pub fn max_blur_radius(&self) -> f32 {
        max_blur_radius(&self.filters)
    }
}

impl From<BackdropFilter> for Primitive {
    fn from(filter: BackdropFilter) -> Self {
        Primitive::BackdropFilter(filter)
    }
}

/// The start or end marker of a group: what is painted between a matched
/// pair is composited into what is beneath it as one picture, filtered,
/// faded and blended as a whole. This is CSS's `filter`, `opacity` and
/// `mix-blend-mode` on an element and its children.
///
/// The renderer isolates a group, rendering it into a target of its own,
/// only when [`Self::isolates`]; the render plan sizes that target to what
/// the group draws.
#[derive(Debug, Clone)]
pub struct GroupBoundary {
    /// The marker's draw order: above everything painted before the group.
    pub order: DrawOrder,
    /// The bounds of the element that made the group, in the viewport.
    pub bounds: Bounds<ScaledPixels>,
    /// What the group's composite is clipped to.
    pub content_mask: ContentMask<ScaledPixels>,
    /// The filter chain applied to the isolated group, in scene (device-pixel) space. Identity
    /// filters are dropped at paint time.
    /// Inline capacity 4 (same struct size as 1 here — see [`BackdropFilter::filters`]).
    pub filters: SmallVec<[ScaledFilter; 4]>,
    /// The opacity the group is composited with, as one picture.
    pub opacity: f32,
    /// How the group's colours mix with what is beneath it.
    pub blend_mode: BlendMode,
    /// Whether the group's first child group is its mask: a group whose
    /// [`Self::mask_mode`] is set, painted before the group's contents.
    pub masked: bool,
    /// Set on a mask group: what it paints is not shown, but masks the group
    /// it is the first child of, by this mode.
    pub mask_mode: Option<MaskMode>,
    /// `true` for the start marker (opens the group), `false` for the end marker (closes it).
    pub is_start: bool,
}

impl GroupBoundary {
    /// Largest gaussian blur radius in this filter chain, in device pixels.
    pub fn max_blur_radius(&self) -> f32 {
        max_blur_radius(&self.filters)
    }

    /// How far the group's filters spread what it draws, in device pixels:
    /// its isolated target covers this much more on every side.
    pub fn filter_extent(&self) -> f32 {
        GAUSSIAN_EXTENT_PER_RADIUS * self.max_blur_radius()
    }

    /// Whether the group has to be rendered on its own and composited: it
    /// is filtered, faded, blended or masked, or it is a mask. Otherwise it
    /// draws in place.
    pub fn isolates(&self) -> bool {
        self.opacity < 1.0
            || self.blend_mode != BlendMode::Normal
            || self.max_blur_radius() > 0.0
            || self.masked
            || self.mask_mode.is_some()
    }
}

/// How a mask group's pixels mask the group it belongs to, as CSS's
/// `mask-mode` names them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(u32)]
pub enum MaskMode {
    /// By the mask's coverage: where it is opaque, the group shows.
    #[default]
    Alpha,
    /// By the mask's luminance, times its coverage: where it is white, the
    /// group shows.
    Luminance,
}

/// How a group's colours mix with the colours beneath it: the blend modes of
/// the W3C Compositing and Blending specification, as CSS's `mix-blend-mode`
/// names them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(u32)]
pub enum BlendMode {
    /// The group's colour, composited over what is beneath it.
    #[default]
    Normal,
    /// The product of the colours: never lighter.
    Multiply,
    /// The inverse of the product of the inverses: never darker.
    Screen,
    /// Multiply or screen, by the colour beneath.
    Overlay,
    /// The darker of the colours.
    Darken,
    /// The lighter of the colours.
    Lighten,
    /// Brightens what is beneath towards the group's colour.
    ColorDodge,
    /// Darkens what is beneath towards the group's colour.
    ColorBurn,
    /// Multiply or screen, by the group's colour.
    HardLight,
    /// A softer hard light.
    SoftLight,
    /// The difference of the colours.
    Difference,
    /// A lower-contrast difference.
    Exclusion,
    /// The group's hue with the saturation and luminosity beneath.
    Hue,
    /// The group's saturation with the hue and luminosity beneath.
    Saturation,
    /// The group's hue and saturation with the luminosity beneath.
    Color,
    /// The group's luminosity with the hue and saturation beneath.
    Luminosity,
}

/// How far a gaussian blur spreads, per unit of its radius: the standard
/// deviation is half the radius, and the kernel is cut off at three of them,
/// but the spread is kept at three radii, as the renderers' blur passes
/// dilate their bounds.
pub const GAUSSIAN_EXTENT_PER_RADIUS: f32 = 3.0;

/// Returns the largest blur radius in a scene-space filter chain.
///
/// This match is deliberately exhaustive so adding a filter requires one shared scheduling
/// decision instead of three backend-specific implementations that can drift.
fn max_blur_radius(filters: &[ScaledFilter]) -> f32 {
    filters.iter().fold(0.0, |radius, filter| match filter {
        ScaledFilter::Blur(filter_radius) => radius.max(filter_radius.0),
    })
}

impl From<GroupBoundary> for Primitive {
    fn from(boundary: GroupBoundary) -> Self {
        Primitive::GroupBoundary(boundary)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(u32)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// An entry of a scene's transform table: from a primitive's own space, where
/// its bounds are, to the viewport, and back. Entry 0 is the identity.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct SceneTransform {
    /// From the primitive's space to the viewport.
    pub transformation: TransformationMatrix,
    /// From the viewport to the primitive's space.
    pub inverse: TransformationMatrix,
}

impl SceneTransform {
    /// The same transform, then `placement`.
    pub fn placed_by(self, placement: &TransformationMatrix) -> Self {
        Self {
            transformation: placement.compose(self.transformation),
            inverse: self
                .inverse
                .compose(placement.inverse().unwrap_or(TransformationMatrix::UNIT)),
        }
    }

    /// The same transform, moved by `offset` in the viewport.
    pub fn moved_by(self, offset: Point<ScaledPixels>) -> Self {
        let [x, y] = self.transformation.translation;
        let [[a, b], [c, d]] = self.inverse.rotation_scale;
        let [inverse_x, inverse_y] = self.inverse.translation;
        Self {
            transformation: TransformationMatrix {
                translation: [x + offset.x.0, y + offset.y.0],
                ..self.transformation
            },
            inverse: TransformationMatrix {
                translation: [
                    inverse_x - (a * offset.x.0 + b * offset.y.0),
                    inverse_y - (c * offset.x.0 + d * offset.y.0),
                ],
                ..self.inverse
            },
        }
    }

    /// The identity, entry 0 of every transform table.
    pub const IDENTITY: Self = Self {
        transformation: TransformationMatrix::UNIT,
        inverse: TransformationMatrix::UNIT,
    };
}

/// What a primitive paints with: `color`, or, where `paint` is not 0, that
/// entry of the scene's paint table faded by `color`'s alpha.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct ScenePaintRef {
    /// The colour, or the paint's opacity in its alpha.
    pub color: SceneHsla,
    /// The paint-table entry: 0 for none.
    pub paint: u32,
}

impl ScenePaintRef {
    /// The same paint with its alpha multiplied by `factor`.
    pub fn opacity(self, factor: f32) -> Self {
        Self {
            color: self.color.opacity(factor),
            ..self
        }
    }

    /// Whether it paints nothing.
    pub fn is_transparent(&self) -> bool {
        self.color.a == 0.
    }
}

impl From<crate::Hsla> for ScenePaintRef {
    fn from(color: crate::Hsla) -> Self {
        Self {
            color: color.into(),
            paint: 0,
        }
    }
}

/// An entry of a scene's clip table: a rounded rectangle in the space of a
/// transform-table entry, and the clip it is nested in. Entry 0 clips nothing
/// and ends every chain. Clips aligned with the viewport fold into a
/// primitive's `content_mask` instead.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[repr(C)]
pub struct SceneClip {
    /// The clip rectangle, in the space of `transform`.
    pub bounds: Bounds<ScaledPixels>,
    /// The rectangle's corner radii.
    pub corner_radii: Corners<ScaledPixels>,
    /// The transform-table entry of the clip's space.
    pub transform: u32,
    /// The clip-table entry this clip is nested in: 0 for none.
    pub parent: u32,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, which has no effect.
    pub const UNIT: Self = Self {
        rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
        translation: [0.0, 0.0],
    };

    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self::UNIT
    }

    /// The transformation that undoes this one, if it can be undone.
    pub fn inverse(&self) -> Option<Self> {
        let [[a, b], [c, d]] = self.rotation_scale;
        let determinant = a * d - b * c;
        if determinant == 0.0 || !determinant.is_finite() {
            return None;
        }
        let rotation_scale = [
            [d / determinant, -b / determinant],
            [-c / determinant, a / determinant],
        ];
        let [x, y] = self.translation;
        Some(Self {
            rotation_scale,
            translation: [
                -(rotation_scale[0][0] * x + rotation_scale[0][1] * y),
                -(rotation_scale[1][0] * x + rotation_scale[1][1] * y),
            ],
        })
    }

    /// Whether this only translates and scales, keeping axes aligned.
    pub fn is_axis_aligned(&self) -> bool {
        self.rotation_scale[0][1] == 0.0 && self.rotation_scale[1][0] == 0.0
    }

    /// Move the origin by a given point
    pub fn translate(mut self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// The same transformation for content moved by `offset`: a sprite is
    /// transformed about points in window space (an SVG rotates about its
    /// centre), so those points move with it. Moving the content, then
    /// applying this, is applying `self` and then moving the result.
    pub fn moved_by(self, offset: Point<ScaledPixels>) -> Self {
        if self == Self::unit() {
            return self;
        }
        Self::unit()
            .translate(offset)
            .compose(self)
            .translate(point(ScaledPixels(-offset.x.0), ScaledPixels(-offset.y.0)))
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    pub padding: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: SceneHsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
    /// The sprite's entry in the scene's transform table, applied after
    /// `transformation`: 0 for the identity.
    pub transform: u32,
    /// The sprite's entry in the scene's clip table: 0 for none.
    pub clip: u32,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    pub padding: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: SceneHsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
    /// The sprite's entry in the scene's transform table, applied after
    /// `transformation`: 0 for the identity.
    pub transform: u32,
    /// The sprite's entry in the scene's clip table: 0 for none.
    pub clip: u32,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub grayscale: ShaderBool,
    pub opacity: f32,
    pub corner_smoothing: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub tile: AtlasTile,
    /// The primitive's entry in the scene's transform table, from its own
    /// space, where `bounds` are, to the viewport: 0 for the identity.
    pub transform: u32,
    /// The primitive's entry in the scene's clip table, for clips not aligned
    /// with the viewport, on top of `content_mask`: 0 for none.
    pub clip: u32,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub source: crate::SurfaceSource,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[expect(missing_docs)]
pub struct PathId(pub usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub id: PathId,
    pub order: DrawOrder,
    pub bounds: Bounds<P>,
    pub content_mask: ContentMask<P>,
    pub vertices: Vec<PathVertex<P>>,
    pub color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// The path under `transformation`, which maps its points.
    pub(crate) fn transformed(mut self, transformation: &TransformationMatrix) -> Self {
        for vertex in &mut self.vertices {
            vertex.xy_position = transformation.apply(vertex.xy_position);
        }
        self.bounds = crate::window::transformed_bounds(transformation, self.bounds);
        self.start = transformation.apply(self.start);
        self.current = transformation.apply(self.current);
        self
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    #[expect(missing_docs)]
    pub fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub xy_position: Point<P>,
    pub st_position: Point<f32>,
    pub content_mask: ContentMask<P>,
}

#[expect(missing_docs)]
impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AtlasTextureKind, DevicePixels, Point, ShaderBool, Size, SurfaceSource, TileId};

    /// Replaying a recording as one run, with one search of the bounds tree,
    /// stacks it as replaying it primitive by primitive does, moved or zoomed:
    /// every pair of overlapping primitives — within the recording, between it
    /// and what it is replayed over, and with what is painted after it — keeps
    /// its order, and primitives sharing a layer keep sharing an order.
    #[test]
    fn a_replayed_run_keeps_the_stacking_of_a_primitive_by_primitive_replay() {
        use rand::{Rng as _, SeedableRng as _};

        for seed in 0..300 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut random_quad = |rng: &mut rand::rngs::StdRng| Quad {
                bounds: Bounds {
                    origin: point(
                        sp(rng.random_range(0.0..300.0)),
                        sp(rng.random_range(0.0..300.0)),
                    ),
                    size: Size {
                        width: sp(rng.random_range(1.0..80.0)),
                        height: sp(rng.random_range(1.0..80.0)),
                    },
                },
                content_mask: ContentMask {
                    bounds: Bounds {
                        origin: point(sp(0.0), sp(0.0)),
                        size: Size {
                            width: sp(400.0),
                            height: sp(400.0),
                        },
                    },
                    ..Default::default()
                },
                ..Default::default()
            };
            let (before, recorded, beneath, after): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = (
                (0..rng.random_range(0..20))
                    .map(|_| random_quad(&mut rng))
                    .collect(),
                (0..rng.random_range(1..30))
                    .map(|_| random_quad(&mut rng))
                    .collect(),
                (0..rng.random_range(0..20))
                    .map(|_| random_quad(&mut rng))
                    .collect(),
                (0..rng.random_range(0..10))
                    .map(|_| random_quad(&mut rng))
                    .collect(),
            );
            let offset = point(
                sp(rng.random_range(-40.0..40.0)),
                sp(rng.random_range(-40.0..40.0)),
            );
            let clip = ContentMask {
                bounds: Bounds {
                    origin: point(sp(20.0), sp(20.0)),
                    size: Size {
                        width: sp(300.0),
                        height: sp(300.0),
                    },
                },
                ..Default::default()
            };

            // The recording, made over other content in an earlier frame.
            let mut rendered = Scene::default();
            for quad in &before {
                rendered.insert_primitive(*quad);
            }
            // Some of the recording is painted in layers, as text is, each
            // covering what is painted in it.
            let start = rendered.len();
            let mut ix = 0;
            while ix < recorded.len() {
                let len = if rng.random_bool(0.3) {
                    rng.random_range(1..=4).min(recorded.len() - ix)
                } else {
                    0
                };
                if len == 0 {
                    rendered.insert_primitive(recorded[ix]);
                    ix += 1;
                    continue;
                }
                let group = &recorded[ix..ix + len];
                let layer = group
                    .iter()
                    .skip(1)
                    .fold(group[0].bounds, |bounds, quad| bounds.union(&quad.bounds));
                rendered.push_layer(layer);
                for quad in group {
                    rendered.insert_primitive(*quad);
                }
                rendered.pop_layer();
                ix += len;
            }
            let range = start..rendered.len();

            // Replayed moved, and zoomed about the offset.
            let zoom = UniformPlacement {
                scale: rng.random_range(0.5..2.0),
                translation: offset,
            };
            let replay = |as_run: bool, zoomed: bool| {
                let mut scene = Scene::default();
                for quad in &beneath {
                    scene.insert_primitive(*quad);
                }
                if as_run && zoomed {
                    scene.replay_placed(
                        range.clone(),
                        &rendered,
                        &zoom.matrix(),
                        &clip,
                        &mut |_| unreachable!(),
                    );
                } else if as_run {
                    scene.replay_at(range.clone(), &rendered, offset, &clip);
                } else if zoomed {
                    for operation in &rendered.paint_operations[range.clone()] {
                        match operation {
                            PaintOperation::Primitive(primitive) => {
                                let mut primitive = primitive.clone();
                                primitive.place_uniformly(&zoom, &clip);
                                scene.insert_primitive(primitive);
                            }
                            PaintOperation::StartLayer(bounds) => {
                                scene.push_layer(zoom.bounds(*bounds).intersect(&clip.bounds))
                            }
                            PaintOperation::EndLayer => scene.pop_layer(),
                            PaintOperation::Surface { .. } | PaintOperation::Raster(..) => {
                                unreachable!()
                            }
                        }
                    }
                } else {
                    for operation in &rendered.paint_operations[range.clone()] {
                        match operation {
                            PaintOperation::Primitive(primitive) => {
                                let mut primitive = primitive.clone();
                                primitive.translate(offset, &clip);
                                scene.insert_primitive(primitive);
                            }
                            PaintOperation::StartLayer(bounds) => {
                                scene.push_layer((*bounds + offset).intersect(&clip.bounds))
                            }
                            PaintOperation::EndLayer => scene.pop_layer(),
                            PaintOperation::Surface { .. } | PaintOperation::Raster(..) => {
                                unreachable!()
                            }
                        }
                    }
                }
                for quad in &after {
                    scene.insert_primitive(*quad);
                }
                scene.quads
            };
            for zoomed in [false, true] {
                let (run, one_by_one) = (replay(true, zoomed), replay(false, zoomed));
                assert_eq!(run.len(), one_by_one.len(), "seed {seed}");
                for (i, a) in one_by_one.iter().enumerate() {
                    assert_eq!(run[i].bounds, a.bounds, "seed {seed}");
                    for (j, b) in one_by_one.iter().enumerate().skip(i + 1) {
                        let overlap = a
                            .bounds
                            .intersect(&a.content_mask.bounds)
                            .intersects(&b.bounds.intersect(&b.content_mask.bounds));
                        if overlap {
                            assert_eq!(
                                run[i].order.cmp(&run[j].order),
                                a.order.cmp(&b.order),
                                "seed {seed}: quads {i} and {j} overlap and changed places"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A glyph replayed under a zoom is rasterized again at the size it then
    /// appears, from where the recording says it came from.
    #[test]
    fn a_glyph_replayed_under_a_zoom_is_rasterized_again() {
        let params = crate::RenderGlyphParams {
            font_id: crate::FontId(0),
            glyph_id: crate::GlyphId(7),
            font_size: crate::px(12.),
            subpixel_variant: Point::default(),
            scale_factor: 2.,
            raster_style: crate::PreparedRasterStyle {
                mode: crate::GlyphRenderMode::Grayscale,
                color_effect: crate::RasterColorEffect::Independent,
                foreground_dependency: crate::ForegroundDependency::Full,
            },
        };
        let tile = |id| AtlasTile {
            texture_id: AtlasTextureId {
                index: 0,
                kind: AtlasTextureKind::Monochrome,
            },
            tile_id: TileId(id),
            padding: 0,
            bounds: Bounds::<DevicePixels>::default(),
        };
        let mut rendered = Scene::default();
        rendered.insert_raster(
            Primitive::MonochromeSprite(MonochromeSprite {
                order: 0,
                padding: 0,
                bounds: Bounds::new(point(sp(10.), sp(10.)), Size::new(sp(8.), sp(12.))),
                content_mask: mask(),
                color: Default::default(),
                tile: tile(1),
                transformation: TransformationMatrix::unit(),
                transform: 0,
                clip: 0,
            }),
            RasterSource::Glyph(GlyphSource {
                params,
                origin: point(sp(10.25), sp(20.)),
                color: false,
            }),
        );

        let mut replayed = Scene::default();
        let mut asked = None;
        let zoom = TransformationMatrix {
            rotation_scale: [[2., 0.], [0., 2.]],
            translation: [5., 5.],
        };
        replayed.replay_placed(
            0..rendered.len(),
            &rendered,
            &zoom,
            &ContentMask {
                bounds: Bounds::new(point(sp(0.), sp(0.)), Size::new(sp(200.), sp(200.))),
                ..Default::default()
            },
            &mut |source| {
                let RasterSource::Glyph(glyph) = &source else {
                    panic!("a glyph's source")
                };
                asked = Some(glyph.clone());
                Some(PlacedRaster {
                    bounds: Bounds::new(point(sp(25.), sp(35.)), Size::new(sp(16.), sp(24.))),
                    tile: tile(2),
                    source,
                })
            },
        );

        let asked = asked.expect("the glyph was rasterized again");
        assert_eq!(asked.params.scale_factor, 4., "at twice the scale");
        assert_eq!(asked.origin, point(sp(25.5), sp(45.)), "where it now is");
        let sprite = replayed.monochrome_sprites[0];
        assert_eq!(sprite.tile.tile_id, TileId(2), "drawn from the new raster");
        assert_eq!(
            sprite.bounds,
            Bounds::new(point(sp(25.), sp(35.)), Size::new(sp(16.), sp(24.)))
        );
        assert!(
            matches!(&replayed.paint_operations[0], PaintOperation::Raster(_, source) if matches!(&**source, RasterSource::Glyph(glyph) if glyph.params.scale_factor == 4.)),
            "and recorded with its new source, for the next replay"
        );
    }

    #[test]
    fn a_moved_transformation_turns_about_the_moved_point() {
        // An SVG rotates about its centre, in window space.
        let center = point(ScaledPixels(50.), ScaledPixels(30.));
        let rotation = TransformationMatrix::unit()
            .translate(center)
            .rotate(Radians(0.7))
            .scale(Size::new(1.5, 0.5))
            .translate(point(ScaledPixels(-50.), ScaledPixels(-30.)));
        let offset = point(ScaledPixels(12.), ScaledPixels(-40.));
        let moved = rotation.moved_by(offset);

        for corner in [(0., 0.), (100., 0.), (0., 60.), (100., 60.)] {
            let before = rotation.apply(point(corner.0.into(), corner.1.into()));
            let after = moved.apply(point((corner.0 + 12.).into(), (corner.1 - 40.).into()));
            assert!((f32::from(after.x) - (f32::from(before.x) + 12.)).abs() < 1e-3);
            assert!((f32::from(after.y) - (f32::from(before.y) - 40.)).abs() < 1e-3);
        }
        assert_eq!(
            TransformationMatrix::unit().moved_by(offset),
            TransformationMatrix::unit()
        );
    }

    fn sp(value: f32) -> ScaledPixels {
        ScaledPixels(value)
    }

    /// All test primitives cover the same region so the bounds tree assigns strictly
    /// increasing orders in insertion order — making the expected batch order deterministic.
    fn full_bounds() -> Bounds<ScaledPixels> {
        Bounds {
            origin: Point {
                x: sp(0.0),
                y: sp(0.0),
            },
            size: Size {
                width: sp(100.0),
                height: sp(100.0),
            },
        }
    }

    fn mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: full_bounds(),
            ..Default::default()
        }
    }

    fn quad() -> Quad {
        Quad {
            bounds: full_bounds(),
            content_mask: mask(),
            ..Default::default()
        }
    }

    fn shadow() -> Shadow {
        Shadow {
            transform: 0,
            clip: 0,
            order: 0,
            blur_radius: sp(0.0),
            bounds: full_bounds(),
            corner_radii: Corners::default(),
            content_mask: mask(),
            color: Default::default(),
            padding: 0,
            element_bounds: full_bounds(),
            element_corner_radii: Corners::default(),
            inset: ShaderBool::Disabled,
            corner_smoothing: 0.0,
        }
    }

    fn polychrome_sprite(texture_index: u32) -> PolychromeSprite {
        PolychromeSprite {
            transform: 0,
            clip: 0,
            order: 0,
            grayscale: ShaderBool::Disabled,
            opacity: 1.0,
            corner_smoothing: 0.0,
            bounds: full_bounds(),
            content_mask: mask(),
            corner_radii: Corners::default(),
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: texture_index,
                    kind: AtlasTextureKind::Polychrome,
                },
                tile_id: TileId(0),
                padding: 0,
                bounds: Bounds::<DevicePixels>::default(),
            },
        }
    }

    fn batches(scene: &mut Scene) -> Vec<PrimitiveBatch> {
        scene.finish();
        scene.batches().collect()
    }

    #[test]
    fn smoothing_and_texture_changes_define_batches() {
        let mut scene = Scene::default();
        for smoothing in [0.0, 0.0, 0.5, 1.0, 0.0] {
            let mut quad = quad();
            quad.corner_smoothing = smoothing;
            scene.insert_primitive(quad);
        }

        for smoothing in [0.0, 0.0, 0.5, 1.0, 0.0] {
            let mut shadow = shadow();
            shadow.corner_smoothing = smoothing;
            scene.insert_primitive(shadow);
        }

        for (texture, smoothing) in [
            (0, 0.0),
            (0, 0.0),
            (0, 0.5),
            (0, 1.0),
            (1, 1.0),
            (1, 0.5),
            (1, 0.0),
        ] {
            let mut sprite = polychrome_sprite(texture);
            sprite.corner_smoothing = smoothing;
            scene.insert_primitive(sprite);
        }

        let batch_signatures = batches(&mut scene)
            .into_iter()
            .map(|batch| match batch {
                PrimitiveBatch::Quads { range, smoothed } => ("quad", None, range, smoothed),
                PrimitiveBatch::Shadows { range, smoothed } => ("shadow", None, range, smoothed),
                PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range,
                    smoothed,
                } => ("polychrome", Some(texture_id.index), range, smoothed),
                other => panic!("unexpected batch: {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            batch_signatures,
            vec![
                ("quad", None, 0..2, false),
                ("quad", None, 2..4, true),
                ("quad", None, 4..5, false),
                ("shadow", None, 0..2, false),
                ("shadow", None, 2..4, true),
                ("shadow", None, 4..5, false),
                ("polychrome", Some(0), 0..2, false),
                ("polychrome", Some(0), 2..4, true),
                ("polychrome", Some(1), 4..6, true),
                ("polychrome", Some(1), 6..7, false),
            ]
        );
    }

    /// A 100x100 quad whose bounds don't overlap `full_bounds()` (used to exercise the
    /// order-reuse path: non-overlapping content reuses low draw-orders).
    fn detached_quad() -> Quad {
        let bounds = Bounds {
            origin: Point {
                x: sp(200.0),
                y: sp(200.0),
            },
            size: Size {
                width: sp(100.0),
                height: sp(100.0),
            },
        };
        Quad {
            bounds,
            content_mask: ContentMask {
                bounds,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn boundary(is_start: bool) -> GroupBoundary {
        GroupBoundary {
            order: 0,
            bounds: full_bounds(),
            content_mask: mask(),
            filters: smallvec::smallvec![ScaledFilter::Blur(sp(8.0))],
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            masked: false,
            mask_mode: None,
            is_start,
        }
    }

    fn backdrop() -> BackdropFilter {
        BackdropFilter {
            bounds: full_bounds(),
            content_mask: mask(),
            corner_radii: Corners::default(),
            filters: smallvec::smallvec![ScaledFilter::Blur(sp(20.0))],
            opacity: 1.0,
            ..Default::default()
        }
    }

    fn surface() -> PaintSurface {
        PaintSurface {
            order: 0,
            bounds: full_bounds(),
            content_mask: mask(),
            source: SurfaceSource::Unsupported(Size::default()),
        }
    }

    fn batch_kinds(scene: &mut Scene) -> Vec<&'static str> {
        scene.finish();
        scene
            .batches()
            .map(|batch| match batch {
                PrimitiveBatch::Quads { .. } => "quad",
                PrimitiveBatch::BackdropFilters(_) => "backdrop",
                PrimitiveBatch::GroupBoundary(ix) => {
                    if scene.group_boundaries[ix].is_start {
                        "start"
                    } else {
                        "end"
                    }
                }
                _ => "other",
            })
            .collect()
    }

    #[test]
    fn content_filter_group_brackets_its_children() {
        let mut scene = Scene::default();
        // Background painted before the filtered element.
        scene.insert_primitive(quad());
        // A content-filtered element: start marker, its child, end marker.
        scene.insert_primitive(boundary(true));
        scene.insert_primitive(quad());
        scene.insert_primitive(boundary(false));

        // The start must precede the group's child and the end must follow it, so the
        // renderer can redirect rendering for exactly the group's span.
        assert_eq!(
            batch_kinds(&mut scene),
            vec!["quad", "start", "quad", "end"]
        );
    }

    #[test]
    fn surface_opacity_is_preserved_without_changing_paint_surface_layout() {
        let mut scene = Scene::default();
        scene.insert_surface(surface(), 0.25);
        scene.insert_primitive(surface());
        scene.finish();

        assert_eq!(scene.surface_opacities(), &[0.25, 1.0]);

        let mut replay = Scene::default();
        replay.replay(0..scene.paint_operations.len(), &scene);
        replay.finish();
        assert_eq!(replay.surface_opacities(), &[0.25, 1.0]);
    }

    // Note: this validates only the *scene ordering* of nested filter boundaries (start/child/
    // end interleaving), not that a renderer actually isolates both levels — that depends on the
    // backend's group-texture pool (see MAX_FILTER_DEPTH) and is exercised by the `blur` example.
    #[test]
    fn nested_content_filters_emit_well_nested_ordering() {
        let mut scene = Scene::default();
        scene.insert_primitive(boundary(true)); // outer start
        scene.insert_primitive(quad()); // outer child
        scene.insert_primitive(boundary(true)); // inner start
        scene.insert_primitive(quad()); // inner child
        scene.insert_primitive(boundary(false)); // inner end
        scene.insert_primitive(boundary(false)); // outer end

        assert_eq!(
            batch_kinds(&mut scene),
            vec!["start", "quad", "start", "quad", "end", "end"]
        );
    }

    #[test]
    fn content_after_a_filter_group_sorts_above_it() {
        let mut scene = Scene::default();
        // A content-filtered element: start marker, its child, end marker.
        scene.insert_primitive(boundary(true));
        scene.insert_primitive(quad());
        scene.insert_primitive(boundary(false));
        // A sibling painted after the group that does NOT overlap it. Without the close-time
        // order-floor it would reuse the lowest order, tie with the start marker, and be swept
        // into the group (start, quad, quad, end); it must instead sort after the end marker.
        scene.insert_primitive(detached_quad());

        assert_eq!(
            batch_kinds(&mut scene),
            vec!["start", "quad", "end", "quad"]
        );
    }

    #[test]
    fn backdrop_filter_sorts_before_a_later_overlapping_quad() {
        let mut scene = Scene::default();
        // Content behind the frosted panel.
        scene.insert_primitive(quad());
        // The panel: its backdrop snapshot, then its (translucent) background quad on top.
        scene.insert_primitive(backdrop());
        scene.insert_primitive(quad());

        assert_eq!(batch_kinds(&mut scene), vec!["quad", "backdrop", "quad"]);
    }

    #[test]
    fn render_commands_pair_nested_filters_and_isolate_each() {
        let mut scene = Scene::default();
        // Three nested groups each receive a target of their own, at any depth.
        scene.insert_primitive(boundary(true));
        scene.insert_primitive(quad());
        scene.insert_primitive(boundary(true));
        scene.insert_primitive(quad());
        scene.insert_primitive(boundary(true));
        scene.insert_primitive(quad());
        scene.insert_primitive(boundary(false));
        scene.insert_primitive(boundary(false));
        scene.insert_primitive(boundary(false));
        scene.finish();

        let commands: Vec<_> = scene
            .render_commands()
            .iter()
            .map(|command| match command {
                RenderCommand::Batch(PrimitiveBatch::Quads { .. }) => "quad".to_string(),
                RenderCommand::BeginGroup {
                    boundary_index,
                    target,
                } => {
                    let boundary = &scene.group_boundaries[*boundary_index];
                    assert!(boundary.is_start);
                    format!("begin:{target:?}")
                }
                // An end command must carry its matched *start* boundary. This lets a
                // renderer use the opening group's bounds, filters, and opacity while
                // composing it, rather than trusting an independently-sorted end marker.
                RenderCommand::EndGroup {
                    boundary_index,
                    target,
                    ..
                } => {
                    let boundary = &scene.group_boundaries[*boundary_index];
                    assert!(boundary.is_start);
                    format!("end:{target:?}")
                }
                RenderCommand::Batch(other) => panic!("unexpected batch: {other:?}"),
            })
            .collect();

        let isolated = format!(
            "{:?}",
            GroupTarget::Isolated {
                region: full_bounds()
            }
        );
        assert_eq!(
            commands,
            vec![
                format!("begin:{isolated}"),
                "quad".into(),
                format!("begin:{isolated}"),
                "quad".into(),
                format!("begin:{isolated}"),
                "quad".into(),
                format!("end:{isolated}"),
                format!("end:{isolated}"),
                format!("end:{isolated}"),
            ]
        );
    }

    #[test]
    fn render_commands_keep_later_content_outside_a_completed_filter() {
        let mut scene = Scene::default();
        scene.insert_primitive(boundary(true));
        scene.insert_primitive(quad());
        scene.insert_primitive(boundary(false));
        // This non-overlapping sibling would reuse a low order without the close-time
        // order floor. The command stream is the renderer contract: it must come after
        // the group has been composited, not be painted into its offscreen target.
        scene.insert_primitive(detached_quad());
        scene.finish();

        let commands: Vec<_> = scene
            .render_commands()
            .iter()
            .map(|command| match command {
                RenderCommand::BeginGroup { target, .. } => {
                    format!("begin:{target:?}")
                }
                RenderCommand::EndGroup { target, .. } => format!("end:{target:?}"),
                RenderCommand::Batch(PrimitiveBatch::Quads { .. }) => "quad".to_string(),
                RenderCommand::Batch(other) => panic!("unexpected batch: {other:?}"),
            })
            .collect();

        let isolated = format!(
            "{:?}",
            GroupTarget::Isolated {
                region: full_bounds()
            }
        );
        assert_eq!(
            commands,
            vec![
                format!("begin:{isolated}"),
                "quad".into(),
                format!("end:{isolated}"),
                "quad".into()
            ]
        );
    }

    /// A group's target covers what it draws, which an enclosing group's
    /// covers in turn; a group with nothing to apply draws in place.
    #[test]
    fn a_group_target_covers_what_the_group_draws() {
        let rect = |x: f32, y: f32, width: f32, height: f32| Bounds {
            origin: point(sp(x), sp(y)),
            size: Size {
                width: sp(width),
                height: sp(height),
            },
        };
        let wide_mask = ContentMask {
            bounds: rect(0., 0., 1000., 1000.),
            ..Default::default()
        };
        let group = |is_start: bool, opacity: f32, blend_mode: BlendMode| GroupBoundary {
            order: 0,
            bounds: rect(0., 0., 1000., 1000.),
            content_mask: wide_mask,
            filters: SmallVec::new(),
            opacity,
            blend_mode,
            masked: false,
            mask_mode: None,
            is_start,
        };
        let quad_at = |bounds: Bounds<ScaledPixels>| Quad {
            bounds,
            content_mask: wide_mask,
            ..Default::default()
        };

        let mut scene = Scene::default();
        scene.insert_primitive(group(true, 0.5, BlendMode::Normal));
        scene.insert_primitive(quad_at(rect(10., 10., 20., 20.)));
        scene.insert_primitive(group(true, 1.0, BlendMode::Multiply));
        scene.insert_primitive(quad_at(rect(100., 100., 10., 10.)));
        scene.insert_primitive(group(true, 1.0, BlendMode::Normal));
        scene.insert_primitive(quad_at(rect(300., 300., 10., 10.)));
        scene.insert_primitive(group(false, 1.0, BlendMode::Normal));
        scene.insert_primitive(group(false, 1.0, BlendMode::Multiply));
        scene.insert_primitive(group(false, 0.5, BlendMode::Normal));
        scene.finish();

        let targets: Vec<GroupTarget> = scene
            .render_commands()
            .iter()
            .filter_map(|command| match command {
                RenderCommand::BeginGroup { target, .. } => Some(*target),
                _ => None,
            })
            .collect();
        assert_eq!(
            targets,
            vec![
                GroupTarget::Isolated {
                    region: rect(10., 10., 300., 300.)
                },
                GroupTarget::Isolated {
                    region: rect(100., 100., 210., 210.)
                },
                GroupTarget::Inline,
            ]
        );
        let requirements = scene.render_plan().requirements();
        assert_eq!(requirements.isolated_group_count, 2);
        assert!(
            requirements.uses_offscreen_target,
            "a blend mode reads what is beneath it"
        );

        let mut faded = Scene::default();
        faded.insert_primitive(group(true, 0.5, BlendMode::Normal));
        faded.insert_primitive(quad_at(rect(10., 10., 20., 20.)));
        faded.insert_primitive(group(false, 0.5, BlendMode::Normal));
        faded.finish();
        assert!(
            !faded.requires_offscreen_rendering(),
            "a faded group does not read what is beneath it"
        );
    }

    /// A masked group shows only within its mask: its target is cut to the
    /// mask's region, the mask draws nothing into it, and a group whose mask
    /// draws nothing has no commands at all.
    #[test]
    fn a_masked_group_is_cut_to_its_mask() {
        let rect = |x: f32, y: f32, width: f32, height: f32| Bounds {
            origin: point(sp(x), sp(y)),
            size: Size {
                width: sp(width),
                height: sp(height),
            },
        };
        let wide_mask = ContentMask {
            bounds: rect(0., 0., 1000., 1000.),
            ..Default::default()
        };
        let group = |is_start: bool, masked: bool, mask_mode: Option<MaskMode>| GroupBoundary {
            order: 0,
            bounds: rect(0., 0., 1000., 1000.),
            content_mask: wide_mask,
            filters: SmallVec::new(),
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            masked,
            mask_mode,
            is_start,
        };
        let quad_at = |bounds: Bounds<ScaledPixels>| Quad {
            bounds,
            content_mask: wide_mask,
            ..Default::default()
        };
        let masked_scene = |mask: Option<Bounds<ScaledPixels>>| {
            let mut scene = Scene::default();
            scene.insert_primitive(group(true, true, None));
            scene.insert_primitive(group(true, false, Some(MaskMode::Alpha)));
            if let Some(mask) = mask {
                scene.insert_primitive(quad_at(mask));
            }
            scene.insert_primitive(group(false, false, Some(MaskMode::Alpha)));
            scene.insert_primitive(quad_at(rect(0., 0., 100., 100.)));
            scene.insert_primitive(group(false, true, None));
            scene.finish();
            scene
        };

        let scene = masked_scene(Some(rect(50., 50., 200., 200.)));
        let targets: Vec<GroupTarget> = scene
            .render_commands()
            .iter()
            .filter_map(|command| match command {
                RenderCommand::BeginGroup { target, .. } => Some(*target),
                _ => None,
            })
            .collect();
        assert_eq!(
            targets,
            vec![
                GroupTarget::Isolated {
                    region: rect(50., 50., 50., 50.)
                },
                GroupTarget::Isolated {
                    region: rect(50., 50., 200., 200.)
                },
            ]
        );

        let scene = masked_scene(None);
        assert!(scene.render_commands().is_empty());
        let scene = masked_scene(Some(rect(500., 500., 10., 10.)));
        assert!(scene.render_commands().is_empty());
    }

    /// A faded group whose primitives don't overlap draws in place, each
    /// primitive faded, once however often the plan is built; one whose
    /// primitives overlap is rendered on its own.
    #[test]
    fn a_faded_group_of_separate_primitives_draws_in_place() {
        let rect = |x: f32, y: f32| Bounds {
            origin: point(sp(x), sp(y)),
            size: Size {
                width: sp(20.),
                height: sp(20.),
            },
        };
        let wide_mask = ContentMask {
            bounds: Bounds {
                origin: point(sp(0.), sp(0.)),
                size: Size {
                    width: sp(1000.),
                    height: sp(1000.),
                },
            },
            ..Default::default()
        };
        let group = |is_start: bool| GroupBoundary {
            order: 0,
            bounds: wide_mask.bounds,
            content_mask: wide_mask,
            filters: SmallVec::new(),
            opacity: 0.5,
            blend_mode: BlendMode::Normal,
            masked: false,
            mask_mode: None,
            is_start,
        };
        let quad_at = |bounds: Bounds<ScaledPixels>| Quad {
            bounds,
            content_mask: wide_mask,
            background: crate::black().into(),
            ..Default::default()
        };
        let faded_scene = |second: Bounds<ScaledPixels>| {
            let mut scene = Scene::default();
            scene.insert_primitive(group(true));
            scene.insert_primitive(quad_at(rect(0., 0.)));
            scene.insert_primitive(quad_at(second));
            scene.insert_primitive(group(false));
            scene.finish();
            scene
        };
        let begin_target = |scene: &Scene| {
            scene
                .render_commands()
                .iter()
                .find_map(|command| match command {
                    RenderCommand::BeginGroup { target, .. } => Some(*target),
                    _ => None,
                })
        };

        let mut scene = faded_scene(rect(50., 0.));
        assert_eq!(begin_target(&scene), Some(GroupTarget::Inline));
        let alphas = |scene: &Scene| {
            scene
                .quads
                .iter()
                .map(|quad| quad.background.color.a)
                .collect::<Vec<_>>()
        };
        assert_eq!(alphas(&scene), vec![0.5, 0.5]);
        scene.finish();
        assert_eq!(alphas(&scene), vec![0.5, 0.5], "faded once");

        let scene = faded_scene(rect(10., 10.));
        assert!(matches!(
            begin_target(&scene),
            Some(GroupTarget::Isolated { .. })
        ));
        assert_eq!(alphas(&scene), vec![1.0, 1.0]);
    }

    #[test]
    fn unmatched_filter_start_renders_inline() {
        let mut scene = Scene::default();
        scene.insert_primitive(boundary(true));
        scene.insert_primitive(quad());
        scene.finish();

        let mut commands = scene.render_commands().iter();
        assert!(matches!(
            commands.next(),
            Some(RenderCommand::BeginGroup {
                target: GroupTarget::Inline,
                ..
            })
        ));
        assert!(matches!(
            commands.next(),
            Some(RenderCommand::Batch(PrimitiveBatch::Quads { .. }))
        ));
        assert!(commands.next().is_none());
        assert!(!scene.requires_offscreen_rendering());
    }

    #[test]
    fn finish_rebuilds_the_plan_after_direct_primitive_mutation() {
        let mut scene = Scene::default();
        scene.insert_primitive(quad());
        scene.finish();
        let commands = scene.render_commands().to_vec();

        // Primitive storage is public, so direct mutation must change the compiled plan.
        scene.quads.push(quad());
        scene.finish();
        assert_ne!(scene.render_commands(), commands);
        assert_eq!(scene.render_plan().requirements().instance_batch_count, 1);
    }

    #[test]
    fn render_plan_reuses_its_command_allocation_across_frames() {
        let mut scene = Scene::default();
        for _ in 0..32 {
            scene.insert_primitive(quad());
        }
        scene.finish();
        let capacity = scene.render_plan.commands.capacity();
        assert!(capacity >= 32);

        scene.clear();
        assert_eq!(scene.render_plan.commands.capacity(), capacity);
        scene.insert_primitive(quad());
        scene.finish();
        assert_eq!(scene.render_plan.commands.capacity(), capacity);
    }

    #[test]
    fn maximum_draw_order_is_not_treated_as_an_empty_batch_cursor() {
        let mut scene = Scene::default();
        let mut quad = quad();
        quad.order = DrawOrder::MAX;
        scene.quads.push(quad);
        scene.finish();

        assert!(matches!(
            scene.render_commands(),
            [RenderCommand::Batch(PrimitiveBatch::Quads { range, smoothed: false })]
                if range == &(0..1)
        ));
    }

    #[test]
    fn plan_requirements_are_collected_with_filter_pairing() {
        let mut scene = Scene::default();
        scene.insert_primitive(backdrop());
        scene.insert_primitive(boundary(true));
        scene.insert_primitive(quad());
        scene.insert_primitive(boundary(false));
        scene.finish();

        let requirements = scene.render_plan().requirements();
        assert_eq!(requirements.backdrop_filter_count, 1);
        assert_eq!(requirements.isolated_group_count, 1);
        assert_eq!(requirements.instance_batch_count, 1);
        assert!(requirements.uses_offscreen_target);
    }
}
