use super::{AtlasTextureId, BatchIterator, BlendMode, GroupBoundary, Scene};
use crate::{Bounds, ScaledPixels, point};
use smallvec::SmallVec;
use std::ops::Range;

/// Where the contents of a group are rendered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GroupTarget {
    /// In place, into the current target: the group has nothing to apply as
    /// a whole, or draws nothing.
    Inline,
    /// Into a target of its own covering `region` of the viewport, from
    /// which it is composited into its parent. The region holds everything
    /// the group draws, spread by its filters, and clipped to its mask; it
    /// is not yet clipped to the viewport.
    ///
    /// A mask group is not composited: its parent's composite samples its
    /// target, and shows only within its region. A masked group whose mask
    /// draws nothing has no commands at all.
    Isolated {
        /// The viewport rectangle the group's target covers.
        region: Bounds<ScaledPixels>,
    },
}

/// Resource totals computed while a scene's render plan is compiled.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[expect(missing_docs)]
pub struct ScenePlanRequirements {
    pub command_count: usize,
    pub instance_batch_count: usize,
    pub path_rasterization_vertex_count: usize,
    pub path_sprite_count: usize,
    pub surface_count: usize,
    pub backdrop_filter_count: usize,
    pub isolated_group_count: usize,
    /// Passes of isolated groups' filters that blur, each up to three draws.
    pub group_blur_count: usize,
    /// Other passes of isolated groups' filters, each one draw.
    pub group_filter_pass_count: usize,
    /// Chunks drawn, not counting those inside them, whose own requirements
    /// are added to these.
    pub chunk_count: usize,
    pub uses_path_target: bool,
    pub uses_offscreen_target: bool,
}

/// A compiled scene command stream, built by [`Scene::finish`] and shared by every renderer.
#[derive(Debug, Default)]
pub struct ScenePlan {
    // Retain this allocation across `Scene::clear`/`Scene::finish`; scenes are rebuilt every frame.
    pub(super) commands: Vec<RenderCommand>,
    /// Groups drawn in place with their opacity folded into what they draw,
    /// for [`Scene::finish`] to fade.
    pub(super) folds: Vec<OpacityFold>,
    requirements: ScenePlanRequirements,
    scene_lengths: SceneLengths,
}

impl ScenePlan {
    pub(super) fn clear(&mut self) {
        self.commands.clear();
        self.folds.clear();
        self.requirements = ScenePlanRequirements::default();
        self.scene_lengths = SceneLengths::default();
    }

    pub(super) fn build(
        scene: &Scene,
        mut commands: Vec<RenderCommand>,
        mut folds: Vec<OpacityFold>,
    ) -> Self {
        commands.clear();
        folds.clear();
        commands.reserve(scene.len());
        let mut matched_starts =
            SmallVec::<[bool; 8]>::from_elem(false, scene.group_boundaries.len());
        let mut pending_starts = SmallVec::<[usize; 4]>::new();
        for (index, boundary) in scene.group_boundaries.iter().enumerate() {
            if boundary.is_start {
                pending_starts.push(index);
            } else if let Some(start_index) = pending_starts.pop() {
                matched_starts[start_index] = true;
            }
        }

        /// A group whose end has not been reached: where its begin command
        /// is, to set its target once its region is known.
        struct OpenGroup {
            command: usize,
            boundary_index: usize,
            region: Option<Bounds<ScaledPixels>>,
            /// The region of its mask group, once drawn, for a masked group.
            mask_region: Option<Bounds<ScaledPixels>>,
            /// The requirements before it began: what they return to if the
            /// group is dropped, with all it holds.
            requirements: ScenePlanRequirements,
        }
        let union = |region: &mut Option<Bounds<ScaledPixels>>, bounds: Bounds<ScaledPixels>| {
            *region = Some(region.map_or(bounds, |region| region.union(&bounds)));
        };
        let mut open = SmallVec::<[OpenGroup; 4]>::new();
        let mut requirements = ScenePlanRequirements::default();

        for batch in BatchIterator::new(scene) {
            match batch {
                PrimitiveBatch::GroupBoundary(boundary_index) => {
                    let boundary = &scene.group_boundaries[boundary_index];
                    if boundary.is_start {
                        open.push(OpenGroup {
                            command: commands.len(),
                            boundary_index,
                            region: None,
                            mask_region: None,
                            requirements,
                        });
                        commands.push(RenderCommand::BeginGroup {
                            boundary_index,
                            target: GroupTarget::Inline,
                        });
                    } else if let Some(group) = open.pop() {
                        let start = &scene.group_boundaries[group.boundary_index];
                        let matched = matched_starts[group.boundary_index];
                        let region = group.region.map(|region| {
                            let outsets = start.filter_outsets();
                            Bounds::from_corners(
                                point(
                                    region.origin.x - ScaledPixels(outsets.left),
                                    region.origin.y - ScaledPixels(outsets.top),
                                ),
                                point(
                                    region.right() + ScaledPixels(outsets.right),
                                    region.bottom() + ScaledPixels(outsets.bottom),
                                ),
                            )
                            .intersect(&start.content_mask.bounds)
                        });
                        let target = match region {
                            Some(region) if matched && start.masked => {
                                // Nothing of a masked group shows outside its
                                // mask, nor anything at all without one.
                                let region = group
                                    .mask_region
                                    .map(|mask| region.intersect(&mask))
                                    .filter(|region| !region.is_empty());
                                let Some(region) = region else {
                                    commands.truncate(group.command);
                                    requirements = group.requirements;
                                    continue;
                                };
                                GroupTarget::Isolated { region }
                            }
                            Some(region) if matched && start.isolates() && !region.is_empty() => {
                                match foldable_batches(scene, start, &commands[group.command + 1..])
                                {
                                    Some(batches) => {
                                        folds.push(OpacityFold {
                                            start: group.boundary_index,
                                            end: boundary_index,
                                            opacity: start.opacity,
                                            batches,
                                        });
                                        GroupTarget::Inline
                                    }
                                    None => GroupTarget::Isolated { region },
                                }
                            }
                            None if matched && start.masked => {
                                commands.truncate(group.command);
                                requirements = group.requirements;
                                continue;
                            }
                            _ => GroupTarget::Inline,
                        };
                        let drawn = match target {
                            GroupTarget::Isolated { region } => {
                                requirements.isolated_group_count += 1;
                                if start.is_filtered() {
                                    for pass in start.filter_plan().passes {
                                        match pass {
                                            crate::FilterPass::Blur { .. } => {
                                                requirements.group_blur_count += 1
                                            }
                                            _ => requirements.group_filter_pass_count += 1,
                                        }
                                    }
                                }
                                requirements.uses_offscreen_target |=
                                    start.blend_mode != BlendMode::Normal;
                                Some(region)
                            }
                            GroupTarget::Inline => group.region,
                        };
                        if let Some(parent) = open.last_mut() {
                            if start.mask_mode.is_some() {
                                // A mask draws nothing into its parent: it is
                                // where its parent shows.
                                if let GroupTarget::Isolated { region } = target {
                                    parent.mask_region = Some(region);
                                }
                            } else if let Some(drawn) = drawn {
                                union(&mut parent.region, drawn);
                            }
                        }
                        commands[group.command] = RenderCommand::BeginGroup {
                            boundary_index: group.boundary_index,
                            target,
                        };
                        commands.push(RenderCommand::EndGroup {
                            boundary_index: group.boundary_index,
                            closing_boundary_index: boundary_index,
                            target,
                        });
                    } else {
                        debug_assert!(false, "group end boundary has no matching start");
                        commands.push(RenderCommand::EndGroup {
                            boundary_index,
                            closing_boundary_index: boundary_index,
                            target: GroupTarget::Inline,
                        });
                    }
                }
                batch => {
                    if let Some(group) = open.last_mut()
                        && let Some(region) = scene.batch_region(&batch)
                    {
                        union(&mut group.region, region);
                    }
                    requirements.include_batch(&batch);
                    commands.push(RenderCommand::Batch(batch));
                }
            }
        }

        for chunk in &scene.chunks {
            requirements.include_chunk(chunk.chunk.scene.render_plan().requirements());
        }
        requirements.command_count = commands.len();
        Self {
            commands,
            folds,
            requirements,
            scene_lengths: SceneLengths::for_scene(scene),
        }
    }

    /// Returns the ordered rendering commands.
    pub fn commands(&self) -> &[RenderCommand] {
        &self.commands
    }

    /// Returns the resource totals collected while compiling this plan.
    pub fn requirements(&self) -> &ScenePlanRequirements {
        &self.requirements
    }

    pub(super) fn assert_matches(&self, scene: &Scene) {
        debug_assert_eq!(
            self.scene_lengths,
            SceneLengths::for_scene(scene),
            "scene primitive storage changed after Scene::finish"
        );
    }
}

/// A faded group drawn in place: its markers, its opacity, and the batches
/// it draws, to be faded by it.
#[derive(Debug)]
pub(super) struct OpacityFold {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) opacity: f32,
    pub(super) batches: SmallVec<[PrimitiveBatch; 4]>,
}

/// Groups that fade up to this many primitives are checked for overlap,
/// pair by pair; larger ones are rendered on their own.
const MAX_FOLDED_PRIMITIVES: usize = 16;

/// The batches of a group that only fades, whose `commands` draw nothing
/// that overlaps, so fading each primitive fades the group as one picture:
/// `None` when the group must be rendered on its own.
fn foldable_batches(
    scene: &Scene,
    start: &GroupBoundary,
    commands: &[RenderCommand],
) -> Option<SmallVec<[PrimitiveBatch; 4]>> {
    if start.blend_mode != BlendMode::Normal
        || start.is_filtered()
        || start.masked
        || start.mask_mode.is_some()
    {
        return None;
    }
    let mut batches = SmallVec::new();
    let mut regions = SmallVec::<[Bounds<ScaledPixels>; MAX_FOLDED_PRIMITIVES]>::new();
    let mut overflowed = false;
    for command in commands {
        // A nested group, or a backdrop filter, which reads what the group
        // has drawn so far, needs the group's own target.
        let RenderCommand::Batch(batch) = command else {
            return None;
        };
        // A chunk is drawn as it was recorded, so it can't be faded.
        if matches!(
            batch,
            PrimitiveBatch::BackdropFilters(_) | PrimitiveBatch::Chunks(_)
        ) {
            return None;
        }
        scene.for_each_primitive_region(batch, &mut |region| {
            if overflowed
                || regions.len() == MAX_FOLDED_PRIMITIVES
                || regions
                    .iter()
                    .any(|other| !other.intersect(&region).is_empty())
            {
                overflowed = true;
            } else {
                regions.push(region);
            }
        });
        if overflowed {
            return None;
        }
        batches.push(batch.clone());
    }
    Some(batches)
}

impl ScenePlanRequirements {
    /// Adds what a chunk drawn in the scene needs.
    fn include_chunk(&mut self, chunk: &ScenePlanRequirements) {
        self.instance_batch_count += chunk.instance_batch_count;
        self.path_rasterization_vertex_count += chunk.path_rasterization_vertex_count;
        self.path_sprite_count += chunk.path_sprite_count;
        self.chunk_count += chunk.chunk_count;
        self.uses_path_target |= chunk.uses_path_target;
    }

    fn include_batch(&mut self, batch: &PrimitiveBatch) {
        match batch {
            PrimitiveBatch::Shadows { range, .. }
            | PrimitiveBatch::Quads { range, .. }
            | PrimitiveBatch::Meshes(range)
            | PrimitiveBatch::Underlines(range) => {
                self.instance_batch_count += usize::from(!range.is_empty());
            }
            PrimitiveBatch::Paths {
                rasterization_vertex_count,
                sprite_count,
                ..
            } => {
                self.path_rasterization_vertex_count += rasterization_vertex_count;
                if *rasterization_vertex_count > 0 {
                    self.path_sprite_count += sprite_count;
                    self.uses_path_target = true;
                    self.instance_batch_count += 2;
                }
            }
            PrimitiveBatch::MonochromeSprites { range, .. }
            | PrimitiveBatch::SubpixelSprites { range, .. }
            | PrimitiveBatch::PolychromeSprites { range, .. } => {
                self.instance_batch_count += usize::from(!range.is_empty());
            }
            PrimitiveBatch::Surfaces(range) => self.surface_count += range.len(),
            PrimitiveBatch::BackdropFilters(range) => {
                self.backdrop_filter_count += range.len();
                self.uses_offscreen_target |= !range.is_empty();
            }
            PrimitiveBatch::Chunks(range) => self.chunk_count += range.len(),
            PrimitiveBatch::GroupBoundary(_) => {
                unreachable!("group boundaries are compiled before requirements are collected")
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SceneLengths {
    shadows: usize,
    quads: usize,
    paths: usize,
    meshes: usize,
    underlines: usize,
    monochrome_sprites: usize,
    subpixel_sprites: usize,
    polychrome_sprites: usize,
    surfaces: usize,
    backdrop_filters: usize,
    group_boundaries: usize,
}

impl SceneLengths {
    fn for_scene(scene: &Scene) -> Self {
        Self {
            shadows: scene.shadows.len(),
            quads: scene.quads.len(),
            paths: scene.paths.len(),
            meshes: scene.meshes.len(),
            underlines: scene.underlines.len(),
            monochrome_sprites: scene.monochrome_sprites.len(),
            subpixel_sprites: scene.subpixel_sprites.len(),
            polychrome_sprites: scene.polychrome_sprites.len(),
            surfaces: scene.surfaces.len(),
            backdrop_filters: scene.backdrop_filters.len(),
            group_boundaries: scene.group_boundaries.len(),
        }
    }
}

/// A contiguous range of one primitive type drawn by a single pipeline invocation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(missing_docs)]
pub enum PrimitiveBatch {
    Shadows {
        range: Range<usize>,
        smoothed: bool,
    },
    Quads {
        range: Range<usize>,
        smoothed: bool,
    },
    Paths {
        range: Range<usize>,
        rasterization_vertex_count: usize,
        sprite_count: usize,
    },
    /// Meshes, each drawn from its own retained vertices.
    Meshes(Range<usize>),
    Underlines(Range<usize>),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    SubpixelSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
        smoothed: bool,
    },
    Surfaces(Range<usize>),
    BackdropFilters(Range<usize>),
    GroupBoundary(usize),
    /// Chunks, each drawn as one unit, by its own batches, at its placement.
    Chunks(Range<usize>),
}

/// Backend-neutral rendering work derived from a [`Scene`].
#[derive(Clone, Debug, PartialEq)]
pub enum RenderCommand {
    /// A normal primitive batch.
    Batch(PrimitiveBatch),
    /// Begin rendering a group.
    BeginGroup {
        /// Index of the opening boundary in [`Scene::group_boundaries`].
        boundary_index: usize,
        /// Whether this group renders in place or into a target of its own.
        target: GroupTarget,
    },
    /// Finish a group, compositing it if it was isolated.
    EndGroup {
        /// Index of the matched opening boundary in [`Scene::group_boundaries`].
        boundary_index: usize,
        /// Index of the closing marker in [`Scene::group_boundaries`].
        closing_boundary_index: usize,
        /// The same target selected by the matching begin command.
        target: GroupTarget,
    },
}

impl RenderCommand {
    /// Returns the opening group boundary carried by a group command.
    pub fn boundary<'a>(&self, scene: &'a Scene) -> Option<&'a GroupBoundary> {
        match self {
            Self::BeginGroup { boundary_index, .. } | Self::EndGroup { boundary_index, .. } => {
                Some(&scene.group_boundaries[*boundary_index])
            }
            Self::Batch(_) => None,
        }
    }

    /// A diagnostic label suitable for GPU debug annotations.
    pub fn label(&self) -> String {
        match self {
            Self::Batch(batch) => batch.label(),
            Self::BeginGroup { target, .. } => match target {
                GroupTarget::Isolated { .. } => "begin isolated group".into(),
                GroupTarget::Inline => "begin inline group".into(),
            },
            Self::EndGroup { target, .. } => match target {
                GroupTarget::Isolated { .. } => "composite isolated group".into(),
                GroupTarget::Inline => "end inline group".into(),
            },
        }
    }
}

impl PrimitiveBatch {
    /// A diagnostic label suitable for GPU debug annotations.
    pub fn label(&self) -> String {
        match self {
            Self::Shadows { range, smoothed } => format!(
                "{}shadows ({})",
                if *smoothed { "smoothed " } else { "" },
                range.len()
            ),
            Self::Quads { range, smoothed } => format!(
                "{}quads ({})",
                if *smoothed { "smoothed " } else { "" },
                range.len()
            ),
            Self::Paths { range, .. } => format!("paths ({})", range.len()),
            Self::Meshes(range) => format!("meshes ({})", range.len()),
            Self::Underlines(range) => format!("underlines ({})", range.len()),
            Self::MonochromeSprites { texture_id, range } => format!(
                "monochrome sprites ({}) on atlas {}",
                range.len(),
                texture_id.index
            ),
            Self::SubpixelSprites { texture_id, range } => format!(
                "subpixel sprites ({}) on atlas {}",
                range.len(),
                texture_id.index
            ),
            Self::PolychromeSprites {
                texture_id,
                range,
                smoothed,
            } => format!(
                "{}polychrome sprites ({}) on atlas {}",
                if *smoothed { "smoothed " } else { "" },
                range.len(),
                texture_id.index
            ),
            Self::Surfaces(range) => format!("surfaces ({})", range.len()),
            Self::BackdropFilters(range) => format!("backdrop filters ({})", range.len()),
            Self::GroupBoundary(index) => format!("group boundary ({index})"),
            Self::Chunks(range) => format!("chunks ({})", range.len()),
        }
    }
}
