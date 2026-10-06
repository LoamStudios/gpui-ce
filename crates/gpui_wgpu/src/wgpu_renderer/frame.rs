use super::{
    WgpuRenderer, begin_color_render_pass,
    buffers::{InstanceTransport, InstanceUpload},
    filters::{FILTER_UNIFORMS_PER_COMPOSITE, FrameUniformRequirements},
    path_types,
    target_pool::PooledTexture,
};
use gpui::{
    Bounds, DevicePixels, GroupTarget, MaskMode, MonochromeSprite, PolychromeSprite,
    PrimitiveBatch, Quad, RenderCommand, Scene, Shadow, SubpixelSprite, Underline, point,
};
use gpui_render::group::group_target_bounds;
use gpui_render::shaders::{
    common::{FontRasterizationUniforms, GlobalUniforms, ShaderBool},
    interface as shader_interface,
};
use smallvec::SmallVec;
use wgsl_rs::std::vec2f;

pub(super) fn render_to_view(
    renderer: &mut WgpuRenderer,
    scene: &Scene,
    frame_view: &wgpu::TextureView,
    readback: Option<ReadbackCopy<'_>>,
) -> Option<wgpu::SubmissionIndex> {
    let Some(targets) = PreparedTargets::prepare(renderer, scene, frame_view) else {
        return None;
    };
    // Prune once against the complete scene. Doing this inside each draw batch would evict a
    // texture that reappears after an intervening primitive and recreate its platform view.
    renderer.retain_surface_cache(&scene.surfaces);

    let encoded = FrameEncoder::new(renderer, scene, targets).encode(readback);
    renderer.resources_mut().end_target_pool_frame();
    match encoded {
        Ok(command_buffers) => Some(renderer.resources().queue.submit(command_buffers)),
        Err(DrawError::ExternalSurface) => None,
        Err(DrawError::CapacityPlanningInvariant) => {
            log::error!("frame storage exceeded its precomputed capacity");
            None
        }
        Err(DrawError::MissingIntermediateTarget) => {
            log::error!("frame preparation did not create a required intermediate target");
            None
        }
    }
}

pub(super) struct ReadbackCopy<'a> {
    pub(super) texture: &'a wgpu::Texture,
    pub(super) buffer: &'a wgpu::Buffer,
    pub(super) bytes_per_row: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

struct PreparedTargets {
    active: wgpu::TextureView,
    presentation: wgpu::TextureView,
    offscreen: Option<wgpu::TextureView>,
    /// The texture behind `offscreen`, which blended groups copy what is beneath them from.
    offscreen_texture: Option<wgpu::Texture>,
    instances: InstanceUpload,
    globals: FrameGlobals,
}

impl PreparedTargets {
    fn prepare(
        renderer: &mut WgpuRenderer,
        scene: &Scene,
        frame_view: &wgpu::TextureView,
    ) -> Option<Self> {
        if !begin_frame(renderer) {
            return None;
        }
        renderer.resources_mut().upload_photo_tiles(scene);
        let requirements = {
            let transport = renderer.resources().instances.transport();
            FrameRequirements::for_scene(scene, transport)
        };
        let device = renderer.resources().device.clone();
        let resources = renderer.resources_mut();
        if !resources.instances.ensure_capacity(
            &device,
            &resources.bind_group_layouts,
            requirements.storage_bytes,
            requirements.instance_batches,
        ) {
            return None;
        }
        let instances = {
            let resources = renderer.resources();
            resources.instances.begin_upload(
                &resources.queue,
                requirements.storage_bytes,
                requirements.instance_batches,
            )?
        };
        if !renderer.resources_mut().upload_scene_tables(scene) {
            return None;
        }
        if !renderer.ensure_uniform_capacity(requirements.uniforms) {
            return None;
        }

        if requirements.uses_path_target {
            renderer.ensure_path_textures();
        }
        if requirements.uses_offscreen_target {
            renderer.ensure_scene_color_texture();
        }
        let globals = write_shader_globals(renderer);

        if requirements.uses_offscreen_target {
            let resources = renderer.resources();
            let offscreen = resources
                .scene_color_view
                .as_ref()
                .expect("offscreen preparation must create a scene target")
                .clone();
            Some(Self {
                active: offscreen.clone(),
                presentation: frame_view.clone(),
                offscreen: Some(offscreen),
                offscreen_texture: resources.scene_color_texture.clone(),
                instances,
                globals,
            })
        } else {
            Some(Self {
                active: frame_view.clone(),
                presentation: frame_view.clone(),
                offscreen: None,
                offscreen_texture: None,
                instances,
                globals,
            })
        }
    }
}

fn begin_frame(renderer: &mut WgpuRenderer) -> bool {
    let Some(error) = renderer.faults.pending_error.lock().unwrap().take() else {
        renderer.faults.consecutive_failed_frames = 0;
        renderer.atlas.before_frame();
        return true;
    };

    renderer.faults.consecutive_failed_frames += 1;
    log::error!(
        "GPU error during frame (failure {} of 10): {error}",
        renderer.faults.consecutive_failed_frames
    );
    if renderer.faults.consecutive_failed_frames > 10 {
        panic!("too many consecutive GPU errors; last error: {error}");
    }
    if renderer.faults.consecutive_failed_frames > 5 {
        if let Some(resources) = renderer.resources.as_mut() {
            resources.invalidate_intermediate_textures();
        }
        renderer.atlas.clear();
        renderer.target.request_redraw();
        renderer.faults.consecutive_failed_frames = 0;
        return false;
    }

    renderer.atlas.before_frame();
    true
}

/// Where this frame's global uniforms are, by dynamic offset into the target globals.
#[derive(Clone, Copy)]
struct FrameGlobals {
    /// The window target's globals, which an isolated group's are made from.
    window: GlobalUniforms,
    window_offset: u32,
    /// The path intermediate's globals: it covers the viewport, as the window does.
    paths_offset: u32,
}

/// Uploads the font rasterization uniforms if they changed, and writes the window's and
/// the path intermediate's globals into the first two target slots.
fn write_shader_globals(renderer: &mut WgpuRenderer) -> FrameGlobals {
    let font = renderer.rendering_params.font_rasterization;
    let font_rasterization = FontRasterizationUniforms {
        gamma_ratios: wgsl_rs::std::vec4f(
            font.gamma_ratios[0],
            font.gamma_ratios[1],
            font.gamma_ratios[2],
            font.gamma_ratios[3],
        ),
        grayscale_enhanced_contrast: font.grayscale_enhanced_contrast,
        subpixel_enhanced_contrast: font.subpixel_enhanced_contrast,
        uses_blue_green_red_subpixel_order: ShaderBool::from(
            renderer.subpixel_order == super::SubpixelOrder::BlueGreenRed,
        ),
        padding: 0,
    };
    let viewport_size = vec2f(
        renderer.target.width() as f32,
        renderer.target.height() as f32,
    );
    let window = GlobalUniforms {
        viewport_size,
        target_origin: vec2f(0.0, 0.0),
        target_size: viewport_size,
        premultiplied_alpha: ShaderBool::from(
            renderer.target.alpha_mode() == wgpu::CompositeAlphaMode::PreMultiplied,
        ),
        padding: 0,
        placement: GlobalUniforms::unplaced(),
        inverse_placement: GlobalUniforms::unplaced(),
        placement_translation: vec2f(0.0, 0.0),
        inverse_placement_translation: vec2f(0.0, 0.0),
    };
    let paths = GlobalUniforms {
        premultiplied_alpha: ShaderBool::Disabled,
        ..window
    };
    if renderer.uploaded_font_rasterization != Some(font_rasterization) {
        let resources = renderer.resources();
        resources.queue.write_buffer(
            &resources.font_rasterization_buffer,
            0,
            shader_interface::bytes_of(&font_rasterization),
        );
        renderer.uploaded_font_rasterization = Some(font_rasterization);
    }

    let target_globals = &renderer.resources().target_globals;
    FrameGlobals {
        window,
        window_offset: target_globals.write(&window),
        paths_offset: target_globals.write(&paths),
    }
}

/// The window's target and the path intermediate have globals every frame.
const FRAME_TARGET_GLOBALS: u64 = 2;

#[derive(Clone, Copy, Default)]
pub(super) struct FrameRequirements {
    storage_bytes: u64,
    /// Instance batches this frame; one downlevel range-uniform slot per batch.
    instance_batches: u64,
    pub(super) uniforms: FrameUniformRequirements,
    uses_path_target: bool,
    uses_offscreen_target: bool,
}

impl FrameRequirements {
    pub(super) fn for_scene(scene: &Scene, transport: InstanceTransport) -> Self {
        let planned = scene.render_plan().requirements();
        let mut storage_bytes = 0_u64;
        let mut instance_batches = 0_u64;
        let mut reserve = |element_size: usize, count: usize| {
            if count > 0 {
                let stride = element_size as u64;
                storage_bytes = storage_bytes.next_multiple_of(transport.batch_alignment(stride));
                storage_bytes = storage_bytes.saturating_add(stride.saturating_mul(count as u64));
                instance_batches += 1;
            }
        };

        for command in scene.render_commands() {
            let RenderCommand::Batch(batch) = command else {
                continue;
            };
            match batch {
                PrimitiveBatch::Shadows { range, .. } => {
                    reserve(std::mem::size_of::<Shadow>(), range.len())
                }
                PrimitiveBatch::Quads { range, .. } => {
                    reserve(std::mem::size_of::<Quad>(), range.len())
                }
                PrimitiveBatch::Paths {
                    rasterization_vertex_count,
                    sprite_count,
                    ..
                } if *rasterization_vertex_count > 0 => {
                    reserve(
                        std::mem::size_of::<path_types::PathRasterizationVertex>(),
                        *rasterization_vertex_count,
                    );
                    reserve(std::mem::size_of::<path_types::PathSprite>(), *sprite_count);
                }
                PrimitiveBatch::Underlines(range) => {
                    reserve(std::mem::size_of::<Underline>(), range.len())
                }
                PrimitiveBatch::MonochromeSprites { range, .. } => {
                    reserve(std::mem::size_of::<MonochromeSprite>(), range.len())
                }
                PrimitiveBatch::SubpixelSprites { range, .. } => {
                    reserve(std::mem::size_of::<SubpixelSprite>(), range.len())
                }
                PrimitiveBatch::PolychromeSprites { range, .. } => {
                    reserve(std::mem::size_of::<PolychromeSprite>(), range.len())
                }
                PrimitiveBatch::Paths { .. }
                | PrimitiveBatch::Surfaces(_)
                | PrimitiveBatch::BackdropFilters(_)
                | PrimitiveBatch::Chunks(_)
                | PrimitiveBatch::GroupBoundary(_) => {}
            }
        }
        debug_assert_eq!(instance_batches as usize, planned.instance_batch_count);

        Self {
            storage_bytes,
            instance_batches,
            uniforms: FrameUniformRequirements {
                // A blurred group takes the three blur passes of a backdrop filter, and no
                // blur composite: the group's composite has uniforms of its own.
                filter_count: FILTER_UNIFORMS_PER_COMPOSITE
                    * (planned.backdrop_filter_count + planned.isolated_group_count) as u64
                    + u64::from(planned.uses_offscreen_target),
                surface_count: planned.surface_count as u64,
                group_count: planned.isolated_group_count as u64,
                target_count: FRAME_TARGET_GLOBALS + planned.isolated_group_count as u64,
            },
            uses_path_target: planned.uses_path_target,
            uses_offscreen_target: planned.uses_offscreen_target,
        }
    }
}

/// A texture a frame draws into: the window's, or an isolated group's.
pub(super) struct FrameTarget {
    pub(super) view: wgpu::TextureView,
    /// The texture behind `view`, when it can be copied from: the scene's offscreen
    /// target, or a group's target, which was taken from the pool and goes back to it.
    pub(super) texture: Option<wgpu::Texture>,
    /// The viewport rectangle the texture covers.
    pub(super) bounds: Bounds<DevicePixels>,
    /// Where the globals describing this target are in the target globals.
    pub(super) globals_offset: u32,
    /// A masked group's mask, once drawn: its target, the viewport rectangle that
    /// covers, and how it masks. Kept out of the pool until the group is composited.
    pub(super) mask: Option<(PooledTexture, Bounds<DevicePixels>, MaskMode)>,
}

impl FrameTarget {
    /// An isolated group's target, to go back to the pool or to be sampled as a mask.
    fn into_pooled(self) -> PooledTexture {
        PooledTexture {
            texture: self
                .texture
                .expect("an isolated group's target is taken from the pool"),
            view: self.view,
        }
    }
}

/// A target of its own for a group covering `bounds` of the viewport, taken from the
/// pool, with globals placing it there; `None` if the pool has no room.
fn group_target(
    renderer: &WgpuRenderer,
    window: GlobalUniforms,
    bounds: Bounds<DevicePixels>,
) -> Option<FrameTarget> {
    let pooled = renderer.take_pooled_texture(bounds.size)?;
    let globals = GlobalUniforms {
        target_origin: vec2f(bounds.origin.x.0 as f32, bounds.origin.y.0 as f32),
        target_size: vec2f(bounds.size.width.0 as f32, bounds.size.height.0 as f32),
        ..window
    };
    Some(FrameTarget {
        view: pooled.view,
        texture: Some(pooled.texture),
        bounds,
        globals_offset: renderer.resources().target_globals.write(&globals),
        mask: None,
    })
}

struct FrameEncoder<'a> {
    renderer: &'a WgpuRenderer,
    scene: &'a Scene,
    encoder: wgpu::CommandEncoder,
    /// The window's target, then one for each isolated group being drawn.
    targets: SmallVec<[FrameTarget; 4]>,
    /// Whether each group being drawn has a target of its own.
    isolated: SmallVec<[bool; 8]>,
    offscreen: Option<wgpu::TextureView>,
    presentation: wgpu::TextureView,
    instances: InstanceUpload,
    globals: FrameGlobals,
}

impl<'a> FrameEncoder<'a> {
    fn new(renderer: &'a WgpuRenderer, scene: &'a Scene, targets: PreparedTargets) -> Self {
        let encoder =
            renderer
                .resources()
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gpui_frame"),
                });
        let window = FrameTarget {
            view: targets.active,
            texture: targets.offscreen_texture,
            bounds: Bounds {
                origin: point(DevicePixels(0), DevicePixels(0)),
                size: renderer.target.viewport_size(),
            },
            globals_offset: targets.globals.window_offset,
            mask: None,
        };
        Self {
            renderer,
            scene,
            encoder,
            targets: smallvec::smallvec![window],
            isolated: SmallVec::new(),
            offscreen: targets.offscreen,
            presentation: targets.presentation,
            instances: targets.instances,
            globals: targets.globals,
        }
    }

    /// Encodes the frame. Instances are written while the passes are recorded, so the
    /// downlevel staging-to-texture copy travels in a command buffer of its own, submitted
    /// before the passes that read the texture.
    fn encode(
        mut self,
        readback: Option<ReadbackCopy<'_>>,
    ) -> Result<[wgpu::CommandBuffer; 2], DrawError> {
        let result = self.encode_commands();
        if result.is_ok() {
            if let Some(offscreen) = &self.offscreen {
                self.renderer.blit_to_frame(
                    &mut self.encoder,
                    offscreen,
                    &self.presentation,
                    self.globals.window_offset,
                );
            }
            if let Some(readback) = readback {
                self.encoder.copy_texture_to_buffer(
                    readback.texture.as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: readback.buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(readback.bytes_per_row),
                            rows_per_image: Some(readback.height),
                        },
                    },
                    wgpu::Extent3d {
                        width: readback.width,
                        height: readback.height,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
        let mut uploads = self.renderer.resources().device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("gpui_frame_uploads"),
            },
        );
        self.instances.finish(&mut uploads);
        self.renderer.resources().finish_frame_uploads();
        let command_buffers = [uploads.finish(), self.encoder.finish()];
        result.map(|()| command_buffers)
    }

    fn encode_commands(&mut self) -> DrawResult {
        let mut pass = begin_scene_render_pass(
            self.renderer,
            &mut self.encoder,
            "main_pass",
            self.targets.last().expect("the window's target is first"),
            wgpu::LoadOp::Clear(self.renderer.target.clear_color()),
        );

        // How many of the groups being drawn are hidden: a mask, or a masked group, that
        // got no target is not drawn at all, nor is what it holds.
        let mut hidden = 0usize;

        for command in self.scene.render_commands() {
            if hidden > 0 {
                match command {
                    RenderCommand::BeginGroup { .. } => hidden += 1,
                    RenderCommand::EndGroup { .. } => hidden -= 1,
                    RenderCommand::Batch(_) => {}
                }
                continue;
            }
            match command {
                RenderCommand::Batch(PrimitiveBatch::Paths {
                    range,
                    rasterization_vertex_count,
                    ..
                }) => {
                    if *rasterization_vertex_count == 0 {
                        continue;
                    }
                    let paths = &self.scene.paths[range.clone()];
                    drop(pass);
                    let rasterized = self.renderer.draw_paths_to_intermediate(
                        &mut self.encoder,
                        paths,
                        &mut self.instances,
                        self.globals.paths_offset,
                    );
                    pass = begin_scene_render_pass(
                        self.renderer,
                        &mut self.encoder,
                        "after_paths",
                        self.targets.last().expect("a target is on the stack"),
                        wgpu::LoadOp::Load,
                    );
                    rasterized?;
                    self.renderer.draw_paths_from_intermediate(
                        paths,
                        &mut self.instances,
                        &mut pass,
                    )?;
                }
                RenderCommand::Batch(PrimitiveBatch::BackdropFilters(range)) => {
                    drop(pass);
                    let target = self.targets.last().expect("a target is on the stack");
                    for filter in &self.scene.backdrop_filters[range.clone()] {
                        self.renderer
                            .draw_backdrop_filter(&mut self.encoder, filter, target);
                    }
                    pass = begin_scene_render_pass(
                        self.renderer,
                        &mut self.encoder,
                        "after_backdrop_filter",
                        self.targets.last().expect("a target is on the stack"),
                        wgpu::LoadOp::Load,
                    );
                }
                RenderCommand::Batch(PrimitiveBatch::GroupBoundary(_)) => {
                    unreachable!("group boundaries must be compiled into render commands")
                }
                RenderCommand::Batch(batch) => encode_inline_batch(
                    self.renderer,
                    self.scene,
                    batch,
                    &mut self.instances,
                    &mut pass,
                )?,
                RenderCommand::BeginGroup {
                    boundary_index,
                    target,
                } => {
                    // A group the plan isolates is drawn in place when it is out of view or
                    // the pool has no room for its target.
                    let group = match target {
                        GroupTarget::Isolated { region } => {
                            group_target_bounds(*region, self.renderer.target.viewport_size())
                        }
                        GroupTarget::Inline => None,
                    }
                    .and_then(|bounds| group_target(self.renderer, self.globals.window, bounds));
                    let boundary = &self.scene.group_boundaries[*boundary_index];
                    if group.is_none() && (boundary.masked || boundary.mask_mode.is_some()) {
                        hidden = 1;
                        continue;
                    }
                    self.isolated.push(group.is_some());
                    if let Some(group) = group {
                        drop(pass);
                        self.targets.push(group);
                        pass = begin_scene_render_pass(
                            self.renderer,
                            &mut self.encoder,
                            "group",
                            self.targets
                                .last()
                                .expect("the group's target was just pushed"),
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        );
                    }
                }
                RenderCommand::EndGroup { boundary_index, .. } => {
                    if self.isolated.pop() == Some(true) {
                        drop(pass);
                        let group = self
                            .targets
                            .pop()
                            .expect("an isolated group's target is on the stack");
                        let boundary = &self.scene.group_boundaries[*boundary_index];
                        if let Some(mode) = boundary.mask_mode {
                            // A mask is not composited: the group it masks samples it.
                            let parent = self
                                .targets
                                .last_mut()
                                .expect("a mask's group is under the mask");
                            let bounds = group.bounds;
                            parent.mask = Some((group.into_pooled(), bounds, mode));
                        } else if boundary.masked && group.mask.is_none() {
                            // Its mask is out of view: so is all of it.
                            self.renderer.give_back_pooled_texture(group.into_pooled());
                        } else {
                            let parent = self
                                .targets
                                .last()
                                .expect("the window's target is under every group's");
                            self.renderer.composite_group(
                                &mut self.encoder,
                                boundary,
                                group,
                                parent,
                            );
                        }
                        pass = begin_scene_render_pass(
                            self.renderer,
                            &mut self.encoder,
                            "after_group",
                            self.targets.last().expect("a target is on the stack"),
                            wgpu::LoadOp::Load,
                        );
                    }
                }
            }
        }
        drop(pass);
        assert!(
            self.targets.len() == 1 && self.isolated.is_empty(),
            "render plan left a group open"
        );
        Ok(())
    }
}

fn begin_scene_render_pass<'a>(
    renderer: &'a WgpuRenderer,
    encoder: &'a mut wgpu::CommandEncoder,
    label: &'a str,
    target: &'a FrameTarget,
    load: wgpu::LoadOp<wgpu::Color>,
) -> wgpu::RenderPass<'a> {
    let mut pass = begin_color_render_pass(encoder, label, &target.view, load);
    pass.set_bind_group(
        shader_interface::GLOBAL_BIND_GROUP,
        &renderer.resources().globals_bind_group,
        &[target.globals_offset],
    );
    pass
}

#[derive(Debug)]
pub(super) enum DrawError {
    CapacityPlanningInvariant,
    ExternalSurface,
    MissingIntermediateTarget,
}

pub(super) type DrawResult = Result<(), DrawError>;

fn encode_inline_batch(
    renderer: &WgpuRenderer,
    scene: &Scene,
    batch: &PrimitiveBatch,
    instances: &mut InstanceUpload,
    pass: &mut wgpu::RenderPass<'_>,
) -> DrawResult {
    match batch {
        PrimitiveBatch::Quads { range, smoothed } => {
            renderer.draw_quads(&scene.quads[range.clone()], *smoothed, instances, pass)
        }
        PrimitiveBatch::Shadows { range, smoothed } => {
            renderer.draw_shadows(&scene.shadows[range.clone()], *smoothed, instances, pass)
        }
        PrimitiveBatch::Underlines(range) => {
            renderer.draw_underlines(&scene.underlines[range.clone()], instances, pass)
        }
        PrimitiveBatch::MonochromeSprites { texture_id, range } => renderer
            .draw_monochrome_sprites(
                &scene.monochrome_sprites[range.clone()],
                *texture_id,
                instances,
                pass,
            ),
        PrimitiveBatch::SubpixelSprites { texture_id, range } => renderer.draw_subpixel_sprites(
            &scene.subpixel_sprites[range.clone()],
            *texture_id,
            instances,
            pass,
        ),
        PrimitiveBatch::PolychromeSprites {
            texture_id,
            range,
            smoothed,
        } => renderer.draw_polychrome_sprites(
            &scene.polychrome_sprites[range.clone()],
            *texture_id,
            *smoothed,
            instances,
            pass,
        ),
        PrimitiveBatch::Surfaces(range) => renderer.draw_surfaces(
            &scene.surfaces[range.clone()],
            &scene.surface_opacities()[range.clone()],
            pass,
        ),
        PrimitiveBatch::Paths { .. }
        | PrimitiveBatch::BackdropFilters(_)
        | PrimitiveBatch::GroupBoundary(_) => {
            unreachable!("pass-interrupting batches are handled by FrameEncoder")
        }
        // Not made for this renderer: `supports_scene_chunks` is false.
        PrimitiveBatch::Chunks(_) => {
            debug_assert!(false, "a scene chunk reached the wgpu renderer");
            Ok(())
        }
    }
}
