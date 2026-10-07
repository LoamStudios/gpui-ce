use gpui::{
    BackdropFilter, BlendMode, Bounds, DevicePixels, FilterImage, FilterPass, FilterPlan,
    GroupBoundary, ScaledPixels, Size,
};
use gpui_render::shaders::interface as shader_interface;
use gpui_render::{
    blur::{
        BlurAxis, BlurKernel, BlurTint, FilterCompositeClip, GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS,
        GroupBlur, ScissorRectangle, downsampled_dimension,
    },
    group::GroupUniforms,
    shaders::blur::BlurUniforms,
};

use super::{
    WgpuRenderer, begin_color_render_pass, frame::FrameTarget, pipelines,
    target_pool::PooledTexture,
};

pub(super) const FILTER_UNIFORMS_PER_COMPOSITE: u64 = 4;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct FrameUniformRequirements {
    pub(super) filter_count: u64,
    pub(super) surface_count: u64,
    /// One composite per isolated group.
    pub(super) group_count: u64,
    /// One set of globals per target drawn into: the window, the path intermediate, and
    /// each isolated group.
    pub(super) target_count: u64,
}

const _: () = assert!(std::mem::size_of::<BlurUniforms>() == 144);
const _: () = assert!(std::mem::size_of::<GroupUniforms>() == 224);

impl WgpuRenderer {
    fn make_blur_bind_group(
        &self,
        uniforms: BlurUniforms,
        source: &wgpu::TextureView,
    ) -> (wgpu::BindGroup, u32) {
        let resources = self.resources();
        let uniform_offset = resources.filter_uniforms.write(&uniforms);
        (resources.blur_bind_group(source), uniform_offset)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_blur_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &pipelines::WgpuRenderPipeline,
        target: &wgpu::TextureView,
        source: &wgpu::TextureView,
        globals_offset: u32,
        uniforms: BlurUniforms,
        scissor: ScissorRectangle,
    ) {
        let (bind_group, uniform_offset) = self.make_blur_bind_group(uniforms, source);
        let resources = self.resources();
        let mut pass = begin_color_render_pass(
            encoder,
            label,
            target,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        pass.set_pipeline(pipeline);
        pass.set_bind_group(
            shader_interface::GLOBAL_BIND_GROUP,
            &resources.globals_bind_group,
            &[globals_offset],
        );
        pass.set_bind_group(
            shader_interface::DATA_BIND_GROUP,
            &bind_group,
            &[uniform_offset],
        );
        pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
        pass.draw(0..pipeline.fixed_vertex_count(), 0..1);
    }

    /// A texture of `size` from the target pool, or `None` if the pool has no room.
    pub(super) fn take_pooled_texture(&self, size: Size<DevicePixels>) -> Option<PooledTexture> {
        let resources = self.resources();
        resources
            .target_pool
            .borrow_mut()
            .take(&resources.device, self.target.format(), size)
    }

    pub(super) fn give_back_pooled_texture(&self, texture: PooledTexture) {
        self.resources().target_pool.borrow_mut().give_back(texture);
    }

    /// Blurs `source`, within `scissor` of its half-resolution copy, into two textures
    /// taken from the pool: the first holds the result, the second is spare. Returns them
    /// with their size, or `None` if the pool has no room.
    fn blur(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &FrameTarget,
        kernel: BlurKernel,
        scissor: ScissorRectangle,
    ) -> Option<(PooledTexture, PooledTexture, [f32; 2])> {
        self.blur_view(
            encoder,
            &source.view,
            source.bounds.size,
            source.globals_offset,
            kernel,
            scissor,
        )
    }

    /// Blurs what `scissor` covers of the half-resolution copy of `source`, a texture of
    /// `size`, into two textures taken from the pool: the first holds the result, the second
    /// is spare. Returns them with the result's size, or `None` if the pool has no room.
    fn blur_view(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        size: Size<DevicePixels>,
        globals_offset: u32,
        kernel: BlurKernel,
        scissor: ScissorRectangle,
    ) -> Option<(PooledTexture, PooledTexture, [f32; 2])> {
        let full_width = size.width.0.max(0) as u32;
        let full_height = size.height.0.max(0) as u32;
        let (blur_width, blur_height) = (
            downsampled_dimension(full_width),
            downsampled_dimension(full_height),
        );
        let blur_size = [blur_width as f32, blur_height as f32];
        let blur_texture_size = Size {
            width: DevicePixels(blur_width as i32),
            height: DevicePixels(blur_height as i32),
        };
        let ping = self.take_pooled_texture(blur_texture_size)?;
        let Some(pong) = self.take_pooled_texture(blur_texture_size) else {
            self.give_back_pooled_texture(ping);
            return None;
        };

        // Downsample source -> ping, then separable gaussian ping -> pong -> ping.
        self.run_blur_pass(
            encoder,
            "blur_downsample",
            &self.resources().pipelines.blur_downsample,
            &ping.view,
            source,
            globals_offset,
            BlurUniforms::downsample([full_width as f32, full_height as f32], blur_size),
            scissor,
        );
        self.run_blur_pass(
            encoder,
            "blur_horizontal",
            &self.resources().pipelines.blur,
            &pong.view,
            &ping.view,
            globals_offset,
            BlurUniforms::gaussian(BlurAxis::Horizontal, blur_size, kernel),
            scissor,
        );
        self.run_blur_pass(
            encoder,
            "blur_vertical",
            &self.resources().pipelines.blur,
            &ping.view,
            &pong.view,
            globals_offset,
            BlurUniforms::gaussian(BlurAxis::Vertical, blur_size, kernel),
            scissor,
        );
        Some((ping, pong, blur_size))
    }

    /// Blurs all of `source`, a group's picture of `size`, as `blur` says, reading it
    /// through `tint` if given: downsampled first, unless it runs at full resolution, then
    /// separably. Returns the result, in a texture from the pool, or `None` if the pool has
    /// no room.
    fn group_blur(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        size: Size<DevicePixels>,
        globals_offset: u32,
        blur: GroupBlur,
        tint: Option<BlurTint>,
    ) -> Option<PooledTexture> {
        let passes = blur.passes(
            [size.width.0.max(0) as u32, size.height.0.max(0) as u32],
            tint,
        );
        let blurred_size = Size {
            width: DevicePixels(passes.size[0] as i32),
            height: DevicePixels(passes.size[1] as i32),
        };
        let scissor = ScissorRectangle {
            x: 0,
            y: 0,
            width: passes.size[0],
            height: passes.size[1],
        };
        let ping = self.take_pooled_texture(blurred_size)?;
        let Some(pong) = self.take_pooled_texture(blurred_size) else {
            self.give_back_pooled_texture(ping);
            return None;
        };
        // [downsample source -> pong], horizontal -> ping, vertical -> pong.
        let mut input = source;
        if let Some(downsample) = passes.downsample {
            self.run_blur_pass(
                encoder,
                "blur_downsample",
                &self.resources().pipelines.blur_downsample,
                &pong.view,
                input,
                globals_offset,
                downsample,
                scissor,
            );
            input = &pong.view;
        }
        self.run_blur_pass(
            encoder,
            "blur_horizontal",
            &self.resources().pipelines.blur,
            &ping.view,
            input,
            globals_offset,
            passes.horizontal,
            scissor,
        );
        self.run_blur_pass(
            encoder,
            "blur_vertical",
            &self.resources().pipelines.blur,
            &pong.view,
            &ping.view,
            globals_offset,
            passes.vertical,
            scissor,
        );
        self.give_back_pooled_texture(ping);
        Some(pong)
    }

    /// Runs `plan`, a group's filters, on `group`, an isolated group's finished target:
    /// each pass reads pictures covering the target and writes one, in a texture from the
    /// pool. Returns the picture to composite, the one its composite merges it over if it
    /// does, and the textures to give back once it is composited.
    fn filter_group(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &FilterPlan,
        group: &FrameTarget,
    ) -> (
        wgpu::TextureView,
        Option<wgpu::TextureView>,
        Vec<PooledTexture>,
    ) {
        let last_reads = plan.last_reads();
        // Each pass's picture, and its texture if it was taken from the pool for it and
        // not yet given back.
        let mut pictures: Vec<(wgpu::TextureView, Option<PooledTexture>)> = Vec::new();
        let mut spare = Vec::new();
        let picture = |pictures: &[(wgpu::TextureView, Option<PooledTexture>)],
                       image: FilterImage| match image {
            FilterImage::Content => group.view.clone(),
            FilterImage::Pass(pass) => pictures[pass].0.clone(),
        };
        for (index, pass) in plan.passes.iter().enumerate() {
            let result = match *pass {
                FilterPass::Blur {
                    input,
                    std_deviation,
                    tint,
                } => {
                    let input = picture(&pictures, input);
                    let tint = tint.map(|tint| BlurTint {
                        color: tint.color,
                        offset: [tint.offset.x.0, tint.offset.y.0],
                    });
                    match GroupBlur::new(std_deviation).and_then(|blur| {
                        self.group_blur(
                            encoder,
                            &input,
                            group.bounds.size,
                            group.globals_offset,
                            blur,
                            tint,
                        )
                    }) {
                        Some(blurred) => (blurred.view.clone(), Some(blurred)),
                        None => (input, None),
                    }
                }
                FilterPass::ColorMatrix {
                    input,
                    ref matrix,
                    offset,
                } => {
                    let input = picture(&pictures, input);
                    let uniforms = GroupUniforms::color_matrix(group.bounds, matrix, offset);
                    self.run_group_filter_pass(encoder, group, &input, &input, uniforms, false)
                }
                FilterPass::Merge { top, bottom } => {
                    let (top, bottom) = (picture(&pictures, top), picture(&pictures, bottom));
                    let uniforms = GroupUniforms::merge(group.bounds);
                    self.run_group_filter_pass(encoder, group, &top, &bottom, uniforms, false)
                }
                FilterPass::Program {
                    input,
                    paint,
                    program,
                    ref to_viewport,
                } => {
                    let input = picture(&pictures, input);
                    // Until its program is linked, the pass leaves the picture as it is.
                    let resources = self.resources();
                    let linked = resources.frame_programs.is_some()
                        && resources.programs.linked().contains(&program);
                    if linked {
                        let uniforms = GroupUniforms::program(group.bounds, paint, to_viewport);
                        self.run_group_filter_pass(encoder, group, &input, &input, uniforms, true)
                    } else {
                        (input, None)
                    }
                }
            };
            pictures.push(result);
            // Let go of the pictures no later pass reads, nor the composite.
            for read in FilterPlan::inputs(pass) {
                if let FilterImage::Pass(read) = read
                    && last_reads[read] == Some(index)
                    && Some(read) != plan.output
                    && plan.composite_inputs() != Some(FilterImage::Pass(read))
                    && let Some(texture) = pictures[read].1.take()
                {
                    spare.push(texture);
                }
            }
        }
        let output = match plan.output {
            Some(output) => pictures[output].0.clone(),
            None => group.view.clone(),
        };
        let beneath = plan
            .composite_inputs()
            .map(|image| picture(&pictures, image));
        spare.extend(pictures.into_iter().filter_map(|(_, texture)| texture));
        (output, beneath, spare)
    }

    /// Runs one pass of a group's filters, other than a blur, into a texture from the pool
    /// covering the group's target, reading `source` and, for a merge, `beneath`. Returns
    /// its picture, or `source` if the pool has no room.
    fn run_group_filter_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        group: &FrameTarget,
        source: &wgpu::TextureView,
        beneath: &wgpu::TextureView,
        uniforms: GroupUniforms,
        linked: bool,
    ) -> (wgpu::TextureView, Option<PooledTexture>) {
        let Some(output) = self.take_pooled_texture(group.bounds.size) else {
            return (source.clone(), None);
        };
        {
            let resources = self.resources();
            let uniform_offset = resources.group_uniforms.write(&uniforms);
            let bind_group = resources.group_bind_group(source, beneath, source);
            let pipeline = match (&resources.frame_programs, linked) {
                (Some(programs), true) => &programs.group_filter,
                _ => &resources.pipelines.group_filter,
            };
            let mut pass = begin_color_render_pass(
                encoder,
                "group_filter",
                &output.view,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            pass.set_pipeline(pipeline);
            pass.set_bind_group(
                shader_interface::GLOBAL_BIND_GROUP,
                &resources.globals_bind_group,
                &[group.globals_offset],
            );
            pass.set_bind_group(
                shader_interface::DATA_BIND_GROUP,
                &bind_group,
                &[uniform_offset],
            );
            pass.draw(0..pipeline.fixed_vertex_count(), 0..1);
        }
        (output.view.clone(), Some(output))
    }

    /// Blurs what is in `target` under the filter's bounds, the backdrop of a backdrop
    /// filter, and composites the result back into `target`, clipped to its rounded
    /// bounds and content mask, and faded by its opacity. `target` may be an isolated
    /// group's, placed anywhere in the viewport.
    pub(super) fn draw_backdrop_filter(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        filter: &BackdropFilter,
        target: &FrameTarget,
    ) {
        let blur_radius = filter.max_blur_radius();
        let Some(kernel) = BlurKernel::for_radius(blur_radius) else {
            return;
        };
        let full_width = target.bounds.size.width.0.max(0) as u32;
        let full_height = target.bounds.size.height.0.max(0) as u32;
        let dilation = GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * blur_radius;
        let origin = target
            .bounds
            .origin
            .map(|value| ScaledPixels(value.0 as f32));
        let scissor = ScissorRectangle::for_blurred_bounds(
            Bounds {
                origin: filter.bounds.origin - origin,
                size: filter.bounds.size,
            },
            dilation,
            full_width,
            full_height,
        );
        if scissor.is_empty() {
            return;
        }
        let Some((ping, pong, blur_size)) = self.blur(encoder, target, kernel, scissor) else {
            return;
        };

        let uniforms = BlurUniforms::composite(
            filter.bounds,
            filter.content_mask.bounds,
            filter.corner_radii,
            filter.corner_smoothing,
            filter.opacity,
            FilterCompositeClip::RoundedBounds,
            blur_size,
            [full_width as f32, full_height as f32],
            [origin.x.0, origin.y.0],
        );
        let (bind_group, uniform_offset) = self.make_blur_bind_group(uniforms, &ping.view);
        {
            let resources = self.resources();
            let pipeline = if uniforms.corner_smoothing > 0.0 {
                &resources.pipelines.smoothed_blur_composite
            } else {
                &resources.pipelines.blur_composite
            };
            let mut pass = begin_color_render_pass(
                encoder,
                "blur_composite",
                &target.view,
                wgpu::LoadOp::Load,
            );
            pass.set_pipeline(pipeline);
            pass.set_bind_group(
                shader_interface::GLOBAL_BIND_GROUP,
                &resources.globals_bind_group,
                &[target.globals_offset],
            );
            pass.set_bind_group(
                shader_interface::DATA_BIND_GROUP,
                &bind_group,
                &[uniform_offset],
            );
            pass.draw(0..pipeline.fixed_vertex_count(), 0..1);
        }
        self.give_back_pooled_texture(ping);
        self.give_back_pooled_texture(pong);
    }

    /// Composites `group`, an isolated group's finished target, into `parent`: through
    /// its filters, faded by its opacity, cut to its mask, and mixed by its blend mode
    /// with a copy of what is beneath it in `parent`. Gives the group's textures, and its
    /// mask's, back to the pool.
    pub(super) fn composite_group(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        boundary: &GroupBoundary,
        group: FrameTarget,
        parent: &FrameTarget,
    ) {
        let plan = boundary.filter_plan();
        let (filtered, beneath, spare) = self.filter_group(encoder, &plan, &group);
        let source = &filtered;

        let backdrop = if boundary.blend_mode == BlendMode::Normal {
            None
        } else {
            self.copy_backdrop(encoder, parent, group.bounds)
        };
        let uniforms = GroupUniforms::composite(
            group.bounds,
            boundary.content_mask.bounds,
            boundary.opacity,
            boundary.blend_mode,
            backdrop.as_ref().map(|_| group.bounds),
            group
                .mask
                .as_ref()
                .map(|(_, bounds, mode)| (*bounds, *mode)),
        )
        .with_composite_filter(&plan.composite, group.bounds);
        {
            let resources = self.resources();
            let uniform_offset = resources.group_uniforms.write(&uniforms);
            // The shader reads the backdrop slot for a blend mode other than normal, which
            // puts the parent's copy there, or for a merge, which puts what the picture is
            // merged over there; and the mask only for a masked group. The bindings must be
            // filled either way.
            let bind_group = resources.group_bind_group(
                source,
                backdrop
                    .as_ref()
                    .map(|backdrop| &backdrop.view)
                    .or(beneath.as_ref())
                    .unwrap_or(source),
                group
                    .mask
                    .as_ref()
                    .map_or(source, |(mask, _, _)| &mask.view),
            );
            let pipeline = &resources.pipelines.group_composite;
            let mut pass = begin_color_render_pass(
                encoder,
                "group_composite",
                &parent.view,
                wgpu::LoadOp::Load,
            );
            pass.set_pipeline(pipeline);
            pass.set_bind_group(
                shader_interface::GLOBAL_BIND_GROUP,
                &resources.globals_bind_group,
                &[parent.globals_offset],
            );
            pass.set_bind_group(
                shader_interface::DATA_BIND_GROUP,
                &bind_group,
                &[uniform_offset],
            );
            pass.draw(0..pipeline.fixed_vertex_count(), 0..1);
        }

        for texture in spare {
            self.give_back_pooled_texture(texture);
        }
        if let Some(backdrop) = backdrop {
            self.give_back_pooled_texture(backdrop);
        }
        if let Some((mask, _, _)) = group.mask {
            self.give_back_pooled_texture(mask);
        }
        if let Some(texture) = group.texture {
            self.give_back_pooled_texture(PooledTexture {
                texture,
                view: group.view,
            });
        }
    }

    /// Copies what `parent` holds under `bounds` into a texture taken from the pool, which
    /// covers `bounds`: what a blend mode mixes a group with. `None` when `parent` cannot
    /// be copied from, holds nothing under `bounds`, or the pool has no room.
    fn copy_backdrop(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        parent: &FrameTarget,
        bounds: Bounds<DevicePixels>,
    ) -> Option<PooledTexture> {
        let parent_texture = parent.texture.as_ref()?;
        let copied = bounds.intersect(&parent.bounds);
        if copied.is_empty() {
            return None;
        }
        let backdrop = self.take_pooled_texture(bounds.size)?;
        let source_origin = copied.origin - parent.bounds.origin;
        let destination_origin = copied.origin - bounds.origin;
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: parent_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: source_origin.x.0 as u32,
                    y: source_origin.y.0 as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &backdrop.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: destination_origin.x.0 as u32,
                    y: destination_origin.y.0 as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: copied.size.width.0 as u32,
                height: copied.size.height.0 as u32,
                depth_or_array_layers: 1,
            },
        );
        Some(backdrop)
    }

    pub(super) fn blit_to_frame(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        frame_view: &wgpu::TextureView,
        globals_offset: u32,
    ) {
        let size = [self.target.width() as f32, self.target.height() as f32];
        let (bind_group, uniform_offset) =
            self.make_blur_bind_group(BlurUniforms::copy(size), source);
        let resources = self.resources();
        let mut pass = begin_color_render_pass(
            encoder,
            "scene_blit",
            frame_view,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        pass.set_pipeline(&resources.pipelines.blur_downsample);
        pass.set_bind_group(
            shader_interface::GLOBAL_BIND_GROUP,
            &resources.globals_bind_group,
            &[globals_offset],
        );
        pass.set_bind_group(
            shader_interface::DATA_BIND_GROUP,
            &bind_group,
            &[uniform_offset],
        );
        pass.draw(
            0..resources.pipelines.blur_downsample.fixed_vertex_count(),
            0..1,
        );
    }
}
