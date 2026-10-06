use gpui::{BackdropFilter, BlendMode, Bounds, DevicePixels, GroupBoundary, ScaledPixels, Size};
use gpui_render::shaders::interface as shader_interface;
use gpui_render::{
    blur::{
        BlurAxis, BlurKernel, FilterCompositeClip, GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS,
        ScissorRectangle, downsampled_dimension,
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

const _: () = assert!(std::mem::size_of::<BlurUniforms>() == 112);
const _: () = assert!(std::mem::size_of::<GroupUniforms>() == 112);

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
        let full_width = source.bounds.size.width.0.max(0) as u32;
        let full_height = source.bounds.size.height.0.max(0) as u32;
        let blur_width = downsampled_dimension(full_width);
        let blur_height = downsampled_dimension(full_height);
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
            &source.view,
            source.globals_offset,
            BlurUniforms::downsample([full_width as f32, full_height as f32], blur_size),
            scissor,
        );
        self.run_blur_pass(
            encoder,
            "blur_horizontal",
            &self.resources().pipelines.blur,
            &pong.view,
            &ping.view,
            source.globals_offset,
            BlurUniforms::gaussian(BlurAxis::Horizontal, blur_size, kernel),
            scissor,
        );
        self.run_blur_pass(
            encoder,
            "blur_vertical",
            &self.resources().pipelines.blur,
            &ping.view,
            &pong.view,
            source.globals_offset,
            BlurUniforms::gaussian(BlurAxis::Vertical, blur_size, kernel),
            scissor,
        );
        Some((ping, pong, blur_size))
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

    /// Composites `group`, an isolated group's finished target, into `parent`: blurred by
    /// its filters, faded by its opacity, and mixed by its blend mode with a copy of what
    /// is beneath it in `parent`. Gives the group's textures back to the pool.
    pub(super) fn composite_group(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        boundary: &GroupBoundary,
        group: FrameTarget,
        parent: &FrameTarget,
    ) {
        let blurred = BlurKernel::for_radius(boundary.max_blur_radius()).and_then(|kernel| {
            let size = group.bounds.size;
            // The whole target is blurred: the render plan sized it to hold what the
            // group draws, spread by its filters.
            let scissor = ScissorRectangle {
                x: 0,
                y: 0,
                width: downsampled_dimension(size.width.0.max(0) as u32),
                height: downsampled_dimension(size.height.0.max(0) as u32),
            };
            self.blur(encoder, &group, kernel, scissor)
        });
        let source = blurred
            .as_ref()
            .map_or(&group.view, |(ping, _, _)| &ping.view);

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
        );
        {
            let resources = self.resources();
            let uniform_offset = resources.group_uniforms.write(&uniforms);
            // The shader reads the backdrop only for blend modes other than normal; the
            // binding must be filled either way.
            let bind_group = resources.group_bind_group(
                source,
                backdrop.as_ref().map_or(source, |backdrop| &backdrop.view),
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

        if let Some((ping, pong, _)) = blurred {
            self.give_back_pooled_texture(ping);
            self.give_back_pooled_texture(pong);
        }
        if let Some(backdrop) = backdrop {
            self.give_back_pooled_texture(backdrop);
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
