use gpui::{
    AtlasTextureId, Bounds, MonochromeSprite, Path, PolychromeSprite, Quad, ScaledPixels, Shadow,
    SubpixelSprite, TransformationMatrix, Underline,
};

use crate::WgpuTextureInfo;
use gpui_render::shaders::interface::{self as shader_interface, BufferData};

use super::{
    WgpuRenderer,
    buffers::{InstanceSlice, InstanceUpload},
    frame, path_types, pipelines,
};

impl WgpuRenderer {
    pub(super) fn draw_quads(
        &self,
        quads: &[Quad],
        smoothed: bool,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let resources = self.resources();
        let pipeline = match (&resources.frame_programs, smoothed) {
            (Some(programs), true) => &programs.smoothed_quads,
            (Some(programs), false) => &programs.quads,
            (None, true) => &resources.pipelines.smoothed_quads,
            (None, false) => &resources.pipelines.quads,
        };
        self.draw_instances(quads, pipeline, instances, pass)
    }

    pub(super) fn draw_shadows(
        &self,
        shadows: &[Shadow],
        smoothed: bool,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let resources = self.resources();
        let pipeline = match (&resources.frame_programs, smoothed) {
            (Some(programs), true) => &programs.smoothed_shadows,
            (Some(programs), false) => &programs.shadows,
            (None, true) => &resources.pipelines.smoothed_shadows,
            (None, false) => &resources.pipelines.shadows,
        };
        self.draw_instances(shadows, pipeline, instances, pass)
    }

    pub(super) fn draw_underlines(
        &self,
        underlines: &[Underline],
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        self.draw_instances(
            underlines,
            &self.resources().pipelines.underlines,
            instances,
            pass,
        )
    }

    pub(super) fn draw_monochrome_sprites(
        &self,
        sprites: &[MonochromeSprite],
        texture_id: AtlasTextureId,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let texture = self.atlas.get_texture_info(texture_id);
        let resources = self.resources();
        let pipeline = match &resources.frame_programs {
            Some(programs) => &programs.monochrome_sprites,
            None => &resources.pipelines.monochrome_sprites,
        };
        self.draw_instances_with_texture(sprites, texture_id, &texture, pipeline, instances, pass)
    }

    pub(super) fn draw_subpixel_sprites(
        &self,
        sprites: &[SubpixelSprite],
        texture_id: AtlasTextureId,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let texture = self.atlas.get_texture_info(texture_id);
        let resources = self.resources();
        let pipeline = resources
            .pipelines
            .subpixel_sprites
            .as_ref()
            .unwrap_or(&resources.pipelines.monochrome_sprites);
        self.draw_instances_with_texture(sprites, texture_id, &texture, pipeline, instances, pass)
    }

    pub(super) fn draw_polychrome_sprites(
        &self,
        sprites: &[PolychromeSprite],
        texture_id: AtlasTextureId,
        smoothed: bool,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let texture = self.atlas.get_texture_info(texture_id);
        let pipelines = &self.resources().pipelines;
        let pipeline = if smoothed {
            &pipelines.smoothed_polychrome_sprites
        } else {
            &pipelines.polychrome_sprites
        };
        self.draw_instances_with_texture(sprites, texture_id, &texture, pipeline, instances, pass)
    }

    fn draw_instances<T: BufferData>(
        &self,
        values: &[T],
        pipeline: &pipelines::WgpuRenderPipeline,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        if values.is_empty() {
            return Ok(());
        }
        let resources = self.resources();
        self.draw_bound_instances(
            values,
            pipeline,
            resources.instances.bind_group(),
            instances,
            pass,
        )
    }

    fn draw_bound_instances<T: BufferData>(
        &self,
        values: &[T],
        pipeline: &pipelines::WgpuRenderPipeline,
        bind_group: &wgpu::BindGroup,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let Some(slice) = instances.write(values) else {
            return Err(frame::DrawError::CapacityPlanningInvariant);
        };
        self.draw_bound_slice(pipeline, bind_group, &slice, pass);
        Ok(())
    }

    fn draw_bound_slice<T>(
        &self,
        pipeline: &pipelines::WgpuRenderPipeline,
        bind_group: &wgpu::BindGroup,
        slice: &InstanceSlice<T>,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        pass.set_pipeline(pipeline);
        slice.set_data_bind_group(pass, bind_group);
        pass.draw(0..pipeline.fixed_vertex_count(), slice.range());
    }

    fn draw_instances_with_texture<T: BufferData>(
        &self,
        values: &[T],
        texture_id: AtlasTextureId,
        texture: &WgpuTextureInfo,
        pipeline: &pipelines::WgpuRenderPipeline,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        if values.is_empty() {
            return Ok(());
        }
        let resources = self.resources();
        let bind_group = resources.instances.textured_bind_group(
            &resources.device,
            &resources.bind_group_layouts,
            pipeline.data_layout(),
            texture_id,
            texture.identity,
            &texture.view,
            &resources.atlas_sampler,
        );
        self.draw_bound_instances(values, pipeline, &bind_group, instances, pass)
    }

    /// Copies `paths` from the intermediate they were rasterized into. Those of a chunk,
    /// rasterized where `placement` puts them, are copied from there, which the pass's
    /// globals must not place again.
    pub(super) fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        placement: Option<&TransformationMatrix>,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let sprite_count = path_types::sprite_count(paths);
        let sprite_slice = match placement {
            None => instances.write_iter(sprite_count, path_types::sprites(paths)),
            // Each pixel must be copied once, for transparent paths: placed sprites that
            // may overlap, as turned ones can, are copied as one.
            Some(placement) if !placement.is_axis_aligned() && sprite_count > 1 => {
                let bounds = path_types::sprites(paths)
                    .map(|sprite| placed_sprite_bounds(placement, sprite))
                    .reduce(|bounds, other| bounds.union(&other));
                instances.write_iter(1, bounds.map(|bounds| path_types::PathSprite { bounds }))
            }
            Some(placement) => instances.write_iter(
                sprite_count,
                path_types::sprites(paths).map(|sprite| path_types::PathSprite {
                    bounds: placed_sprite_bounds(placement, sprite),
                }),
            ),
        };
        let Some(sprite_slice) = sprite_slice else {
            return Err(frame::DrawError::CapacityPlanningInvariant);
        };
        let resources = self.resources();
        let Some(path_intermediate_view) = resources.path_intermediate_view.as_ref() else {
            return Err(frame::DrawError::MissingIntermediateTarget);
        };
        let bind_group = resources.instances.path_bind_group(
            &resources.device,
            &resources.bind_group_layouts,
            resources.pipelines.paths.data_layout(),
            path_intermediate_view,
            &resources.atlas_sampler,
        );
        self.draw_bound_slice(&resources.pipelines.paths, &bind_group, &sprite_slice, pass);
        Ok(())
    }

    /// Rasterizes `paths` into the viewport-sized intermediate texture, drawn with
    /// `globals`, the group-0 bind group of the scene they are in, at `globals_offset`,
    /// globals which describe that texture.
    pub(super) fn draw_paths_to_intermediate(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        paths: &[Path<ScaledPixels>],
        instances: &mut InstanceUpload,
        globals: &wgpu::BindGroup,
        globals_offset: u32,
    ) -> frame::DrawResult {
        let vertex_count = path_types::rasterization_vertex_count(paths);
        if vertex_count == 0 {
            return Ok(());
        }

        let Some(vertex_slice) =
            instances.write_iter(vertex_count, path_types::rasterization_vertices(paths))
        else {
            return Err(frame::DrawError::CapacityPlanningInvariant);
        };
        let resources = self.resources();
        let Some(path_intermediate_view) = resources.path_intermediate_view.as_ref() else {
            return Err(frame::DrawError::MissingIntermediateTarget);
        };
        let (target_view, resolve_target) = if let Some(msaa_view) = &resources.path_msaa_view {
            (msaa_view, Some(path_intermediate_view))
        } else {
            (path_intermediate_view, None)
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("path_rasterization_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    // Only the resolved texture is sampled later. The MSAA
                    // attachment is transient and cannot use StoreOp::Store.
                    store: if resolve_target.is_some() {
                        wgpu::StoreOp::Discard
                    } else {
                        wgpu::StoreOp::Store
                    },
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(match &resources.frame_programs {
            Some(programs) => &programs.path_rasterization,
            None => &resources.pipelines.path_rasterization,
        });
        pass.set_bind_group(
            shader_interface::GLOBAL_BIND_GROUP,
            globals,
            &[globals_offset],
        );
        vertex_slice.set_data_bind_group(&mut pass, resources.instances.bind_group());
        pass.draw(vertex_slice.range(), 0..1);
        Ok(())
    }
}

/// The viewport bounds a path sprite of a chunk covers, placed by `placement`.
fn placed_sprite_bounds(
    placement: &TransformationMatrix,
    sprite: path_types::PathSprite,
) -> Bounds<ScaledPixels> {
    frame::transformed_bounds(placement, sprite.bounds)
}
