#[wgsl_rs::wgsl]
pub mod surface {
    use super::super::common::*;
    use wgsl_rs::std::*;

    #[repr(C)]
    #[derive(Clone, Copy, Wgsl)]
    pub struct SurfaceUniforms {
        pub bounds: Bounds,
        pub content_mask: ContentMask,
        pub color_format: SurfaceColorFormat,
        pub opacity: f32,
        pub padding0: u32,
        pub padding1: u32,
        pub padding2: u32,
        pub padding3: u32,
        pub padding4: u32,
        pub padding5: u32,
    }
    uniform!(group(1), binding(0), SURFACE_LOCALS: SurfaceUniforms);
    texture!(group(1), binding(1), SURFACE_TEXTURE: Texture2D<f32>);
    texture!(group(1), binding(2), SURFACE_CHROMA_TEXTURE: Texture2D<f32>);
    sampler!(group(1), binding(3), SURFACE_SAMPLER: Sampler);

    pub const YCBCR_TO_LINEAR_RGB: Mat4x4f = mat4x4f(
        vec4f(1.0000, 1.0000, 1.0000, 0.0),
        vec4f(0.0000, -0.3441, 1.7720, 0.0),
        vec4f(1.4020, -0.7141, 0.0000, 0.0),
        vec4f(-0.7010, 0.5291, -0.8860, 1.0),
    );

    pub fn sample_yuv_surface(texture_position: Vec2f) -> Vec4f {
        let luma = texture_sample_level(SURFACE_TEXTURE, SURFACE_SAMPLER, texture_position, 0.0).x;
        let chroma = texture_sample_level(
            SURFACE_CHROMA_TEXTURE,
            SURFACE_SAMPLER,
            texture_position,
            0.0,
        )
        .xy();
        YCBCR_TO_LINEAR_RGB * vec4f(luma, chroma.x, chroma.y, 1.0)
    }

    #[derive(Wgsl)]
    pub struct SurfaceVarying {
        #[builtin(position)]
        pub position: Vec4f,
        #[location(0)]
        pub texture_position: Vec2f,
        #[location(3)]
        pub clip_distances: Vec4f,
    }

    #[vertex]
    pub fn vertex_surface(#[builtin(vertex_index)] vertex_id: u32) -> SurfaceVarying {
        let vertex = rectangle_vertex(vertex_id, get!(SURFACE_LOCALS).bounds);
        SurfaceVarying {
            position: vertex.clip_position,
            texture_position: vertex.unit_position,
            clip_distances: clip_distances(
                vertex.viewport_position,
                get!(SURFACE_LOCALS).content_mask.bounds,
            ),
        }
    }

    #[fragment]
    pub fn fragment_surface(input: SurfaceVarying) -> Vec4f {
        if is_clipped(input.clip_distances) {
            return transparent();
        }
        let locals = get!(SURFACE_LOCALS);
        let fade = ContentMask::alpha(locals.content_mask, scene_position(input.position.xy()));
        if locals.color_format == SurfaceColorFormat::Yuv {
            return sample_yuv_surface(input.texture_position) * locals.opacity * fade;
        }
        texture_sample_level(
            SURFACE_TEXTURE,
            SURFACE_SAMPLER,
            input.texture_position,
            0.0,
        ) * locals.opacity
            * fade
    }
}

#[wgsl_rs::wgsl]
pub mod blur {
    use super::super::common::*;
    use super::super::corner_smoothing::*;
    use wgsl_rs::std::*;

    #[repr(C)]
    #[derive(Clone, Copy, Wgsl)]
    pub struct BlurUniforms {
        pub bounds: Bounds,
        pub content_mask: Bounds,
        pub corner_radii: Corners,
        pub direction: Vec2f,
        pub standard_deviation: f32,
        pub opacity: f32,
        pub sample_count: u32,
        pub sample_step: f32,
        pub composite_clip: BlurCompositeClip,
        pub downsample_mode: DownsampleMode,
        pub source_size: Vec2f,
        pub target_size: Vec2f,
        pub corner_smoothing: f32,
        /// For a downsample: how many source texels, along each axis, make
        /// one of the target's (2, 4, 8 or 16).
        pub downsample_factor: u32,
        /// Where the blurred source's first texel sits in the viewport.
        pub source_origin: Vec2f,
        /// With `tinted` 1, the pass reads the source's coverage in this
        /// colour, premultiplied, moved by `tint_offset` (in the source's
        /// texture coordinates): a drop shadow, cast as it is blurred.
        pub tint: Vec4f,
        pub tint_offset: Vec2f,
        pub tinted: u32,
        pub padding1: u32,
    }
    uniform!(group(1), binding(0), BLUR_LOCALS: BlurUniforms);
    texture!(group(1), binding(1), BLUR_TEXTURE: Texture2D<f32>);
    sampler!(group(1), binding(2), BLUR_SAMPLER: Sampler);

    #[derive(Clone, Copy, Wgsl)]
    pub struct BlurCompositeVertexData {
        pub position: Vec4f,
        pub texture_coordinates: Vec2f,
        pub clip_distances: Vec4f,
    }

    pub fn prepare_blur_composite_vertex(vertex_id: u32) -> BlurCompositeVertexData {
        let vertex = rectangle_vertex(vertex_id, get!(BLUR_LOCALS).bounds);
        BlurCompositeVertexData {
            position: vertex.clip_position,
            texture_coordinates: vertex.unit_position,
            clip_distances: clip_distances(
                vertex.viewport_position,
                get!(BLUR_LOCALS).content_mask,
            ),
        }
    }

    pub fn blur_composite_color(position: Vec2f, coverage: f32) -> Vec4f {
        let blurred = texture_sample_level(
            BLUR_TEXTURE,
            BLUR_SAMPLER,
            (position - get!(BLUR_LOCALS).source_origin) / get!(BLUR_LOCALS).target_size,
            0.0,
        );
        let factor = coverage * get!(BLUR_LOCALS).opacity;
        vec4f(
            blurred.x * factor,
            blurred.y * factor,
            blurred.z * factor,
            blurred.w * factor,
        )
    }

    #[derive(Wgsl)]
    pub struct BlurVarying {
        #[builtin(position)]
        pub position: Vec4f,
        #[location(0)]
        pub texture_coordinates: Vec2f,
        #[location(3)]
        pub clip_distances: Vec4f,
    }

    #[vertex]
    pub fn vertex_blur_fullscreen(#[builtin(vertex_index)] vertex_id: u32) -> BlurVarying {
        let vertex = fullscreen_vertex(vertex_id);
        BlurVarying {
            position: vertex.clip_position,
            texture_coordinates: vertex.texture_coordinates,
            clip_distances: unclipped_distances(),
        }
    }

    /// Each target texel averages the `downsample_factor`² source texels it
    /// covers, a box prefilter: one bilinear sample at the middle of each
    /// 2×2 of them.
    #[fragment]
    pub fn fragment_blur_downsample(input: BlurVarying) -> Vec4f {
        let uniforms = get!(BLUR_LOCALS);
        if uniforms.downsample_mode == DownsampleMode::HalfResolution {
            let factor = max(uniforms.downsample_factor, 2u32);
            let taps = factor / 2u32;
            let corner = floor(input.position.xy()) * (factor as f32);
            let last = (uniforms.source_size - 0.5) / uniforms.source_size;
            let mut sum = vec4f(0.0, 0.0, 0.0, 0.0);
            let mut row = 0u32;
            while row < taps {
                let mut column = 0u32;
                while column < taps {
                    let texel =
                        corner + vec2f(2.0 * (column as f32) + 1.0, 2.0 * (row as f32) + 1.0);
                    sum = sum + sample_blur_texture(min(texel / uniforms.source_size, last));
                    column += 1u32;
                }
                row += 1u32;
            }
            return sum / ((taps * taps) as f32);
        }
        texture_sample_level(BLUR_TEXTURE, BLUR_SAMPLER, input.texture_coordinates, 0.0)
    }

    /// The source at `texture_coordinates`, or, for a tinted pass, its
    /// coverage there in the tint, moved: transparent where that falls
    /// outside the source.
    pub fn sample_blur_texture(texture_coordinates: Vec2f) -> Vec4f {
        let uniforms = get!(BLUR_LOCALS);
        if uniforms.tinted == 0u32 {
            return texture_sample_level(BLUR_TEXTURE, BLUR_SAMPLER, texture_coordinates, 0.0);
        }
        let moved = texture_coordinates - uniforms.tint_offset;
        if moved.x < 0.0 || moved.y < 0.0 || moved.x > 1.0 || moved.y > 1.0 {
            return transparent();
        }
        uniforms.tint * texture_sample_level(BLUR_TEXTURE, BLUR_SAMPLER, moved, 0.0).w
    }

    /// A separable Gaussian along `direction`, its weights unnormalized
    /// (`exp(-d² / 2σ²)` at `d` texels), as the sum normalizes them: each is
    /// the last times a factor that shrinks by `exp(-1 / σ²)` a texel, for
    /// no `exp` a tap. Where taps are a texel apart, each two neighbouring
    /// taps on a side are read as one bilinear sample between them, weighted
    /// by both: half the reads.
    pub fn gaussian_blur(texture_coordinates: Vec2f) -> Vec4f {
        let uniforms = get!(BLUR_LOCALS);
        let variance = uniforms.standard_deviation * uniforms.standard_deviation;
        let mut weighted_color = sample_blur_texture(texture_coordinates);
        let mut weight_sum = 1.0;
        let mut sample_index = 1u32;

        if uniforms.sample_step == 1.0 {
            // The weight at the texel, and the factor to the next one's.
            let shrink = exp(-1.0 / variance);
            let mut factor = exp(-0.5 / variance);
            let mut weight = 1.0;
            while sample_index <= uniforms.sample_count {
                weight *= factor;
                factor *= shrink;
                let near_weight = weight;
                weight *= factor;
                factor *= shrink;
                let far_weight = select(0.0, weight, sample_index < uniforms.sample_count);
                let pair_weight = near_weight + far_weight;
                let distance = sample_index as f32 + far_weight / pair_weight;
                let coordinate_offset = uniforms.direction * distance;
                let symmetric_pair = sample_blur_texture(texture_coordinates - coordinate_offset)
                    + sample_blur_texture(texture_coordinates + coordinate_offset);
                weighted_color = weighted_color + symmetric_pair * pair_weight;
                weight_sum += 2.0 * pair_weight;
                sample_index += 2u32;
            }
            return weighted_color / max(weight_sum, MIN_NORMALIZED_WEIGHT);
        }

        while sample_index <= uniforms.sample_count {
            let distance = sample_index as f32 * uniforms.sample_step;
            let weight = exp(-(distance * distance) / (2.0 * variance));
            let coordinate_offset = uniforms.direction * distance;
            let symmetric_pair = sample_blur_texture(texture_coordinates - coordinate_offset)
                + sample_blur_texture(texture_coordinates + coordinate_offset);
            weighted_color = weighted_color + symmetric_pair * weight;
            weight_sum += 2.0 * weight;
            sample_index += 1u32;
        }
        weighted_color / max(weight_sum, MIN_NORMALIZED_WEIGHT)
    }

    #[fragment]
    pub fn fragment_blur(input: BlurVarying) -> Vec4f {
        gaussian_blur(input.texture_coordinates)
    }

    #[vertex]
    pub fn vertex_blur_composite(#[builtin(vertex_index)] vertex_id: u32) -> BlurVarying {
        let vertex = prepare_blur_composite_vertex(vertex_id);
        BlurVarying {
            position: vertex.position,
            texture_coordinates: vertex.texture_coordinates,
            clip_distances: vertex.clip_distances,
        }
    }

    #[fragment]
    pub fn fragment_blur_composite(input: BlurVarying) -> Vec4f {
        if is_clipped(input.clip_distances) {
            return transparent();
        }
        let coverage = select(
            1.0,
            antialiased_coverage(rounded_rectangle_signed_distance(
                scene_position(input.position.xy()),
                get!(BLUR_LOCALS).bounds,
                get!(BLUR_LOCALS).corner_radii,
            )),
            get!(BLUR_LOCALS).composite_clip == BlurCompositeClip::RoundedBounds,
        );
        blur_composite_color(scene_position(input.position.xy()), coverage)
    }

    #[derive(Wgsl)]
    pub struct SmoothedBlurVarying {
        #[builtin(position)]
        pub position: Vec4f,
        #[location(0)]
        pub texture_coordinates: Vec2f,
        #[location(1)]
        #[interpolate(flat)]
        pub horizontal_corner_reaches: Vec4f,
        #[location(2)]
        #[interpolate(flat)]
        pub vertical_corner_reaches: Vec4f,
        #[location(3)]
        pub clip_distances: Vec4f,
        #[location(4)]
        #[interpolate(flat)]
        pub smoothing_factors: Vec4f,
        #[location(5)]
        #[interpolate(flat)]
        pub superellipse_power: f32,
    }

    #[vertex]
    pub fn vertex_smoothed_blur_composite(
        #[builtin(vertex_index)] vertex_id: u32,
    ) -> SmoothedBlurVarying {
        let vertex = prepare_blur_composite_vertex(vertex_id);
        let prepared = prepare_corners(
            get!(BLUR_LOCALS).bounds.size,
            get!(BLUR_LOCALS).corner_radii,
            get!(BLUR_LOCALS).corner_smoothing,
            true,
        );
        SmoothedBlurVarying {
            position: vertex.position,
            texture_coordinates: vertex.texture_coordinates,
            horizontal_corner_reaches: prepared.horizontal_reaches,
            vertical_corner_reaches: prepared.vertical_reaches,
            clip_distances: vertex.clip_distances,
            smoothing_factors: prepared.smoothing_factors,
            superellipse_power: prepared.superellipse_power,
        }
    }

    #[fragment]
    pub fn fragment_smoothed_blur_composite(input: SmoothedBlurVarying) -> Vec4f {
        if is_clipped(input.clip_distances) {
            return transparent();
        }
        let coverage = select(
            1.0,
            antialiased_coverage(prepared_corner_signed_distance(
                scene_position(input.position.xy()),
                get!(BLUR_LOCALS).bounds,
                get!(BLUR_LOCALS).corner_radii,
                get!(BLUR_LOCALS).corner_smoothing,
                PreparedCorners {
                    horizontal_reaches: input.horizontal_corner_reaches,
                    vertical_reaches: input.vertical_corner_reaches,
                    smoothing_factors: input.smoothing_factors,
                    superellipse_power: input.superellipse_power,
                },
            )),
            get!(BLUR_LOCALS).composite_clip == BlurCompositeClip::RoundedBounds,
        );
        blur_composite_color(scene_position(input.position.xy()), coverage)
    }
}
