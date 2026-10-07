#[wgsl_rs::wgsl]
pub mod group {
    use super::super::common::*;
    use wgsl_rs::std::*;

    /// How an isolated group is composited into its parent.
    #[repr(C)]
    #[derive(Clone, Copy, Wgsl)]
    pub struct GroupUniforms {
        /// The viewport rectangle the composite covers.
        pub bounds: Bounds,
        /// The viewport rectangle it is clipped to.
        pub content_mask: Bounds,
        /// Where the group's texture sits in the viewport, and its size there.
        pub source_origin: Vec2f,
        pub source_size: Vec2f,
        /// Where the copy of the parent's pixels under the group sits in the
        /// viewport, and its size there, for blend modes other than normal.
        pub backdrop_origin: Vec2f,
        pub backdrop_size: Vec2f,
        /// Where the group's mask target sits in the viewport, and its size
        /// there, when `mask` asks for one: outside it, nothing shows.
        pub mask_origin: Vec2f,
        pub mask_size: Vec2f,
        pub opacity: f32,
        pub blend_mode: GroupBlendMode,
        pub mask: GroupMask,
        pub padding: u32,
        /// For a filter pass (`fragment_group_filter`), which writes a new
        /// picture covering `bounds` from the group's texture: what it does.
        pub filter_kind: GroupFilter,
        /// For a colour matrix: 1 to apply it in linear light.
        pub filter_linear: u32,
        /// For a program: its paint-table entry.
        pub filter_paint: u32,
        pub filter_padding: u32,
        /// For a colour matrix: how far the picture it reads is moved, in
        /// device pixels.
        pub filter_offset: Vec2f,
        /// For a program: the translation from the element's logical pixels
        /// to the viewport, and the linear part, row-major.
        pub input_translation: Vec2f,
        pub input_matrix: Vec4f,
        /// For a colour matrix: the coefficients of R, G, B and A in each
        /// output channel, then the offsets.
        pub matrix_red: Vec4f,
        pub matrix_green: Vec4f,
        pub matrix_blue: Vec4f,
        pub matrix_alpha: Vec4f,
        pub matrix_offset: Vec4f,
    }
    uniform!(group(1), binding(0), GROUP_LOCALS: GroupUniforms);
    texture!(group(1), binding(1), GROUP_TEXTURE: Texture2D<f32>);
    texture!(group(1), binding(2), BACKDROP_TEXTURE: Texture2D<f32>);
    sampler!(group(1), binding(3), GROUP_SAMPLER: Sampler);
    texture!(group(1), binding(4), MASK_TEXTURE: Texture2D<f32>);

    #[derive(Wgsl)]
    pub struct GroupVarying {
        #[builtin(position)]
        pub position: Vec4f,
        #[location(3)]
        pub clip_distances: Vec4f,
    }

    #[vertex]
    pub fn vertex_group_composite(#[builtin(vertex_index)] vertex_id: u32) -> GroupVarying {
        let vertex = rectangle_vertex(vertex_id, get!(GROUP_LOCALS).bounds);
        GroupVarying {
            position: vertex.clip_position,
            clip_distances: clip_distances(
                vertex.viewport_position,
                get!(GROUP_LOCALS).content_mask,
            ),
        }
    }

    pub fn unpremultiply(color: Vec4f) -> Vec3f {
        color.xyz() / max(color.w, MIN_UNPREMULTIPLY_ALPHA)
    }

    // The separable blend functions of the W3C Compositing and Blending
    // specification, per channel: `backdrop` is the parent's colour, `source`
    // the group's.

    pub fn blend_multiply(backdrop: f32, source: f32) -> f32 {
        backdrop * source
    }

    pub fn blend_screen(backdrop: f32, source: f32) -> f32 {
        backdrop + source - backdrop * source
    }

    pub fn blend_hard_light(backdrop: f32, source: f32) -> f32 {
        if source <= 0.5 {
            return blend_multiply(backdrop, 2.0 * source);
        }
        blend_screen(backdrop, 2.0 * source - 1.0)
    }

    pub fn blend_color_dodge(backdrop: f32, source: f32) -> f32 {
        if backdrop <= 0.0 {
            return 0.0;
        }
        if source >= 1.0 {
            return 1.0;
        }
        min(1.0, backdrop / (1.0 - source))
    }

    pub fn blend_color_burn(backdrop: f32, source: f32) -> f32 {
        if backdrop >= 1.0 {
            return 1.0;
        }
        if source <= 0.0 {
            return 0.0;
        }
        1.0 - min(1.0, (1.0 - backdrop) / source)
    }

    pub fn blend_soft_light(backdrop: f32, source: f32) -> f32 {
        if source <= 0.5 {
            return backdrop - (1.0 - 2.0 * source) * backdrop * (1.0 - backdrop);
        }
        let mut d = sqrt(backdrop);
        if backdrop <= 0.25 {
            d = ((16.0 * backdrop - 12.0) * backdrop + 4.0) * backdrop;
        }
        backdrop + (2.0 * source - 1.0) * (d - backdrop)
    }

    pub fn blend_channel(mode: GroupBlendMode, backdrop: f32, source: f32) -> f32 {
        if mode == GroupBlendMode::Multiply {
            return blend_multiply(backdrop, source);
        }
        if mode == GroupBlendMode::Screen {
            return blend_screen(backdrop, source);
        }
        if mode == GroupBlendMode::Overlay {
            return blend_hard_light(source, backdrop);
        }
        if mode == GroupBlendMode::Darken {
            return min(backdrop, source);
        }
        if mode == GroupBlendMode::Lighten {
            return max(backdrop, source);
        }
        if mode == GroupBlendMode::ColorDodge {
            return blend_color_dodge(backdrop, source);
        }
        if mode == GroupBlendMode::ColorBurn {
            return blend_color_burn(backdrop, source);
        }
        if mode == GroupBlendMode::HardLight {
            return blend_hard_light(backdrop, source);
        }
        if mode == GroupBlendMode::SoftLight {
            return blend_soft_light(backdrop, source);
        }
        if mode == GroupBlendMode::Difference {
            return abs(backdrop - source);
        }
        if mode == GroupBlendMode::Exclusion {
            return backdrop + source - 2.0 * backdrop * source;
        }
        source
    }

    // The non-separable blend functions, which mix hue, saturation and
    // luminosity across channels.

    pub fn luminosity(color: Vec3f) -> f32 {
        dot(color, vec3f(0.3, 0.59, 0.11))
    }

    pub fn clip_color(color: Vec3f) -> Vec3f {
        let l = luminosity(color);
        let n = min(color.x, min(color.y, color.z));
        let x = max(color.x, max(color.y, color.z));
        let mut clipped = color;
        if n < 0.0 {
            clipped = vec3f(l, l, l)
                + (clipped - vec3f(l, l, l)) * l / max(l - n, MIN_UNPREMULTIPLY_ALPHA);
        }
        if x > 1.0 {
            clipped = vec3f(l, l, l)
                + (clipped - vec3f(l, l, l)) * (1.0 - l) / max(x - l, MIN_UNPREMULTIPLY_ALPHA);
        }
        clipped
    }

    pub fn with_luminosity(color: Vec3f, l: f32) -> Vec3f {
        let d = l - luminosity(color);
        clip_color(color + vec3f(d, d, d))
    }

    pub fn saturation(color: Vec3f) -> f32 {
        max(color.x, max(color.y, color.z)) - min(color.x, min(color.y, color.z))
    }

    pub fn with_saturation(color: Vec3f, s: f32) -> Vec3f {
        let low = min(color.x, min(color.y, color.z));
        let high = max(color.x, max(color.y, color.z));
        if high <= low {
            return vec3f(0.0, 0.0, 0.0);
        }
        (color - vec3f(low, low, low)) * s / (high - low)
    }

    pub fn blend(mode: GroupBlendMode, backdrop: Vec3f, source: Vec3f) -> Vec3f {
        if mode == GroupBlendMode::Hue {
            return with_luminosity(
                with_saturation(source, saturation(backdrop)),
                luminosity(backdrop),
            );
        }
        if mode == GroupBlendMode::Saturation {
            return with_luminosity(
                with_saturation(backdrop, saturation(source)),
                luminosity(backdrop),
            );
        }
        if mode == GroupBlendMode::Color {
            return with_luminosity(source, luminosity(backdrop));
        }
        if mode == GroupBlendMode::Luminosity {
            return with_luminosity(backdrop, luminosity(source));
        }
        vec3f(
            blend_channel(mode, backdrop.x, source.x),
            blend_channel(mode, backdrop.y, source.y),
            blend_channel(mode, backdrop.z, source.z),
        )
    }

    /// The group's colour at `position`, premultiplied, to be composited over
    /// its parent: for a blend mode other than normal, its colour mixed with
    /// the parent's under it, as the W3C specification mixes them before
    /// compositing source-over.
    #[fragment]
    pub fn fragment_group_composite(input: GroupVarying) -> Vec4f {
        if is_clipped(input.clip_distances) {
            return transparent();
        }
        let locals = get!(GROUP_LOCALS);
        let position = scene_position(input.position.xy());
        let mut coverage = locals.opacity;
        if locals.mask != GroupMask::None {
            let mask_position = (position - locals.mask_origin) / locals.mask_size;
            if mask_position.x < 0.0
                || mask_position.y < 0.0
                || mask_position.x > 1.0
                || mask_position.y > 1.0
            {
                return transparent();
            }
            let mask = texture_sample_level(MASK_TEXTURE, GROUP_SAMPLER, mask_position, 0.0);
            if locals.mask == GroupMask::Alpha {
                coverage *= mask.w;
            } else {
                // The luminance of the premultiplied colour: the mask's
                // luminance times its coverage.
                coverage *= dot(mask.xyz(), vec3f(0.2125, 0.7154, 0.0721));
            }
        }
        let source = texture_sample_level(
            GROUP_TEXTURE,
            GROUP_SAMPLER,
            (position - locals.source_origin) / locals.source_size,
            0.0,
        );
        if locals.blend_mode == GroupBlendMode::Normal {
            return source * coverage;
        }
        let backdrop = texture_sample_level(
            BACKDROP_TEXTURE,
            GROUP_SAMPLER,
            (position - locals.backdrop_origin) / locals.backdrop_size,
            0.0,
        );
        let backdrop_color = unpremultiply(backdrop);
        let source_color = unpremultiply(source);
        let mixed = source_color * (1.0 - backdrop.w)
            + blend(locals.blend_mode, backdrop_color, source_color) * backdrop.w;
        vec4f(mixed.x, mixed.y, mixed.z, 1.0) * source.w * coverage
    }

    /// The group's texture at viewport position `position`: transparent
    /// outside it.
    pub fn group_source(position: Vec2f) -> Vec4f {
        let locals = get!(GROUP_LOCALS);
        let uv = (position - locals.source_origin) / locals.source_size;
        if uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 {
            return transparent();
        }
        texture_sample_level(GROUP_TEXTURE, GROUP_SAMPLER, uv, 0.0)
    }

    /// The picture a filter program reads, at `position` in the element's
    /// logical pixels: what `program_input` returns where filters run.
    pub fn group_filter_input(position: Vec2f) -> Vec4f {
        let locals = get!(GROUP_LOCALS);
        let m = locals.input_matrix;
        let viewport = vec2f(
            m.x * position.x + m.y * position.y,
            m.z * position.x + m.w * position.y,
        ) + locals.input_translation;
        group_source(viewport)
    }

    /// `color`, premultiplied, through the pass's colour matrix: applied to
    /// straight colour, in linear light if the pass asks, and clamped.
    pub fn group_color_matrix(color: Vec4f) -> Vec4f {
        let locals = get!(GROUP_LOCALS);
        let mut rgb = unpremultiply(color);
        if locals.filter_linear != 0u32 {
            rgb = srgb_to_linear(rgb);
        }
        let straight = vec4f(rgb.x, rgb.y, rgb.z, color.w);
        let transformed = vec4f(
            dot(locals.matrix_red, straight),
            dot(locals.matrix_green, straight),
            dot(locals.matrix_blue, straight),
            dot(locals.matrix_alpha, straight),
        ) + locals.matrix_offset;
        let clamped = clamp(
            transformed,
            vec4f(0.0, 0.0, 0.0, 0.0),
            vec4f(1.0, 1.0, 1.0, 1.0),
        );
        let mut out = clamped.xyz();
        if locals.filter_linear != 0u32 {
            out = linear_to_srgb(out);
        }
        vec4f(
            out.x * clamped.w,
            out.y * clamped.w,
            out.z * clamped.w,
            clamped.w,
        )
    }

    /// The colour of filter program `id`, premultiplied, at a fragment of
    /// the element it runs over, as [`program_color`] takes one.
    ///
    /// The standard shaders run no programs, so this is `fallback`. A
    /// renderer that links programs replaces this function, which must stay
    /// the only one by its name, with one that runs them.
    pub fn program_filter_color(
        _id: u32,
        _uv: Vec2f,
        _position: Vec2f,
        _size: Vec2f,
        _origin: Vec2f,
        _scale: f32,
        _stroke: Vec2f,
        _base: u32,
        fallback: Vec4f,
    ) -> Vec4f {
        vec4f(
            fallback.x * fallback.w,
            fallback.y * fallback.w,
            fallback.z * fallback.w,
            fallback.w,
        )
    }

    /// The colour of the filter program of paint-table entry `index` at
    /// `viewport_position`, premultiplied.
    pub fn group_filter_program(index: u32, viewport_position: Vec2f) -> Vec4f {
        let paint = scene_paint(index);
        let point =
            TransformationMatrix::transform_position(paint.transformation, viewport_position);
        let size = paint.geometry.xy();
        let units_per_pixel = sqrt(max(
            abs(determinant(paint.transformation.rotation_scale)),
            MIN_PROGRAM_BOX_SIZE,
        ));
        let scale = 1.0 / units_per_pixel;
        program_filter_color(
            u32(paint.geometry.z),
            point / max(size, vec2f(MIN_PROGRAM_BOX_SIZE, MIN_PROGRAM_BOX_SIZE)),
            point,
            size,
            viewport_position - point * scale,
            scale,
            vec2f(0.0, 0.0),
            paint.first_stop,
            paint.radii,
        )
    }

    /// One pass of a group's filters: the group's picture, in its texture,
    /// recoloured by a matrix, merged over another picture, or run through a
    /// program, into a new picture covering the same viewport rectangle.
    #[fragment]
    pub fn fragment_group_filter(input: GroupVarying) -> Vec4f {
        let locals = get!(GROUP_LOCALS);
        let position = scene_position(input.position.xy());
        if locals.filter_kind == GroupFilter::Program {
            return group_filter_program(locals.filter_paint, position);
        }
        let source = group_source(position - locals.filter_offset);
        if locals.filter_kind == GroupFilter::ColorMatrix {
            return group_color_matrix(source);
        }
        if locals.filter_kind == GroupFilter::Merge {
            let beneath = texture_sample_level(
                BACKDROP_TEXTURE,
                GROUP_SAMPLER,
                (position - locals.backdrop_origin) / locals.backdrop_size,
                0.0,
            );
            return source + beneath * (1.0 - source.w);
        }
        source
    }
}
