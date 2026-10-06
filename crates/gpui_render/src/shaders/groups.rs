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
}
