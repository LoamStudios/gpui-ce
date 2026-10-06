#[wgsl_rs::wgsl]
mod source {
    use wgsl_rs::std::*;

    // Defined here as a literal so the Rust-to-WGSL translator emits it, not `std`.
    #[allow(clippy::approx_constant)]
    pub const PI: f32 = 3.141592653589793;
    pub const HALF_TURN_DEGREES: f32 = 180.0;
    pub const FULL_TURN_DEGREES: f32 = 360.0;
    pub const CSS_GRADIENT_OFFSET_DEGREES: f32 = 90.0;
    pub const PIXEL_ANTIALIAS_RADIUS: f32 = 0.5;
    pub const GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS: f32 = 3.0;
    pub const PATTERN_COMPONENT_SCALE: f32 = 255.0;
    pub const PATTERN_PACKING_RADIX: f32 = 65535.0;
    pub const MIN_NORMALIZED_WEIGHT: f32 = 0.00001;
    pub const MIN_PATH_GRADIENT: f32 = 0.001;
    /// The smallest alpha a colour is divided by to unpremultiply it.
    pub const MIN_UNPREMULTIPLY_ALPHA: f32 = 0.00001;
    pub const UNDERLINE_WAVE_FREQUENCY: f32 = 2.0;
    pub const UNDERLINE_WAVE_HEIGHT_RATIO: f32 = 0.8;
    pub const GRADIENT_DITHER_SCALE: f32 = 0.6180339887;
    pub const GRADIENT_DITHER_RGB_STRENGTH: f32 = 2.0 / PATTERN_COMPONENT_SCALE;
    pub const GRADIENT_DITHER_ALPHA_STRENGTH: f32 = 3.0 / PATTERN_COMPONENT_SCALE;
    pub const GRADIENT_DITHER_SEED_A: Vec2f = vec2f(12.9898, 78.233);
    pub const GRADIENT_DITHER_SEED_B: Vec2f = vec2f(39.3460, 11.135);
    pub const GRADIENT_DITHER_MULTIPLIER_A: f32 = 43758.5453;
    pub const GRADIENT_DITHER_MULTIPLIER_B: f32 = 24634.6345;
    pub const DIAGONAL_STRIPE_ANGLE: f32 = PI / 4.0;
    pub const REC_601_LUMA_WEIGHTS: Vec3f = vec3f(0.30, 0.59, 0.11);
    pub const LINEAR_RGB_LUMA_WEIGHTS: Vec3f = vec3f(0.2126, 0.7152, 0.0722);
    pub const SRGB_DECODE_CUTOFF: f32 = 0.04045;
    pub const SRGB_ENCODE_CUTOFF: f32 = 0.0031308;
    pub const SRGB_TRANSFER_OFFSET: f32 = 0.055;
    pub const SRGB_TRANSFER_SCALE: f32 = 1.055;
    pub const SRGB_DECODE_EXPONENT: f32 = 2.4;
    pub const SRGB_LINEAR_SCALE: f32 = 12.92;
    pub const ERROR_FUNCTION_LINEAR_COEFFICIENT: f32 = 0.278393;
    pub const ERROR_FUNCTION_QUADRATIC_COEFFICIENT: f32 = 0.230389;
    pub const ERROR_FUNCTION_CUBIC_COEFFICIENT: f32 = 0.000972;
    pub const ERROR_FUNCTION_QUARTIC_COEFFICIENT: f32 = 0.078108;
    pub const DARK_TEXT_CONTRAST_SCALE: f32 = 4.0;
    pub const DARK_TEXT_BRIGHTNESS_CUTOFF: f32 = 0.75;
    pub const LINEAR_SRGB_TO_CONE_RESPONSE: Mat3x3f = mat3x3f(
        vec3f(0.4122214708, 0.2119034982, 0.0883024619),
        vec3f(0.5363325363, 0.6806995451, 0.2817188376),
        vec3f(0.0514459929, 0.1073969566, 0.6299787005),
    );
    pub const CONE_RESPONSE_TO_OKLAB: Mat3x3f = mat3x3f(
        vec3f(0.2104542553, 1.9779984951, 0.0259040371),
        vec3f(0.7936177850, -2.4285922050, 0.7827717662),
        vec3f(-0.0040720468, 0.4505938995, -0.8086757660),
    );
    pub const OKLAB_TO_CONE_RESPONSE: Mat3x3f = mat3x3f(
        vec3f(1.0, 1.0, 1.0),
        vec3f(0.3963377774, -0.1055613458, -0.0894841775),
        vec3f(0.2158037573, -0.0638541728, -1.2914855480),
    );
    pub const CONE_RESPONSE_TO_LINEAR_SRGB: Mat3x3f = mat3x3f(
        vec3f(4.0767416621, -1.2684380046, -0.0041960863),
        vec3f(-3.3077115913, 2.6097574011, -0.7034186147),
        vec3f(0.2309699292, -0.3413193965, 1.7076147010),
    );

    #[repr(u32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Wgsl)]
    pub enum ShaderBool {
        Disabled = 0,
        Enabled = 1,
    }

    #[repr(u32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Wgsl)]
    pub enum BackgroundTag {
        Solid = 0,
        LinearGradient = 1,
        PatternSlash = 2,
        Checkerboard = 3,
        Paint = 4,
    }

    /// The shape of a paint-table gradient.
    #[repr(u32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Wgsl)]
    pub enum PaintKind {
        Linear = 0,
        Radial = 1,
        Sweep = 2,
        Stripes = 3,
        Checkerboard = 4,
    }

    /// How a paint-table gradient continues past its ends.
    #[repr(u32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Wgsl)]
    pub enum PaintExtend {
        Pad = 0,
        Repeat = 1,
        Reflect = 2,
    }

    /// The colour space a paint-table gradient's stops are interpolated in.
    #[repr(u32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Wgsl)]
    pub enum PaintColorSpace {
        Srgb = 0,
        LinearSrgb = 1,
        Oklab = 2,
        Oklch = 3,
        LegacySrgb = 4,
        LegacyOklab = 5,
    }

    #[repr(u32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Wgsl)]
    pub enum ColorSpace {
        Srgb = 0,
        Oklab = 1,
    }

    #[repr(u32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Wgsl)]
    pub enum BorderStyle {
        Solid = 0,
        Dashed = 1,
    }

    #[repr(u32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Wgsl)]
    pub enum SurfaceColorFormat {
        Rgba = 0,
        Yuv = 1,
    }

    /// How a group's colours mix with its parent's: the blend modes of the
    /// W3C Compositing and Blending specification.
    #[repr(u32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Wgsl)]
    pub enum GroupBlendMode {
        Normal = 0,
        Multiply = 1,
        Screen = 2,
        Overlay = 3,
        Darken = 4,
        Lighten = 5,
        ColorDodge = 6,
        ColorBurn = 7,
        HardLight = 8,
        SoftLight = 9,
        Difference = 10,
        Exclusion = 11,
        Hue = 12,
        Saturation = 13,
        Color = 14,
        Luminosity = 15,
    }

    /// How a group's composite is masked: not at all, or by its mask
    /// target's coverage or luminance.
    #[repr(u32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Wgsl)]
    pub enum GroupMask {
        None = 0,
        Alpha = 1,
        Luminance = 2,
    }

    #[repr(u32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Wgsl)]
    pub enum BlurCompositeClip {
        None = 0,
        RoundedBounds = 1,
    }

    #[repr(u32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Wgsl)]
    pub enum DownsampleMode {
        Copy = 0,
        HalfResolution = 1,
    }

    #[repr(C)]
    #[derive(Clone, Copy, PartialEq, Wgsl)]
    pub struct GlobalUniforms {
        /// The scene's viewport, in device pixels.
        pub viewport_size: Vec2f,
        /// Where the render target's first texel sits in the viewport: zero
        /// for the window, a group's origin for a group's target.
        pub target_origin: Vec2f,
        /// The render target's size, in device pixels.
        pub target_size: Vec2f,
        pub premultiplied_alpha: ShaderBool,
        pub padding: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, PartialEq, Wgsl)]
    pub struct FontRasterizationUniforms {
        pub gamma_ratios: Vec4f,
        pub grayscale_enhanced_contrast: f32,
        pub subpixel_enhanced_contrast: f32,
        pub uses_blue_green_red_subpixel_order: ShaderBool,
        pub padding: u32,
    }

    uniform!(group(0), binding(0), GLOBALS: GlobalUniforms);
    uniform!(group(0), binding(1), FONT_RASTERIZATION: FontRasterizationUniforms);

    #[repr(C)]
    #[derive(Clone, Copy, Wgsl)]
    pub struct Bounds {
        pub origin: Vec2f,
        pub size: Vec2f,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Wgsl)]
    pub struct ContentMask {
        pub bounds: Bounds,
        pub fade_out: Edges,
    }

    impl ContentMask {
        /// The mask's coverage at the given window-space position: 1 inside,
        /// 0 outside, and a linear ramp to 0 across each edge's fade distance.
        pub fn alpha(content_mask: ContentMask, position: Vec2f) -> f32 {
            let bounds = content_mask.bounds;
            let fade = content_mask.fade_out;
            let mut alpha = 1.0;
            if fade.left > 0.0 {
                alpha *= saturate((position.x - bounds.origin.x) / fade.left);
            }
            if fade.right > 0.0 {
                alpha *= saturate((bounds.origin.x + bounds.size.x - position.x) / fade.right);
            }
            if fade.top > 0.0 {
                alpha *= saturate((position.y - bounds.origin.y) / fade.top);
            }
            if fade.bottom > 0.0 {
                alpha *= saturate((bounds.origin.y + bounds.size.y - position.y) / fade.bottom);
            }
            alpha
        }
    }

    impl Bounds {
        pub fn position(bounds: Bounds, unit_position: Vec2f) -> Vec2f {
            bounds.origin + unit_position * bounds.size
        }

        pub fn half_size(bounds: Bounds) -> Vec2f {
            bounds.size / 2.0
        }

        pub fn center(bounds: Bounds) -> Vec2f {
            bounds.origin + Bounds::half_size(bounds)
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Wgsl)]
    pub struct Corners {
        pub top_left: f32,
        pub top_right: f32,
        pub bottom_right: f32,
        pub bottom_left: f32,
    }

    impl Corners {
        pub fn is_zero(corners: Corners) -> bool {
            corners.top_left == 0.0
                && corners.top_right == 0.0
                && corners.bottom_right == 0.0
                && corners.bottom_left == 0.0
        }
    }

    #[derive(Clone, Copy, Wgsl)]
    pub struct Edges {
        pub top: f32,
        pub right: f32,
        pub bottom: f32,
        pub left: f32,
    }

    impl Edges {
        pub fn is_zero(edges: Edges) -> bool {
            edges.top == 0.0 && edges.right == 0.0 && edges.bottom == 0.0 && edges.left == 0.0
        }
    }
    #[derive(Clone, Copy, Wgsl)]
    pub struct Hsla {
        pub h: f32,
        pub s: f32,
        pub l: f32,
        pub a: f32,
    }
    #[derive(Clone, Copy, Wgsl)]
    pub struct AtlasTextureId {
        pub index: u32,
        pub kind: u32,
    }
    #[derive(Clone, Copy, Wgsl)]
    pub struct AtlasBounds {
        pub origin: Vec2i,
        pub size: Vec2i,
    }
    #[derive(Clone, Copy, Wgsl)]
    pub struct AtlasTile {
        pub texture_id: AtlasTextureId,
        pub tile_id: u32,
        pub padding: u32,
        pub bounds: AtlasBounds,
    }
    #[derive(Clone, Copy, Wgsl)]
    pub struct TransformationMatrix {
        pub rotation_scale: Mat2x2f,
        pub translation: Vec2f,
    }

    impl TransformationMatrix {
        pub fn transform_position(transform: TransformationMatrix, position: Vec2f) -> Vec2f {
            transpose(transform.rotation_scale) * position + transform.translation
        }
    }

    /// An entry of the scene's transform table: from a primitive's own space,
    /// where its bounds are, to the viewport, and back. Entry 0 is the
    /// identity, which most primitives use.
    #[derive(Clone, Copy, Wgsl)]
    pub struct SceneTransform {
        pub transformation: TransformationMatrix,
        pub inverse: TransformationMatrix,
    }

    /// An entry of the scene's clip table: a rounded rectangle in the space of
    /// a transform-table entry, and the clip it is nested in. Entry 0 clips
    /// nothing and ends every chain. Clips aligned with the viewport fold into
    /// a primitive's `content_mask` instead.
    #[derive(Clone, Copy, Wgsl)]
    pub struct SceneClip {
        pub bounds: Bounds,
        pub corner_radii: Corners,
        pub transform: u32,
        pub parent: u32,
    }

    storage!(group(0), binding(2), TRANSFORMS: RuntimeArray<SceneTransform>);
    storage!(group(0), binding(3), CLIPS: RuntimeArray<SceneClip>);

    /// An entry of the scene's paint table: a gradient, placed by the
    /// transformation from viewport positions to its own space.
    ///
    /// Its geometry is, by kind: for linear, the start and end points; for
    /// radial, the start and end centres, and in `radii` their radii; for
    /// sweep, the centre, then the start and end angles in radians.
    #[derive(Clone, Copy, Wgsl)]
    pub struct ScenePaint {
        pub transformation: TransformationMatrix,
        pub kind: PaintKind,
        pub extend: PaintExtend,
        pub color_space: PaintColorSpace,
        pub first_stop: u32,
        pub stop_count: u32,
        pub padding: u32,
        pub geometry: Vec4f,
        pub radii: Vec4f,
    }

    /// A colour stop of the scene's stop table: its colour in its gradient's
    /// interpolation space, premultiplied except for hue, and its offset.
    #[derive(Clone, Copy, Wgsl)]
    pub struct SceneColorStop {
        pub color: Vec4f,
        pub offset: f32,
        pub padding0: u32,
        pub padding1: u32,
        pub padding2: u32,
    }

    storage!(group(0), binding(4), PAINTS: RuntimeArray<ScenePaint>);
    storage!(group(0), binding(5), COLOR_STOPS: RuntimeArray<SceneColorStop>);

    /// Where a gradient is at a point in its own space, from 0 at its start to
    /// 1 at its end, before it is extended; `x` is that offset, and `y` is 0
    /// where the gradient is not defined, as outside a cone.
    pub fn gradient_offset(paint: ScenePaint, point: Vec2f) -> Vec2f {
        if paint.kind == PaintKind::Linear {
            let start = paint.geometry.xy();
            let direction = paint.geometry.zw() - start;
            let length_squared = dot(direction, direction);
            if length_squared == 0.0 {
                return vec2f(0.0, 0.0);
            }
            return vec2f(dot(point - start, direction) / length_squared, 1.0);
        }
        if paint.kind == PaintKind::Sweep {
            let relative = point - paint.geometry.xy();
            let mut angle = atan2(relative.y, relative.x);
            if angle < 0.0 {
                angle += 2.0 * PI;
            }
            let span = paint.geometry.w - paint.geometry.z;
            if span == 0.0 {
                return vec2f(0.0, 0.0);
            }
            return vec2f((angle - paint.geometry.z) / span, 1.0);
        }
        // Two-point conical: the largest t whose circle, between the start
        // and end circles, passes through the point, with a radius of zero or
        // more.
        let start = paint.geometry.xy();
        let center_step = paint.geometry.zw() - start;
        let radius = paint.radii.x;
        let radius_step = paint.radii.y - radius;
        let relative = point - start;
        let a = dot(center_step, center_step) - radius_step * radius_step;
        let b = dot(relative, center_step) + radius * radius_step;
        let c = dot(relative, relative) - radius * radius;
        if abs(a) < 1e-6 {
            if b == 0.0 {
                return vec2f(0.0, 0.0);
            }
            let t = c / (2.0 * b);
            return vec2f(t, select(0.0, 1.0, radius + t * radius_step >= 0.0));
        }
        let discriminant = b * b - a * c;
        if discriminant < 0.0 {
            return vec2f(0.0, 0.0);
        }
        let root = sqrt(discriminant);
        let larger = max((b + root) / a, (b - root) / a);
        if radius + larger * radius_step >= 0.0 {
            return vec2f(larger, 1.0);
        }
        let smaller = min((b + root) / a, (b - root) / a);
        vec2f(
            smaller,
            select(0.0, 1.0, radius + smaller * radius_step >= 0.0),
        )
    }

    /// A gradient offset continued past the ends by `extend`, into 0 to 1.
    pub fn extend_offset(extend: PaintExtend, offset: f32) -> f32 {
        if extend == PaintExtend::Repeat {
            return offset - floor(offset);
        }
        if extend == PaintExtend::Reflect {
            let period = offset - 2.0 * floor(offset / 2.0);
            return select(period, 2.0 - period, period > 1.0);
        }
        saturate(offset)
    }

    /// The coverage of diagonal stripes `width` wide and `interval` apart at
    /// `point`, measured from where they start.
    pub fn stripes_coverage(point: Vec2f, width: f32, interval: f32) -> f32 {
        let height = width + interval;
        let period = height * sin(DIAGONAL_STRIPE_ANGLE);
        let rotation = mat2x2f(
            vec2f(cos(DIAGONAL_STRIPE_ANGLE), -sin(DIAGONAL_STRIPE_ANGLE)),
            vec2f(sin(DIAGONAL_STRIPE_ANGLE), cos(DIAGONAL_STRIPE_ANGLE)),
        );
        let pattern = (rotation * point).x % period;
        let distance = min(pattern, period - pattern) - period * (width / height) / 2.0;
        antialiased_coverage(distance)
    }

    /// The colour of a paint's stops at `offset`, interpolated in its space.
    pub fn stops_color(paint: ScenePaint, offset: f32) -> Vec4f {
        let first = get!(COLOR_STOPS)[paint.first_stop as usize];
        if offset <= first.offset || paint.stop_count < 2u32 {
            return first.color;
        }
        let mut previous = first;
        let mut index = 1u32;
        while index < paint.stop_count {
            let stop_index = paint.first_stop + index;
            let stop = get!(COLOR_STOPS)[stop_index as usize];
            if offset <= stop.offset {
                let span = stop.offset - previous.offset;
                let mut t = 1.0;
                if span > 0.0 {
                    t = (offset - previous.offset) / span;
                }
                return mix(previous.color, stop.color, vec4f(t, t, t, t));
            }
            previous = stop;
            index += 1u32;
        }
        previous.color
    }

    /// A colour in `space`, premultiplied except for hue, as unpremultiplied
    /// sRGB-encoded RGBA, as paints are drawn.
    pub fn paint_space_to_srgba(space: PaintColorSpace, color: Vec4f) -> Vec4f {
        // Legacy gradients interpolate unpremultiplied, as they always have.
        if space == PaintColorSpace::LegacySrgb {
            return srgba_to_linear(color);
        }
        if space == PaintColorSpace::LegacyOklab {
            return oklab_to_linear_srgb(color);
        }
        let alpha = color.w;
        if alpha <= 0.0 {
            return transparent();
        }
        let mut components = color.xyz();
        if space == PaintColorSpace::Oklch {
            let lightness = components.x / alpha;
            let chroma = components.y / alpha;
            let hue = components.z * PI / HALF_TURN_DEGREES;
            components = vec3f(lightness, chroma * cos(hue), chroma * sin(hue));
        } else {
            components = components / alpha;
        }
        if space == PaintColorSpace::Srgb {
            return vec4f(components.x, components.y, components.z, alpha);
        }
        let mut linear = components;
        if space == PaintColorSpace::Oklab || space == PaintColorSpace::Oklch {
            let cone_root = OKLAB_TO_CONE_RESPONSE * components;
            linear = CONE_RESPONSE_TO_LINEAR_SRGB * (cone_root * cone_root * cone_root);
        }
        let encoded = linear_to_srgb(max(linear, vec3f(0.0, 0.0, 0.0)));
        vec4f(encoded.x, encoded.y, encoded.z, alpha)
    }

    /// The colour of paint-table entry `index` at a viewport position.
    pub fn table_paint_color(index: u32, viewport_position: Vec2f) -> Vec4f {
        let paint = get!(PAINTS)[index as usize];
        let point =
            TransformationMatrix::transform_position(paint.transformation, viewport_position);
        if paint.kind == PaintKind::Stripes || paint.kind == PaintKind::Checkerboard {
            let mut color = paint_space_to_srgba(
                PaintColorSpace::Srgb,
                get!(COLOR_STOPS)[paint.first_stop as usize].color,
            );
            if paint.kind == PaintKind::Stripes {
                color.w *= stripes_coverage(point, paint.geometry.x, paint.geometry.y);
            } else {
                let square = paint.geometry.x;
                color.w *= saturate((floor(point.x / square) + floor(point.y / square)) % 2.0);
            }
            return color;
        }
        let offset = gradient_offset(paint, point);
        if offset.y == 0.0 {
            return transparent();
        }
        let color = stops_color(paint, extend_offset(paint.extend, offset.x));
        paint_space_to_srgba(paint.color_space, color) + gradient_dither(viewport_position)
    }

    /// How many clips a primitive's chain may hold; deeper nesting is clipped
    /// to its innermost clips.
    pub const MAX_CLIP_CHAIN: u32 = 4;

    /// The transformation of a transform-table entry.
    pub fn scene_transformation(transform: u32) -> TransformationMatrix {
        get!(TRANSFORMS)[transform as usize].transformation
    }

    /// A viewport position in the space of a transform-table entry.
    pub fn local_position(transform: u32, viewport_position: Vec2f) -> Vec2f {
        if transform == 0u32 {
            return viewport_position;
        }
        TransformationMatrix::transform_position(
            get!(TRANSFORMS)[transform as usize].inverse,
            viewport_position,
        )
    }

    /// The antialiased coverage of a clip chain at a viewport position: 1
    /// inside every clip of the chain, 0 outside any.
    pub fn clip_coverage(clip: u32, viewport_position: Vec2f) -> f32 {
        let mut coverage = 1.0;
        let mut index = clip;
        let mut depth = 0u32;
        while index != 0u32 && depth < MAX_CLIP_CHAIN {
            let entry = get!(CLIPS)[index as usize];
            let point = local_position(entry.transform, viewport_position);
            coverage *= antialiased_coverage(rounded_rectangle_signed_distance(
                point,
                entry.bounds,
                entry.corner_radii,
            ));
            index = entry.parent;
            depth += 1u32;
        }
        coverage
    }

    #[derive(Clone, Copy, Wgsl)]
    pub struct RectangleVertex {
        pub unit_position: Vec2f,
        pub viewport_position: Vec2f,
        pub clip_position: Vec4f,
    }

    #[derive(Clone, Copy, Wgsl)]
    pub struct FullscreenVertex {
        pub texture_coordinates: Vec2f,
        pub clip_position: Vec4f,
    }

    pub fn is_enabled(value: ShaderBool) -> bool {
        value == ShaderBool::Enabled
    }

    pub fn transparent() -> Vec4f {
        vec4f(0.0, 0.0, 0.0, 0.0)
    }

    pub fn antialiased_coverage(signed_distance: f32) -> f32 {
        saturate(PIXEL_ANTIALIAS_RADIUS - signed_distance)
    }

    pub fn premultiply(color: Vec4f, coverage: f32) -> Vec4f {
        let alpha = color.w * coverage;
        vec4f(color.x * alpha, color.y * alpha, color.z * alpha, alpha)
    }

    pub fn atlas_texture_coordinates(
        unit_position: Vec2f,
        tile: AtlasTile,
        atlas_size: Vec2u,
    ) -> Vec2f {
        let tile_origin = vec2f(tile.bounds.origin.x as f32, tile.bounds.origin.y as f32);
        let tile_size = vec2f(tile.bounds.size.x as f32, tile.bounds.size.y as f32);
        let texture_size = vec2f(atlas_size.x as f32, atlas_size.y as f32);
        (tile_origin + unit_position * tile_size) / texture_size
    }

    pub fn color_brightness(color: Vec3f) -> f32 {
        dot(color, REC_601_LUMA_WEIGHTS)
    }
    pub fn light_on_dark_contrast(enhanced_contrast: f32, color: Vec3f) -> f32 {
        let darkness = saturate(
            DARK_TEXT_CONTRAST_SCALE * (DARK_TEXT_BRIGHTNESS_CUTOFF - color_brightness(color)),
        );
        enhanced_contrast * darkness
    }

    pub fn enhance_contrast<T>(alpha: T, contrast: f32) -> T
    where
        T: Copy
            + std::ops::Mul<f32, Output = T>
            + std::ops::Add<f32, Output = T>
            + std::ops::Div<T, Output = T>,
    {
        alpha * (contrast + 1.0) / (alpha * contrast + 1.0)
    }

    pub fn apply_alpha_correction<T>(alpha: T, brightness: T, gamma_ratios: Vec4f) -> T
    where
        T: Copy
            + std::ops::Mul<f32, Output = T>
            + std::ops::Mul<T, Output = T>
            + std::ops::Add<f32, Output = T>
            + std::ops::Add<T, Output = T>,
        f32: std::ops::Sub<T, Output = T>,
    {
        let brightness_adjustment = brightness * gamma_ratios.x + gamma_ratios.y;
        let correction =
            brightness_adjustment * alpha + (brightness * gamma_ratios.z + gamma_ratios.w);
        alpha + alpha * (1.0 - alpha) * correction
    }
    pub fn apply_contrast_and_gamma_correction(
        sample: f32,
        color: Vec3f,
        enhanced_contrast_factor: f32,
        gamma_ratios: Vec4f,
    ) -> f32 {
        let enhanced_contrast = light_on_dark_contrast(enhanced_contrast_factor, color);
        apply_alpha_correction::<f32>(
            enhance_contrast::<f32>(sample, enhanced_contrast),
            color_brightness(color),
            gamma_ratios,
        )
    }
    pub fn apply_contrast_and_gamma_correction3(
        sample: Vec3f,
        color: Vec3f,
        enhanced_contrast_factor: f32,
        gamma_ratios: Vec4f,
    ) -> Vec3f {
        let contrasted = enhance_contrast::<Vec3f>(
            sample,
            light_on_dark_contrast(enhanced_contrast_factor, color),
        );
        apply_alpha_correction::<Vec3f>(contrasted, color, gamma_ratios)
    }
    pub fn rectangle_corner(vertex_id: u32) -> Vec2f {
        vec2f((vertex_id & 1u32) as f32, 0.5 * (vertex_id & 2u32) as f32)
    }

    pub fn viewport_to_clip_position(position: Vec2f) -> Vec4f {
        let globals = get!(GLOBALS);
        let clip_position = (position - globals.target_origin) / globals.target_size
            * vec2f(2.0, -2.0)
            + vec2f(-1.0, 1.0);
        vec4f(clip_position.x, clip_position.y, 0.0, 1.0)
    }

    /// The viewport position of a fragment, from its position in the render
    /// target, which may be a group's target placed anywhere in the viewport.
    pub fn scene_position(fragment_position: Vec2f) -> Vec2f {
        fragment_position + get!(GLOBALS).target_origin
    }

    pub fn rectangle_vertex(vertex_id: u32, bounds: Bounds) -> RectangleVertex {
        let unit_position = rectangle_corner(vertex_id);
        let viewport_position = Bounds::position(bounds, unit_position);
        RectangleVertex {
            unit_position,
            viewport_position,
            clip_position: viewport_to_clip_position(viewport_position),
        }
    }

    pub fn transformed_rectangle_vertex(
        vertex_id: u32,
        bounds: Bounds,
        transform: TransformationMatrix,
    ) -> RectangleVertex {
        let unit_position = rectangle_corner(vertex_id);
        let viewport_position = TransformationMatrix::transform_position(
            transform,
            Bounds::position(bounds, unit_position),
        );
        RectangleVertex {
            unit_position,
            viewport_position,
            clip_position: viewport_to_clip_position(viewport_position),
        }
    }

    /// The transformation that applies `inner`, then `outer`.
    pub fn compose_transformations(
        outer: TransformationMatrix,
        inner: TransformationMatrix,
    ) -> TransformationMatrix {
        // `rotation_scale` is stored row-major and applied transposed, so the
        // stored product of the composition runs inner, then outer.
        TransformationMatrix {
            rotation_scale: inner.rotation_scale * outer.rotation_scale,
            translation: TransformationMatrix::transform_position(outer, inner.translation),
        }
    }

    pub fn fullscreen_vertex(vertex_id: u32) -> FullscreenVertex {
        let texture_coordinates = vec2f(
            ((vertex_id << 1u32) & 2u32) as f32,
            (vertex_id & 2u32) as f32,
        );
        FullscreenVertex {
            texture_coordinates,
            clip_position: vec4f(
                texture_coordinates.x * 2.0 - 1.0,
                1.0 - texture_coordinates.y * 2.0,
                0.0,
                1.0,
            ),
        }
    }

    pub fn clip_distances(position: Vec2f, clip_bounds: Bounds) -> Vec4f {
        let distance_from_top_left = position - clip_bounds.origin;
        let distance_from_bottom_right = clip_bounds.origin + clip_bounds.size - position;
        vec4f(
            distance_from_top_left.x,
            distance_from_bottom_right.x,
            distance_from_top_left.y,
            distance_from_bottom_right.y,
        )
    }

    pub fn is_clipped(distances: Vec4f) -> bool {
        min(min(distances.x, distances.y), min(distances.z, distances.w)) < 0.0
    }

    pub fn unclipped_distances() -> Vec4f {
        vec4f(1.0, 1.0, 1.0, 1.0)
    }
    pub fn srgb_to_linear(srgb: Vec3f) -> Vec3f {
        let cutoff = vec3b(
            srgb.x < SRGB_DECODE_CUTOFF,
            srgb.y < SRGB_DECODE_CUTOFF,
            srgb.z < SRGB_DECODE_CUTOFF,
        );
        let higher = pow(
            (srgb + SRGB_TRANSFER_OFFSET) / SRGB_TRANSFER_SCALE,
            vec3f(
                SRGB_DECODE_EXPONENT,
                SRGB_DECODE_EXPONENT,
                SRGB_DECODE_EXPONENT,
            ),
        );
        select(higher, srgb / SRGB_LINEAR_SCALE, cutoff)
    }
    pub fn linear_to_srgb(linear: Vec3f) -> Vec3f {
        let cutoff = vec3b(
            linear.x < SRGB_ENCODE_CUTOFF,
            linear.y < SRGB_ENCODE_CUTOFF,
            linear.z < SRGB_ENCODE_CUTOFF,
        );
        let inverse_exponent = 1.0 / SRGB_DECODE_EXPONENT;
        let higher = SRGB_TRANSFER_SCALE
            * pow(
                linear,
                vec3f(inverse_exponent, inverse_exponent, inverse_exponent),
            )
            - SRGB_TRANSFER_OFFSET;
        select(higher, linear * SRGB_LINEAR_SCALE, cutoff)
    }
    pub fn linear_to_srgba(color: Vec4f) -> Vec4f {
        let red_green_blue = linear_to_srgb(color.rgb());
        vec4f(
            red_green_blue.x,
            red_green_blue.y,
            red_green_blue.z,
            color.w,
        )
    }
    pub fn srgba_to_linear(color: Vec4f) -> Vec4f {
        let red_green_blue = srgb_to_linear(color.rgb());
        vec4f(
            red_green_blue.x,
            red_green_blue.y,
            red_green_blue.z,
            color.w,
        )
    }
    pub fn hue_to_unit_rgb(hue: f32) -> Vec3f {
        let phases = fract(vec3f(hue, hue + 2.0 / 3.0, hue + 1.0 / 3.0));
        saturate(abs(phases * 6.0 - 3.0) - 1.0)
    }

    pub fn hsla_to_rgba(hsla: Hsla) -> Vec4f {
        let chroma = hsla.s * (1.0 - abs(2.0 * hsla.l - 1.0));
        let red_green_blue = (hue_to_unit_rgb(hsla.h) - 0.5) * chroma + hsla.l;
        vec4f(red_green_blue.x, red_green_blue.y, red_green_blue.z, hsla.a)
    }
    pub fn linear_srgb_to_oklab(color: Vec4f) -> Vec4f {
        // Keeps the retired Metal shader's 2.2 transfer curve for pixel parity.
        let linear_srgb = pow(color.rgb(), vec3f(2.2, 2.2, 2.2));
        let cone_response = LINEAR_SRGB_TO_CONE_RESPONSE * linear_srgb;
        let cube_root = vec3f(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
        let oklab = CONE_RESPONSE_TO_OKLAB * pow(cone_response, cube_root);
        vec4f(oklab.x, oklab.y, oklab.z, color.w)
    }
    pub fn oklab_to_linear_srgb(color: Vec4f) -> Vec4f {
        let cone_root = OKLAB_TO_CONE_RESPONSE * color.rgb();
        let cone_response = cone_root * cone_root * cone_root;
        let linear_srgb = CONE_RESPONSE_TO_LINEAR_SRGB * cone_response;
        let srgb = pow(linear_srgb, vec3f(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
        vec4f(srgb.x, srgb.y, srgb.z, color.w)
    }
    pub fn over(below: Vec4f, above: Vec4f) -> Vec4f {
        let alpha = above.w + below.w * (1.0 - above.w);
        if alpha == 0.0 {
            return transparent();
        }
        let red_green_blue =
            (above.rgb() * above.w + below.rgb() * below.w * (1.0 - above.w)) / alpha;
        vec4f(red_green_blue.x, red_green_blue.y, red_green_blue.z, alpha)
    }
    pub fn gaussian(position: f32, standard_deviation: f32) -> f32 {
        let variance = standard_deviation * standard_deviation;
        exp(-(position * position) / (2.0 * variance)) / (sqrt(2.0 * PI) * standard_deviation)
    }
    pub fn approximate_error_function(value: Vec2f) -> Vec2f {
        let signs = sign(value);
        let absolute_value = abs(value);
        let polynomial = 1.0
            + absolute_value
                * (ERROR_FUNCTION_LINEAR_COEFFICIENT
                    + absolute_value
                        * (ERROR_FUNCTION_QUADRATIC_COEFFICIENT
                            + absolute_value
                                * (ERROR_FUNCTION_CUBIC_COEFFICIENT
                                    + absolute_value * ERROR_FUNCTION_QUARTIC_COEFFICIENT)));
        let polynomial_squared = polynomial * polynomial;
        signs - signs / (polynomial_squared * polynomial_squared)
    }
    pub fn integrated_rounded_rectangle_coverage(
        horizontal_position: f32,
        vertical_position: f32,
        standard_deviation: f32,
        corner_radius: f32,
        half_size: Vec2f,
    ) -> f32 {
        let distance_beyond_corner = min(half_size.y - corner_radius - abs(vertical_position), 0.0);
        let curved_extent = half_size.x - corner_radius
            + sqrt(max(
                0.0,
                corner_radius * corner_radius - distance_beyond_corner * distance_beyond_corner,
            ));
        let integration_bounds = horizontal_position + vec2f(-curved_extent, curved_extent);
        let normalized_bounds = integration_bounds * (sqrt(0.5) / standard_deviation);
        let integral = 0.5 + 0.5 * approximate_error_function(normalized_bounds);
        integral.y - integral.x
    }
    pub fn pick_corner_radius(center_to_point: Vec2f, radii: Corners) -> f32 {
        let left = select(radii.bottom_left, radii.top_left, center_to_point.y < 0.0);
        let right = select(radii.bottom_right, radii.top_right, center_to_point.y < 0.0);
        select(right, left, center_to_point.x < 0.0)
    }
    pub fn rounded_rectangle_signed_distance_from_corner(
        corner_center_to_point: Vec2f,
        corner_radius: f32,
    ) -> f32 {
        if corner_radius == 0.0 {
            max(corner_center_to_point.x, corner_center_to_point.y)
        } else {
            length(max(vec2f(0.0, 0.0), corner_center_to_point))
                + min(0.0, max(corner_center_to_point.x, corner_center_to_point.y))
                - corner_radius
        }
    }
    pub fn rounded_rectangle_signed_distance(
        point: Vec2f,
        bounds: Bounds,
        corner_radii: Corners,
    ) -> f32 {
        let half_size = Bounds::half_size(bounds);
        let center_to_point = point - Bounds::center(bounds);
        let corner_radius = pick_corner_radius(center_to_point, corner_radii);
        rounded_rectangle_signed_distance_from_corner(
            abs(center_to_point) - half_size + corner_radius,
            corner_radius,
        )
    }

    pub fn gaussian_signed_distance_coverage(distance: f32, standard_deviation: f32) -> f32 {
        let normalized = distance / (sqrt(2.0) * standard_deviation);
        saturate(0.5 - 0.5 * approximate_error_function(vec2f(normalized, normalized)).x)
    }

    pub fn blend_color(color: Vec4f, alpha_factor: f32) -> Vec4f {
        let alpha = color.w * alpha_factor;
        let multiplier = select(1.0, alpha, is_enabled(get!(GLOBALS).premultiplied_alpha));
        vec4f(
            color.x * multiplier,
            color.y * multiplier,
            color.z * multiplier,
            alpha,
        )
    }

    /// What a primitive paints with: a colour, or, where `paint` is not 0,
    /// that entry of the scene's paint table faded by the colour's alpha.
    #[derive(Clone, Copy, Wgsl)]
    pub struct PaintRef {
        pub color: Hsla,
        pub paint: u32,
    }

    /// The colour a paint reference carries, as the vertex stage passes it
    /// on.
    pub fn prepare_paint(paint: PaintRef) -> Vec4f {
        hsla_to_rgba(paint.color)
    }

    pub fn gradient_dither(position: Vec2f) -> Vec4f {
        let seed = position * GRADIENT_DITHER_SCALE;
        let noise_a = fract(sin(dot(seed, GRADIENT_DITHER_SEED_A)) * GRADIENT_DITHER_MULTIPLIER_A);
        let noise_b = fract(sin(dot(seed, GRADIENT_DITHER_SEED_B)) * GRADIENT_DITHER_MULTIPLIER_B);
        let triangular_noise = noise_a + noise_b - 1.0;
        vec4f(
            triangular_noise * GRADIENT_DITHER_RGB_STRENGTH,
            triangular_noise * GRADIENT_DITHER_RGB_STRENGTH,
            triangular_noise * GRADIENT_DITHER_RGB_STRENGTH,
            triangular_noise * GRADIENT_DITHER_ALPHA_STRENGTH,
        )
    }

    /// The colour `paint` draws at `viewport_position`, given `solid`, its
    /// prepared colour.
    pub fn paint_color(paint: PaintRef, viewport_position: Vec2f, solid: Vec4f) -> Vec4f {
        if paint.paint == 0u32 {
            return solid;
        }
        let mut color = table_paint_color(paint.paint, viewport_position);
        color.w *= solid.w;
        color
    }
    pub fn corner_dash_velocity(first: f32, second: f32) -> f32 {
        if first == 0.0 {
            second
        } else if second == 0.0 {
            first
        } else {
            min(first, second)
        }
    }
    pub fn signed_modulo(value: f32, modulus: f32) -> f32 {
        value - modulus * trunc(value / modulus)
    }
    pub fn dash_coverage(position: f32, period: f32, dash_length: f32, dash_velocity: f32) -> f32 {
        let half_period = period / 2.0;
        let half_dash_length = dash_length / 2.0;
        let centered =
            signed_modulo(position + half_period - half_dash_length, period) - half_period;
        let signed_distance = abs(centered) - half_dash_length;
        antialiased_coverage(signed_distance / dash_velocity)
    }
    pub fn quarter_ellipse_signed_distance(point: Vec2f, radii: Vec2f) -> f32 {
        (length(point / radii) - 1.0) * (radii.x + radii.y) * -0.5
    }
}

pub use source::*;
