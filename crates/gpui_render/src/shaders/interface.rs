/// Shader-declared data that can be uploaded without serialization.
///
/// # Safety
/// Implementors need a stable C-compatible layout, no padding, matching the WGSL declaration.
pub unsafe trait BufferData: Sized {
    const WGSL_TYPE: &'static str;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StorageAbi {
    pub wgsl_type: &'static str,
    pub rust_stride: usize,
}

pub const fn storage_abi<T: BufferData>() -> StorageAbi {
    StorageAbi {
        wgsl_type: T::WGSL_TYPE,
        rust_stride: std::mem::size_of::<T>(),
    }
}

pub fn bytes_of<T: BufferData>(value: &T) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(value).cast::<u8>(),
            std::mem::size_of::<T>(),
        )
    }
}

pub fn slice_as_bytes<T: BufferData>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DataLayout {
    Instances,
    TexturedInstances,
    MonochromeSprites,
    SubpixelSprites,
    NativeOnly,
    Surface,
    Blur,
    Group,
    /// A frame's mesh instances, and the vertices of the mesh being drawn,
    /// drawn indexed.
    Meshes,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VertexCount {
    Rectangle,
    FullscreenTriangle,
    Dynamic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrimitiveTopology {
    TriangleList,
    TriangleStrip,
}

impl VertexCount {
    pub const fn fixed(self) -> Option<u32> {
        match self {
            Self::Rectangle => Some(RECTANGLE_VERTEX_COUNT),
            Self::FullscreenTriangle => Some(FULLSCREEN_TRIANGLE_VERTEX_COUNT),
            Self::Dynamic => None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Pipeline {
    pub label: &'static str,
    pub vertex_entry: &'static str,
    pub fragment_entry: &'static str,
    pub topology: PrimitiveTopology,
    pub data_layout: DataLayout,
    pub vertex_count: VertexCount,
}

macro_rules! define_pipelines {
        ($($name:ident: $label:literal, $vertex:ident, $fragment:ident, $topology:ident, $layout:ident, $vertices:ident;)*) => {
            $(
                pub const $name: Pipeline = Pipeline {
                    label: $label,
                    vertex_entry: stringify!($vertex),
                    fragment_entry: stringify!($fragment),
                    topology: PrimitiveTopology::$topology,
                    data_layout: DataLayout::$layout,
                    vertex_count: VertexCount::$vertices,
                };
            )*

            pub const ALL: &[Pipeline] = &[$($name),*];
        };
    }

define_pipelines! {
    QUADS: "quads", vertex_quad, fragment_quad, TriangleStrip, Instances, Rectangle;
    SMOOTHED_QUADS: "smoothed_quads", vertex_smoothed_quad, fragment_smoothed_quad, TriangleStrip, Instances, Rectangle;
    SHADOWS: "shadows", vertex_shadow, fragment_shadow, TriangleStrip, Instances, Rectangle;
    SMOOTHED_SHADOWS: "smoothed_shadows", vertex_smoothed_shadow, fragment_smoothed_shadow, TriangleStrip, Instances, Rectangle;
    PATH_RASTERIZATION: "path_rasterization", vertex_path_rasterization, fragment_path_rasterization, TriangleList, Instances, Dynamic;
    PATHS: "paths", vertex_path, fragment_path, TriangleStrip, TexturedInstances, Rectangle;
    MESHES: "meshes", vertex_mesh, fragment_mesh, TriangleList, Meshes, Dynamic;
    UNDERLINES: "underlines", vertex_underline, fragment_underline, TriangleStrip, Instances, Rectangle;
    MONOCHROME_SPRITES: "monochrome_sprites", vertex_monochrome_sprite, fragment_monochrome_sprite, TriangleStrip, MonochromeSprites, Rectangle;
    SUBPIXEL_SPRITES: "subpixel_sprites", vertex_subpixel_sprite, fragment_subpixel_sprite, TriangleStrip, SubpixelSprites, Rectangle;
    POLYCHROME_SPRITES: "polychrome_sprites", vertex_polychrome_sprite, fragment_polychrome_sprite, TriangleStrip, TexturedInstances, Rectangle;
    SMOOTHED_POLYCHROME_SPRITES: "smoothed_polychrome_sprites", vertex_smoothed_polychrome_sprite, fragment_smoothed_polychrome_sprite, TriangleStrip, TexturedInstances, Rectangle;
    SURFACES: "surfaces", vertex_surface, fragment_surface, TriangleStrip, Surface, Rectangle;
    BLUR_DOWNSAMPLE: "blur_downsample", vertex_blur_fullscreen, fragment_blur_downsample, TriangleList, Blur, FullscreenTriangle;
    BLUR: "blur", vertex_blur_fullscreen, fragment_blur, TriangleList, Blur, FullscreenTriangle;
    BLUR_COMPOSITE: "blur_composite", vertex_blur_composite, fragment_blur_composite, TriangleStrip, Blur, Rectangle;
    SMOOTHED_BLUR_COMPOSITE: "smoothed_blur_composite", vertex_smoothed_blur_composite, fragment_smoothed_blur_composite, TriangleStrip, Blur, Rectangle;
    GROUP_COMPOSITE: "group_composite", vertex_group_composite, fragment_group_composite, TriangleStrip, Group, Rectangle;
    GROUP_FILTER: "group_filter", vertex_group_composite, fragment_group_filter, TriangleStrip, Group, Rectangle;
}

pub const EMOJI_RASTERIZATION: Pipeline = Pipeline {
    label: "emoji_rasterization",
    vertex_entry: "vertex_emoji_rasterization",
    fragment_entry: "fragment_emoji_rasterization",
    topology: PrimitiveTopology::TriangleStrip,
    data_layout: DataLayout::NativeOnly,
    vertex_count: VertexCount::Rectangle,
};

pub const GLOBAL_BIND_GROUP: u32 = 0;
pub const DATA_BIND_GROUP: u32 = 1;
pub const GLOBAL_UNIFORMS_BINDING: u32 = 0;
pub const FONT_RASTERIZATION_BINDING: u32 = 1;
pub const DATA_BUFFER_BINDING: u32 = 0;
pub const PRIMARY_TEXTURE_BINDING: u32 = 1;
pub const SECONDARY_TEXTURE_BINDING: u32 = 2;
pub const PRIMARY_SAMPLER_BINDING: u32 = 2;
pub const SURFACE_SAMPLER_BINDING: u32 = 3;
/// Group-1 storage array of the vertices of the mesh being drawn, beside the
/// frame's mesh instances at [`DATA_BUFFER_BINDING`].
pub const MESH_VERTICES_BINDING: u32 = 1;
/// Group-0 storage table of the scene's transforms, shared by every pipeline.
pub const TRANSFORMS_BINDING: u32 = 2;
/// Group-0 storage table of the scene's clips, shared by every pipeline.
pub const CLIPS_BINDING: u32 = 3;
/// Group-0 storage table of the scene's paints and their colour stops, as `vec4<f32>`
/// words, shared by every pipeline.
pub const PAINTS_BINDING: u32 = 4;
/// Group-0 texture array of the window's resident photo tiles, shared by every pipeline.
pub const PHOTO_TILES_BINDING: u32 = 5;
/// Group-0 sampler photo tiles are filtered with: linear, clamped to their edges.
pub const PHOTO_SAMPLER_BINDING: u32 = 6;
/// How many bindings group 0 declares. Native backends have one flat slot space per
/// resource class, so group 0 takes the first slots and group 1 follows it.
pub const GLOBAL_BINDING_COUNT: u32 = 7;
pub const RECTANGLE_VERTEX_COUNT: u32 = 4;
pub const FULLSCREEN_TRIANGLE_VERTEX_COUNT: u32 = 3;
/// D3D11 constant-buffer register of the per-draw instance base for instanced pipelines.
/// Group-0 cbuffers occupy b0 and b1, and the group-1 uniform lands on b7, after group 0.
pub const DX11_DRAW_CONSTANTS_REGISTER: u32 = 3;
/// Metal buffer index of the runtime-array sizes Naga's MSL declares, after every buffer slot:
/// the data buffer, and the mesh vertices beside it.
pub const MSL_BUFFER_SIZES_SLOT: u32 = native_slot(DATA_BIND_GROUP, MESH_VERTICES_BINDING) + 1;
/// Bytes a renderer binds at [`MSL_BUFFER_SIZES_SLOT`]. The generated MSL declares one `uint`
/// per runtime-sized array it can reach, but never reads them: array accesses are unchecked,
/// which the build asserts along with this bound.
pub const MSL_BUFFER_SIZES_BYTES: u32 = 32;

/// The native slot of a WGSL binding: its HLSL register (`b`, `t` or `s`) and, for
/// buffers, its Metal buffer index.
pub const fn native_slot(group: u32, binding: u32) -> u32 {
    if group == GLOBAL_BIND_GROUP {
        binding
    } else {
        GLOBAL_BINDING_COUNT + binding
    }
}

macro_rules! buffer_data {
        ($($rust:ty => $wgsl:literal),* $(,)?) => {
            $(
                unsafe impl BufferData for $rust {
                    const WGSL_TYPE: &'static str = $wgsl;
                }
            )*
        };
    }

buffer_data! {
    super::common::GlobalUniforms => "GlobalUniforms",
    super::common::FontRasterizationUniforms => "FontRasterizationUniforms",
    super::surface::SurfaceUniforms => "SurfaceUniforms",
    super::blur::BlurUniforms => "BlurUniforms",
    super::group::GroupUniforms => "GroupUniforms",
    gpui::Quad => "Quad",
    gpui::Shadow => "Shadow",
    gpui::Underline => "Underline",
    gpui::MonochromeSprite => "MonochromeSprite",
    gpui::SubpixelSprite => "SubpixelSprite",
    gpui::PolychromeSprite => "PolychromeSprite",
    gpui::SceneTransform => "SceneTransform",
    gpui::SceneClip => "SceneClip",
    gpui::PaintWord => "PaintWord",
    gpui::MeshInstance => "MeshInstance",
    gpui::MeshVertex => "MeshVertex",
}

pub const SCENE_STORAGE_ABI: &[StorageAbi] = &[
    storage_abi::<gpui::Quad>(),
    storage_abi::<gpui::Shadow>(),
    storage_abi::<gpui::Underline>(),
    storage_abi::<gpui::MonochromeSprite>(),
    storage_abi::<gpui::SubpixelSprite>(),
    storage_abi::<gpui::PolychromeSprite>(),
    storage_abi::<gpui::SceneTransform>(),
    storage_abi::<gpui::SceneClip>(),
    storage_abi::<gpui::PaintWord>(),
    storage_abi::<gpui::MeshInstance>(),
    storage_abi::<gpui::MeshVertex>(),
];

macro_rules! render_layout {
    ($ty:ty, $name:literal, $($field:ident),+ $(,)?) => {
        gpui::SceneBufferLayout {
            name: $name,
            size: std::mem::size_of::<$ty>(),
            fields: &[$((stringify!($field), std::mem::offset_of!($ty, $field))),+],
        }
    };
}

#[doc(hidden)]
pub const RENDER_BUFFER_LAYOUTS: &[gpui::SceneBufferLayout] = &[
    render_layout!(
        super::common::GlobalUniforms,
        "GlobalUniforms",
        viewport_size,
        target_origin,
        target_size,
        premultiplied_alpha,
        padding,
        placement,
        inverse_placement,
        placement_translation,
        inverse_placement_translation
    ),
    render_layout!(
        super::common::FontRasterizationUniforms,
        "FontRasterizationUniforms",
        gamma_ratios,
        grayscale_enhanced_contrast,
        subpixel_enhanced_contrast,
        uses_blue_green_red_subpixel_order,
        padding
    ),
    render_layout!(
        super::surface::SurfaceUniforms,
        "SurfaceUniforms",
        bounds,
        content_mask,
        color_format,
        opacity,
        padding0,
        padding1,
        padding2,
        padding3,
        padding4,
        padding5
    ),
    render_layout!(
        super::blur::BlurUniforms,
        "BlurUniforms",
        bounds,
        content_mask,
        corner_radii,
        direction,
        standard_deviation,
        opacity,
        sample_count,
        sample_step,
        composite_clip,
        downsample_mode,
        source_size,
        target_size,
        corner_smoothing,
        padding0,
        source_origin
    ),
    render_layout!(
        super::group::GroupUniforms,
        "GroupUniforms",
        bounds,
        content_mask,
        source_origin,
        source_size,
        backdrop_origin,
        backdrop_size,
        mask_origin,
        mask_size,
        opacity,
        blend_mode,
        mask,
        padding,
        filter_kind,
        filter_linear,
        filter_paint,
        filter_padding,
        filter_offset,
        input_translation,
        input_matrix,
        matrix_red,
        matrix_green,
        matrix_blue,
        matrix_alpha,
        matrix_offset
    ),
    render_layout!(crate::path_types::PathSprite, "PathSprite", bounds),
    render_layout!(
        crate::path_types::PathRasterizationVertex,
        "PathRasterizationVertex",
        xy_position,
        curve_position,
        color,
        padding,
        bounds
    ),
];
