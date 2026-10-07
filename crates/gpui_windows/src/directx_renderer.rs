use std::{
    slice,
    sync::{Arc, OnceLock},
};

use anyhow::{Context, Result};
use collections::FxHashMap;
use gpui_render::{
    InstanceRange,
    artifacts::{Dx11DrawConstants, Dx11DrawConstantsBinding},
    blur::{
        BlurAxis, BlurKernel, BlurUniforms, FilterCompositeClip,
        GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS, ScissorRectangle, downsampled_dimension,
    },
    group::{GroupUniforms, group_target_bounds},
    path_types::{PathRasterizationVertex, PathSprite},
    shaders::{
        common::{FontRasterizationUniforms, GlobalUniforms, ShaderBool, SurfaceColorFormat},
        interface as shader_interface,
        surface::SurfaceUniforms,
    },
};
use gpui_util::ResultExt;
use smallvec::SmallVec;
use wgsl_rs::std::{vec2f, vec4f};
use windows::{
    Win32::{
        Foundation::{FreeLibrary, HMODULE, HWND, RECT},
        Graphics::{
            Direct3D::*,
            Direct3D11::*,
            DirectComposition::*,
            DirectWrite::*,
            Dxgi::{Common::*, *},
        },
        System::LibraryLoader::LoadLibraryA,
    },
    core::{HSTRING, Interface, PCSTR},
};
use windows_061::core::Interface as _;

use crate::directx_renderer::shader_resources::ShaderModule;
use crate::*;
use gpui::*;

pub(crate) const DISABLE_DIRECT_COMPOSITION: &str = "GPUI_DISABLE_DIRECT_COMPOSITION";
const RENDER_TARGET_FORMAT: DXGI_FORMAT = DXGI_FORMAT_B8G8R8A8_UNORM;
// This configuration is used for MSAA rendering on paths only, and it's guaranteed to be supported by DirectX 11.
const PATH_MULTISAMPLE_COUNT: u32 = 4;
const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;
// Native shaders number group 0's registers first and group 1's after them.
const fn global_register(binding: u32) -> u32 {
    shader_interface::native_slot(shader_interface::GLOBAL_BIND_GROUP, binding)
}
const fn data_register(binding: u32) -> u32 {
    shader_interface::native_slot(shader_interface::DATA_BIND_GROUP, binding)
}
/// The scene's transform, clip and paint tables, bound together from this register in
/// that order.
const SCENE_TABLES_REGISTER: u32 = global_register(shader_interface::TRANSFORMS_BINDING);
const _: () = assert!(
    global_register(shader_interface::CLIPS_BINDING) == SCENE_TABLES_REGISTER + 1
        && global_register(shader_interface::PAINTS_BINDING) == SCENE_TABLES_REGISTER + 2,
    "the clip and paint tables must follow the transform table"
);
const _: () = assert!(
    SCENE_TABLES_REGISTER + 3 <= DATA_REGISTER,
    "the scene tables must not reach group 1's registers"
);
const DATA_REGISTER: u32 = data_register(shader_interface::DATA_BUFFER_BINDING);
/// The vertices of the mesh being drawn, beside the frame's mesh instances.
const MESH_VERTICES_REGISTER: u32 = data_register(shader_interface::MESH_VERTICES_BINDING);
/// The window's photo tile array, a texture beside the scene tables, and the sampler that
/// filters it, which group 1's samplers follow.
const PHOTO_TILES_REGISTER: u32 = global_register(shader_interface::PHOTO_TILES_BINDING);
const PHOTO_SAMPLER_REGISTER: u32 = global_register(shader_interface::PHOTO_SAMPLER_BINDING);
const _: () = assert!(
    (PHOTO_TILES_REGISTER < SCENE_TABLES_REGISTER
        || PHOTO_TILES_REGISTER >= SCENE_TABLES_REGISTER + 3)
        && PHOTO_TILES_REGISTER < DATA_REGISTER
        && PHOTO_SAMPLER_REGISTER < data_register(0),
    "the photo tiles and their sampler must sit apart from the scene tables, below group 1"
);
const PRIMARY_TEXTURE_REGISTER: u32 = data_register(shader_interface::PRIMARY_TEXTURE_BINDING);
const PRIMARY_SAMPLER_REGISTER: u32 = data_register(shader_interface::PRIMARY_SAMPLER_BINDING);
const SURFACE_SAMPLER_REGISTER: u32 = data_register(shader_interface::SURFACE_SAMPLER_BINDING);
/// The group composite's textures, its group's and the backdrop's, and its sampler: bindings
/// 1, 2 and 3 of `GROUP_TEXTURE`, `BACKDROP_TEXTURE` and `GROUP_SAMPLER` in `shaders/groups.rs`.
const GROUP_TEXTURE_REGISTER: u32 = data_register(shader_interface::PRIMARY_TEXTURE_BINDING);
const _: () = assert!(
    data_register(shader_interface::SECONDARY_TEXTURE_BINDING) == GROUP_TEXTURE_REGISTER + 1,
    "the backdrop texture must follow the group texture"
);
const GROUP_SAMPLER_REGISTER: u32 = data_register(3);
/// The group composite's mask texture: binding 4 of `MASK_TEXTURE` in `shaders/groups.rs`.
const MASK_TEXTURE_REGISTER: u32 = data_register(4);

pub(crate) struct FontInfo {
    pub gamma_ratios: [f32; 4],
    pub grayscale_enhanced_contrast: f32,
    pub subpixel_enhanced_contrast: f32,
    pub is_bgr: bool,
}

pub(crate) struct DirectXRenderer {
    hwnd: HWND,
    atlas: Arc<DirectXAtlas>,
    devices: Option<DirectXRendererDevices>,
    resources: Option<DirectXResources>,
    globals: DirectXGlobalElements,
    pipelines: DirectXRenderPipelines,
    /// The shader programs linked into the shaders that read paints.
    programs: gpui_render::linked::LinkedPrograms<crate::directx_programs::LinkedShaders>,
    /// The shaders the frame being drawn draws paints with, when its scene
    /// runs programs: `None` for the pipelines' own.
    frame_programs: Option<Arc<crate::directx_programs::LinkedShaders>>,
    /// Whether the last frame drew programs still being linked.
    programs_pending: bool,
    direct_composition: Option<DirectComposition>,
    font_info: &'static FontInfo,

    width: u32,
    height: u32,

    /// Whether we want to skip drawing due to device lost events.
    ///
    /// In that case we want to discard the first frame that we draw as we got reset in the middle of a frame
    /// meaning we lost all the allocated gpu textures and scene resources.
    skip_draws: bool,

    /// The targets this frame is drawing into: the window's (the offscreen `scene_color` when
    /// the scene reads back what it has painted, the swapchain's buffer otherwise), then one for
    /// each isolated group being drawn. Batches draw into the last.
    targets: Vec<FrameTarget>,
    path_rasterization_vertices: Vec<PathRasterizationVertex>,
    path_sprites: Vec<PathSprite>,
    /// The meshes kept on the GPU, by id.
    meshes: gpui_render::meshes::MeshCache<DirectXMesh>,

    /// The scene whose commands are being drawn: the window's, or a chunk inside it.
    level: DrawLevel,
    /// Where each chunk scene drawn this frame keeps its data in the frame's buffers, by the
    /// address of its scene, which the frame's scene keeps alive.
    chunk_slots: FxHashMap<*const Scene, SceneSlot>,
}

/// Where a scene drawn this frame, the window's or a chunk's, keeps its data in the frame's
/// buffers: each scene's instances and tables follow the window's.
#[derive(Clone, Default)]
struct SceneSlot {
    /// Where its instances start in each batch pipeline's buffer.
    shadows: u32,
    quads: u32,
    underlines: u32,
    monochrome_sprites: u32,
    subpixel_sprites: u32,
    polychrome_sprites: u32,
    meshes: u32,
    /// Its transform, clip and paint tables, for registers from [`SCENE_TABLES_REGISTER`].
    tables: [Option<ID3D11ShaderResourceView>; 3],
}

/// The scene whose commands are being drawn, and how.
#[derive(Clone, Default)]
struct DrawLevel {
    slot: SceneSlot,
    /// For a chunk: from its viewport to the window's, in device pixels, and the window
    /// viewport rectangle it is clipped to.
    chunk: Option<(TransformationMatrix, Bounds<ScaledPixels>)>,
}

/// Direct3D objects
#[derive(Clone)]
pub(crate) struct DirectXRendererDevices {
    pub(crate) adapter: IDXGIAdapter1,
    pub(crate) dxgi_factory: IDXGIFactory6,
    pub(crate) device: ID3D11Device,
    pub(crate) device_context: ID3D11DeviceContext,
    dxgi_device: Option<IDXGIDevice>,
    annotation: Option<ID3DUserDefinedAnnotation>,
}

struct DirectXResources {
    // Direct3D rendering objects
    swap_chain: IDXGISwapChain1,
    render_target: Option<ID3D11Texture2D>,
    render_target_view: Option<ID3D11RenderTargetView>,

    // Path intermediates are absent until a scene contains a path batch.
    path: Option<PathResources>,

    // The window's offscreen target, absent until a scene reads back what it has painted:
    // backdrop filters blur it, and blend modes mix groups with it.
    scene_color: Option<ColorTarget>,

    // Targets for isolated groups, the blurs of groups and backdrops, and copies of what is
    // beneath a blended group.
    target_pool: TexturePool,

    // Views for capture textures that are referenced by the current scene. Keeping the
    // underlying texture alive makes its COM pointer a stable cache key.
    surface_views: FxHashMap<usize, CachedSurfaceView>,

    // Cached viewport
    viewport: D3D11_VIEWPORT,

    // The rasterizer state for everything but chunks, and the one chunks are drawn with,
    // which cuts what they draw to their scissor rectangle.
    rasterizer_state: ID3D11RasterizerState,
    scissored_rasterizer_state: ID3D11RasterizerState,
}

struct CachedSurfaceView {
    #[expect(dead_code)]
    texture: ID3D11Texture2D,
    srv: Option<ID3D11ShaderResourceView>,
}

struct PathResources {
    texture: ID3D11Texture2D,
    srv: Option<ID3D11ShaderResourceView>,
    msaa_texture: ID3D11Texture2D,
    msaa_view: Option<ID3D11RenderTargetView>,
}

impl PathResources {
    fn new(device: &ID3D11Device, width: u32, height: u32) -> Result<Self> {
        let (texture, srv) = create_path_intermediate_texture(device, width, height)?;
        let (msaa_texture, msaa_view) =
            create_path_intermediate_msaa_texture_and_view(device, width, height)?;
        Ok(Self {
            texture,
            srv,
            msaa_texture,
            msaa_view,
        })
    }
}

/// A texture a frame draws into: the window's, or an isolated group's.
struct FrameTarget {
    color: ColorTarget,
    /// The viewport rectangle the texture covers.
    bounds: Bounds<DevicePixels>,
    /// The viewport covering the whole texture, set by every draw into it.
    viewport: D3D11_VIEWPORT,
    /// A masked group's mask, once drawn: its target, the viewport rectangle it covers, and
    /// how it masks.
    mask: Option<(ColorTarget, Bounds<DevicePixels>, MaskMode)>,
}

impl FrameTarget {
    fn new(color: ColorTarget, origin: Point<DevicePixels>) -> Self {
        let viewport = D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: color.size.width.0 as f32,
            Height: color.size.height.0 as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        Self {
            bounds: Bounds {
                origin,
                size: color.size,
            },
            color,
            viewport,
            mask: None,
        }
    }
}

struct DirectXRenderPipelines {
    shadow_pipeline: PipelineState<Shadow>,
    quad_pipeline: PipelineState<Quad>,
    path_rasterization_pipeline: PipelineState<PathRasterizationVertex>,
    path_sprite_pipeline: PipelineState<PathSprite>,
    underline_pipeline: PipelineState<Underline>,
    mono_sprites: PipelineState<MonochromeSprite>,
    subpixel_sprites: PipelineState<SubpixelSprite>,
    poly_sprites: PipelineState<PolychromeSprite>,
    meshes: PipelineState<MeshInstance>,
    surfaces: SurfacePipeline,
    // Blur: not the generic PipelineState, since these sample a texture instead of
    // reading a structured instance buffer; parameters live in a cbuffer at [`DATA_REGISTER`].
    blur_downsample_vertex: ID3D11VertexShader,
    blur_downsample_fragment: ID3D11PixelShader,
    blur_vertex: ID3D11VertexShader,
    blur_fragment: ID3D11PixelShader,
    blur_composite_vertex: ID3D11VertexShader,
    blur_composite_fragment: ID3D11PixelShader,
    smoothed_blur_composite_vertex: ID3D11VertexShader,
    smoothed_blur_composite_fragment: ID3D11PixelShader,
    blur_params_buffer: ID3D11Buffer,
    blur_blend_replace: ID3D11BlendState,
    blur_blend_composite: ID3D11BlendState,
    group_composite: GroupCompositePipeline,
}

/// The generated `group_composite` pipeline: one draw per isolated group, compositing its
/// target into its parent's (premultiplied).
struct GroupCompositePipeline {
    vertex: ID3D11VertexShader,
    fragment: ID3D11PixelShader,
    params_buffer: ID3D11Buffer,
    blend: ID3D11BlendState,
}

/// The generated `surfaces` pipeline: one draw per surface, per-draw uniforms.
struct SurfacePipeline {
    vertex: ID3D11VertexShader,
    fragment: ID3D11PixelShader,
    params_buffer: ID3D11Buffer,
    blend: ID3D11BlendState,
}

struct DirectXGlobalElements {
    globals_buffer: Option<ID3D11Buffer>,
    font_buffer: Option<ID3D11Buffer>,
    /// Per-draw [`Dx11DrawConstants`]; rewritten before every instanced draw.
    draw_constants_buffer: ID3D11Buffer,
    sampler: Option<ID3D11SamplerState>,
    transforms: SceneTableBuffer<SceneTransform>,
    clips: SceneTableBuffer<SceneClip>,
    paints: SceneTableBuffer<PaintWord>,
    /// The window's resident photo tiles, which every batch pipeline's paints may sample.
    photo_tiles: PhotoTiles,
}

impl DirectXGlobalElements {
    /// Global constant buffers at registers b0 (globals) and b1 (font rasterization).
    fn cbuffers(&self) -> [Option<ID3D11Buffer>; 2] {
        [self.globals_buffer.clone(), self.font_buffer.clone()]
    }

    /// The transform, clip and paint tables, for registers from
    /// [`SCENE_TABLES_REGISTER`].
    fn scene_tables(&self) -> [Option<ID3D11ShaderResourceView>; 3] {
        [
            self.transforms.view.clone(),
            self.clips.view.clone(),
            self.paints.view.clone(),
        ]
    }

    /// Binds the photo tile array and its sampler for the pixel stage, where paints are
    /// evaluated.
    ///
    /// # Safety
    ///
    /// `device_context` must belong to the device the tiles were created on.
    unsafe fn bind_photo_tiles(&self, device_context: &ID3D11DeviceContext) {
        unsafe {
            device_context.PSSetShaderResources(
                PHOTO_TILES_REGISTER,
                Some(slice::from_ref(self.photo_tiles.view())),
            );
            device_context.PSSetSamplers(
                PHOTO_SAMPLER_REGISTER,
                Some(slice::from_ref(self.photo_tiles.sampler())),
            );
        }
    }
}

/// A per-frame scene table that every batch pipeline reads in both stages, entry zero first.
struct SceneTableBuffer<T> {
    label: &'static str,
    buffer: ID3D11Buffer,
    view: Option<ID3D11ShaderResourceView>,
    capacity: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T> SceneTableBuffer<T> {
    // Most frames hold only entry zero; transformed UI grows the table geometrically.
    const INITIAL_CAPACITY: usize = 64;

    fn new(device: &ID3D11Device, label: &'static str) -> Result<Self> {
        let buffer = create_buffer(device, std::mem::size_of::<T>(), Self::INITIAL_CAPACITY)?;
        let view = create_buffer_view(device, &buffer)?;
        Ok(Self {
            label,
            buffer,
            view,
            capacity: Self::INITIAL_CAPACITY,
            _marker: std::marker::PhantomData,
        })
    }

    /// Uploads `sections`, one scene's table each, end to end, each starting on a boundary a
    /// view can start at, and returns where each one is.
    fn update(
        &mut self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
        sections: &[&[T]],
    ) -> Result<SmallVec<[TableSection; 4]>> {
        let (offsets, entries) = section_offsets(sections, TABLE_SECTION_ALIGNMENT);
        if self.capacity < entries {
            let element_size = std::mem::size_of::<T>();
            let max_entries = MAX_INSTANCE_BUFFER_SIZE / element_size;
            anyhow::ensure!(
                entries <= max_entries,
                "{} needs {} entries, above the {MAX_INSTANCE_BUFFER_SIZE}-byte limit",
                self.label,
                entries,
            );
            let capacity = entries
                .checked_next_power_of_two()
                .unwrap_or(max_entries)
                .min(max_entries);
            self.buffer = create_buffer(device, element_size, capacity)?;
            self.view = create_buffer_view(device, &self.buffer)?;
            self.capacity = capacity;
        }
        update_buffer_sections(device_context, &self.buffer, sections, &offsets)?;
        Ok(offsets
            .iter()
            .zip(sections)
            .map(|(&offset, section)| TableSection {
                offset,
                len: section.len(),
            })
            .collect())
    }

    /// A view of one scene's table, uploaded by [`Self::update`], that its shaders index from
    /// its own entry zero.
    fn section_view(
        &self,
        device: &ID3D11Device,
        section: &TableSection,
    ) -> Result<Option<ID3D11ShaderResourceView>> {
        let element_size = std::mem::size_of::<T>();
        // Every scene's table holds at least entry zero.
        let len = section.len.max(1);
        anyhow::ensure!(
            section.offset + len <= self.capacity,
            "{} section {}..{} exceeds its buffer of {} entries",
            self.label,
            section.offset,
            section.offset + len,
            self.capacity,
        );
        create_buffer_section_view(
            device,
            &self.buffer,
            section.offset * element_size,
            len * element_size,
        )
    }
}

/// Where one scene's table sits in a [`SceneTableBuffer`], in entries.
struct TableSection {
    offset: usize,
    len: usize,
}

/// The byte boundary each scene's table starts on, so that a raw view can start there.
const TABLE_SECTION_ALIGNMENT: usize = 16;

/// The `T`s of each of `scenes`, picked by `section`.
fn sections<'a, T>(
    scenes: &[&'a Scene],
    section: impl Fn(&'a Scene) -> &'a [T],
) -> SmallVec<[&'a [T]; 4]> {
    scenes.iter().map(|scene| section(scene)).collect()
}

/// Where each of `sections` starts, in elements, laid end to end with each starting at a
/// multiple of `alignment` bytes, a power of two; and how many elements they take in all.
fn section_offsets<T>(sections: &[&[T]], alignment: usize) -> (SmallVec<[usize; 4]>, usize) {
    let element_size = std::mem::size_of::<T>().max(1);
    // The fewest elements whose size is a multiple of the alignment.
    let step = alignment
        >> element_size
            .trailing_zeros()
            .min(alignment.trailing_zeros());
    let mut end = 0usize;
    let offsets = sections
        .iter()
        .map(|section| {
            let start = end.next_multiple_of(step);
            end = start.saturating_add(section.len());
            start
        })
        .collect();
    (offsets, end)
}

/// Frame-wide state that every batch draw binds alongside its own pipeline.
struct FrameBindings<'a> {
    device_context: &'a ID3D11DeviceContext,
    viewport: &'a D3D11_VIEWPORT,
    globals: &'a DirectXGlobalElements,
    /// The tables of the scene being drawn, the window's or a chunk's.
    scene_tables: &'a [Option<ID3D11ShaderResourceView>; 3],
}

struct Annotation<'a>(&'a ID3DUserDefinedAnnotation);

impl<'a> Annotation<'a> {
    fn new(annotation: &'a ID3DUserDefinedAnnotation, label: HSTRING) -> Self {
        unsafe { annotation.BeginEvent(&label) };
        Self(annotation)
    }
}

impl Drop for Annotation<'_> {
    fn drop(&mut self) {
        unsafe { self.0.EndEvent() };
    }
}

struct DirectComposition {
    comp_device: IDCompositionDevice,
    comp_target: IDCompositionTarget,
    comp_visual: IDCompositionVisual,
}

impl DirectXRendererDevices {
    pub(crate) fn new(
        directx_devices: &DirectXDevices,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        let DirectXDevices {
            adapter,
            dxgi_factory,
            device,
            device_context,
        } = directx_devices;
        let dxgi_device = if disable_direct_composition {
            None
        } else {
            Some(device.cast().context("Creating DXGI device")?)
        };
        let annotation = device_context.cast().ok();

        Ok(Self {
            adapter: adapter.clone(),
            dxgi_factory: dxgi_factory.clone(),
            device: device.clone(),
            device_context: device_context.clone(),
            dxgi_device,
            annotation,
        })
    }
}

impl DirectXRenderer {
    pub(crate) fn new(
        hwnd: HWND,
        directx_devices: &DirectXDevices,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        if disable_direct_composition {
            log::info!("Direct Composition is disabled.");
        }

        let devices = DirectXRendererDevices::new(directx_devices, disable_direct_composition)
            .context("Creating DirectX devices")?;
        let atlas = Arc::new(DirectXAtlas::new(&devices.device, &devices.device_context));

        let resources = DirectXResources::new(&devices, 1, 1, hwnd, disable_direct_composition)
            .context("Creating DirectX resources")?;
        let globals = DirectXGlobalElements::new(&devices.device)
            .context("Creating DirectX global elements")?;
        let pipelines = DirectXRenderPipelines::new(&devices.device)
            .context("Creating DirectX render pipelines")?;

        let direct_composition = if disable_direct_composition {
            None
        } else {
            let composition = DirectComposition::new(devices.dxgi_device.as_ref().unwrap(), hwnd)
                .context("Creating DirectComposition")?;
            composition
                .set_swap_chain(&resources.swap_chain)
                .context("Setting swap chain for DirectComposition")?;
            Some(composition)
        };

        Ok(DirectXRenderer {
            hwnd,
            atlas,
            devices: Some(devices),
            resources: Some(resources),
            globals,
            pipelines,
            programs: gpui_render::linked::LinkedPrograms::new(),
            frame_programs: None,
            programs_pending: false,
            direct_composition,
            font_info: Self::get_font_info(),
            width: 1,
            height: 1,
            skip_draws: false,
            targets: Vec::new(),
            path_rasterization_vertices: Vec::new(),
            meshes: gpui_render::meshes::MeshCache::new(),
            path_sprites: Vec::new(),
            level: DrawLevel::default(),
            chunk_slots: FxHashMap::default(),
        })
    }

    pub(crate) fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        self.atlas.clone()
    }

    fn pre_draw(&self, clear_color: &[f32; 4]) -> Result<()> {
        let resources = self.resources.as_ref().expect("resources missing");
        let device_context = &self
            .devices
            .as_ref()
            .expect("devices missing")
            .device_context;
        self.write_globals(resources.window_bounds())?;
        update_buffer(
            device_context,
            self.globals.font_buffer.as_ref().unwrap(),
            &[FontRasterizationUniforms {
                gamma_ratios: vec4f(
                    self.font_info.gamma_ratios[0],
                    self.font_info.gamma_ratios[1],
                    self.font_info.gamma_ratios[2],
                    self.font_info.gamma_ratios[3],
                ),
                grayscale_enhanced_contrast: self.font_info.grayscale_enhanced_contrast,
                subpixel_enhanced_contrast: self.font_info.subpixel_enhanced_contrast,
                uses_blue_green_red_subpixel_order: ShaderBool::from(self.font_info.is_bgr),
                padding: 0,
            }],
        )?;
        unsafe {
            device_context.ClearRenderTargetView(
                resources
                    .render_target_view
                    .as_ref()
                    .context("missing render target view")?,
                clear_color,
            );
            device_context
                .OMSetRenderTargets(Some(slice::from_ref(&resources.render_target_view)), None);
            device_context.RSSetViewports(Some(slice::from_ref(&resources.viewport)));
        }
        Ok(())
    }

    /// Points the global uniforms at a target covering `bounds` of the viewport. Shaders place
    /// what they draw relative to the target's origin, so this changes with the target; inside
    /// a chunk, they place it from the chunk's viewport too.
    fn write_globals(&self, bounds: Bounds<DevicePixels>) -> Result<()> {
        let resources = self.resources.as_ref().context("resources missing")?;
        let device_context = &self
            .devices
            .as_ref()
            .context("devices missing")?
            .device_context;
        let globals = GlobalUniforms {
            viewport_size: vec2f(resources.viewport.Width, resources.viewport.Height),
            target_origin: vec2f(bounds.origin.x.0 as f32, bounds.origin.y.0 as f32),
            target_size: vec2f(bounds.size.width.0 as f32, bounds.size.height.0 as f32),
            // DirectComposition wants premultiplied output, but path rasterization
            // premultiplies in-shader; scene geometry blends straight alpha as before.
            premultiplied_alpha: ShaderBool::Disabled,
            padding: 0,
            placement: GlobalUniforms::unplaced(),
            inverse_placement: GlobalUniforms::unplaced(),
            placement_translation: vec2f(0.0, 0.0),
            inverse_placement_translation: vec2f(0.0, 0.0),
        };
        let globals = match &self.level.chunk {
            Some((placement, _)) => globals.placed(placement),
            None => globals,
        };
        update_buffer(
            device_context,
            self.globals
                .globals_buffer
                .as_ref()
                .context("globals buffer missing")?,
            &[globals],
        )
    }

    /// Binds the current target for the batches that follow: its view, its viewport and the
    /// globals describing it; inside a chunk, with the chunk's placement and its scissor.
    fn bind_current_target(&self) -> Result<()> {
        let target = self.targets.last().context("no render target is bound")?;
        let device_context = &self
            .devices
            .as_ref()
            .context("devices missing")?
            .device_context;
        let resources = self.resources.as_ref().context("resources missing")?;
        unsafe {
            device_context.OMSetRenderTargets(Some(slice::from_ref(&target.color.rtv)), None);
            device_context.RSSetViewports(Some(slice::from_ref(&target.viewport)));
            match &self.level.chunk {
                None => device_context.RSSetState(&resources.rasterizer_state),
                Some((_, clip)) => {
                    device_context.RSSetState(&resources.scissored_rasterizer_state);
                    device_context.RSSetScissorRects(Some(&[chunk_scissor(target, *clip)]));
                }
            }
        }
        self.write_globals(target.bounds)
    }

    /// The viewport of the target batches draw into.
    fn current_viewport(&self) -> Result<D3D11_VIEWPORT> {
        Ok(self
            .targets
            .last()
            .context("no render target is bound")?
            .viewport)
    }

    #[inline]
    fn present(&mut self) -> Result<()> {
        let result = unsafe {
            self.resources
                .as_ref()
                .expect("resources missing")
                .swap_chain
                .Present(0, DXGI_PRESENT(0))
        };
        result.ok().context("Presenting swap chain failed")
    }

    pub(crate) fn handle_device_lost(&mut self, directx_devices: &DirectXDevices) -> Result<()> {
        try_to_recover_from_device_lost(|| {
            self.handle_device_lost_impl(directx_devices)
                .context("DirectXRenderer handling device lost")
        })
    }

    fn handle_device_lost_impl(&mut self, directx_devices: &DirectXDevices) -> Result<()> {
        let disable_direct_composition = self.direct_composition.is_none();

        unsafe {
            #[cfg(debug_assertions)]
            if let Some(devices) = &self.devices {
                report_live_objects(&devices.device)
                    .context("Failed to report live objects after device lost")
                    .log_err();
            }

            self.resources.take();
            // Its meshes live on the lost device.
            self.meshes.clear();
            if let Some(devices) = &self.devices {
                devices.device_context.OMSetRenderTargets(None, None);
                devices.device_context.ClearState();
                devices.device_context.Flush();
                #[cfg(debug_assertions)]
                report_live_objects(&devices.device)
                    .context("Failed to report live objects after device lost")
                    .log_err();
            }

            self.direct_composition.take();
            self.devices.take();
        }

        let devices = DirectXRendererDevices::new(directx_devices, disable_direct_composition)
            .context("Recreating DirectX devices")?;
        let resources = DirectXResources::new(
            &devices,
            self.width,
            self.height,
            self.hwnd,
            disable_direct_composition,
        )
        .context("Creating DirectX resources")?;
        let globals = DirectXGlobalElements::new(&devices.device)
            .context("Creating DirectXGlobalElements")?;
        let pipelines = DirectXRenderPipelines::new(&devices.device)
            .context("Creating DirectXRenderPipelines")?;

        let direct_composition = if disable_direct_composition {
            None
        } else {
            let composition =
                DirectComposition::new(devices.dxgi_device.as_ref().unwrap(), self.hwnd)?;
            composition.set_swap_chain(&resources.swap_chain)?;
            Some(composition)
        };

        self.atlas
            .handle_device_lost(&devices.device, &devices.device_context);

        unsafe {
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(&resources.render_target_view)), None);
        }
        self.devices = Some(devices);
        self.resources = Some(resources);
        self.globals = globals;
        self.pipelines = pipelines;
        // Shaders linked on the lost device are linked again on the new one.
        self.programs.clear();
        self.frame_programs = None;
        self.direct_composition = direct_composition;
        self.skip_draws = true;
        Ok(())
    }

    /// Draws `scene` and presents it. Returns whether it should be drawn
    /// again soon: it runs shader programs still being linked, which it drew
    /// with their fallback colours.
    pub(crate) fn draw(
        &mut self,
        scene: &Scene,
        background_appearance: WindowBackgroundAppearance,
    ) -> Result<bool> {
        if self.skip_draws {
            // skip drawing this frame, we just recovered from a device lost event
            // and so likely do not have the textures anymore that are required for drawing
            return Ok(false);
        }
        self.render(scene, background_appearance)?;
        self.present()?;
        Ok(self.programs_pending)
    }

    /// Picks the shaders `scene`'s paints draw with, starting to link the
    /// shader programs it runs that they lack.
    fn prepare_programs(&mut self, scene: &Scene) -> Result<()> {
        let device = &self.devices.as_ref().context("devices missing")?.device;
        let programs = self.programs.prepare(scene, || {
            let device = crate::directx_programs::LinkingDevice(device.clone());
            move |programs: Vec<gpui::shader::Program>| {
                crate::directx_programs::link_shaders(&device, &programs)
            }
        });
        self.frame_programs = programs.pipelines;
        self.programs_pending = programs.pending;
        Ok(())
    }

    /// Encodes a complete frame without presenting it. Window drawing and test readback share
    /// this path so batching, filters, and resource-retention behavior cannot diverge.
    fn render(
        &mut self,
        scene: &Scene,
        background_appearance: WindowBackgroundAppearance,
    ) -> Result<()> {
        // Drawn from the window's scene, until a chunk is entered.
        self.level = DrawLevel::default();
        self.pre_draw(&match background_appearance {
            appearance if appearance.is_opaque() => [1.0f32; 4],
            _ => [0.0f32; 4],
        })?;

        self.upload_scene_buffers(scene)?;
        self.prepare_programs(scene)?;
        self.meshes.begin_frame();

        // Render the scene into an offscreen texture when backdrop filters or blend modes read
        // what is already painted, then blit it to the swapchain; otherwise render straight to
        // the swapchain, with no extra blit.
        let use_offscreen = scene.requires_offscreen_rendering();
        let requirements = scene.render_plan().requirements();
        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_mut().context("resources missing")?;
        resources.retain_surface_views(&scene.surfaces);
        if requirements.uses_path_target {
            resources.ensure_path_resources(&devices.device)?;
        }
        let swapchain_rtv = resources.render_target_view.clone();
        let viewport_size = resources.viewport_size();
        let window = if use_offscreen {
            let scene_color = resources.ensure_scene_color(&devices.device)?.clone();
            unsafe {
                devices.device_context.ClearRenderTargetView(
                    scene_color
                        .rtv
                        .as_ref()
                        .context("scene color view missing")?,
                    &[0.0; 4],
                );
            }
            scene_color
        } else {
            ColorTarget {
                texture: resources
                    .render_target
                    .clone()
                    .context("render target missing")?,
                rtv: swapchain_rtv.clone(),
                // The swapchain's buffer is only ever drawn into and copied from.
                srv: None,
                size: viewport_size,
            }
        };
        let scene_srv = window.srv.clone();

        self.targets.clear();
        self.targets.push(FrameTarget::new(
            window,
            point(DevicePixels(0), DevicePixels(0)),
        ));
        let result = self
            .bind_current_target()
            .and_then(|()| self.encode_commands(scene, viewport_size));

        // Give back what an error left on the stack; a finished frame leaves only the window's.
        while self.targets.len() > 1 {
            if let Some(mut target) = self.targets.pop() {
                if let Some((mask, _, _)) = target.mask.take() {
                    self.give_back(mask);
                }
                self.give_back(target.color);
            }
        }
        self.targets.clear();
        if let Some(resources) = self.resources.as_mut() {
            resources.target_pool.end_frame();
        }
        self.meshes.end_frame();
        result?;

        // Present the offscreen scene by blitting it into the swapchain.
        if use_offscreen {
            self.dx_blit(&scene_srv, &swapchain_rtv)?;
        }
        Ok(())
    }

    /// Draws the scene's render commands into the targets on the stack, starting with the
    /// window's.
    fn encode_commands(&mut self, scene: &Scene, viewport_size: Size<DevicePixels>) -> Result<()> {
        // Whether each group being drawn has a target of its own.
        let mut isolated = SmallVec::<[bool; 8]>::new();
        // How many of the groups being drawn are hidden: a mask, or a masked group, that got
        // no target is not drawn at all, nor is what it holds.
        let mut hidden = 0usize;
        let annotation = self
            .devices
            .as_ref()
            .and_then(|devices| devices.annotation.clone())
            .filter(|annotation| unsafe { annotation.GetStatus().as_bool() });
        for command in scene.render_commands() {
            if hidden > 0 {
                match command {
                    RenderCommand::BeginGroup { .. } => hidden += 1,
                    RenderCommand::EndGroup { .. } => hidden -= 1,
                    RenderCommand::Batch(_) => {}
                }
                continue;
            }
            let _annotation = annotation
                .as_ref()
                .map(|annotation| Annotation::new(annotation, HSTRING::from(command.label())));
            match command {
                RenderCommand::Batch(PrimitiveBatch::Shadows { range, smoothed }) => self
                    .draw_shadows(
                        instance_range(range, self.level.slot.shadows)?,
                        *smoothed,
                    ),
                RenderCommand::Batch(PrimitiveBatch::Quads { range, smoothed }) => {
                    self.draw_quads(instance_range(range, self.level.slot.quads)?, *smoothed)
                }
                RenderCommand::Batch(PrimitiveBatch::Paths {
                    range,
                    rasterization_vertex_count,
                    sprite_count,
                }) => {
                    if *rasterization_vertex_count == 0 {
                        continue;
                    }
                    let paths = &scene.paths[range.clone()];
                    self.draw_paths_to_intermediate(paths, *rasterization_vertex_count)?;
                    self.draw_paths_from_intermediate(paths, *sprite_count)
                }
                RenderCommand::Batch(PrimitiveBatch::Underlines(range)) => {
                    self.draw_underlines(instance_range(range, self.level.slot.underlines)?)
                }
                RenderCommand::Batch(PrimitiveBatch::Meshes(range)) => self.draw_meshes(
                    &scene.meshes[range.clone()],
                    instance_range(range, self.level.slot.meshes)?,
                ),
                RenderCommand::Batch(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range,
                }) => self.draw_monochrome_sprites(
                    *texture_id,
                    instance_range(range, self.level.slot.monochrome_sprites)?,
                ),
                RenderCommand::Batch(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range,
                }) => self.draw_subpixel_sprites(
                    *texture_id,
                    instance_range(range, self.level.slot.subpixel_sprites)?,
                ),
                RenderCommand::Batch(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range,
                    smoothed,
                }) => self.draw_polychrome_sprites(
                    *texture_id,
                    instance_range(range, self.level.slot.polychrome_sprites)?,
                    *smoothed,
                ),
                RenderCommand::Batch(PrimitiveBatch::Surfaces(range)) => {
                    self.draw_surfaces(
                        &scene.surfaces[range.clone()],
                        &scene.surface_opacities()[range.clone()],
                    )
                }
                RenderCommand::Batch(PrimitiveBatch::BackdropFilters(range)) => {
                    let result = (|| {
                        for filter in &scene.backdrop_filters[range.clone()] {
                            self.dx_blur_and_composite(
                                filter.bounds,
                                filter.content_mask.bounds,
                                filter.corner_radii,
                                filter.corner_smoothing,
                                filter.max_blur_radius(),
                                filter.opacity,
                            )?;
                        }
                        Ok::<(), anyhow::Error>(())
                    })();
                    // The blurs drew elsewhere; draw into the current target again.
                    self.bind_current_target()?;
                    result
                }
                RenderCommand::BeginGroup {
                    boundary_index,
                    target,
                } => {
                    let has_target = self.begin_group(target, viewport_size)?;
                    let boundary = &scene.group_boundaries[*boundary_index];
                    if !has_target && (boundary.masked || boundary.mask_mode.is_some()) {
                        hidden = 1;
                        continue;
                    }
                    isolated.push(has_target);
                    Ok(())
                }
                RenderCommand::EndGroup { boundary_index, .. } => {
                    if isolated.pop() == Some(true) {
                        self.end_group(&scene.group_boundaries[*boundary_index])
                    } else {
                        Ok(())
                    }
                }
                RenderCommand::Batch(PrimitiveBatch::GroupBoundary(_)) => {
                    unreachable!("group boundaries are resolved by the render plan")
                }
                RenderCommand::Batch(PrimitiveBatch::Chunks(range)) => {
                    self.draw_chunks(&scene.chunks[range.clone()], viewport_size)
                }
            }
            .with_context(|| {
                format!(
                    "scene too large:\
                    {} paths, {} shadows, {} quads, {} underlines, {} mono, {} subpixel, {} poly, {} surfaces",
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.subpixel_sprites.len(),
                    scene.polychrome_sprites.len(),
                    scene.surfaces.len(),
                )
            })?;
        }
        Ok(())
    }

    /// Draws each of `chunks` in turn, as a unit: its scene's commands, with its tables, placed
    /// from its viewport into this one and cut to its clip.
    fn draw_chunks(
        &mut self,
        chunks: &[PlacedChunk],
        viewport_size: Size<DevicePixels>,
    ) -> Result<()> {
        for placed in chunks {
            let chunk_scene = &placed.chunk.scene;
            let slot = self
                .chunk_slots
                .get(&(chunk_scene as *const Scene))
                .cloned()
                .context("a chunk was drawn that was not uploaded")?;
            let (parent_placement, parent_clip) = match &self.level.chunk {
                Some((placement, clip)) => (*placement, Some(*clip)),
                None => (TransformationMatrix::unit(), None),
            };
            let placement = parent_placement.compose(placed.placement);
            let mut clip = transformed_bounds(&parent_placement, placed.content_mask.bounds);
            if let Some(outer) = parent_clip {
                clip = clip.intersect(&outer);
            }
            let parent = std::mem::replace(
                &mut self.level,
                DrawLevel {
                    slot,
                    chunk: Some((placement, clip)),
                },
            );
            let result = self
                .bind_current_target()
                .and_then(|()| self.encode_commands(chunk_scene, viewport_size));
            self.level = parent;
            result?;
        }
        // Draw the parent's batches that follow with its placement, tables and clip.
        self.bind_current_target()
    }

    /// Starts drawing a group: into a target of its own, cleared to transparent, when the plan
    /// isolates it and the pool has room for one; in place otherwise. Returns whether it has a
    /// target of its own.
    fn begin_group(
        &mut self,
        target: &GroupTarget,
        viewport_size: Size<DevicePixels>,
    ) -> Result<bool> {
        let GroupTarget::Isolated { region } = target else {
            return Ok(false);
        };
        let Some(bounds) = group_target_bounds(*region, viewport_size) else {
            return Ok(false);
        };
        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_mut().context("resources missing")?;
        let Some(color) = resources.target_pool.take(&devices.device, bounds.size) else {
            return Ok(false);
        };
        unsafe {
            devices.device_context.ClearRenderTargetView(
                color.rtv.as_ref().context("group target view missing")?,
                &[0.0; 4],
            );
        }
        self.targets.push(FrameTarget::new(color, bounds.origin));
        self.bind_current_target()?;
        Ok(true)
    }

    /// Finishes a group drawn into a target of its own: composites it into its parent, which
    /// the batches that follow draw into, and gives its texture, and its mask's, back to the
    /// pool. A mask is not composited but kept on its parent, the group it masks, which
    /// samples it; a masked group whose mask drew nothing is dropped.
    fn end_group(&mut self, boundary: &GroupBoundary) -> Result<()> {
        let mut group = self
            .targets
            .pop()
            .context("an isolated group's target is on the stack")?;
        if let Some(mode) = boundary.mask_mode {
            let parent = self
                .targets
                .last_mut()
                .context("a mask's group is on the stack")?;
            if let Some((stale, _, _)) = parent.mask.replace((group.color, group.bounds, mode)) {
                self.give_back(stale);
            }
            return self.bind_current_target();
        }
        let result = if boundary.masked && group.mask.is_none() {
            // Its mask is out of view: so is all of it.
            self.bind_current_target()
        } else {
            self.composite_group(boundary, &group)
                .and_then(|()| self.bind_current_target())
        };
        if let Some((mask, _, _)) = group.mask.take() {
            self.give_back(mask);
        }
        self.give_back(group.color);
        result
    }

    fn give_back(&mut self, target: ColorTarget) {
        if let Some(resources) = self.resources.as_mut() {
            resources.target_pool.give_back(target);
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn render_to_image(
        &mut self,
        scene: &Scene,
        background_appearance: WindowBackgroundAppearance,
    ) -> Result<image::RgbaImage> {
        anyhow::ensure!(
            !self.skip_draws,
            "render_to_image unavailable while recovering from a lost device"
        );
        self.render(scene, background_appearance)?;

        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        let render_target = resources
            .render_target
            .as_ref()
            .context("render target missing")?;

        let mut source_desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { render_target.GetDesc(&mut source_desc) };
        let width = source_desc.Width;
        let height = source_desc.Height;
        let staging_desc = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
            MipLevels: 1,
            ArraySize: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..source_desc
        };
        let mut staging = None;
        unsafe {
            devices
                .device
                .CreateTexture2D(&staging_desc, None, Some(&mut staging))?
        };
        let staging = staging.context("creating staging texture")?;
        unsafe {
            devices.device_context.CopyResource(&staging, render_target);
        }

        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            devices
                .device_context
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?
        };
        let row_bytes = width as usize * 4;
        let mut pixels = vec![0u8; row_bytes * height as usize];
        // SAFETY: a successful `Map` exposes `RowPitch * height` readable bytes until `Unmap`.
        // D3D11 guarantees RowPitch is at least the logical row width, and each destination row
        // is disjoint within the exactly-sized output allocation.
        unsafe {
            let source = mapped.pData.cast::<u8>();
            for row in 0..height as usize {
                std::ptr::copy_nonoverlapping(
                    source.add(row * mapped.RowPitch as usize),
                    pixels.as_mut_ptr().add(row * row_bytes),
                    row_bytes,
                );
            }
            devices.device_context.Unmap(&staging, 0);
        }
        for pixel in pixels.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
        image::RgbaImage::from_raw(width, height, pixels)
            .context("failed to build RGBA image from DirectX staging readback")
    }

    pub(crate) fn resize(&mut self, new_size: Size<DevicePixels>) -> Result<()> {
        let width = new_size.width.0.max(1) as u32;
        let height = new_size.height.0.max(1) as u32;
        if self.width == width && self.height == height {
            return Ok(());
        }
        self.width = width;
        self.height = height;

        // Clear the render target before resizing
        let devices = self.devices.as_ref().context("devices missing")?;
        unsafe { devices.device_context.OMSetRenderTargets(None, None) };
        let resources = self.resources.as_mut().context("resources missing")?;
        resources.render_target.take();
        resources.render_target_view.take();

        // Resizing the swap chain requires a call to the underlying DXGI adapter, which can return the device removed error.
        // The app might have moved to a monitor that's attached to a different graphics device.
        // When a graphics device is removed or reset, the desktop resolution often changes, resulting in a window size change.
        // But here we just return the error, because we are handling device lost scenarios elsewhere.
        unsafe {
            resources
                .swap_chain
                .ResizeBuffers(
                    BUFFER_COUNT as u32,
                    width,
                    height,
                    RENDER_TARGET_FORMAT,
                    DXGI_SWAP_CHAIN_FLAG(0),
                )
                .context("Failed to resize swap chain")?;
        }

        resources.recreate_resources(devices, width, height)?;

        unsafe {
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(&resources.render_target_view)), None);
        }

        Ok(())
    }

    /// Uploads the frame's tables and instances: the window scene's, then each chunk scene's
    /// drawn in it, once however often it is drawn, after it in the same buffers.
    fn upload_scene_buffers(&mut self, scene: &Scene) -> Result<()> {
        let devices = self.devices.as_ref().context("devices missing")?;
        let (device, device_context) = (&devices.device, &devices.device_context);

        // The window's scene, then each chunk scene inside it.
        let mut scenes: Vec<&Scene> = vec![scene];
        let mut index = 0;
        self.chunk_slots.clear();
        while let Some(&parent) = scenes.get(index) {
            for placed in &parent.chunks {
                let chunk_scene = &placed.chunk.scene;
                if self
                    .chunk_slots
                    .insert(chunk_scene as *const Scene, SceneSlot::default())
                    .is_none()
                {
                    scenes.push(chunk_scene);
                }
            }
            index += 1;
        }

        let transforms = self.globals.transforms.update(
            device,
            device_context,
            &sections(&scenes, |scene| scene.transforms()),
        )?;
        let clips = self.globals.clips.update(
            device,
            device_context,
            &sections(&scenes, |scene| scene.clips()),
        )?;
        let paints = self.globals.paints.update(
            device,
            device_context,
            &sections(&scenes, |scene| scene.paint_table()),
        )?;
        self.globals
            .photo_tiles
            .upload(device, device_context, scene)?;

        let pipelines = &mut self.pipelines;
        let shadows = pipelines.shadow_pipeline.update_sections(
            device,
            device_context,
            &sections(&scenes, |scene| &scene.shadows),
        )?;
        let quads = pipelines.quad_pipeline.update_sections(
            device,
            device_context,
            &sections(&scenes, |scene| &scene.quads),
        )?;
        let underlines = pipelines.underline_pipeline.update_sections(
            device,
            device_context,
            &sections(&scenes, |scene| &scene.underlines),
        )?;
        let monochrome_sprites = pipelines.mono_sprites.update_sections(
            device,
            device_context,
            &sections(&scenes, |scene| &scene.monochrome_sprites),
        )?;
        let subpixel_sprites = pipelines.subpixel_sprites.update_sections(
            device,
            device_context,
            &sections(&scenes, |scene| &scene.subpixel_sprites),
        )?;
        let polychrome_sprites = pipelines.poly_sprites.update_sections(
            device,
            device_context,
            &sections(&scenes, |scene| &scene.polychrome_sprites),
        )?;
        let mesh_instances: Vec<Vec<MeshInstance>> = scenes
            .iter()
            .map(|scene| scene.meshes.iter().map(|mesh| mesh.instance).collect())
            .collect();
        let meshes = pipelines.meshes.update_sections(
            device,
            device_context,
            &mesh_instances.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        )?;

        for (index, scene) in scenes.iter().enumerate() {
            let tables = if index == 0 {
                // The window's tables start each buffer: its whole view reads them.
                self.globals.scene_tables()
            } else {
                [
                    self.globals
                        .transforms
                        .section_view(device, &transforms[index])?,
                    self.globals.clips.section_view(device, &clips[index])?,
                    self.globals.paints.section_view(device, &paints[index])?,
                ]
            };
            let slot = SceneSlot {
                shadows: shadows[index],
                quads: quads[index],
                underlines: underlines[index],
                monochrome_sprites: monochrome_sprites[index],
                subpixel_sprites: subpixel_sprites[index],
                polychrome_sprites: polychrome_sprites[index],
                meshes: meshes[index],
                tables,
            };
            if index == 0 {
                self.level.slot = slot;
            } else {
                self.chunk_slots.insert(*scene as *const Scene, slot);
            }
        }
        Ok(())
    }

    /// Frame-wide bindings for the batch draws of the current frame, into the current target.
    fn frame_bindings(&self) -> Result<FrameBindings<'_>> {
        Ok(FrameBindings {
            device_context: &self
                .devices
                .as_ref()
                .context("devices missing")?
                .device_context,
            viewport: &self
                .targets
                .last()
                .context("no render target is bound")?
                .viewport,
            globals: &self.globals,
            scene_tables: &self.level.slot.tables,
        })
    }

    fn draw_shadows(&mut self, instances: InstanceRange, smoothed: bool) -> Result<()> {
        let pipeline = &self.pipelines.shadow_pipeline;
        match &self.frame_programs {
            Some(linked) => pipeline.draw_instances_linked(
                &self.frame_bindings()?,
                None,
                instances,
                if smoothed {
                    &linked.smoothed_shadows
                } else {
                    &linked.shadows
                },
            ),
            None => {
                pipeline.draw_instances_variant(&self.frame_bindings()?, None, instances, smoothed)
            }
        }
    }

    fn draw_quads(&mut self, instances: InstanceRange, smoothed: bool) -> Result<()> {
        let pipeline = &self.pipelines.quad_pipeline;
        match &self.frame_programs {
            Some(linked) => pipeline.draw_instances_linked(
                &self.frame_bindings()?,
                None,
                instances,
                if smoothed {
                    &linked.smoothed_quads
                } else {
                    &linked.quads
                },
            ),
            None => {
                pipeline.draw_instances_variant(&self.frame_bindings()?, None, instances, smoothed)
            }
        }
    }

    fn draw_paths_to_intermediate(
        &mut self,
        paths: &[Path<ScaledPixels>],
        rasterization_vertex_count: usize,
    ) -> Result<()> {
        if paths.is_empty() {
            return Ok(());
        }

        self.path_rasterization_vertices.clear();
        self.path_rasterization_vertices
            .reserve(rasterization_vertex_count);
        for path in paths {
            self.path_rasterization_vertices
                .extend(path.vertices.iter().map(|vertex| PathRasterizationVertex {
                    xy_position: vertex.xy_position,
                    curve_position: vertex.st_position,
                    color: path.color.paint_ref(),
                    padding: 0,
                    bounds: path.clipped_bounds(),
                }));
        }
        debug_assert_eq!(
            self.path_rasterization_vertices.len(),
            rasterization_vertex_count
        );

        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        let path = resources
            .path
            .as_ref()
            .context("path resources were not prepared")?;
        // The intermediate covers the window, whatever target the sprites are drawn into, so
        // paths are rasterized with the window's globals and viewport.
        self.write_globals(resources.window_bounds())?;
        // Clear intermediate MSAA texture
        unsafe {
            devices.device_context.ClearRenderTargetView(
                path.msaa_view.as_ref().context("path MSAA view missing")?,
                &[0.0; 4],
            );
            // Set intermediate MSAA texture as render target
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(&path.msaa_view)), None);
            // A chunk's scissor is in its target's coordinates, not the intermediate's: its
            // paths are cut to it when their sprites are drawn.
            devices
                .device_context
                .RSSetState(&resources.rasterizer_state);
        }

        self.pipelines.path_rasterization_pipeline.update_buffer(
            &devices.device,
            &devices.device_context,
            &self.path_rasterization_vertices,
        )?;
        self.pipelines.path_rasterization_pipeline.draw_vertices(
            &FrameBindings {
                device_context: &devices.device_context,
                viewport: &resources.viewport,
                globals: &self.globals,
                scene_tables: &self.level.slot.tables,
            },
            u32::try_from(rasterization_vertex_count)
                .context("path rasterization vertex count exceeds the D3D11 draw limit")?,
            self.frame_programs
                .as_deref()
                .map(|linked| &linked.path_rasterization),
        )?;

        // Resolve MSAA to non-MSAA intermediate texture
        unsafe {
            devices.device_context.ResolveSubresource(
                &path.texture,
                0,
                &path.msaa_texture,
                0,
                RENDER_TARGET_FORMAT,
            );
        }
        // Draw the path sprites into the current target again.
        self.bind_current_target()
    }

    fn draw_paths_from_intermediate(
        &mut self,
        paths: &[Path<ScaledPixels>],
        sprite_count: usize,
    ) -> Result<()> {
        let Some(first_path) = paths.first() else {
            return Ok(());
        };

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        self.path_sprites.clear();
        self.path_sprites.reserve(sprite_count);
        if paths.last().unwrap().order == first_path.order {
            self.path_sprites
                .extend(paths.iter().map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                }));
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            self.path_sprites.push(PathSprite { bounds });
        }
        debug_assert_eq!(self.path_sprites.len(), sprite_count);

        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        let path = resources
            .path
            .as_ref()
            .context("path resources were not prepared")?;
        self.pipelines.path_sprite_pipeline.update_buffer(
            &devices.device,
            &devices.device_context,
            &self.path_sprites,
        )?;
        let instances = InstanceRange::from_start(sprite_count)
            .context("path sprite count exceeds the D3D11 instance limit")?;
        self.pipelines.path_sprite_pipeline.draw_instances(
            &self.frame_bindings()?,
            Some(slice::from_ref(&path.srv)),
            instances,
        )
    }

    /// Draws `meshes`, whose instances are `instances` of the frame's, each
    /// from its vertices and indices kept on the GPU, uploaded the first
    /// time it is drawn.
    fn draw_meshes(&mut self, meshes: &[MeshPrimitive], instances: InstanceRange) -> Result<()> {
        let device = self
            .devices
            .as_ref()
            .context("devices missing")?
            .device
            .clone();
        let linked = self.frame_programs.clone();
        let variant = linked.as_ref().map(|linked| &linked.meshes);
        for (index, primitive) in meshes.iter().enumerate() {
            let Some(mesh) = self
                .meshes
                .buffers(&primitive.mesh, |mesh| {
                    DirectXMesh::upload(&device, mesh).log_err()
                })
                .cloned()
            else {
                continue;
            };
            let instance = instances.first() as usize + index;
            self.pipelines.meshes.draw_mesh(
                &self.frame_bindings()?,
                &mesh,
                InstanceRange::new(instance..instance + 1)
                    .context("a mesh instance past the D3D11 instance limit")?,
                variant,
            )?;
        }
        Ok(())
    }

    fn draw_underlines(&mut self, instances: InstanceRange) -> Result<()> {
        self.pipelines
            .underline_pipeline
            .draw_instances(&self.frame_bindings()?, None, instances)
    }

    fn draw_monochrome_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        instances: InstanceRange,
    ) -> Result<()> {
        let texture_view = self.atlas.get_texture_view(texture_id);
        let pipeline = &self.pipelines.mono_sprites;
        match &self.frame_programs {
            Some(linked) => pipeline.draw_instances_linked(
                &self.frame_bindings()?,
                Some(&texture_view),
                instances,
                &linked.monochrome_sprites,
            ),
            None => {
                pipeline.draw_instances(&self.frame_bindings()?, Some(&texture_view), instances)
            }
        }
    }

    fn draw_subpixel_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        instances: InstanceRange,
    ) -> Result<()> {
        let texture_view = self.atlas.get_texture_view(texture_id);
        self.pipelines.subpixel_sprites.draw_instances(
            &self.frame_bindings()?,
            Some(&texture_view),
            instances,
        )
    }

    fn draw_polychrome_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        instances: InstanceRange,
        smoothed: bool,
    ) -> Result<()> {
        let texture_view = self.atlas.get_texture_view(texture_id);
        self.pipelines.poly_sprites.draw_instances_variant(
            &self.frame_bindings()?,
            Some(&texture_view),
            instances,
            smoothed,
        )
    }

    fn draw_surfaces(&mut self, surfaces: &[PaintSurface], opacities: &[f32]) -> Result<()> {
        if surfaces.is_empty() {
            return Ok(());
        }
        let viewport = self.current_viewport()?;
        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_mut().context("resources missing")?;
        let ctx = &devices.device_context;
        let cbuffers = self.globals.cbuffers();
        let surface_cb = [Some(self.pipelines.surfaces.params_buffer.clone())];
        let sampler = [self.globals.sampler.clone()];

        for (index, surface) in surfaces.iter().enumerate() {
            let gpui::SurfaceSource::WindowsCapture(frame) = &surface.source else {
                log::error!("DirectX renderer cannot import this surface source");
                anyhow::bail!("unsupported surface source");
            };
            let key = frame.texture().as_raw() as usize;
            if let std::collections::hash_map::Entry::Vacant(entry) =
                resources.surface_views.entry(key)
            {
                let mut srv = None;
                // Screen capture uses windows 0.61 while this renderer uses 0.62. COM interface
                // pointers are ABI-stable; transferring an owned clone keeps the texture alive.
                let texture =
                    unsafe { ID3D11Texture2D::from_raw(frame.texture().clone().into_raw()) };
                unsafe {
                    devices
                        .device
                        .CreateShaderResourceView(&texture, None, Some(&mut srv))?
                };
                entry.insert(CachedSurfaceView { texture, srv });
            }
            let texture_srv = &resources
                .surface_views
                .get(&key)
                .context("capture surface view cache insertion failed")?
                .srv;
            // The surface shader declares both planes; RGBA captures bind one view to both.
            let texture_srvs = [texture_srv.clone(), texture_srv.clone()];

            let uniforms = SurfaceUniforms {
                bounds: surface.bounds.into(),
                content_mask: surface.content_mask.into(),
                color_format: SurfaceColorFormat::Rgba,
                opacity: opacities.get(index).copied().unwrap_or(1.0),
                padding0: 0,
                padding1: 0,
                padding2: 0,
                padding3: 0,
                padding4: 0,
                padding5: 0,
            };
            update_buffer(ctx, &self.pipelines.surfaces.params_buffer, &[uniforms])?;

            unsafe {
                ctx.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
                ctx.RSSetViewports(Some(slice::from_ref(&viewport)));
                ctx.VSSetShader(&self.pipelines.surfaces.vertex, None);
                ctx.PSSetShader(&self.pipelines.surfaces.fragment, None);
                ctx.VSSetConstantBuffers(0, Some(&cbuffers));
                ctx.PSSetConstantBuffers(0, Some(&cbuffers));
                ctx.VSSetConstantBuffers(DATA_REGISTER, Some(&surface_cb));
                ctx.PSSetConstantBuffers(DATA_REGISTER, Some(&surface_cb));
                ctx.PSSetShaderResources(PRIMARY_TEXTURE_REGISTER, Some(&texture_srvs));
                ctx.PSSetSamplers(SURFACE_SAMPLER_REGISTER, Some(&sampler));
                ctx.OMSetBlendState(&self.pipelines.surfaces.blend, None, 0xFFFFFFFF);
                ctx.DrawInstanced(4, 1, 0, 0);
            }
        }
        Ok(())
    }

    /// Run a single blur pass: a full-screen (or composite) draw sampling `source_srv` into
    /// `target_rtv`, with `params` in the blur constant buffer (at [`DATA_REGISTER`]).
    #[allow(clippy::too_many_arguments)]
    fn dx_blur_pass(
        &self,
        vertex: &ID3D11VertexShader,
        fragment: &ID3D11PixelShader,
        blend: &ID3D11BlendState,
        target_rtv: &Option<ID3D11RenderTargetView>,
        source_srv: &Option<ID3D11ShaderResourceView>,
        params: BlurUniforms,
        viewport: &D3D11_VIEWPORT,
        topology: D3D_PRIMITIVE_TOPOLOGY,
        vertex_count: u32,
        clear: bool,
    ) -> Result<()> {
        let devices = self.devices.as_ref().context("devices missing")?;
        let ctx = &devices.device_context;
        update_buffer(ctx, &self.pipelines.blur_params_buffer, &[params])?;
        let null_srv: [Option<ID3D11ShaderResourceView>; 1] = [None];
        let cbuffers = self.globals.cbuffers();
        let blur_params = [Some(self.pipelines.blur_params_buffer.clone())];
        unsafe {
            // Unbind any SRV at the blur slot; the target must not be bound as input.
            ctx.PSSetShaderResources(PRIMARY_TEXTURE_REGISTER, Some(&null_srv));
            if clear {
                ctx.ClearRenderTargetView(
                    target_rtv.as_ref().context("blur target view missing")?,
                    &[0.0; 4],
                );
            }
            ctx.OMSetRenderTargets(Some(slice::from_ref(target_rtv)), None);
            ctx.RSSetViewports(Some(slice::from_ref(viewport)));
            ctx.IASetPrimitiveTopology(topology);
            ctx.VSSetShader(vertex, None);
            ctx.PSSetShader(fragment, None);
            ctx.VSSetConstantBuffers(0, Some(&cbuffers));
            ctx.PSSetConstantBuffers(0, Some(&cbuffers));
            ctx.VSSetConstantBuffers(DATA_REGISTER, Some(&blur_params));
            ctx.PSSetConstantBuffers(DATA_REGISTER, Some(&blur_params));
            ctx.PSSetSamplers(
                PRIMARY_SAMPLER_REGISTER,
                Some(slice::from_ref(&self.globals.sampler)),
            );
            ctx.PSSetShaderResources(PRIMARY_TEXTURE_REGISTER, Some(slice::from_ref(source_srv)));
            ctx.OMSetBlendState(blend, None, 0xFFFFFFFF);
            ctx.DrawInstanced(vertex_count, 1, 0, 0);
            // Unbind the source so the target can be rebound as a render target next.
            ctx.PSSetShaderResources(PRIMARY_TEXTURE_REGISTER, Some(&null_srv));
        }
        Ok(())
    }

    /// Blurs what the current target holds under `bounds`, the backdrop of a backdrop filter,
    /// and composites the result back into it, clipped to `bounds`, `corner_radii` and
    /// `content_mask`, and faded by `opacity`. The blur runs at half resolution in textures
    /// taken from the pool for it.
    #[allow(clippy::too_many_arguments)]
    fn dx_blur_and_composite(
        &mut self,
        bounds: Bounds<ScaledPixels>,
        content_mask: Bounds<ScaledPixels>,
        corner_radii: Corners<ScaledPixels>,
        corner_smoothing: f32,
        blur_radius: f32,
        opacity: f32,
    ) -> Result<()> {
        let Some(kernel) = BlurKernel::for_radius(blur_radius) else {
            return Ok(());
        };
        let target = self.targets.last().context("no render target is bound")?;
        let full_width = target.bounds.size.width.0.max(0) as u32;
        let full_height = target.bounds.size.height.0.max(0) as u32;
        let origin = target
            .bounds
            .origin
            .map(|value| ScaledPixels(value.0 as f32));
        // Nothing the blur could reach is in this target.
        let scissor = ScissorRectangle::for_blurred_bounds(
            Bounds {
                origin: bounds.origin - origin,
                size: bounds.size,
            },
            GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * blur_radius,
            full_width,
            full_height,
        );
        if scissor.is_empty() {
            return Ok(());
        }
        let source = target.color.clone();
        let Some((ping, pong, blur_size)) = self.blur(&source, kernel)? else {
            return Ok(());
        };

        let composite_uniforms = BlurUniforms::composite(
            bounds,
            content_mask,
            corner_radii,
            corner_smoothing,
            opacity,
            FilterCompositeClip::RoundedBounds,
            blur_size,
            [full_width as f32, full_height as f32],
            [origin.x.0, origin.y.0],
        );
        let (composite_vertex, composite_fragment) = if composite_uniforms.corner_smoothing > 0.0 {
            (
                &self.pipelines.smoothed_blur_composite_vertex,
                &self.pipelines.smoothed_blur_composite_fragment,
            )
        } else {
            (
                &self.pipelines.blur_composite_vertex,
                &self.pipelines.blur_composite_fragment,
            )
        };
        // Composite the blurred result into the target (preserving its contents), with the
        // target's globals, which are still bound.
        self.dx_blur_pass(
            composite_vertex,
            composite_fragment,
            &self.pipelines.blur_blend_composite,
            &source.rtv,
            &ping.srv,
            composite_uniforms,
            &self.current_viewport()?,
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            4,
            false,
        )?;
        self.give_back(ping);
        self.give_back(pong);
        Ok(())
    }

    /// Blurs the whole of `source` into two half-resolution textures taken from the pool: the
    /// first holds the result, the second is spare. Returns them with their size, or `None` if
    /// the pool has no room.
    fn blur(
        &mut self,
        source: &ColorTarget,
        kernel: BlurKernel,
    ) -> Result<Option<(ColorTarget, ColorTarget, [f32; 2])>> {
        let full_width = source.size.width.0.max(0) as u32;
        let full_height = source.size.height.0.max(0) as u32;
        let blur_size = [
            downsampled_dimension(full_width) as f32,
            downsampled_dimension(full_height) as f32,
        ];
        let half_size = size(
            DevicePixels(blur_size[0] as i32),
            DevicePixels(blur_size[1] as i32),
        );
        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_mut().context("resources missing")?;
        let pool = &mut resources.target_pool;
        let Some(ping) = pool.take(&devices.device, half_size) else {
            return Ok(None);
        };
        let Some(pong) = pool.take(&devices.device, half_size) else {
            pool.give_back(ping);
            return Ok(None);
        };
        let half_viewport = D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: blur_size[0],
            Height: blur_size[1],
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };

        // Downsample source -> ping, then separable gaussian ping -> pong -> ping.
        self.dx_blur_pass(
            &self.pipelines.blur_downsample_vertex,
            &self.pipelines.blur_downsample_fragment,
            &self.pipelines.blur_blend_replace,
            &ping.rtv,
            &source.srv,
            BlurUniforms::downsample([full_width as f32, full_height as f32], blur_size),
            &half_viewport,
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            3,
            true,
        )?;
        self.dx_blur_pass(
            &self.pipelines.blur_vertex,
            &self.pipelines.blur_fragment,
            &self.pipelines.blur_blend_replace,
            &pong.rtv,
            &ping.srv,
            BlurUniforms::gaussian(BlurAxis::Horizontal, blur_size, kernel),
            &half_viewport,
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            3,
            true,
        )?;
        self.dx_blur_pass(
            &self.pipelines.blur_vertex,
            &self.pipelines.blur_fragment,
            &self.pipelines.blur_blend_replace,
            &ping.rtv,
            &pong.srv,
            BlurUniforms::gaussian(BlurAxis::Vertical, blur_size, kernel),
            &half_viewport,
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            3,
            true,
        )?;
        Ok(Some((ping, pong, blur_size)))
    }

    /// Composites `group`, an isolated group's finished target, into the current target, its
    /// parent: blurred by its filters, faded by its opacity, and mixed by its blend mode with a
    /// copy of what is beneath it in the parent, and cut to its mask, if it has one. Gives back
    /// the textures it takes; the caller gives back the group's and its mask's.
    fn composite_group(&mut self, boundary: &GroupBoundary, group: &FrameTarget) -> Result<()> {
        let blurred = match BlurKernel::for_radius(boundary.max_blur_radius()) {
            Some(kernel) => self.blur(&group.color, kernel)?,
            None => None,
        };
        // The blur covers the same viewport rectangle as the group, at half resolution.
        let source = blurred
            .as_ref()
            .map_or(&group.color.srv, |(blurred, _, _)| &blurred.srv)
            .clone();
        let backdrop = if boundary.blend_mode != BlendMode::Normal {
            self.copy_backdrop(group.bounds)?
        } else {
            None
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
        );
        // Normal blending never reads the backdrop, nor an unmasked group its mask; bind the
        // source in their place.
        let backdrop_srv = backdrop.as_ref().map_or(&source, |backdrop| &backdrop.srv);
        let mask_srv = group
            .mask
            .as_ref()
            .map_or(&source, |(mask, _, _)| &mask.srv);
        self.draw_group_composite(uniforms, &source, backdrop_srv, mask_srv)?;

        if let Some((blurred, spare, _)) = blurred {
            self.give_back(blurred);
            self.give_back(spare);
        }
        if let Some(backdrop) = backdrop {
            self.give_back(backdrop);
        }
        Ok(())
    }

    /// Copies what the current target holds under `bounds` into a texture taken from the pool,
    /// which covers `bounds`: what a blend mode mixes a group with. `None` if none of `bounds`
    /// is in the target or the pool has no room.
    fn copy_backdrop(&mut self, bounds: Bounds<DevicePixels>) -> Result<Option<ColorTarget>> {
        let parent = self.targets.last().context("no render target is bound")?;
        let copied = bounds.intersect(&parent.bounds);
        if copied.is_empty() {
            return Ok(None);
        }
        let source_origin = copied.origin - parent.bounds.origin;
        let destination_origin = copied.origin - bounds.origin;
        let parent_texture = parent.color.texture.clone();
        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_mut().context("resources missing")?;
        let Some(backdrop) = resources.target_pool.take(&devices.device, bounds.size) else {
            return Ok(None);
        };
        let region = D3D11_BOX {
            left: source_origin.x.0 as u32,
            top: source_origin.y.0 as u32,
            front: 0,
            right: (source_origin.x.0 + copied.size.width.0) as u32,
            bottom: (source_origin.y.0 + copied.size.height.0) as u32,
            back: 1,
        };
        unsafe {
            devices.device_context.CopySubresourceRegion(
                &backdrop.texture,
                0,
                destination_origin.x.0 as u32,
                destination_origin.y.0 as u32,
                0,
                &parent_texture,
                0,
                Some(&region),
            );
        }
        Ok(Some(backdrop))
    }

    /// Draws the group composite into the current target, sampling the group from `source`,
    /// what is beneath it from `backdrop`, and its mask from `mask`.
    fn draw_group_composite(
        &self,
        uniforms: GroupUniforms,
        source: &Option<ID3D11ShaderResourceView>,
        backdrop: &Option<ID3D11ShaderResourceView>,
        mask: &Option<ID3D11ShaderResourceView>,
    ) -> Result<()> {
        let target = self.targets.last().context("no render target is bound")?;
        let ctx = &self
            .devices
            .as_ref()
            .context("devices missing")?
            .device_context;
        let pipeline = &self.pipelines.group_composite;
        update_buffer(ctx, &pipeline.params_buffer, &[uniforms])?;
        let cbuffers = self.globals.cbuffers();
        let params = [Some(pipeline.params_buffer.clone())];
        let scene_tables = self.level.slot.tables.clone();
        let textures = [source.clone(), backdrop.clone()];
        let unbound: [Option<ID3D11ShaderResourceView>; 2] = [None, None];
        unsafe {
            ctx.OMSetRenderTargets(Some(slice::from_ref(&target.color.rtv)), None);
            ctx.RSSetViewports(Some(slice::from_ref(&target.viewport)));
            ctx.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
            ctx.VSSetShader(&pipeline.vertex, None);
            ctx.PSSetShader(&pipeline.fragment, None);
            ctx.VSSetConstantBuffers(0, Some(&cbuffers));
            ctx.PSSetConstantBuffers(0, Some(&cbuffers));
            ctx.VSSetConstantBuffers(DATA_REGISTER, Some(&params));
            ctx.PSSetConstantBuffers(DATA_REGISTER, Some(&params));
            // A composite clipped to a rounded rectangle finds it through the transform table.
            ctx.VSSetShaderResources(SCENE_TABLES_REGISTER, Some(&scene_tables));
            ctx.PSSetShaderResources(SCENE_TABLES_REGISTER, Some(&scene_tables));
            self.globals.bind_photo_tiles(ctx);
            ctx.PSSetShaderResources(GROUP_TEXTURE_REGISTER, Some(&textures));
            ctx.PSSetShaderResources(MASK_TEXTURE_REGISTER, Some(slice::from_ref(mask)));
            ctx.PSSetSamplers(
                GROUP_SAMPLER_REGISTER,
                Some(slice::from_ref(&self.globals.sampler)),
            );
            ctx.OMSetBlendState(&pipeline.blend, None, 0xFFFFFFFF);
            ctx.DrawInstanced(4, 1, 0, 0);
            // Unbind the pooled textures, which later groups may draw into.
            ctx.PSSetShaderResources(GROUP_TEXTURE_REGISTER, Some(&unbound));
            ctx.PSSetShaderResources(MASK_TEXTURE_REGISTER, Some(&unbound[..1]));
        }
        Ok(())
    }

    /// Copy the offscreen scene texture into the swapchain render target.
    fn dx_blit(
        &self,
        source_srv: &Option<ID3D11ShaderResourceView>,
        target_rtv: &Option<ID3D11RenderTargetView>,
    ) -> Result<()> {
        let full_vp = self
            .resources
            .as_ref()
            .context("resources missing")?
            .viewport;
        self.dx_blur_pass(
            &self.pipelines.blur_downsample_vertex,
            &self.pipelines.blur_downsample_fragment,
            &self.pipelines.blur_blend_replace,
            target_rtv,
            source_srv,
            BlurUniforms::copy([self.width as f32, self.height as f32]),
            &full_vp,
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            3,
            true,
        )
    }

    pub(crate) fn gpu_specs(&self) -> Result<GpuSpecs> {
        let devices = self.devices.as_ref().context("devices missing")?;
        let desc = unsafe { devices.adapter.GetDesc1() }?;
        let is_software_emulated = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
        let device_name = String::from_utf16_lossy(&desc.Description)
            .trim_matches(char::from(0))
            .to_string();
        let driver_name = match desc.VendorId {
            0x10DE => "NVIDIA Corporation".to_string(),
            0x1002 => "AMD Corporation".to_string(),
            0x8086 => "Intel Corporation".to_string(),
            id => format!("Unknown Vendor (ID: {:#X})", id),
        };
        let driver_version = match desc.VendorId {
            0x10DE => nvidia::get_driver_version(),
            0x1002 => amd::get_driver_version(),
            // For Intel and other vendors, we use the DXGI API to get the driver version.
            _ => dxgi::get_driver_version(&devices.adapter),
        }
        .context("Failed to get gpu driver info")
        .log_err()
        .unwrap_or("Unknown Driver".to_string());
        Ok(GpuSpecs {
            is_software_emulated,
            device_name,
            driver_name,
            driver_info: driver_version,
        })
    }

    pub(crate) fn get_font_info() -> &'static FontInfo {
        static CACHED_FONT_INFO: OnceLock<FontInfo> = OnceLock::new();
        CACHED_FONT_INFO.get_or_init(|| unsafe {
            let factory: IDWriteFactory5 = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED).unwrap();
            let render_params: IDWriteRenderingParams1 =
                factory.CreateRenderingParams().unwrap().cast().unwrap();
            FontInfo {
                gamma_ratios: gpui::get_gamma_correction_ratios(render_params.GetGamma()),
                grayscale_enhanced_contrast: render_params.GetGrayscaleEnhancedContrast(),
                subpixel_enhanced_contrast: render_params.GetEnhancedContrast(),
                is_bgr: render_params.GetPixelGeometry() == DWRITE_PIXEL_GEOMETRY_BGR,
            }
        })
    }

    pub(crate) fn mark_drawable(&mut self) {
        self.skip_draws = false;
    }
}

impl DirectXResources {
    pub fn new(
        devices: &DirectXRendererDevices,
        width: u32,
        height: u32,
        hwnd: HWND,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        let swap_chain = if disable_direct_composition {
            create_swap_chain(&devices.dxgi_factory, &devices.device, hwnd, width, height)?
        } else {
            create_swap_chain_for_composition(
                &devices.dxgi_factory,
                &devices.device,
                width,
                height,
            )?
        };

        let (render_target, render_target_view, viewport) =
            create_resources(devices, &swap_chain, width, height)?;
        let rasterizer_state = create_rasterizer_state(&devices.device, false)?;
        let scissored_rasterizer_state = create_rasterizer_state(&devices.device, true)?;
        unsafe { devices.device_context.RSSetState(&rasterizer_state) };
        Ok(Self {
            swap_chain,
            render_target: Some(render_target),
            render_target_view,
            path: None,
            scene_color: None,
            target_pool: TexturePool::new(RENDER_TARGET_FORMAT),
            surface_views: FxHashMap::default(),
            viewport,
            rasterizer_state,
            scissored_rasterizer_state,
        })
    }

    #[inline]
    fn recreate_resources(
        &mut self,
        devices: &DirectXRendererDevices,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let (render_target, render_target_view, viewport) =
            create_resources(devices, &self.swap_chain, width, height)?;
        self.render_target = Some(render_target);
        self.render_target_view = render_target_view;
        // Intermediate textures are size-dependent and recreated lazily if a later scene needs
        // them. Ordinary scenes therefore pay neither the allocation nor resize cost. Group
        // targets are sized to their groups, not the window, so the pool keeps them.
        self.path = None;
        self.scene_color = None;
        self.viewport = viewport;
        Ok(())
    }

    /// The window's size, in device pixels.
    fn viewport_size(&self) -> Size<DevicePixels> {
        size(
            DevicePixels(self.viewport.Width as i32),
            DevicePixels(self.viewport.Height as i32),
        )
    }

    /// The viewport rectangle the window's target covers: all of it.
    fn window_bounds(&self) -> Bounds<DevicePixels> {
        Bounds {
            origin: point(DevicePixels(0), DevicePixels(0)),
            size: self.viewport_size(),
        }
    }

    fn ensure_scene_color(&mut self, device: &ID3D11Device) -> Result<&ColorTarget> {
        if self.scene_color.is_none() {
            self.scene_color = Some(ColorTarget::new(
                device,
                RENDER_TARGET_FORMAT,
                self.viewport_size(),
            )?);
        }
        self.scene_color
            .as_ref()
            .context("scene color target was inserted above")
    }

    fn ensure_path_resources(&mut self, device: &ID3D11Device) -> Result<()> {
        if self.path.is_none() {
            self.path = Some(PathResources::new(
                device,
                self.viewport.Width as u32,
                self.viewport.Height as u32,
            )?);
        }
        Ok(())
    }

    fn retain_surface_views(&mut self, surfaces: &[PaintSurface]) {
        let active_keys = surfaces
            .iter()
            .filter_map(|surface| match &surface.source {
                gpui::SurfaceSource::WindowsCapture(frame) => {
                    Some(frame.texture().as_raw() as usize)
                }
                _ => None,
            })
            .collect::<SmallVec<[usize; 4]>>();
        self.surface_views
            .retain(|key, _| active_keys.contains(key));
    }
}

impl DirectXRenderPipelines {
    pub fn new(device: &ID3D11Device) -> Result<Self> {
        let shadow_pipeline = PipelineState::new(
            device,
            "shadow_pipeline",
            ShaderModule::Shadow,
            4,
            create_blend_state(device)?,
        )?
        .with_variant(device, ShaderModule::SmoothedShadow)?;
        let quad_pipeline = PipelineState::new(
            device,
            "quad_pipeline",
            ShaderModule::Quad,
            64,
            create_blend_state(device)?,
        )?
        .with_variant(device, ShaderModule::SmoothedQuad)?;
        let path_rasterization_pipeline = PipelineState::new(
            device,
            "path_rasterization_pipeline",
            ShaderModule::PathRasterization,
            32,
            create_premultiplied_blend_state(device)?,
        )?;
        let path_sprite_pipeline = PipelineState::new(
            device,
            "path_sprite_pipeline",
            ShaderModule::PathSprite,
            4,
            create_premultiplied_blend_state(device)?,
        )?;
        let underline_pipeline = PipelineState::new(
            device,
            "underline_pipeline",
            ShaderModule::Underline,
            4,
            create_blend_state(device)?,
        )?;
        let mono_sprites = PipelineState::new(
            device,
            "monochrome_sprite_pipeline",
            ShaderModule::MonochromeSprite,
            512,
            create_blend_state(device)?,
        )?;
        let subpixel_sprites = PipelineState::new(
            device,
            "subpixel_sprite_pipeline",
            ShaderModule::SubpixelSprite,
            512,
            create_blend_state_for_subpixel_rendering(device)?,
        )?;
        let poly_sprites = PipelineState::new(
            device,
            "polychrome_sprite_pipeline",
            ShaderModule::PolychromeSprite,
            16,
            create_blend_state(device)?,
        )?
        .with_variant(device, ShaderModule::SmoothedPolychromeSprite)?;
        let meshes = PipelineState::new(
            device,
            "mesh_pipeline",
            ShaderModule::Mesh,
            16,
            create_blend_state(device)?,
        )?;

        let blur_downsample = ShaderModule::BlurDownsample.bytecode()?;
        let blur_downsample_vertex = create_vertex_shader(device, blur_downsample.vertex)?;
        let blur_downsample_fragment = create_fragment_shader(device, blur_downsample.fragment)?;
        let blur = ShaderModule::Blur.bytecode()?;
        let blur_vertex = create_vertex_shader(device, blur.vertex)?;
        let blur_fragment = create_fragment_shader(device, blur.fragment)?;
        let blur_composite = ShaderModule::BlurComposite.bytecode()?;
        let blur_composite_vertex = create_vertex_shader(device, blur_composite.vertex)?;
        let blur_composite_fragment = create_fragment_shader(device, blur_composite.fragment)?;
        let smoothed_blur_composite = ShaderModule::SmoothedBlurComposite.bytecode()?;
        let smoothed_blur_composite_vertex =
            create_vertex_shader(device, smoothed_blur_composite.vertex)?;
        let smoothed_blur_composite_fragment =
            create_fragment_shader(device, smoothed_blur_composite.fragment)?;
        let blur_params_buffer =
            create_constant_buffer(device, std::mem::size_of::<BlurUniforms>())?;
        let blur_blend_replace = create_blend_state_no_blend(device)?;
        // Premultiplied (One / InvSrcAlpha) — the composite outputs a premultiplied blurred sample;
        // straight-alpha blending would darken the faded edges.
        let blur_blend_composite = create_premultiplied_blend_state(device)?;

        let group_composite = ShaderModule::GroupComposite.bytecode()?;
        let group_composite = GroupCompositePipeline {
            vertex: create_vertex_shader(device, group_composite.vertex)?,
            fragment: create_fragment_shader(device, group_composite.fragment)?,
            params_buffer: create_constant_buffer(device, std::mem::size_of::<GroupUniforms>())?,
            // The composite outputs the group premultiplied, as its target holds it.
            blend: create_premultiplied_blend_state(device)?,
        };

        let surface = ShaderModule::Surface.bytecode()?;
        let surfaces = SurfacePipeline {
            vertex: create_vertex_shader(device, surface.vertex)?,
            fragment: create_fragment_shader(device, surface.fragment)?,
            params_buffer: create_constant_buffer(device, std::mem::size_of::<SurfaceUniforms>())?,
            blend: create_blend_state(device)?,
        };

        Ok(Self {
            shadow_pipeline,
            quad_pipeline,
            path_rasterization_pipeline,
            path_sprite_pipeline,
            underline_pipeline,
            mono_sprites,
            subpixel_sprites,
            poly_sprites,
            meshes,
            surfaces,
            blur_downsample_vertex,
            blur_downsample_fragment,
            blur_vertex,
            blur_fragment,
            blur_composite_vertex,
            blur_composite_fragment,
            smoothed_blur_composite_vertex,
            smoothed_blur_composite_fragment,
            blur_params_buffer,
            blur_blend_replace,
            blur_blend_composite,
            group_composite,
        })
    }
}

impl DirectComposition {
    pub fn new(dxgi_device: &IDXGIDevice, hwnd: HWND) -> Result<Self> {
        let comp_device = get_comp_device(dxgi_device)?;
        let comp_target = unsafe { comp_device.CreateTargetForHwnd(hwnd, true) }?;
        let comp_visual = unsafe { comp_device.CreateVisual() }?;

        Ok(Self {
            comp_device,
            comp_target,
            comp_visual,
        })
    }

    pub fn set_swap_chain(&self, swap_chain: &IDXGISwapChain1) -> Result<()> {
        unsafe {
            self.comp_visual.SetContent(swap_chain)?;
            self.comp_target.SetRoot(&self.comp_visual)?;
            self.comp_device.Commit()?;
        }
        Ok(())
    }
}

impl DirectXGlobalElements {
    pub fn new(device: &ID3D11Device) -> Result<Self> {
        let globals_buffer = create_constant_buffer(device, std::mem::size_of::<GlobalUniforms>())?;
        let font_buffer =
            create_constant_buffer(device, std::mem::size_of::<FontRasterizationUniforms>())?;
        let draw_constants_buffer =
            create_constant_buffer(device, std::mem::size_of::<Dx11DrawConstants>())?;

        let sampler = unsafe {
            let desc = D3D11_SAMPLER_DESC {
                Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D11_TEXTURE_ADDRESS_WRAP,
                AddressV: D3D11_TEXTURE_ADDRESS_WRAP,
                AddressW: D3D11_TEXTURE_ADDRESS_WRAP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D11_COMPARISON_ALWAYS,
                BorderColor: [0.0; 4],
                MinLOD: 0.0,
                MaxLOD: D3D11_FLOAT32_MAX,
            };
            let mut output = None;
            device.CreateSamplerState(&desc, Some(&mut output))?;
            output
        };

        Ok(Self {
            globals_buffer: Some(globals_buffer),
            font_buffer: Some(font_buffer),
            draw_constants_buffer,
            sampler,
            transforms: SceneTableBuffer::new(device, "scene_transforms")?,
            clips: SceneTableBuffer::new(device, "scene_clips")?,
            paints: SceneTableBuffer::new(device, "scene_paints")?,
            photo_tiles: PhotoTiles::new(device)?,
        })
    }
}

/// One generated instanced pipeline plus its whole-frame instance buffer.
///
/// The scene uploads every `T` of the frame once; batches then address sub-ranges of that
/// buffer. Direct3D 11 leaves `SV_InstanceID` zero-based for every draw regardless of
/// `StartInstanceLocation`, so the batch base reaches the shader through the draw-constants
/// cbuffer instead. [`PipelineState::draw`] is the single place that issues a draw, and it
/// always writes those constants first.
struct PipelineState<T> {
    label: &'static str,
    specification: &'static shader_interface::Pipeline,
    vertex: ID3D11VertexShader,
    fragment: ID3D11PixelShader,
    variant: Option<PipelineVariant>,
    draw_constants: Dx11DrawConstantsBinding,
    buffer: ID3D11Buffer,
    buffer_size: usize,
    view: Option<ID3D11ShaderResourceView>,
    blend_state: ID3D11BlendState,
    _marker: std::marker::PhantomData<T>,
}

pub(crate) struct PipelineVariant {
    pub(crate) specification: &'static shader_interface::Pipeline,
    pub(crate) vertex: ID3D11VertexShader,
    pub(crate) fragment: ID3D11PixelShader,
}

impl<T> PipelineState<T> {
    fn new(
        device: &ID3D11Device,
        label: &'static str,
        shader_module: ShaderModule,
        buffer_size: usize,
        blend_state: ID3D11BlendState,
    ) -> Result<Self> {
        let shader = shader_module.shader();
        let bytecode = shader_module.bytecode()?;
        let draw_constants = bytecode.draw_constants.with_context(|| {
            format!("{label} was generated without DX11 draw constants and cannot draw batches")
        })?;
        let vertex = create_vertex_shader(device, bytecode.vertex)?;
        let fragment = create_fragment_shader(device, bytecode.fragment)?;
        let buffer = create_buffer(device, std::mem::size_of::<T>(), buffer_size)?;
        let view = create_buffer_view(device, &buffer)?;

        Ok(PipelineState {
            label,
            specification: shader.pipeline,
            vertex,
            fragment,
            variant: None,
            draw_constants,
            buffer,
            buffer_size,
            view,
            blend_state,
            _marker: std::marker::PhantomData,
        })
    }

    fn with_variant(mut self, device: &ID3D11Device, shader_module: ShaderModule) -> Result<Self> {
        let shader = shader_module.shader();
        let bytecode = shader_module.bytecode()?;
        anyhow::ensure!(
            shader.pipeline.data_layout == self.specification.data_layout
                && shader.pipeline.topology == self.specification.topology
                && shader.pipeline.vertex_count == self.specification.vertex_count,
            "{} variant has an incompatible pipeline layout",
            self.label,
        );
        let draw_constants = bytecode.draw_constants.with_context(|| {
            format!(
                "{} variant was generated without DX11 draw constants",
                self.label
            )
        })?;
        anyhow::ensure!(
            draw_constants == self.draw_constants,
            "{} variant uses a different DX11 draw-constants register",
            self.label,
        );
        self.variant = Some(PipelineVariant {
            specification: shader.pipeline,
            vertex: create_vertex_shader(device, bytecode.vertex)?,
            fragment: create_fragment_shader(device, bytecode.fragment)?,
        });
        Ok(self)
    }

    fn update_buffer(
        &mut self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
        data: &[T],
    ) -> Result<()> {
        self.update_sections(device, device_context, &[data])
            .map(drop)
    }

    /// Uploads `sections`, one scene's instances each, end to end, and returns where each one
    /// starts, the base its batches' instance ranges are drawn from.
    fn update_sections(
        &mut self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
        sections: &[&[T]],
    ) -> Result<SmallVec<[u32; 4]>> {
        let (offsets, len) = section_offsets(sections, 1);
        let bases = offsets
            .iter()
            .map(|&offset| u32::try_from(offset))
            .collect::<Result<_, _>>()
            .with_context(|| format!("{} instances exceed the D3D11 instance limit", self.label))?;
        if len == 0 {
            return Ok(bases);
        }
        if self.buffer_size < len {
            let element_size = std::mem::size_of::<T>();
            anyhow::ensure!(
                element_size > 0,
                "{} cannot store zero-sized instances",
                self.label
            );
            let required_size = element_size
                .checked_mul(len)
                .context("instance-buffer byte size overflow")?;
            anyhow::ensure!(
                required_size <= MAX_INSTANCE_BUFFER_SIZE,
                "{} buffer needs {required_size} bytes, above the {MAX_INSTANCE_BUFFER_SIZE}-byte limit",
                self.label,
            );
            let max_elements = MAX_INSTANCE_BUFFER_SIZE / element_size;
            let new_buffer_size = len
                .checked_next_power_of_two()
                .unwrap_or(max_elements)
                .min(max_elements);
            anyhow::ensure!(new_buffer_size >= len, "instance-buffer capacity overflow");
            log::debug!(
                "Updating {} buffer size from {} to {}",
                self.label,
                self.buffer_size,
                new_buffer_size
            );
            let buffer = create_buffer(device, std::mem::size_of::<T>(), new_buffer_size)?;
            let view = create_buffer_view(device, &buffer)?;
            self.buffer = buffer;
            self.view = view;
            self.buffer_size = new_buffer_size;
        }
        update_buffer_sections(device_context, &self.buffer, sections, &offsets)?;
        Ok(bases)
    }

    /// Draws `instances` of the uploaded frame data as the pipeline's fixed rectangle,
    /// optionally sampling `texture` from the primary texture slot.
    fn draw_instances(
        &self,
        frame: &FrameBindings<'_>,
        texture: Option<&[Option<ID3D11ShaderResourceView>]>,
        instances: InstanceRange,
    ) -> Result<()> {
        let vertex_count = self
            .specification
            .vertex_count
            .fixed()
            .with_context(|| format!("{} has no fixed vertex count", self.label))?;
        self.draw(frame, texture, vertex_count, instances)
    }

    fn draw_instances_variant(
        &self,
        frame: &FrameBindings<'_>,
        texture: Option<&[Option<ID3D11ShaderResourceView>]>,
        instances: InstanceRange,
        use_variant: bool,
    ) -> Result<()> {
        let variant = use_variant.then(|| {
            self.variant
                .as_ref()
                .unwrap_or_else(|| panic!("{} has no shader variant", self.label))
        });
        let specification = variant
            .map(|variant| variant.specification)
            .unwrap_or(self.specification);
        let vertex_count = specification
            .vertex_count
            .fixed()
            .with_context(|| format!("{} has no fixed vertex count", self.label))?;
        self.draw_with_variant(frame, texture, vertex_count, instances, variant)
    }

    /// Draws `instances` with `linked`, the pipeline's shaders with shader
    /// programs linked in, in place of its own.
    fn draw_instances_linked(
        &self,
        frame: &FrameBindings<'_>,
        texture: Option<&[Option<ID3D11ShaderResourceView>]>,
        instances: InstanceRange,
        linked: &PipelineVariant,
    ) -> Result<()> {
        let vertex_count = linked
            .specification
            .vertex_count
            .fixed()
            .with_context(|| format!("{} has no fixed vertex count", self.label))?;
        self.draw_with_variant(frame, texture, vertex_count, instances, Some(linked))
    }

    /// Draws `vertex_count` vertex-pulled vertices as a single instance, with
    /// `linked` shaders, with shader programs linked in, in place of the
    /// pipeline's own if given.
    fn draw_vertices(
        &self,
        frame: &FrameBindings<'_>,
        vertex_count: u32,
        linked: Option<&PipelineVariant>,
    ) -> Result<()> {
        anyhow::ensure!(
            self.specification.vertex_count.fixed().is_none(),
            "{} draws a fixed vertex count per instance",
            self.label
        );
        self.draw_with_variant(frame, None, vertex_count, InstanceRange::SINGLE, linked)
    }

    fn draw(
        &self,
        frame: &FrameBindings<'_>,
        texture: Option<&[Option<ID3D11ShaderResourceView>]>,
        vertex_count: u32,
        instances: InstanceRange,
    ) -> Result<()> {
        self.draw_with_variant(frame, texture, vertex_count, instances, None)
    }

    fn draw_with_variant(
        &self,
        frame: &FrameBindings<'_>,
        texture: Option<&[Option<ID3D11ShaderResourceView>]>,
        vertex_count: u32,
        instances: InstanceRange,
        variant: Option<&PipelineVariant>,
    ) -> Result<()> {
        if instances.is_empty() || vertex_count == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            instances.end() as usize <= self.buffer_size,
            "DirectX instance range {}..{} exceeds the {} buffer of {} elements",
            instances.first(),
            instances.end(),
            self.label,
            self.buffer_size,
        );
        let ctx = frame.device_context;
        update_buffer(
            ctx,
            &frame.globals.draw_constants_buffer,
            &[Dx11DrawConstants::for_instances(instances.first())],
        )?;
        let specification = variant
            .map(|variant| variant.specification)
            .unwrap_or(self.specification);
        let topology = match specification.topology {
            shader_interface::PrimitiveTopology::TriangleList => {
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST
            }
            shader_interface::PrimitiveTopology::TriangleStrip => {
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP
            }
        };
        let draw_constants = [Some(frame.globals.draw_constants_buffer.clone())];
        unsafe {
            ctx.VSSetShaderResources(SCENE_TABLES_REGISTER, Some(frame.scene_tables));
            ctx.PSSetShaderResources(SCENE_TABLES_REGISTER, Some(frame.scene_tables));
            frame.globals.bind_photo_tiles(ctx);
            ctx.VSSetShaderResources(DATA_REGISTER, Some(slice::from_ref(&self.view)));
            ctx.PSSetShaderResources(DATA_REGISTER, Some(slice::from_ref(&self.view)));
            ctx.IASetPrimitiveTopology(topology);
            ctx.RSSetViewports(Some(slice::from_ref(frame.viewport)));
            ctx.VSSetShader(
                variant
                    .map(|variant| &variant.vertex)
                    .unwrap_or(&self.vertex),
                None,
            );
            ctx.PSSetShader(
                variant
                    .map(|variant| &variant.fragment)
                    .unwrap_or(&self.fragment),
                None,
            );
            ctx.VSSetConstantBuffers(0, Some(&frame.globals.cbuffers()));
            ctx.PSSetConstantBuffers(0, Some(&frame.globals.cbuffers()));
            ctx.VSSetConstantBuffers(self.draw_constants.register, Some(&draw_constants));
            ctx.OMSetBlendState(&self.blend_state, None, 0xFFFFFFFF);
            if let Some(texture) = texture {
                ctx.PSSetSamplers(
                    PRIMARY_SAMPLER_REGISTER,
                    Some(slice::from_ref(&frame.globals.sampler)),
                );
                // The vertex stage reads the atlas dimensions for tile coordinates.
                ctx.VSSetShaderResources(PRIMARY_TEXTURE_REGISTER, Some(texture));
                ctx.PSSetShaderResources(PRIMARY_TEXTURE_REGISTER, Some(texture));
            }
            // `StartInstanceLocation` stays zero: the shader adds the base itself.
            ctx.DrawInstanced(vertex_count, instances.count(), 0, 0);
        }
        Ok(())
    }
}

impl PipelineState<MeshInstance> {
    /// Draws `mesh` as instance `instance` of the frame's mesh instances,
    /// with `linked` shaders, with shader programs linked in, in place of
    /// the pipeline's own if given.
    fn draw_mesh(
        &self,
        frame: &FrameBindings<'_>,
        mesh: &DirectXMesh,
        instance: InstanceRange,
        linked: Option<&PipelineVariant>,
    ) -> Result<()> {
        anyhow::ensure!(
            instance.end() as usize <= self.buffer_size,
            "mesh instance {} exceeds the mesh buffer of {} elements",
            instance.first(),
            self.buffer_size,
        );
        let ctx = frame.device_context;
        update_buffer(
            ctx,
            &frame.globals.draw_constants_buffer,
            &[Dx11DrawConstants::for_instances(instance.first())],
        )?;
        let draw_constants = [Some(frame.globals.draw_constants_buffer.clone())];
        unsafe {
            ctx.VSSetShaderResources(SCENE_TABLES_REGISTER, Some(frame.scene_tables));
            ctx.PSSetShaderResources(SCENE_TABLES_REGISTER, Some(frame.scene_tables));
            frame.globals.bind_photo_tiles(ctx);
            ctx.VSSetShaderResources(DATA_REGISTER, Some(slice::from_ref(&self.view)));
            ctx.PSSetShaderResources(DATA_REGISTER, Some(slice::from_ref(&self.view)));
            ctx.VSSetShaderResources(
                MESH_VERTICES_REGISTER,
                Some(slice::from_ref(&mesh.vertices)),
            );
            ctx.IASetIndexBuffer(&mesh.indices, DXGI_FORMAT_R32_UINT, 0);
            ctx.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.RSSetViewports(Some(slice::from_ref(frame.viewport)));
            ctx.VSSetShader(
                linked.map(|linked| &linked.vertex).unwrap_or(&self.vertex),
                None,
            );
            ctx.PSSetShader(
                linked
                    .map(|linked| &linked.fragment)
                    .unwrap_or(&self.fragment),
                None,
            );
            ctx.VSSetConstantBuffers(0, Some(&frame.globals.cbuffers()));
            ctx.PSSetConstantBuffers(0, Some(&frame.globals.cbuffers()));
            ctx.VSSetConstantBuffers(self.draw_constants.register, Some(&draw_constants));
            ctx.OMSetBlendState(&self.blend_state, None, 0xFFFFFFFF);
            // `StartInstanceLocation` stays zero: the shader adds the base itself.
            ctx.DrawIndexedInstanced(mesh.index_count, 1, 0, 0, 0);
        }
        Ok(())
    }
}

/// A mesh kept on the GPU: a raw view of its vertices, and its indices.
#[derive(Clone)]
struct DirectXMesh {
    vertices: Option<ID3D11ShaderResourceView>,
    indices: ID3D11Buffer,
    index_count: u32,
}

impl DirectXMesh {
    fn upload(device: &ID3D11Device, mesh: &gpui::Mesh) -> Result<Self> {
        // SAFETY: `MeshVertex` is `repr(C)` floats and indices are `u32`s.
        let vertex_bytes = unsafe {
            slice::from_raw_parts(
                mesh.vertices().as_ptr().cast::<u8>(),
                std::mem::size_of_val(mesh.vertices()),
            )
        };
        let index_bytes = unsafe {
            slice::from_raw_parts(
                mesh.indices().as_ptr().cast::<u8>(),
                std::mem::size_of_val(mesh.indices()),
            )
        };
        let immutable = |bytes: &[u8], bind: u32, misc: u32| -> Result<ID3D11Buffer> {
            let desc = D3D11_BUFFER_DESC {
                ByteWidth: u32::try_from(bytes.len())
                    .context("a mesh past the D3D11 buffer limit")?,
                Usage: D3D11_USAGE_IMMUTABLE,
                BindFlags: bind,
                MiscFlags: misc,
                ..Default::default()
            };
            let data = D3D11_SUBRESOURCE_DATA {
                pSysMem: bytes.as_ptr().cast(),
                ..Default::default()
            };
            let mut buffer = None;
            unsafe { device.CreateBuffer(&desc, Some(&data), Some(&mut buffer)) }?;
            buffer.context("CreateBuffer returned no mesh buffer")
        };
        let vertices = immutable(
            vertex_bytes,
            D3D11_BIND_SHADER_RESOURCE.0 as u32,
            D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS.0 as u32,
        )?;
        let indices = immutable(index_bytes, D3D11_BIND_INDEX_BUFFER.0 as u32, 0)?;
        Ok(Self {
            vertices: create_buffer_view(device, &vertices)?,
            indices,
            index_count: mesh.indices().len() as u32,
        })
    }
}

impl Drop for DirectXRenderer {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        if let Some(devices) = &self.devices {
            report_live_objects(&devices.device).ok();
        }
    }
}

#[inline]
fn get_comp_device(dxgi_device: &IDXGIDevice) -> Result<IDCompositionDevice> {
    Ok(unsafe { DCompositionCreateDevice(dxgi_device)? })
}

fn create_swap_chain_for_composition(
    dxgi_factory: &IDXGIFactory6,
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<IDXGISwapChain1> {
    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: RENDER_TARGET_FORMAT,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: BUFFER_COUNT as u32,
        // Composition SwapChains only support the DXGI_SCALING_STRETCH Scaling.
        Scaling: DXGI_SCALING_STRETCH,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
        AlphaMode: DXGI_ALPHA_MODE_PREMULTIPLIED,
        Flags: 0,
    };
    Ok(unsafe { dxgi_factory.CreateSwapChainForComposition(device, &desc, None)? })
}

fn create_swap_chain(
    dxgi_factory: &IDXGIFactory6,
    device: &ID3D11Device,
    hwnd: HWND,
    width: u32,
    height: u32,
) -> Result<IDXGISwapChain1> {
    use windows::Win32::Graphics::Dxgi::DXGI_MWA_NO_ALT_ENTER;

    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: RENDER_TARGET_FORMAT,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: BUFFER_COUNT as u32,
        Scaling: DXGI_SCALING_NONE,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
        AlphaMode: DXGI_ALPHA_MODE_IGNORE,
        Flags: 0,
    };
    let swap_chain =
        unsafe { dxgi_factory.CreateSwapChainForHwnd(device, hwnd, &desc, None, None) }?;
    unsafe { dxgi_factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER) }?;
    Ok(swap_chain)
}

#[inline]
fn create_resources(
    devices: &DirectXRendererDevices,
    swap_chain: &IDXGISwapChain1,
    width: u32,
    height: u32,
) -> Result<(
    ID3D11Texture2D,
    Option<ID3D11RenderTargetView>,
    D3D11_VIEWPORT,
)> {
    let (render_target, render_target_view) =
        create_render_target_and_its_view(swap_chain, &devices.device)?;
    let viewport = set_viewport(&devices.device_context, width as f32, height as f32);
    Ok((render_target, render_target_view, viewport))
}

fn create_render_target_and_its_view(
    swap_chain: &IDXGISwapChain1,
    device: &ID3D11Device,
) -> Result<(ID3D11Texture2D, Option<ID3D11RenderTargetView>)> {
    let render_target: ID3D11Texture2D = unsafe { swap_chain.GetBuffer(0) }?;
    let mut render_target_view = None;
    unsafe { device.CreateRenderTargetView(&render_target, None, Some(&mut render_target_view))? };
    Ok((render_target, render_target_view))
}

#[inline]
fn create_path_intermediate_texture(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<(ID3D11Texture2D, Option<ID3D11ShaderResourceView>)> {
    let texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: RENDER_TARGET_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output.unwrap()
    };

    let mut shader_resource_view = None;
    unsafe { device.CreateShaderResourceView(&texture, None, Some(&mut shader_resource_view))? };

    Ok((texture, Some(shader_resource_view.unwrap())))
}

#[inline]
fn create_path_intermediate_msaa_texture_and_view(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<(ID3D11Texture2D, Option<ID3D11RenderTargetView>)> {
    let msaa_texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: RENDER_TARGET_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: PATH_MULTISAMPLE_COUNT,
                Quality: D3D11_STANDARD_MULTISAMPLE_PATTERN.0 as u32,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output.unwrap()
    };
    let mut msaa_view = None;
    unsafe { device.CreateRenderTargetView(&msaa_texture, None, Some(&mut msaa_view))? };
    Ok((msaa_texture, Some(msaa_view.unwrap())))
}

#[inline]
fn set_viewport(device_context: &ID3D11DeviceContext, width: f32, height: f32) -> D3D11_VIEWPORT {
    let viewport = [D3D11_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: width,
        Height: height,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    }];
    unsafe { device_context.RSSetViewports(Some(&viewport)) };
    viewport[0]
}

/// The rasterizer state scene geometry is drawn with, cutting it to the scissor rectangle if
/// `scissor`.
fn create_rasterizer_state(device: &ID3D11Device, scissor: bool) -> Result<ID3D11RasterizerState> {
    let desc = D3D11_RASTERIZER_DESC {
        FillMode: D3D11_FILL_SOLID,
        CullMode: D3D11_CULL_NONE,
        FrontCounterClockwise: false.into(),
        DepthBias: 0,
        DepthBiasClamp: 0.0,
        SlopeScaledDepthBias: 0.0,
        DepthClipEnable: true.into(),
        ScissorEnable: scissor.into(),
        MultisampleEnable: true.into(),
        AntialiasedLineEnable: false.into(),
    };
    let mut state = None;
    unsafe { device.CreateRasterizerState(&desc, Some(&mut state))? };
    state.context("creating a rasterizer state")
}

// https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ns-d3d11-d3d11_blend_desc
#[inline]
fn create_blend_state(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    // Source-over alpha, so that a group's target holds the coverage of what it draws.
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_for_subpixel_rendering(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC1_COLOR;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC1_COLOR;
    // It does not make sense to draw transparent subpixel-rendered text, since it cannot be meaningfully alpha-blended onto anything else.
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    desc.RenderTarget[0].RenderTargetWriteMask =
        D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8 & !D3D11_COLOR_WRITE_ENABLE_ALPHA.0 as u8;

    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
pub(crate) fn create_premultiplied_blend_state(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    // If the feature level is set to greater than D3D_FEATURE_LEVEL_9_3, the display
    // device performs the blend in linear space, which is ideal.
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    // Source-over alpha, so that a group's target holds the coverage of what it draws.
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

/// Create a CPU-writable dynamic constant buffer of the given byte size (rounded up to 16).
#[inline]
pub(crate) fn create_constant_buffer(
    device: &ID3D11Device,
    byte_size: usize,
) -> Result<ID3D11Buffer> {
    let desc = D3D11_BUFFER_DESC {
        ByteWidth: byte_size.next_multiple_of(16) as u32,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        ..Default::default()
    };
    let mut buffer = None;
    unsafe { device.CreateBuffer(&desc, None, Some(&mut buffer)) }?;
    Ok(buffer.unwrap())
}

/// A blend state that overwrites the target (no blending) — used for the blur downsample and
/// gaussian passes.
#[inline]
fn create_blend_state_no_blend(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = false.into();
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
pub(crate) fn create_vertex_shader(
    device: &ID3D11Device,
    bytes: &[u8],
) -> Result<ID3D11VertexShader> {
    unsafe {
        let mut shader = None;
        device.CreateVertexShader(bytes, None, Some(&mut shader))?;
        Ok(shader.unwrap())
    }
}

#[inline]
pub(crate) fn create_fragment_shader(
    device: &ID3D11Device,
    bytes: &[u8],
) -> Result<ID3D11PixelShader> {
    unsafe {
        let mut shader = None;
        device.CreatePixelShader(bytes, None, Some(&mut shader))?;
        Ok(shader.unwrap())
    }
}

#[inline]
fn create_buffer(
    device: &ID3D11Device,
    element_size: usize,
    buffer_size: usize,
) -> Result<ID3D11Buffer> {
    anyhow::ensure!(
        element_size > 0,
        "cannot create a buffer for zero-sized elements"
    );
    let byte_width = element_size
        .checked_mul(buffer_size)
        .context("instance-buffer byte size overflow")?;
    anyhow::ensure!(
        byte_width <= u32::MAX as usize,
        "instance-buffer byte size exceeds the D3D11 buffer limit"
    );
    anyhow::ensure!(
        byte_width % 4 == 0,
        "instance-buffer byte size must be four-byte aligned"
    );
    let desc = D3D11_BUFFER_DESC {
        // The HLSL reads instances through a raw `ByteAddressBuffer` view, which needs
        // a raw-view-enabled buffer with 4-byte-aligned contents.
        ByteWidth: byte_width as u32,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        MiscFlags: D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS.0 as u32,
        ..Default::default()
    };
    let mut buffer = None;
    unsafe { device.CreateBuffer(&desc, None, Some(&mut buffer)) }?;
    Ok(buffer.unwrap())
}

#[inline]
fn create_buffer_view(
    device: &ID3D11Device,
    buffer: &ID3D11Buffer,
) -> Result<Option<ID3D11ShaderResourceView>> {
    let mut buffer_desc = D3D11_BUFFER_DESC::default();
    unsafe { buffer.GetDesc(&mut buffer_desc) };
    create_buffer_section_view(device, buffer, 0, buffer_desc.ByteWidth as usize)
}

/// A raw view of `byte_len` bytes of `buffer` from `byte_offset`, both multiples of four.
fn create_buffer_section_view(
    device: &ID3D11Device,
    buffer: &ID3D11Buffer,
    byte_offset: usize,
    byte_len: usize,
) -> Result<Option<ID3D11ShaderResourceView>> {
    anyhow::ensure!(
        byte_offset.is_multiple_of(4) && byte_len.is_multiple_of(4),
        "a raw buffer view must start and end on four-byte boundaries"
    );
    let desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D11_SRV_DIMENSION_BUFFEREX,
        Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
            BufferEx: D3D11_BUFFEREX_SRV {
                FirstElement: u32::try_from(byte_offset / 4)
                    .context("buffer view offset exceeds the D3D11 limit")?,
                NumElements: u32::try_from(byte_len / 4)
                    .context("buffer view length exceeds the D3D11 limit")?,
                Flags: D3D11_BUFFEREX_SRV_FLAG_RAW.0 as u32,
            },
        },
    };
    let mut view = None;
    unsafe { device.CreateShaderResourceView(buffer, Some(&desc), Some(&mut view)) }?;
    Ok(view)
}

#[inline]
fn update_buffer<T>(
    device_context: &ID3D11DeviceContext,
    buffer: &ID3D11Buffer,
    data: &[T],
) -> Result<()> {
    unsafe {
        let mut dest = std::mem::zeroed();
        device_context.Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut dest))?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), dest.pData as _, data.len());
        device_context.Unmap(buffer, 0);
    }
    Ok(())
}

/// Writes each of `sections` into `buffer` from its element offset in `offsets`, discarding
/// what was there; what lies between them is left undefined.
fn update_buffer_sections<T>(
    device_context: &ID3D11DeviceContext,
    buffer: &ID3D11Buffer,
    sections: &[&[T]],
    offsets: &[usize],
) -> Result<()> {
    debug_assert_eq!(sections.len(), offsets.len());
    unsafe {
        let mut dest = std::mem::zeroed();
        device_context.Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut dest))?;
        let dest = dest.pData.cast::<T>();
        // SAFETY: the buffer holds the sections' extent, which the callers grew it to.
        for (section, &offset) in sections.iter().zip(offsets) {
            std::ptr::copy_nonoverlapping(section.as_ptr(), dest.add(offset), section.len());
        }
        device_context.Unmap(buffer, 0);
    }
    Ok(())
}

/// Converts a render-plan slice of a scene whose instances start at `base` in the frame's
/// buffer into draw arguments, refusing ranges D3D11 cannot address.
fn instance_range(range: &std::ops::Range<usize>, base: u32) -> Result<InstanceRange> {
    let base = base as usize;
    range
        .start
        .checked_add(base)
        .zip(range.end.checked_add(base))
        .and_then(|(start, end)| InstanceRange::new(start..end))
        .with_context(|| format!("batch {range:?} from {base} exceeds the D3D11 instance limit"))
}

/// The scissor rectangle, in `target`'s pixels, that keeps what is drawn into it inside
/// `clip`, a window viewport rectangle.
fn chunk_scissor(target: &FrameTarget, clip: Bounds<ScaledPixels>) -> RECT {
    let (width, height) = (
        target.bounds.size.width.0.max(0) as f32,
        target.bounds.size.height.0.max(0) as f32,
    );
    let origin = (
        target.bounds.origin.x.0 as f32,
        target.bounds.origin.y.0 as f32,
    );
    let left = (clip.origin.x.0 - origin.0).floor().clamp(0., width);
    let top = (clip.origin.y.0 - origin.1).floor().clamp(0., height);
    let right = (clip.origin.x.0 + clip.size.width.0 - origin.0)
        .ceil()
        .clamp(left, width);
    let bottom = (clip.origin.y.0 + clip.size.height.0 - origin.1)
        .ceil()
        .clamp(top, height);
    RECT {
        left: left as i32,
        top: top as i32,
        right: right as i32,
        bottom: bottom as i32,
    }
}

/// The viewport bounds that contain `bounds` moved by `transformation`.
fn transformed_bounds(
    transformation: &TransformationMatrix,
    bounds: Bounds<ScaledPixels>,
) -> Bounds<ScaledPixels> {
    let corners = [
        bounds.origin,
        point(bounds.origin.x + bounds.size.width, bounds.origin.y),
        point(bounds.origin.x, bounds.origin.y + bounds.size.height),
        point(
            bounds.origin.x + bounds.size.width,
            bounds.origin.y + bounds.size.height,
        ),
    ]
    .map(|corner| transformation.apply(corner.map(|value| px(value.0))));
    let (mut left, mut top, mut right, mut bottom) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for corner in corners {
        left = left.min(f32::from(corner.x));
        top = top.min(f32::from(corner.y));
        right = right.max(f32::from(corner.x));
        bottom = bottom.max(f32::from(corner.y));
    }
    Bounds {
        origin: point(ScaledPixels(left), ScaledPixels(top)),
        size: size(ScaledPixels(right - left), ScaledPixels(bottom - top)),
    }
}

#[cfg(debug_assertions)]
fn report_live_objects(device: &ID3D11Device) -> Result<()> {
    let debug_device: ID3D11Debug = device.cast()?;
    unsafe {
        debug_device.ReportLiveDeviceObjects(D3D11_RLDO_DETAIL)?;
    }
    Ok(())
}

const BUFFER_COUNT: usize = 3;

pub(crate) mod shader_resources {
    //! D3D11 bytecode generated from the shared Rust shader sources at build time.

    use anyhow::Result;
    use gpui_render::artifacts::{Dx11Bytecode, Dx11Shader, NATIVE_SHADERS, NativeShader};

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub(crate) enum ShaderModule {
        Quad,
        SmoothedQuad,
        Shadow,
        SmoothedShadow,
        Underline,
        PathRasterization,
        PathSprite,
        MonochromeSprite,
        SubpixelSprite,
        PolychromeSprite,
        SmoothedPolychromeSprite,
        Mesh,
        Surface,
        BlurDownsample,
        Blur,
        BlurComposite,
        SmoothedBlurComposite,
        GroupComposite,
    }

    impl ShaderModule {
        pub(crate) fn shader(self) -> &'static NativeShader {
            let label = match self {
                Self::Quad => "quads",
                Self::SmoothedQuad => "smoothed_quads",
                Self::Shadow => "shadows",
                Self::SmoothedShadow => "smoothed_shadows",
                Self::Underline => "underlines",
                Self::PathRasterization => "path_rasterization",
                Self::PathSprite => "paths",
                Self::MonochromeSprite => "monochrome_sprites",
                Self::SubpixelSprite => "subpixel_sprites",
                Self::PolychromeSprite => "polychrome_sprites",
                Self::SmoothedPolychromeSprite => "smoothed_polychrome_sprites",
                Self::Mesh => "meshes",
                Self::Surface => "surfaces",
                Self::BlurDownsample => "blur_downsample",
                Self::Blur => "blur",
                Self::BlurComposite => "blur_composite",
                Self::SmoothedBlurComposite => "smoothed_blur_composite",
                Self::GroupComposite => "group_composite",
            };
            NATIVE_SHADERS
                .iter()
                .find(|shader| shader.label == label)
                .unwrap_or_else(|| panic!("missing generated native shader {label}"))
        }

        /// Both compiled stages plus the draw-constants contract the vertex stage expects.
        pub(crate) fn bytecode(self) -> Result<Dx11Bytecode> {
            dx11_bytecode(self.shader())
        }
    }

    fn dx11_bytecode(shader: &NativeShader) -> Result<Dx11Bytecode> {
        match shader.dx11 {
            Dx11Shader::Sm50(bytecode) => Ok(bytecode),
            Dx11Shader::NativeWindowsBuildRequired => anyhow::bail!(
                "{} has no DX11 bytecode: build the Windows target on a Windows host; runtime HLSL compilation is intentionally unsupported",
                shader.label,
            ),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use gpui_render::shaders::interface::DataLayout;

        #[test]
        fn every_generated_dx11_artifact_is_available() {
            for shader in NATIVE_SHADERS {
                dx11_bytecode(shader).unwrap_or_else(|error| {
                    panic!("missing bytecode for {}: {error:#}", shader.label)
                });
            }
        }

        /// Instanced pipelines index a whole-frame buffer, so they must carry the base.
        #[test]
        fn instanced_pipelines_declare_draw_constants() {
            for shader in NATIVE_SHADERS {
                let instanced = matches!(
                    shader.pipeline.data_layout,
                    DataLayout::Instances
                        | DataLayout::TexturedInstances
                        | DataLayout::MonochromeSprites
                        | DataLayout::SubpixelSprites
                        | DataLayout::Meshes
                );
                let bytecode = dx11_bytecode(shader).unwrap();
                assert_eq!(
                    bytecode.draw_constants.is_some(),
                    instanced,
                    "{} draw-constants contract does not match its data layout",
                    shader.label,
                );
            }
        }
    }
}

fn with_dll_library<R>(dll_name: PCSTR, f: impl FnOnce(HMODULE) -> Result<R>) -> Result<R> {
    let library = unsafe {
        LoadLibraryA(dll_name).with_context(|| format!("Loading DLL: {}", dll_name.display()))?
    };
    let result = f(library);
    unsafe {
        FreeLibrary(library)
            .with_context(|| format!("Freeing DLL: {}", dll_name.display()))
            .log_err();
    }
    result
}

mod nvidia {
    use std::{
        ffi::CStr,
        os::raw::{c_char, c_int, c_uint},
    };

    use anyhow::Result;
    use windows::{Win32::System::LibraryLoader::GetProcAddress, core::s};

    use super::with_dll_library;

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L180
    const NVAPI_SHORT_STRING_MAX: usize = 64;

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L235
    #[allow(non_camel_case_types)]
    type NvAPI_ShortString = [c_char; NVAPI_SHORT_STRING_MAX];

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L447
    #[allow(non_camel_case_types)]
    type NvAPI_SYS_GetDriverAndBranchVersion_t = unsafe extern "C" fn(
        driver_version: *mut c_uint,
        build_branch_string: *mut NvAPI_ShortString,
    ) -> c_int;

    pub(super) fn get_driver_version() -> Result<String> {
        #[cfg(target_pointer_width = "64")]
        let nvidia_dll_name = s!("nvapi64.dll");
        #[cfg(target_pointer_width = "32")]
        let nvidia_dll_name = s!("nvapi.dll");

        with_dll_library(nvidia_dll_name, |nvidia_dll| unsafe {
            let nvapi_query_addr = GetProcAddress(nvidia_dll, s!("nvapi_QueryInterface"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get nvapi_QueryInterface address"))?;
            let nvapi_query: extern "C" fn(u32) -> *mut () = std::mem::transmute(nvapi_query_addr);

            // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_interface.h#L41
            let nvapi_get_driver_version_ptr = nvapi_query(0x2926aaad);
            if nvapi_get_driver_version_ptr.is_null() {
                anyhow::bail!("Failed to get NVIDIA driver version function pointer");
            }
            let nvapi_get_driver_version: NvAPI_SYS_GetDriverAndBranchVersion_t =
                std::mem::transmute(nvapi_get_driver_version_ptr);

            let mut driver_version: c_uint = 0;
            let mut build_branch_string: NvAPI_ShortString = [0; NVAPI_SHORT_STRING_MAX];
            let result = nvapi_get_driver_version(
                &mut driver_version as *mut c_uint,
                &mut build_branch_string as *mut NvAPI_ShortString,
            );

            if result != 0 {
                anyhow::bail!(
                    "Failed to get NVIDIA driver version, error code: {}",
                    result
                );
            }
            let major = driver_version / 100;
            let minor = driver_version % 100;
            let branch_string = CStr::from_ptr(build_branch_string.as_ptr());
            Ok(format!(
                "{}.{} {}",
                major,
                minor,
                branch_string.to_string_lossy()
            ))
        })
    }
}

mod amd {
    use std::os::raw::{c_char, c_int, c_void};

    use anyhow::Result;
    use windows::{Win32::System::LibraryLoader::GetProcAddress, core::s};

    use super::with_dll_library;

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L145
    const AGS_CURRENT_VERSION: i32 = (6 << 22) | (3 << 12);

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L204
    // This is an opaque type, using struct to represent it properly for FFI
    #[repr(C)]
    struct AGSContext {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct AGSGPUInfo {
        pub driver_version: *const c_char,
        pub radeon_software_version: *const c_char,
        pub num_devices: c_int,
        pub devices: *mut c_void,
    }

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L429
    #[allow(non_camel_case_types)]
    type agsInitialize_t = unsafe extern "C" fn(
        version: c_int,
        config: *const c_void,
        context: *mut *mut AGSContext,
        gpu_info: *mut AGSGPUInfo,
    ) -> c_int;

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L436
    #[allow(non_camel_case_types)]
    type agsDeInitialize_t = unsafe extern "C" fn(context: *mut AGSContext) -> c_int;

    pub(super) fn get_driver_version() -> Result<String> {
        #[cfg(target_pointer_width = "64")]
        let amd_dll_name = s!("amd_ags_x64.dll");
        #[cfg(target_pointer_width = "32")]
        let amd_dll_name = s!("amd_ags_x86.dll");

        with_dll_library(amd_dll_name, |amd_dll| unsafe {
            let ags_initialize_addr = GetProcAddress(amd_dll, s!("agsInitialize"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get agsInitialize address"))?;
            let ags_deinitialize_addr = GetProcAddress(amd_dll, s!("agsDeInitialize"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get agsDeInitialize address"))?;

            let ags_initialize: agsInitialize_t = std::mem::transmute(ags_initialize_addr);
            let ags_deinitialize: agsDeInitialize_t = std::mem::transmute(ags_deinitialize_addr);

            let mut context: *mut AGSContext = std::ptr::null_mut();
            let mut gpu_info: AGSGPUInfo = AGSGPUInfo {
                driver_version: std::ptr::null(),
                radeon_software_version: std::ptr::null(),
                num_devices: 0,
                devices: std::ptr::null_mut(),
            };

            let result = ags_initialize(
                AGS_CURRENT_VERSION,
                std::ptr::null(),
                &mut context,
                &mut gpu_info,
            );
            if result != 0 {
                anyhow::bail!("Failed to initialize AMD AGS, error code: {}", result);
            }

            // Vulkan actually returns this as the driver version
            let software_version = if !gpu_info.radeon_software_version.is_null() {
                std::ffi::CStr::from_ptr(gpu_info.radeon_software_version)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Unknown Radeon Software Version".to_string()
            };

            let driver_version = if !gpu_info.driver_version.is_null() {
                std::ffi::CStr::from_ptr(gpu_info.driver_version)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Unknown Radeon Driver Version".to_string()
            };

            ags_deinitialize(context);
            Ok(format!("{} ({})", software_version, driver_version))
        })
    }
}

mod dxgi {
    use windows::{
        Win32::Graphics::Dxgi::{IDXGIAdapter1, IDXGIDevice},
        core::Interface,
    };

    pub(super) fn get_driver_version(adapter: &IDXGIAdapter1) -> anyhow::Result<String> {
        let number = unsafe { adapter.CheckInterfaceSupport(&IDXGIDevice::IID as _) }?;
        Ok(format!(
            "{}.{}.{}.{}",
            number >> 48,
            (number >> 32) & 0xFFFF,
            (number >> 16) & 0xFFFF,
            number & 0xFFFF
        ))
    }
}

#[cfg(test)]
mod tests {
    //! Draws through the real Direct3D 11 renderer on a hidden window and reads pixels back.
    //! The scene deliberately splits one primitive kind across two batches so the second
    //! batch starts past the beginning of the frame's instance buffer.

    // Explicit imports: a glob of `super` would also pull in gpui's `#[test]` proc macro.
    use super::DirectXRenderer;
    use crate::directx_devices::DirectXDevices;
    use anyhow::{Context as _, Result};
    use gpui::{
        AtlasKey, AtlasTile, BorderStyle, Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla,
        ImageId, MonochromeSprite, PlacedChunk, PlatformAtlas, Point, PolychromeSprite, Primitive,
        PrimitiveBatch, Quad, RenderCommand, RenderImageParams, RenderSvgParams, ScaledPixels,
        Scene, SceneChunk, SceneClip, ScenePaintRef, SceneTransform, ShaderBool, Size,
        TransformationMatrix, WindowBackgroundAppearance, hsla, rgb, rgb_to_hsla,
    };
    use std::borrow::Cow;
    use std::rc::Rc;
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DestroyWindow, WINDOW_EX_STYLE, WS_OVERLAPPED,
    };
    use windows::core::w;

    struct HiddenWindow(HWND);

    impl HiddenWindow {
        fn new() -> Result<Self> {
            let hwnd = unsafe {
                CreateWindowExW(
                    WINDOW_EX_STYLE(0),
                    w!("STATIC"),
                    w!("gpui directx renderer test"),
                    WS_OVERLAPPED,
                    0,
                    0,
                    200,
                    100,
                    None,
                    None,
                    None,
                    None,
                )
            }?;
            Ok(Self(hwnd))
        }
    }

    impl Drop for HiddenWindow {
        fn drop(&mut self) {
            unsafe { DestroyWindow(self.0) }.ok();
        }
    }

    fn scaled(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: Point {
                x: ScaledPixels(x),
                y: ScaledPixels(y),
            },
            size: Size {
                width: ScaledPixels(width),
                height: ScaledPixels(height),
            },
        }
    }

    fn full_mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: scaled(0.0, 0.0, 200.0, 100.0),
            ..Default::default()
        }
    }

    fn dashed_border_scene(dash_length: f32, dash_gap: f32) -> Scene {
        let mut scene = Scene::default();

        for (bounds, corner_smoothing) in [
            (scaled(4.0, 4.0, 12.0, 12.0), 0.0),
            (scaled(24.0, 4.0, 12.0, 12.0), 0.6),
        ] {
            scene.insert_primitive(Quad {
                bounds,
                content_mask: full_mask(),
                background: gpui::ScenePaintRef::from(hsla(0.05, 0.8, 0.45, 1.0)),
                border_style: BorderStyle::Dashed,
                border_dashed_length: dash_length,
                border_dashed_gap: dash_gap,
                border_color: hsla(0.6, 0.9, 0.7, 1.0).into(),
                corner_radii: Corners::all(ScaledPixels(4.0)),
                border_widths: Edges::all(ScaledPixels(2.0)),
                corner_smoothing,
                ..Default::default()
            });
        }

        scene.finish();

        scene
    }

    fn images_differ_in_region(
        first: &image::RgbaImage,
        second: &image::RgbaImage,
        left: u32,
        right: u32,
    ) -> bool {
        (4..16).any(|y| (left..right).any(|x| first.get_pixel(x, y) != second.get_pixel(x, y)))
    }

    fn tile(atlas: &dyn PlatformAtlas, key: AtlasKey, bytes: Vec<u8>) -> AtlasTile {
        atlas
            .get_or_insert_with(&key, &mut || {
                Ok(Some((
                    Size {
                        width: DevicePixels(8),
                        height: DevicePixels(8),
                    },
                    Cow::Owned(bytes.clone()),
                )))
            })
            .expect("atlas insert must succeed")
            .expect("atlas insert must produce a tile")
    }

    #[test]
    fn configurable_dashes_reach_both_directx_quad_pipelines() -> Result<()> {
        let window = HiddenWindow::new()?;
        let devices = DirectXDevices::new()?;
        let mut renderer = DirectXRenderer::new(window.0, &devices, true)?;
        renderer.resize(Size {
            width: DevicePixels(40),
            height: DevicePixels(20),
        })?;

        let default_image = renderer.render_to_image(
            &dashed_border_scene(2.0, 1.0),
            WindowBackgroundAppearance::Opaque,
        )?;
        let custom_image = renderer.render_to_image(
            &dashed_border_scene(4.0, 0.5),
            WindowBackgroundAppearance::Opaque,
        )?;

        assert!(
            images_differ_in_region(&default_image, &custom_image, 4, 16),
            "custom dash length and gap must change the ordinary quad pipeline"
        );
        assert!(
            images_differ_in_region(&default_image, &custom_image, 24, 36),
            "custom dash length and gap must change the smoothed quad pipeline"
        );

        Ok(())
    }

    #[test]
    fn every_batch_reads_its_own_instances() -> Result<()> {
        let window = HiddenWindow::new()?;
        let devices = DirectXDevices::new()?;
        let mut renderer = DirectXRenderer::new(window.0, &devices, true)?;
        renderer.resize(Size {
            width: DevicePixels(200),
            height: DevicePixels(100),
        })?;
        let atlas = renderer.sprite_atlas();
        let mono = tile(
            atlas.as_ref(),
            AtlasKey::Svg(RenderSvgParams {
                path: "test-mono".into(),
                size: Size {
                    width: DevicePixels(8),
                    height: DevicePixels(8),
                },
            }),
            vec![255; 64],
        );
        let poly = tile(
            atlas.as_ref(),
            AtlasKey::Image(RenderImageParams {
                image_id: ImageId(1),
                frame_index: 0,
            }),
            // BGRA red, opaque.
            (0..64).flat_map(|_| [0u8, 0, 255, 255]).collect(),
        );

        let green = rgb_to_hsla(rgb(0x00ff00));
        let blue = rgb_to_hsla(rgb(0x0000ff));
        let mut scene = Scene::default();
        scene.insert_primitive(Quad {
            order: 0,
            bounds: scaled(10.0, 10.0, 30.0, 30.0),
            content_mask: full_mask(),
            background: gpui::ScenePaintRef::from(green),
            ..Default::default()
        });
        scene.insert_primitive(MonochromeSprite {
            transform: 0,
            clip: 0,
            order: 0,
            paint: 0,
            bounds: scaled(10.0, 60.0, 30.0, 30.0),
            content_mask: full_mask(),
            color: green.into(),
            tile: mono,
            transformation: Default::default(),
        });
        scene.insert_primitive(PolychromeSprite {
            transform: 0,
            clip: 0,
            order: 0,
            grayscale: ShaderBool::Disabled,
            opacity: 1.0,
            corner_smoothing: 0.0,
            bounds: scaled(50.0, 60.0, 30.0, 30.0),
            content_mask: full_mask(),
            corner_radii: Default::default(),
            tile: poly,
        });
        // Overlapping the image lifts this quad above it, splitting the quads into two
        // batches. The second one starts at instance 1 of the frame's quad buffer.
        scene.insert_primitive(Quad {
            order: 0,
            bounds: scaled(60.0, 70.0, 30.0, 30.0),
            content_mask: full_mask(),
            background: gpui::ScenePaintRef::from(blue),
            ..Default::default()
        });
        scene.finish();
        let quad_batches: Vec<_> = scene
            .render_commands()
            .iter()
            .filter_map(|command| match command {
                RenderCommand::Batch(PrimitiveBatch::Quads { range, .. }) => Some(range.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(quad_batches, vec![0..1, 1..2]);

        let image = renderer.render_to_image(&scene, WindowBackgroundAppearance::Opaque)?;
        let expect = |name: &str, x: u32, y: u32, expected: [u8; 3]| {
            let [r, g, b, _] = image.get_pixel(x, y).0;
            let close = |actual: u8, wanted: u8| actual.abs_diff(wanted) <= 8;
            assert!(
                close(r, expected[0]) && close(g, expected[1]) && close(b, expected[2]),
                "{name} at ({x},{y}) rendered ({r},{g},{b}), expected {expected:?}"
            );
        };
        expect("first quad batch", 25, 25, [0, 255, 0]);
        expect("monochrome sprite", 25, 75, [0, 255, 0]);
        expect("polychrome sprite", 55, 65, [255, 0, 0]);
        expect("second quad batch", 85, 85, [0, 0, 255]);
        expect("background", 190, 20, [255, 255, 255]);
        Ok(())
    }

    /// A transform-table entry moves a quad, and a transformed clip-table entry cuts one;
    /// neither would land here if the tables were not bound or not uploaded.
    #[test]
    fn scene_tables_reach_directx_shaders() -> Result<()> {
        let window = HiddenWindow::new()?;
        let devices = DirectXDevices::new()?;
        let mut renderer = DirectXRenderer::new(window.0, &devices, true)?;
        renderer.resize(Size {
            width: DevicePixels(200),
            height: DevicePixels(100),
        })?;
        let green = rgb_to_hsla(rgb(0x00ff00));
        // A quarter turn clockwise about (50, 50): a point at (x, y) lands at (100 - y, x).
        let turn = TransformationMatrix {
            rotation_scale: [[0.0, -1.0], [1.0, 0.0]],
            translation: [100.0, 0.0],
        };
        let quarter_turn = SceneTransform {
            transformation: turn,
            inverse: turn.inverse().context("a rotation is invertible")?,
        };

        let mut scene = Scene::default();
        let transform = scene.push_transform(quarter_turn);
        // A horizontal bar, x 10..40 and y 40..60, turned into x 40..60 and y 10..40.
        scene.insert_primitive(Quad {
            transform,
            bounds: scaled(10.0, 40.0, 30.0, 20.0),
            content_mask: full_mask(),
            background: gpui::ScenePaintRef::from(green),
            ..Default::default()
        });
        // A full quad, x 100..200, clipped to x 130..170 and y 40..60 in the space of a
        // quarter turn about (150, 50): x 140..160 and y 30..70 in the viewport.
        let shifted = TransformationMatrix {
            rotation_scale: turn.rotation_scale,
            translation: [200.0, -100.0],
        };
        let clip_transform = scene.push_transform(SceneTransform {
            transformation: shifted,
            inverse: shifted.inverse().context("a rotation is invertible")?,
        });
        let clip = scene.push_clip(SceneClip {
            bounds: scaled(130.0, 40.0, 40.0, 20.0),
            corner_radii: Corners::default(),
            transform: clip_transform,
            parent: 0,
        });
        scene.insert_primitive(Quad {
            clip,
            bounds: scaled(100.0, 0.0, 100.0, 100.0),
            content_mask: full_mask(),
            background: gpui::ScenePaintRef::from(green),
            ..Default::default()
        });
        scene.finish();

        let image = renderer.render_to_image(&scene, WindowBackgroundAppearance::Opaque)?;
        for (what, x, y, painted) in [
            ("transformed bar", 50, 15, true),
            ("transformed bar", 55, 35, true),
            ("untransformed bar", 15, 50, false),
            ("untransformed bar", 35, 55, false),
            ("transformed clip", 150, 35, true),
            ("transformed clip", 155, 65, true),
            ("untransformed clip", 135, 50, false),
            ("outside the clip", 190, 90, false),
        ] {
            let [r, g, b, _] = image.get_pixel(x, y).0;
            assert_eq!(
                (r, g, b) == (0, 255, 0),
                painted,
                "{what} at ({x},{y}) rendered ({r},{g},{b})"
            );
        }
        Ok(())
    }

    /// A chunk drawn twice, scaled and clipped, then moved, lands where each placement puts
    /// it, cut to its clip; the window's own quads, before and after it, still read theirs.
    #[test]
    fn chunks_draw_at_their_placements() -> Result<()> {
        let window = HiddenWindow::new()?;
        let devices = DirectXDevices::new()?;
        let mut renderer = DirectXRenderer::new(window.0, &devices, true)?;
        renderer.resize(Size {
            width: DevicePixels(200),
            height: DevicePixels(100),
        })?;
        let red = rgb_to_hsla(rgb(0xff0000));
        let green = rgb_to_hsla(rgb(0x00ff00));
        let blue = rgb_to_hsla(rgb(0x0000ff));
        let quad = |bounds, color: Hsla| Quad {
            bounds,
            content_mask: full_mask(),
            background: ScenePaintRef {
                color: color.into(),
                paint: 0,
            },
            ..Default::default()
        };

        // Green then blue, side by side, 20 pixels square, at the chunk's origin.
        let mut chunk_scene = Scene::default();
        chunk_scene.insert_primitive(quad(scaled(0.0, 0.0, 20.0, 20.0), green));
        chunk_scene.insert_primitive(quad(scaled(20.0, 0.0, 20.0, 20.0), blue));
        chunk_scene.finish();
        let chunk = Rc::new(SceneChunk {
            scene: chunk_scene,
            bounds: scaled(0.0, 0.0, 40.0, 20.0),
        });
        let placed = |placement: TransformationMatrix, bounds, clip| {
            Primitive::Chunk(PlacedChunk {
                order: 0,
                chunk: chunk.clone(),
                placement,
                bounds,
                content_mask: ContentMask {
                    bounds: clip,
                    ..Default::default()
                },
            })
        };

        let mut scene = Scene::default();
        scene.insert_primitive(quad(scaled(0.0, 30.0, 10.0, 10.0), red));
        // Doubled at (100, 10): green over x 100..140, blue over 140..180, cut at 160.
        scene.insert_primitive(placed(
            TransformationMatrix {
                rotation_scale: [[2.0, 0.0], [0.0, 2.0]],
                translation: [100.0, 10.0],
            },
            scaled(100.0, 10.0, 80.0, 40.0),
            scaled(100.0, 10.0, 60.0, 40.0),
        ));
        // As it is, at (0, 60).
        scene.insert_primitive(placed(
            TransformationMatrix {
                rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
                translation: [0.0, 60.0],
            },
            scaled(0.0, 60.0, 40.0, 20.0),
            scaled(0.0, 0.0, 200.0, 100.0),
        ));
        scene.insert_primitive(quad(scaled(180.0, 80.0, 20.0, 20.0), red));
        scene.finish();
        assert!(
            scene
                .render_commands()
                .iter()
                .any(|command| matches!(command, RenderCommand::Batch(PrimitiveBatch::Chunks(_)))),
            "the scene must draw its chunks as chunks"
        );

        let image = renderer.render_to_image(&scene, WindowBackgroundAppearance::Opaque)?;
        for (what, x, y, expected) in [
            ("doubled chunk, green", 110, 30, [0, 255, 0]),
            ("doubled chunk, blue", 150, 30, [0, 0, 255]),
            ("doubled chunk, past its clip", 170, 30, [255, 255, 255]),
            ("moved chunk, green", 10, 70, [0, 255, 0]),
            ("moved chunk, blue", 30, 70, [0, 0, 255]),
            ("where the chunk was recorded", 10, 10, [255, 255, 255]),
            ("window quad before the chunks", 5, 35, [255, 0, 0]),
            ("window quad after the chunks", 190, 90, [255, 0, 0]),
        ] {
            let [r, g, b, _] = image.get_pixel(x, y).0;
            let close = |actual: u8, wanted: u8| actual.abs_diff(wanted) <= 8;
            assert!(
                close(r, expected[0]) && close(g, expected[1]) && close(b, expected[2]),
                "{what} at ({x},{y}) rendered ({r},{g},{b}), expected {expected:?}"
            );
        }
        Ok(())
    }
}
