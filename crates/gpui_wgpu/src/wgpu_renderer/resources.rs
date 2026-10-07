use std::{
    cell::RefCell,
    num::NonZeroU64,
    sync::{Arc, Mutex},
};

use collections::FxHashMap;
use smallvec::SmallVec;

use crate::WgpuContext;
use gpui_render::linked::LinkedPrograms;
use gpui_render::shaders::{
    blur::BlurUniforms,
    common::{FontRasterizationUniforms, GlobalUniforms},
    group::GroupUniforms,
    surface::SurfaceUniforms,
};

use super::{
    WgpuRenderer,
    buffers::{DynamicUniformBuffer, InstanceBufferArena, InstanceTransport, SceneTable},
    filters::FrameUniformRequirements,
    photos::PhotoTiles,
    pipelines::{
        LinkedPaintPipelines, PhotoBindings, SceneTableBindings, WgpuBindGroupLayouts,
        WgpuPipelines,
    },
    settings::RenderingParameters,
    surfaces::SurfaceCache,
    target_pool::TexturePool,
};

const INITIAL_FILTER_UNIFORM_CAPACITY: u64 = 16;
const INITIAL_SURFACE_UNIFORM_CAPACITY: u64 = 8;
const INITIAL_GROUP_UNIFORM_CAPACITY: u64 = 4;
/// The window's target, the path intermediate, and a few isolated groups.
const INITIAL_TARGET_GLOBALS_CAPACITY: u64 = 8;

/// Device-owned state that is replaced atomically during GPU recovery.
pub(super) struct WgpuResources {
    pub(super) device: Arc<wgpu::Device>,
    pub(super) queue: Arc<wgpu::Queue>,
    pub(super) renderer_tier: crate::RendererTier,
    pub(super) surface: Option<wgpu::Surface<'static>>,
    pub(super) pipelines: WgpuPipelines,
    /// The shader programs linked into the pipelines that read paints.
    pub(super) programs: LinkedPrograms<LinkedPaintPipelines>,
    /// The pipelines the frame being drawn draws paints with, when its
    /// scene runs programs: `None` for the standard ones.
    pub(super) frame_programs: Option<Arc<LinkedPaintPipelines>>,
    /// Whether the last frame drew programs still being linked.
    pub(super) programs_pending: bool,
    pub(super) bind_group_layouts: WgpuBindGroupLayouts,
    pub(super) atlas_sampler: wgpu::Sampler,
    pub(super) surface_sampler: wgpu::Sampler,
    pub(super) surface_uniforms: DynamicUniformBuffer<SurfaceUniforms>,
    #[cfg_attr(
        not(any(
            all(target_family = "wasm", feature = "custom-gpu"),
            target_os = "macos",
            target_os = "linux",
            target_os = "freebsd",
            all(target_os = "windows", feature = "wgpu-surfaces")
        )),
        allow(dead_code)
    )]
    pub(super) surface_cache: RefCell<SurfaceCache>,
    pub(super) filter_uniforms: DynamicUniformBuffer<BlurUniforms>,
    blur_bind_groups: RefCell<BlurBindGroups>,
    pub(super) group_uniforms: DynamicUniformBuffer<GroupUniforms>,
    group_bind_groups: RefCell<GroupBindGroups>,
    /// The global uniforms of each target a frame draws into, bound by dynamic offset:
    /// the window's, the path intermediate's, and each isolated group's.
    pub(super) target_globals: DynamicUniformBuffer<GlobalUniforms>,
    pub(super) font_rasterization_buffer: wgpu::Buffer,
    scene_tables: SceneTables,
    /// Tables of the scene chunks drawn this frame, one slot for each distinct chunk.
    chunk_tables: ChunkTables,
    /// The window's resident photo tiles, bound in group 0.
    photo_tiles: PhotoTiles,
    /// Group 0, whose global uniforms are chosen per target by dynamic offset.
    pub(super) globals_bind_group: wgpu::BindGroup,
    /// The `target_globals` generation `globals_bind_group` was made for.
    globals_bind_group_generation: u64,
    pub(super) instances: InstanceBufferArena,
    /// The meshes kept on the GPU, by id.
    pub(super) meshes:
        RefCell<gpui_render::meshes::MeshCache<std::rc::Rc<super::drawing::WgpuMesh>>>,
    pub(super) path_intermediate_texture: Option<wgpu::Texture>,
    pub(super) path_intermediate_view: Option<wgpu::TextureView>,
    pub(super) path_msaa_texture: Option<wgpu::Texture>,
    pub(super) path_msaa_view: Option<wgpu::TextureView>,
    pub(super) scene_color_texture: Option<wgpu::Texture>,
    pub(super) scene_color_view: Option<wgpu::TextureView>,
    /// Targets for isolated groups, the blurs of groups and backdrops, and copies of what
    /// is beneath a blended group.
    pub(super) target_pool: RefCell<TexturePool>,
}

/// The scene's transform, clip, paint and colour-stop tables, bound in group 0 beside the
/// frame uniforms.
struct SceneTables {
    transforms: SceneTable<gpui::SceneTransform>,
    clips: SceneTable<gpui::SceneClip>,
    paints: SceneTable<gpui::PaintWord>,
}

impl SceneTables {
    fn new(device: &wgpu::Device, transport: InstanceTransport) -> Self {
        Self {
            transforms: SceneTable::new(device, "scene_transforms", transport),
            clips: SceneTable::new(device, "scene_clips", transport),
            paints: SceneTable::new(device, "scene_paints", transport),
        }
    }

    fn bindings(&self) -> SceneTableBindings<'_> {
        SceneTableBindings {
            transforms: self.transforms.binding(),
            clips: self.clips.binding(),
            paints: self.paints.binding(),
        }
    }
}

/// The tables of the scene chunks a frame draws, each with a group-0 bind group of its
/// own that binds them in place of the scene's. Slots are reused from frame to frame.
#[derive(Default)]
struct ChunkTables {
    slots: Vec<ChunkTableSlot>,
    /// This frame's slot for each chunk scene, by its address: the chunks are held by the
    /// scene being drawn, so the addresses are stable and distinct for the frame.
    by_scene: FxHashMap<usize, usize>,
}

struct ChunkTableSlot {
    tables: SceneTables,
    /// Made once the slot's tables and the group-0 resources it shares are in place;
    /// dropped when either is replaced.
    bind_group: Option<wgpu::BindGroup>,
}

fn chunk_key(scene: &gpui::Scene) -> usize {
    std::ptr::from_ref(scene) as usize
}

#[derive(Default)]
struct BlurBindGroups {
    uniform_generation: u64,
    groups: FxHashMap<wgpu::TextureView, wgpu::BindGroup>,
}

/// Group-composite bind groups, by the group's texture and the backdrop's.
#[derive(Default)]
struct GroupBindGroups {
    uniform_generation: u64,
    groups: FxHashMap<(wgpu::TextureView, wgpu::TextureView, wgpu::TextureView), wgpu::BindGroup>,
}

pub(super) struct ResourceMetadata {
    pub(super) globals: GlobalBufferLayout,
    pub(super) last_error: Arc<Mutex<Option<String>>>,
}

pub(super) struct GlobalBufferLayout {
    pub(super) maximum_uniform_buffer_size: u64,
}

impl WgpuResources {
    pub(super) fn new(
        context: &WgpuContext,
        surface: Option<wgpu::Surface<'static>>,
        surface_config: &wgpu::SurfaceConfiguration,
        rendering: &RenderingParameters,
        dual_source_blending: bool,
    ) -> anyhow::Result<(Self, ResourceMetadata)> {
        let device = Arc::clone(&context.device);
        let queue = Arc::clone(&context.queue);
        let renderer_tier = context.renderer_tier();
        let bind_group_layouts = WgpuBindGroupLayouts::new(&device, renderer_tier);
        let pipelines = WgpuPipelines::new(
            &device,
            &bind_group_layouts,
            surface_config.format,
            surface_config.alpha_mode,
            rendering.path_sample_count,
            dual_source_blending,
            renderer_tier,
        );
        let linear_sampler = |label| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            })
        };
        let atlas_sampler = linear_sampler("atlas_sampler");
        let surface_sampler = linear_sampler("surface_sampler");

        let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let surface_uniforms = DynamicUniformBuffer::new(
            &device,
            "surface_uniforms",
            INITIAL_SURFACE_UNIFORM_CAPACITY,
            uniform_alignment,
        );
        let filter_uniforms = DynamicUniformBuffer::new(
            &device,
            "filter_uniforms",
            INITIAL_FILTER_UNIFORM_CAPACITY,
            uniform_alignment,
        );
        let group_uniforms = DynamicUniformBuffer::new(
            &device,
            "group_uniforms",
            INITIAL_GROUP_UNIFORM_CAPACITY,
            uniform_alignment,
        );
        let target_globals = DynamicUniformBuffer::new(
            &device,
            "target_globals",
            INITIAL_TARGET_GLOBALS_CAPACITY,
            uniform_alignment,
        );
        let font_rasterization_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("font_rasterization_buffer"),
            size: std::mem::size_of::<FontRasterizationUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scene_tables = SceneTables::new(&device, InstanceTransport::from_tier(renderer_tier));
        let photo_tiles = PhotoTiles::new(&device, renderer_tier);
        let globals_bind_group = create_globals_bind_group(
            &device,
            &bind_group_layouts,
            &target_globals,
            &font_rasterization_buffer,
            &scene_tables,
            &photo_tiles,
        );
        let globals_bind_group_generation = target_globals.generation();
        let last_error = context.uncaptured_error_slot();
        let surface_cache = SurfaceCache::new(&device)?;

        let metadata = ResourceMetadata {
            globals: GlobalBufferLayout {
                maximum_uniform_buffer_size: device.limits().max_buffer_size.min(u32::MAX as u64),
            },
            last_error,
        };
        let resources = Self {
            instances: InstanceBufferArena::new(&device, &bind_group_layouts, renderer_tier),
            meshes: RefCell::default(),
            renderer_tier,
            device,
            queue,
            surface,
            pipelines,
            programs: LinkedPrograms::new(),
            frame_programs: None,
            programs_pending: false,
            bind_group_layouts,
            atlas_sampler,
            surface_sampler,
            surface_uniforms,
            surface_cache: RefCell::new(surface_cache),
            filter_uniforms,
            blur_bind_groups: RefCell::default(),
            group_uniforms,
            group_bind_groups: RefCell::default(),
            target_globals,
            font_rasterization_buffer,
            scene_tables,
            chunk_tables: ChunkTables::default(),
            photo_tiles,
            globals_bind_group,
            globals_bind_group_generation,
            path_intermediate_texture: None,
            path_intermediate_view: None,
            path_msaa_texture: None,
            path_msaa_view: None,
            scene_color_texture: None,
            scene_color_view: None,
            target_pool: RefCell::new(TexturePool::new()),
        };
        Ok((resources, metadata))
    }

    /// Picks the pipelines `scene`'s paints draw with, starting to link the
    /// shader programs it runs that they lack.
    pub(super) fn prepare_programs(&mut self, scene: &gpui::Scene) {
        let linker = &self.pipelines.linker;
        let programs = self.programs.prepare(scene, || {
            let linker = linker.clone();
            move |programs: Vec<gpui::shader::Program>| linker.link(&programs)
        });
        self.frame_programs = programs.pipelines;
        self.programs_pending = programs.pending;
    }

    pub(super) fn invalidate_intermediate_textures(&mut self) {
        self.instances.invalidate_texture_bindings();
        self.blur_bind_groups.get_mut().groups.clear();
        self.group_bind_groups.get_mut().groups.clear();
        self.target_pool.get_mut().clear();
        self.path_intermediate_texture = None;
        self.path_intermediate_view = None;
        self.path_msaa_texture = None;
        self.path_msaa_view = None;
        self.scene_color_texture = None;
        self.scene_color_view = None;
    }

    /// Uploads the scene's transform, clip and paint tables, growing them, and
    /// rebuilding the group-0 bind groups that reference them, as needed. Returns false when
    /// the device cannot hold them.
    pub(super) fn upload_scene_tables(&mut self, scene: &gpui::Scene) -> bool {
        let tables = &mut self.scene_tables;
        let device = &self.device;
        let (Some(transforms_grew), Some(clips_grew), Some(paints_grew)) = (
            tables
                .transforms
                .ensure_capacity(device, scene.transforms().len() as u64),
            tables
                .clips
                .ensure_capacity(device, scene.clips().len() as u64),
            tables
                .paints
                .ensure_capacity(device, scene.paint_table().len() as u64),
        ) else {
            return false;
        };
        if transforms_grew || clips_grew || paints_grew {
            self.rebuild_globals_bind_group();
        }
        let tables = &self.scene_tables;
        tables.transforms.write(&self.queue, scene.transforms());
        tables.clips.write(&self.queue, scene.clips());
        tables.paints.write(&self.queue, scene.paint_table());
        true
    }

    /// Uploads the tables of every chunk `scene` draws, at any depth, each into a slot of
    /// its own with a group-0 bind group for it. Called once the frame's target globals
    /// are in place, as the bind groups reference them. Returns false when the device
    /// cannot hold a chunk's tables.
    pub(super) fn upload_chunk_tables(&mut self, scene: &gpui::Scene) -> bool {
        self.chunk_tables.by_scene.clear();
        if scene.chunks.is_empty() {
            return true;
        }
        let transport = self.instances.transport();
        let mut pending: SmallVec<[&gpui::Scene; 8]> = scene
            .chunks
            .iter()
            .map(|placed| &placed.chunk.scene)
            .collect();
        while let Some(chunk) = pending.pop() {
            let key = chunk_key(chunk);
            if self.chunk_tables.by_scene.contains_key(&key) {
                continue;
            }
            let index = self.chunk_tables.by_scene.len();
            self.chunk_tables.by_scene.insert(key, index);
            if index == self.chunk_tables.slots.len() {
                self.chunk_tables.slots.push(ChunkTableSlot {
                    tables: SceneTables::new(&self.device, transport),
                    bind_group: None,
                });
            }
            let slot = &mut self.chunk_tables.slots[index];
            let tables = &mut slot.tables;
            let (Some(transforms_grew), Some(clips_grew), Some(paints_grew)) = (
                tables
                    .transforms
                    .ensure_capacity(&self.device, chunk.transforms().len() as u64),
                tables
                    .clips
                    .ensure_capacity(&self.device, chunk.clips().len() as u64),
                tables
                    .paints
                    .ensure_capacity(&self.device, chunk.paint_table().len() as u64),
            ) else {
                return false;
            };
            if transforms_grew || clips_grew || paints_grew {
                slot.bind_group = None;
            }
            tables.transforms.write(&self.queue, chunk.transforms());
            tables.clips.write(&self.queue, chunk.clips());
            tables.paints.write(&self.queue, chunk.paint_table());
            if slot.bind_group.is_none() {
                slot.bind_group = Some(create_globals_bind_group(
                    &self.device,
                    &self.bind_group_layouts,
                    &self.target_globals,
                    &self.font_rasterization_buffer,
                    &slot.tables,
                    &self.photo_tiles,
                ));
            }
            pending.extend(chunk.chunks.iter().map(|placed| &placed.chunk.scene));
        }
        true
    }

    /// The group-0 bind group that binds `chunk`'s tables, uploaded this frame by
    /// [`Self::upload_chunk_tables`].
    pub(super) fn chunk_bind_group(&self, chunk: &gpui::Scene) -> Option<&wgpu::BindGroup> {
        let index = *self.chunk_tables.by_scene.get(&chunk_key(chunk))?;
        self.chunk_tables.slots[index].bind_group.as_ref()
    }

    /// Copies the tiles the scene's photos placed since the last frame into the photo
    /// tile array, growing it, and rebuilding the group-0 bind group that holds it, as
    /// needed. Called before anything else of the frame is uploaded, as growing the array
    /// submits a copy of what it held.
    pub(super) fn upload_photo_tiles(&mut self, scene: &gpui::Scene) {
        if self.photo_tiles.upload(&self.device, &self.queue, scene) {
            self.rebuild_globals_bind_group();
        }
    }

    fn rebuild_globals_bind_group(&mut self) {
        self.globals_bind_group = create_globals_bind_group(
            &self.device,
            &self.bind_group_layouts,
            &self.target_globals,
            &self.font_rasterization_buffer,
            &self.scene_tables,
            &self.photo_tiles,
        );
        self.globals_bind_group_generation = self.target_globals.generation();
        // The chunks' bind groups share what was just replaced.
        for slot in &mut self.chunk_tables.slots {
            slot.bind_group = None;
        }
    }

    pub(super) fn finish_frame_uploads(&self) {
        self.filter_uniforms.finish_upload();
        self.surface_uniforms.finish_upload();
        self.group_uniforms.finish_upload();
        self.target_globals.finish_upload();
    }

    /// Ends a frame's use of the target pool, letting go of the bind groups that hold
    /// textures it lets go of.
    pub(super) fn end_target_pool_frame(&mut self) {
        if self.target_pool.get_mut().end_frame() {
            self.blur_bind_groups.get_mut().groups.clear();
            self.group_bind_groups.get_mut().groups.clear();
        }
    }

    /// The bind group compositing a group from `source`, mixed, for a blend mode other
    /// than normal, with `backdrop`, and cut, when the group is masked, to `mask`.
    pub(super) fn group_bind_group(
        &self,
        source: &wgpu::TextureView,
        backdrop: &wgpu::TextureView,
        mask: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let mut cache = self.group_bind_groups.borrow_mut();
        let uniform_generation = self.group_uniforms.generation();
        if cache.uniform_generation != uniform_generation {
            cache.groups.clear();
            cache.uniform_generation = uniform_generation;
        }
        cache
            .groups
            .entry((source.clone(), backdrop.clone(), mask.clone()))
            .or_insert_with(|| {
                self.bind_group_layouts.create_group(
                    &self.device,
                    wgpu::BufferBinding {
                        buffer: &self.group_uniforms.buffer,
                        offset: 0,
                        size: NonZeroU64::new(std::mem::size_of::<GroupUniforms>() as u64),
                    },
                    source,
                    backdrop,
                    &self.surface_sampler,
                    mask,
                )
            })
            .clone()
    }

    pub(super) fn blur_bind_group(&self, source: &wgpu::TextureView) -> wgpu::BindGroup {
        let mut cache = self.blur_bind_groups.borrow_mut();
        let uniform_generation = self.filter_uniforms.generation();
        if cache.uniform_generation != uniform_generation {
            cache.groups.clear();
            cache.uniform_generation = uniform_generation;
        }
        cache
            .groups
            .entry(source.clone())
            .or_insert_with(|| {
                self.bind_group_layouts.create_blur(
                    &self.device,
                    wgpu::BufferBinding {
                        buffer: &self.filter_uniforms.buffer,
                        offset: 0,
                        size: NonZeroU64::new(std::mem::size_of::<BlurUniforms>() as u64),
                    },
                    source,
                    &self.surface_sampler,
                )
            })
            .clone()
    }
}

/// Creates the group-0 bind group. Its global uniforms are bound by dynamic offset, to
/// the slot of the target being drawn into.
fn create_globals_bind_group(
    device: &wgpu::Device,
    layouts: &WgpuBindGroupLayouts,
    target_globals: &DynamicUniformBuffer<GlobalUniforms>,
    font_rasterization_buffer: &wgpu::Buffer,
    tables: &SceneTables,
    photo_tiles: &PhotoTiles,
) -> wgpu::BindGroup {
    layouts.create_globals(
        device,
        "globals_bind_group",
        wgpu::BufferBinding {
            buffer: &target_globals.buffer,
            offset: 0,
            size: NonZeroU64::new(std::mem::size_of::<GlobalUniforms>() as u64),
        },
        wgpu::BufferBinding {
            buffer: font_rasterization_buffer,
            offset: 0,
            size: NonZeroU64::new(std::mem::size_of::<FontRasterizationUniforms>() as u64),
        },
        tables.bindings(),
        PhotoBindings {
            tiles: photo_tiles.view(),
            sampler: photo_tiles.sampler(),
        },
    )
}

impl WgpuRenderer {
    pub(super) fn ensure_path_textures(&mut self) {
        if self.resources().path_intermediate_texture.is_some() {
            return;
        }
        let format = self.target.format();
        let width = self.target.width();
        let height = self.target.height();
        let sample_count = self.rendering_params.path_sample_count;
        let resources = self.resources_mut();
        let (texture, view) = sampled_render_texture(&resources.device, format, width, height);
        resources.path_intermediate_texture = Some(texture);
        resources.path_intermediate_view = Some(view);
        if let Some((texture, view)) =
            msaa_texture(&resources.device, format, width, height, sample_count)
        {
            resources.path_msaa_texture = Some(texture);
            resources.path_msaa_view = Some(view);
        }
    }

    /// Creates the offscreen target the scene is drawn into when backdrop filters or blend
    /// modes read what is already painted.
    pub(super) fn ensure_scene_color_texture(&mut self) {
        let format = self.target.format();
        let width = self.target.width();
        let height = self.target.height();
        let resources = self.resources_mut();
        if resources.scene_color_texture.is_none() {
            let (texture, view) = sampled_render_texture(&resources.device, format, width, height);
            resources.scene_color_texture = Some(texture);
            resources.scene_color_view = Some(view);
        }
    }

    pub(super) fn ensure_uniform_capacity(
        &mut self,
        requirements: FrameUniformRequirements,
    ) -> bool {
        let maximum_buffer_size = self.globals.maximum_uniform_buffer_size;
        let resources = self.resources_mut();
        let filters = resources.filter_uniforms.ensure_capacity(
            &resources.device,
            requirements.filter_count,
            maximum_buffer_size,
        );
        let surfaces = resources.surface_uniforms.ensure_capacity(
            &resources.device,
            requirements.surface_count,
            maximum_buffer_size,
        );
        let groups = resources.group_uniforms.ensure_capacity(
            &resources.device,
            requirements.group_count,
            maximum_buffer_size,
        );
        let targets = resources.target_globals.ensure_capacity(
            &resources.device,
            requirements.target_count,
            maximum_buffer_size,
        );
        let capacity_available = filters && surfaces && groups && targets;
        if !capacity_available {
            log::error!(
                "scene uniform data exceeds the GPU buffer limit: {} filter uniforms, {} surface uniforms, {} group uniforms and {} target globals",
                requirements.filter_count,
                requirements.surface_count,
                requirements.group_count,
                requirements.target_count,
            );
            return false;
        }
        if resources.globals_bind_group_generation != resources.target_globals.generation() {
            resources.rebuild_globals_bind_group();
        }
        let filters = resources
            .filter_uniforms
            .begin_upload(&resources.queue, requirements.filter_count);
        let surfaces = resources
            .surface_uniforms
            .begin_upload(&resources.queue, requirements.surface_count);
        let groups = resources
            .group_uniforms
            .begin_upload(&resources.queue, requirements.group_count);
        let targets = resources
            .target_globals
            .begin_upload(&resources.queue, requirements.target_count);
        if !(filters && surfaces && groups && targets) {
            resources.finish_frame_uploads();
            log::error!("failed to map frame uniform staging memory");
            return false;
        }
        true
    }
}

fn sampled_render_texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sampled_render_texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        // The scene's offscreen target is copied from when a blended group reads what is
        // beneath it.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn msaa_texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    sample_count: u32,
) -> Option<(wgpu::Texture, wgpu::TextureView)> {
    if sample_count <= 1 {
        return None;
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("path_msaa"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Some((texture, view))
}
