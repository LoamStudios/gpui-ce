//! Shader programs linked into the Metal pipelines that read the paint
//! table.
//!
//! A scene whose paints run no programs draws with the standard pipelines.
//! One that does draws with pipelines whose `program_color` runs every
//! program linked so far, and draws its paint's fallback colour for any
//! other. The set linked only grows, so linked pipelines never stop drawing
//! a program they drew: when a scene brings programs they lack, the
//! renderer links all of them again, on a thread of its own, and draws with
//! the pipelines it has, which show those paints' fallback colours, until
//! the new ones are ready.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

use gpui::{Scene, shader::Program};
use gpui_render::{artifacts::NativeShader, link::link_msl};
use metal::MTLPixelFormat;

use crate::metal_renderer::{
    PATH_SAMPLE_COUNT, build_path_rasterization_pipeline_state, build_pipeline_state, native_shader,
};

/// The pipelines that read the paint table, with programs linked in.
pub(crate) struct ProgramPipelines {
    pub(crate) quads: metal::RenderPipelineState,
    pub(crate) smoothed_quads: metal::RenderPipelineState,
    pub(crate) shadows: metal::RenderPipelineState,
    pub(crate) smoothed_shadows: metal::RenderPipelineState,
    pub(crate) path_rasterization: metal::RenderPipelineState,
}

/// What preparing a frame's programs found.
pub(crate) struct FramePrograms {
    /// The pipelines to draw paints with: `None` for the standard ones.
    pub(crate) pipelines: Option<Arc<ProgramPipelines>>,
    /// Whether the scene runs programs those pipelines lack, which are being
    /// linked: drawn with their fallback colours, the frame should be drawn
    /// again once they are ready.
    pub(crate) pending: bool,
}

struct PendingLink {
    ids: BTreeSet<u32>,
    result: mpsc::Receiver<Result<(ProgramPipelines, Duration), String>>,
}

/// The programs a renderer has linked, and is linking.
pub(crate) struct LinkedPrograms {
    device: metal::Device,
    /// Every program seen and not failed, by id.
    programs: BTreeMap<u32, Program>,
    /// The programs `pipelines` run.
    linked: BTreeSet<u32>,
    pipelines: Option<Arc<ProgramPipelines>>,
    pending: Option<PendingLink>,
    /// Programs that failed to link, which are not tried again.
    failed: BTreeSet<u32>,
    /// Whether a frame waits for the programs it needs to be linked, rather
    /// than drawing their fallback colours meanwhile.
    synchronous: bool,
    /// How long the last link took.
    last_link_time: Option<Duration>,
}

impl LinkedPrograms {
    pub(crate) fn new(device: metal::Device) -> Self {
        Self {
            device,
            programs: BTreeMap::new(),
            linked: BTreeSet::new(),
            pipelines: None,
            pending: None,
            failed: BTreeSet::new(),
            synchronous: std::env::var_os("GPUI_LINK_PROGRAMS_SYNCHRONOUSLY").is_some(),
            last_link_time: None,
        }
    }

    pub(crate) fn set_synchronous(&mut self, synchronous: bool) {
        self.synchronous = synchronous;
    }

    pub(crate) fn last_link_time(&self) -> Option<Duration> {
        self.last_link_time
    }

    /// The pipelines to draw `scene` with, starting to link the programs it
    /// runs that they lack, and in synchronous mode waiting for them.
    pub(crate) fn prepare(&mut self, scene: &Scene) -> FramePrograms {
        let mut needed = BTreeSet::new();
        self.collect(scene, &mut needed);
        if needed.is_empty() {
            return FramePrograms {
                pipelines: None,
                pending: false,
            };
        }
        self.poll(false);
        // A link in progress finishes before the next starts, so that
        // programs arriving every frame still get linked.
        if self.pending.is_none() && self.lacks_any(&needed) {
            self.start_link();
        }
        while self.synchronous && self.pending.is_some() {
            self.poll(true);
            if self.pending.is_none() && self.lacks_any(&needed) {
                self.start_link();
            }
        }
        FramePrograms {
            pipelines: self.pipelines.clone(),
            pending: self.lacks_any(&needed),
        }
    }

    /// Whether the linked pipelines lack any of `ids` that has not failed.
    fn lacks_any(&self, ids: &BTreeSet<u32>) -> bool {
        ids.iter()
            .any(|id| !self.linked.contains(id) && !self.failed.contains(id))
    }

    /// Adds the programs `scene` and the chunks it draws run to those seen,
    /// and their ids, but for those that failed, to `needed`.
    fn collect(&mut self, scene: &Scene, needed: &mut BTreeSet<u32>) {
        for program in scene.programs() {
            let id = program.id();
            if self.failed.contains(&id) {
                continue;
            }
            self.programs.entry(id).or_insert_with(|| program.clone());
            needed.insert(id);
        }
        for placed in &scene.chunks {
            self.collect(&placed.chunk.scene, needed);
        }
    }

    /// Links every program seen, on a thread of its own.
    fn start_link(&mut self) {
        let ids: BTreeSet<u32> = self.programs.keys().copied().collect();
        let programs: Vec<Program> = self.programs.values().cloned().collect();
        let device = self.device.clone();
        let (sender, result) = mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("gpui-link-programs".into())
            .spawn(move || {
                let start = Instant::now();
                let pipelines = link_pipelines(&device, &programs);
                sender
                    .send(pipelines.map(|pipelines| (pipelines, start.elapsed())))
                    .ok();
            });
        match spawned {
            Ok(_) => self.pending = Some(PendingLink { ids, result }),
            Err(error) => log::error!("failed to start linking shader programs: {error}"),
        }
    }

    /// Takes the result of the link in progress, if it is done, or, if
    /// `wait`, once it is.
    fn poll(&mut self, wait: bool) {
        let Some(pending) = &self.pending else {
            return;
        };
        let result = if wait {
            pending.result.recv().map_err(|_| ())
        } else {
            match pending.result.try_recv() {
                Ok(result) => Ok(result),
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => Err(()),
            }
        };
        let pending = self.pending.take().expect("a link is in progress");
        match result {
            Ok(Ok((pipelines, elapsed))) => {
                log::debug!(
                    "linked {} shader programs in {:.1} ms",
                    pending.ids.len(),
                    elapsed.as_secs_f64() * 1000.
                );
                self.last_link_time = Some(elapsed);
                self.pipelines = Some(Arc::new(pipelines));
                self.linked = pending.ids;
            }
            Ok(Err(error)) => {
                log::error!("failed to link shader programs: {error}");
                self.fail(&pending.ids);
            }
            Err(()) => {
                log::error!("linking shader programs stopped without a result");
                self.fail(&pending.ids);
            }
        }
    }

    /// Gives up on those of `ids` not already linked: their paints draw
    /// their fallback colours.
    fn fail(&mut self, ids: &BTreeSet<u32>) {
        for id in ids.difference(&self.linked) {
            self.failed.insert(*id);
            self.programs.remove(id);
        }
    }
}

/// The pipelines that read the paint table, with `programs` linked in.
///
/// Quads and smoothed quads share one shader module, as do shadows; the
/// three modules are linked and compiled in parallel, as compiling them is
/// most of the work.
fn link_pipelines(
    device: &metal::Device,
    programs: &[Program],
) -> Result<ProgramPipelines, String> {
    let modules: [&[&str]; 3] = [
        &["quads", "smoothed_quads"],
        &["shadows", "smoothed_shadows"],
        &["path_rasterization"],
    ];
    let mut linked = std::thread::scope(|scope| {
        let threads = modules.map(|labels| {
            scope.spawn(move || {
                let shaders: Vec<&NativeShader> =
                    labels.iter().map(|label| native_shader(label)).collect();
                link_module(device, &shaders, programs)
            })
        });
        threads.map(|thread| {
            thread
                .join()
                .unwrap_or_else(|_| Err("linking shader programs panicked".into()))
        })
    })
    .into_iter()
    .collect::<Result<Vec<_>, String>>()?
    .into_iter()
    .flatten();
    let mut next = || linked.next().expect("a pipeline per shader");
    Ok(ProgramPipelines {
        quads: next(),
        smoothed_quads: next(),
        shadows: next(),
        smoothed_shadows: next(),
        path_rasterization: next(),
    })
}

/// The pipelines of `shaders`, which share a module, with `programs`
/// linked into it.
fn link_module(
    device: &metal::Device,
    shaders: &[&NativeShader],
    programs: &[Program],
) -> Result<Vec<metal::RenderPipelineState>, String> {
    let msl = link_msl(shaders[0], programs).map_err(|error| error.to_string())?;
    // Programs are written once for the CPU and every GPU: compiled
    // precisely, they compute what the CPU does.
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(false);
    let library = device
        .new_library_with_source(&msl, &options)
        .map_err(|error| format!("{}: {error}", shaders[0].label))?;
    Ok(shaders
        .iter()
        .map(|shader| {
            if shader.label == "path_rasterization" {
                build_path_rasterization_pipeline_state(
                    device,
                    &library,
                    shader,
                    MTLPixelFormat::BGRA8Unorm,
                    PATH_SAMPLE_COUNT,
                )
            } else {
                build_pipeline_state(device, &library, shader, MTLPixelFormat::BGRA8Unorm)
            }
        })
        .collect())
}
