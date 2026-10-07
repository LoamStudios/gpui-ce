//! Shader programs linked into the Metal pipelines that read the paint
//! table.
//!
//! Which programs are linked, and when, is [`gpui_render::linked`]'s
//! business; this builds the Metal pipelines for a set of them.

use gpui::shader::Program;
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
    pub(crate) monochrome_sprites: metal::RenderPipelineState,
}

/// The pipelines that read the paint table, with `programs` linked in.
///
/// Quads and smoothed quads share one shader module, as do shadows; the
/// four modules are linked and compiled in parallel, as compiling them is
/// most of the work.
pub(crate) fn link_pipelines(
    device: &metal::Device,
    programs: &[Program],
) -> Result<ProgramPipelines, String> {
    let modules: [&[&str]; 4] = [
        &["quads", "smoothed_quads"],
        &["shadows", "smoothed_shadows"],
        &["path_rasterization"],
        &["monochrome_sprites"],
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
        monochrome_sprites: next(),
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
