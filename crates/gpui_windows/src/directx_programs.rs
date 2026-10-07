//! Shader programs linked into the Direct3D 11 shaders that read the paint
//! table.
//!
//! Which programs are linked, and when, is [`gpui_render::linked`]'s
//! business; this builds the shaders for a set of them. Each shader's WGSL
//! has the programs linked in, the floor Naga translates it to shader-model
//! 5.0 HLSL with the build's registers and draw constants
//! ([`gpui_render::link::link_hlsl`]), and `D3DCompile` compiles that at
//! run time, as the build compiles the standard shaders.

use std::ffi::CString;

use gpui::shader::Program;
use gpui_render::{
    artifacts::{NATIVE_SHADERS, NativeShader},
    link::link_hlsl,
};
use windows::{
    Win32::Graphics::{
        Direct3D::{
            Fxc::{D3DCOMPILE_IEEE_STRICTNESS, D3DCompile},
            ID3DInclude,
        },
        Direct3D11::ID3D11Device,
    },
    core::PCSTR,
};

use crate::directx_renderer::{PipelineVariant, create_fragment_shader, create_vertex_shader};

/// The shaders that read the paint table, with programs linked in, each in
/// place of its pipeline's own.
pub(crate) struct LinkedShaders {
    pub(crate) quads: PipelineVariant,
    pub(crate) smoothed_quads: PipelineVariant,
    pub(crate) shadows: PipelineVariant,
    pub(crate) smoothed_shadows: PipelineVariant,
    pub(crate) path_rasterization: PipelineVariant,
    pub(crate) meshes: PipelineVariant,
    pub(crate) monochrome_sprites: PipelineVariant,
}

// SAFETY: Direct3D 11 devices are free-threaded, and shader objects are
// immutable once created: they are made on the linking thread and only
// bound on the renderer's.
unsafe impl Send for LinkedShaders {}
unsafe impl Sync for LinkedShaders {}

/// The device, to create shaders with on the linking thread.
pub(crate) struct LinkingDevice(pub(crate) ID3D11Device);

// SAFETY: as above, `ID3D11Device`'s creation methods are free-threaded.
unsafe impl Send for LinkingDevice {}

/// The shaders that read the paint table, with `programs` linked in.
///
/// Quads and smoothed quads share one module, as do shadows; the five
/// modules are translated and compiled in parallel, as compiling them is
/// most of the work.
pub(crate) fn link_shaders(
    device: &LinkingDevice,
    programs: &[Program],
) -> Result<LinkedShaders, String> {
    let modules: [&[&str]; 5] = [
        &["quads", "smoothed_quads"],
        &["shadows", "smoothed_shadows"],
        &["path_rasterization"],
        &["meshes"],
        &["monochrome_sprites"],
    ];
    let mut linked = std::thread::scope(|scope| {
        let threads = modules.map(|labels| {
            scope.spawn(move || {
                let shaders: Vec<&NativeShader> =
                    labels.iter().map(|label| native_shader(label)).collect();
                link_module(&device.0, &shaders, programs)
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
    let mut next = || linked.next().expect("a variant per shader");
    Ok(LinkedShaders {
        quads: next(),
        smoothed_quads: next(),
        shadows: next(),
        smoothed_shadows: next(),
        path_rasterization: next(),
        meshes: next(),
        monochrome_sprites: next(),
    })
}

fn native_shader(label: &str) -> &'static NativeShader {
    NATIVE_SHADERS
        .iter()
        .find(|shader| shader.label == label)
        .unwrap_or_else(|| panic!("missing generated native shader {label}"))
}

/// The shaders of `shaders`, which share a module, with `programs` linked
/// into it.
fn link_module(
    device: &ID3D11Device,
    shaders: &[&NativeShader],
    programs: &[Program],
) -> Result<Vec<PipelineVariant>, String> {
    let hlsl = link_hlsl(shaders[0], programs).map_err(|error| error.to_string())?;
    shaders
        .iter()
        .map(|shader| {
            let vertex = compile(&hlsl, shader.label, shader.vertex_entry, b"vs_5_0\0")?;
            let fragment = compile(&hlsl, shader.label, shader.fragment_entry, b"ps_5_0\0")?;
            Ok(PipelineVariant {
                specification: shader.pipeline,
                vertex: create_vertex_shader(device, &vertex)
                    .map_err(|error| format!("{}: {error}", shader.label))?,
                fragment: create_fragment_shader(device, &fragment)
                    .map_err(|error| format!("{}: {error}", shader.label))?,
            })
        })
        .collect()
}

/// DXBC for `entry` of `source` in `profile`, a NUL-terminated shader
/// profile. Programs are written once for the CPU and every GPU: compiled
/// with IEEE strictness, they compute what the CPU does.
fn compile(source: &str, label: &str, entry: &str, profile: &[u8]) -> Result<Vec<u8>, String> {
    let entry_name = CString::new(entry).map_err(|_| format!("{label}: entry point has a NUL"))?;
    let mut blob = None;
    let mut errors = None;
    let result = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            PCSTR::from_raw(c"gpui_linked_shaders.hlsl".as_ptr().cast()),
            None,
            None::<&ID3DInclude>,
            PCSTR::from_raw(entry_name.as_ptr().cast()),
            PCSTR::from_raw(profile.as_ptr()),
            D3DCOMPILE_IEEE_STRICTNESS,
            0,
            &mut blob,
            Some(&mut errors),
        )
    };
    if let Err(error) = result {
        let details = errors
            .as_ref()
            .map(|errors| unsafe {
                std::ffi::CStr::from_ptr(errors.GetBufferPointer().cast())
                    .to_string_lossy()
                    .into_owned()
            })
            .unwrap_or_default();
        return Err(format!(
            "failed to compile linked HLSL for {label} ({entry}): {error}\n{details}"
        ));
    }
    let blob = blob.ok_or_else(|| format!("{label} ({entry}): D3DCompile returned no bytecode"))?;
    Ok(unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
            .to_vec()
    })
}
