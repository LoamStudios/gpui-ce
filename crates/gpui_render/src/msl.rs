//! Metal Shading Language from validated WGSL, with GPUI's buffer, texture
//! and sampler slots. Shared by the build script, which writes the standard
//! shaders' MSL, and by renderers linking shader programs at run time.

use crate::shaders::interface;

/// The Metal sampler slot of a pipeline's own sampler.
pub const MSL_SAMPLER_SLOT: u8 = 0;
/// The Metal sampler slot of group 0's sampler, which filters photo paints.
pub const MSL_SCENE_SAMPLER_SLOT: u8 = 1;

/// MSL for every entry point of `module`, which `info` validated, binding
/// its resources at GPUI's native slots. `label` names it in errors.
pub fn write_msl(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    label: &str,
) -> Result<String, String> {
    let mut resources = naga::back::msl::BindingMap::default();
    for (_, variable) in module.global_variables.iter() {
        let Some(binding) = &variable.binding else {
            continue;
        };
        let target = match variable.space {
            naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. } => {
                let slot = interface::native_slot(binding.group, binding.binding);
                if slot >= interface::MSL_BUFFER_SIZES_SLOT {
                    return Err(format!(
                        "MSL buffer slot {slot} of {label} collides with the buffer sizes"
                    ));
                }
                naga::back::msl::BindTarget {
                    buffer: Some(slot as u8),
                    ..Default::default()
                }
            }
            naga::AddressSpace::Handle => match module.types[variable.ty].inner {
                naga::TypeInner::Image { .. } => naga::back::msl::BindTarget {
                    texture: Some(binding.binding.saturating_sub(1) as u8),
                    ..Default::default()
                },
                // A pipeline's own sampler takes slot 0, and group 0's, which
                // filters photo paints, slot 1, so a pipeline with a texture
                // of its own can also draw photos.
                naga::TypeInner::Sampler { .. } => naga::back::msl::BindTarget {
                    sampler: Some(naga::back::msl::BindSamplerTarget::Resource(
                        if binding.group == interface::GLOBAL_BIND_GROUP {
                            MSL_SCENE_SAMPLER_SLOT
                        } else {
                            MSL_SAMPLER_SLOT
                        },
                    )),
                    ..Default::default()
                },
                ref ty => return Err(format!("unsupported MSL resource type {ty:?} in {label}")),
            },
            space => {
                return Err(format!(
                    "unsupported MSL resource space {space:?} in {label}"
                ));
            }
        };
        resources.insert(*binding, target);
    }
    let entry_resources = naga::back::msl::EntryPointResources {
        resources,
        sizes_buffer: Some(interface::MSL_BUFFER_SIZES_SLOT as u8),
        ..Default::default()
    };
    let per_entry_point_map = module
        .entry_points
        .iter()
        .map(|entry| (entry.name.clone(), entry_resources.clone()))
        .collect();
    let options = naga::back::msl::Options {
        lang_version: (2, 0),
        per_entry_point_map,
        fake_missing_bindings: false,
        ..Default::default()
    };
    let source = naga::back::msl::write_string(
        module,
        info,
        &options,
        &naga::back::msl::PipelineOptions::default(),
    )
    .map(|(source, _)| source)
    .map_err(|error| format!("failed to generate MSL for {label}: {error}"))?;
    check_buffer_sizes_unread(&source, label)?;
    Ok(source)
}

/// Renderers bind `MSL_BUFFER_SIZES_BYTES` of placeholder sizes rather than tracking which
/// runtime arrays each entry point reaches, which is sound only while nothing reads them.
fn check_buffer_sizes_unread(source: &str, label: &str) -> Result<(), String> {
    if source.contains("_buffer_sizes.") {
        return Err(format!(
            "{label}: generated MSL reads runtime-array sizes, which renderers do not bind"
        ));
    }
    let sizes = source
        .split_once("struct _mslBufferSizes {")
        .and_then(|(_, rest)| rest.split_once("};"))
        .map_or(0, |(members, _)| members.matches("uint size").count());
    if sizes as u32 * 4 > interface::MSL_BUFFER_SIZES_BYTES {
        return Err(format!(
            "{label}: generated MSL declares {sizes} runtime-array sizes, more than renderers bind"
        ));
    }
    Ok(())
}

/// The math functions `precise_functions` calls the precise forms of.
#[allow(
    dead_code,
    reason = "the build script shares this module, and links no programs"
)]
const PRECISE_FUNCTIONS: &[&str] = &[
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "exp",
    "exp2",
    "exp10",
    "log",
    "log2",
    "log10",
    "pow",
    "powr",
    "sqrt",
    "rsqrt",
    "fmod",
    "fract",
    "length",
    "normalize",
    "distance",
];

/// `msl`, as Naga writes it, with the bodies of the functions named in
/// `names` (WGSL names: Naga may add a trailing `_`) computing precisely,
/// whatever the library is compiled with: each opts out of fast math, and
/// calls the precise form of each math function.
#[allow(
    dead_code,
    reason = "the build script shares this module, and links no programs"
)]
pub fn precise_functions(msl: &str, names: &std::collections::HashSet<&str>) -> String {
    let mut out = String::with_capacity(msl.len() + 4096);
    // Whether the function being declared is one of them, and whether its
    // body is being written.
    let (mut named, mut inside) = (false, false);
    for line in msl.lines() {
        if inside {
            if line == "}" {
                inside = false;
                out.push_str(line);
            } else {
                let mut line = line.to_string();
                for function in PRECISE_FUNCTIONS {
                    line = line.replace(
                        &format!("metal::{function}("),
                        &format!("metal::precise::{function}("),
                    );
                }
                out.push_str(&line);
            }
            out.push('\n');
            continue;
        }
        // Naga writes a function's declaration from its return type and
        // name to the line `) {`, and its body to the line `}`.
        if !line.starts_with([' ', '#', '}']) && line.ends_with('(') {
            let declared = line[..line.len() - 1].rsplit(' ').next().unwrap_or("");
            named = names.contains(declared)
                || declared
                    .strip_suffix('_')
                    .is_some_and(|name| names.contains(name));
        }
        out.push_str(line);
        out.push('\n');
        if named && line == ") {" {
            out.push_str("#pragma METAL fp math_mode(safe)\n");
            named = false;
            inside = true;
        }
    }
    out
}
