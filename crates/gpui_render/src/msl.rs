//! Metal Shading Language from validated WGSL, with GPUI's buffer, texture
//! and sampler slots. Shared by the build script, which writes the standard
//! shaders' MSL, and by renderers linking shader programs at run time.

use crate::shaders::interface;

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
                naga::TypeInner::Sampler { .. } => naga::back::msl::BindTarget {
                    sampler: Some(naga::back::msl::BindSamplerTarget::Resource(0)),
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
