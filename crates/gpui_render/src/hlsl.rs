//! HLSL for Direct3D 11, shader model 5.0, from validated WGSL. Shared by
//! the build script, which compiles the standard shaders' HLSL to DXBC, and
//! by the DirectX renderer, which links shader programs into the shaders
//! that read the paint table at run time and compiles them then.
//!
//! Both go through the floor Naga (`naga_old`, 24), whose shader-model 5.0
//! output is what the standard shaders have always been compiled from, so a
//! shader with programs linked in is translated exactly as it is without.

use crate::shaders::interface;

/// Instanced pipelines read their batch base from the draw-constants cbuffer.
///
/// Direct3D 11 never folds `StartInstanceLocation` into `SV_InstanceID`, so a shader that
/// indexes a whole-frame instance buffer needs the base delivered some other way. Naga's
/// "special constants" cbuffer is that way: `instance_index` becomes
/// `first_instance + SV_InstanceID`. Fullscreen and per-draw-uniform pipelines draw one
/// instance from vertex zero and carry no such cbuffer.
pub fn dx11_draw_constants_register(pipeline: &interface::Pipeline) -> Option<u32> {
    use interface::DataLayout;
    match pipeline.data_layout {
        DataLayout::Instances
        | DataLayout::TexturedInstances
        | DataLayout::MonochromeSprites
        | DataLayout::SubpixelSprites
        | DataLayout::Meshes => Some(interface::DX11_DRAW_CONSTANTS_REGISTER),
        DataLayout::NativeOnly | DataLayout::Surface | DataLayout::Blur | DataLayout::Group => None,
    }
}

/// Shader-model 5.0 HLSL for every entry point of `module`, which `info`
/// validated, from `wgsl`, its source, for `pipeline`: resources at GPUI's
/// native registers, and, for instanced pipelines, the draw constants as a
/// classic `cbuffer`.
pub fn write_hlsl(
    module: &naga_old::Module,
    info: &naga_old::valid::ModuleInfo,
    wgsl: &str,
    pipeline: &interface::Pipeline,
) -> Result<String, String> {
    let label = pipeline.label;
    let mut binding_map = naga_old::back::hlsl::BindingMap::default();
    for (_, variable) in module.global_variables.iter() {
        if let Some(binding) = &variable.binding {
            binding_map.insert(
                binding.clone(),
                naga_old::back::hlsl::BindTarget {
                    space: 0,
                    register: interface::native_slot(binding.group, binding.binding),
                    ..Default::default()
                },
            );
        }
    }
    let draw_constants_register = dx11_draw_constants_register(pipeline);
    let options = naga_old::back::hlsl::Options {
        shader_model: naga_old::back::hlsl::ShaderModel::V5_0,
        binding_map,
        fake_missing_bindings: false,
        special_constants_binding: draw_constants_register.map(|register| {
            naga_old::back::hlsl::BindTarget {
                space: 0,
                register,
                ..Default::default()
            }
        }),
        ..Default::default()
    };
    let mut output = String::new();
    naga_old::back::hlsl::Writer::new(&mut output, &options)
        .write(module, info, None)
        .map_err(|error| format!("failed to generate HLSL for {label}: {error}"))?;
    match draw_constants_register {
        Some(register) => lower_draw_constants_to_sm50(output, wgsl, label, register),
        None if wgsl.contains("instance_index") => Err(format!(
            "{label} reads instance_index but has no DX11 draw constants"
        )),
        None => Ok(output),
    }
}

/// Naga declares its special constants with `ConstantBuffer<T>`, a shader-model 5.1 form
/// that `vs_5_0` rejects. Rewrite that one declaration into the classic `cbuffer` block and
/// prove the base actually reaches every instance lookup.
fn lower_draw_constants_to_sm50(
    hlsl: String,
    wgsl: &str,
    label: &str,
    register: u32,
) -> Result<String, String> {
    let declaration =
        format!("ConstantBuffer<NagaConstants> _NagaConstants: register(b{register});");
    if hlsl.matches(&declaration).count() != 1 {
        return Err(format!(
            "{label}: expected exactly one Naga special-constants declaration"
        ));
    }
    if hlsl.matches(&format!("register(b{register})")).count() != 1 {
        return Err(format!(
            "{label}: draw-constants register b{register} collides with another cbuffer"
        ));
    }
    let lowered = hlsl.replace(
        &declaration,
        &format!(
            "cbuffer DrawConstants : register(b{register}) {{ NagaConstants _NagaConstants; }}"
        ),
    );
    if wgsl.contains("instance_index") && !lowered.contains("_NagaConstants.first_instance + ") {
        return Err(format!(
            "{label}: instance_index must be offset by the draw-constants base"
        ));
    }
    Ok(lowered)
}
