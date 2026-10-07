//! Links shader programs into the standard shaders that read the paint
//! table, at run time.
//!
//! Each such shader's WGSL is generated without `program_color` (see
//! [`NativeShader::linkable_wgsl`]). Linking appends the shader prelude the
//! programs are written against, a `paint_param` that reads their
//! parameters from the paint table, a `program_color` that dispatches on a
//! program's id to the program and returns the fallback colour for any id
//! it does not hold, and the programs themselves; Naga validates the result
//! and translates it for the backend.

use std::fmt;

use gpui::shader::{Program, prelude_source};

use crate::artifacts::NativeShader;

/// Why programs could not be linked into a shader.
#[derive(Clone, Debug)]
pub struct LinkError(String);

impl fmt::Display for LinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LinkError {}

/// Where a linkable shader reads the paint table from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dialect {
    /// A storage buffer, `PAINTS`.
    Modern,
    /// The downlevel (WebGL2/GLES) data texture, through `dl_load_PAINTS`.
    Downlevel,
}

/// WGSL without `program_color`, to link programs into.
#[derive(Clone, Copy, Debug)]
pub struct Linkable {
    pub label: &'static str,
    pub wgsl: &'static str,
    pub dialect: Dialect,
}

impl Linkable {
    /// `shader`'s linkable WGSL, if it reads the paint table.
    pub fn native(shader: &NativeShader) -> Option<Self> {
        Some(Self {
            label: shader.label,
            wgsl: shader.linkable_wgsl?,
            dialect: Dialect::Modern,
        })
    }

    /// The wgpu renderers' module of every pipeline, in `dialect`.
    pub fn base(dialect: Dialect) -> Self {
        match dialect {
            Dialect::Modern => Self {
                label: "gpui_shaders",
                wgsl: crate::artifacts::BASE_LINKABLE_WGSL,
                dialect,
            },
            Dialect::Downlevel => Self {
                label: "gpui_shaders_downlevel",
                wgsl: crate::artifacts::BASE_DOWNLEVEL_LINKABLE_WGSL,
                dialect,
            },
        }
    }
}

/// The WGSL of `linkable` with `programs` linked in. Programs run unchanged
/// wherever a paint names their id; any other program id draws its paint's
/// fallback colour.
pub fn link_source<'a>(
    linkable: &Linkable,
    programs: impl IntoIterator<Item = &'a Program>,
) -> String {
    let programs: Vec<&Program> = programs.into_iter().collect();
    let prelude = prelude_source();
    let mut source = String::with_capacity(
        linkable.wgsl.len()
            + prelude.len()
            + programs
                .iter()
                .map(|program| program.source().len() + 128)
                .sum::<usize>()
            + 1024,
    );
    // Programs may take derivatives, as UI shaders do, inside the per-pixel
    // control flow that clips and picks paints.
    source.push_str("diagnostic(off, derivative_uniformity);\n");
    source.push_str(linkable.wgsl);
    source.push('\n');
    source.push_str(prelude);
    source.push_str(match linkable.dialect {
        Dialect::Modern => {
            "
fn paint_param(index: u32) -> vec4<f32> {
    return PAINTS[index].value;
}
"
        }
        Dialect::Downlevel => {
            "
fn paint_param(index: u32) -> vec4<f32> {
    return dl_load_PAINTS(index).value;
}
"
        }
    });
    source.push_str(
        "
fn program_color(id: u32, uv: vec2<f32>, position: vec2<f32>, size: vec2<f32>, origin: vec2<f32>, scale: f32, base: u32, fallback: vec4<f32>) -> vec4<f32> {
    let fragment = Fragment(uv, position, size, origin, scale);
    switch id {
",
    );
    for program in &programs {
        source.push_str(&format!(
            "        case {}u: {{ return Color_unpremultiply({}(fragment, base)); }}\n",
            program.id(),
            program.entry_point(),
        ));
    }
    source.push_str(
        "        default: { return fallback; }
    }
}

",
    );
    for program in &programs {
        source.push_str(program.source());
        source.push('\n');
    }
    source
}

/// The WGSL of `shader` with `programs` linked in, or `None` if it does not
/// read the paint table.
pub fn linked_wgsl<'a>(
    shader: &NativeShader,
    programs: impl IntoIterator<Item = &'a Program>,
) -> Option<String> {
    Some(link_source(&Linkable::native(shader)?, programs))
}

/// `linkable` with `programs` linked in, parsed and validated, with its
/// source.
pub fn link<'a>(
    linkable: &Linkable,
    programs: impl IntoIterator<Item = &'a Program>,
) -> Result<(String, naga::Module, naga::valid::ModuleInfo), LinkError> {
    let source = link_source(linkable, programs);
    let module = naga::front::wgsl::parse_str(&source).map_err(|error| {
        LinkError(format!(
            "{} with programs linked in: {}",
            linkable.label,
            error.emit_to_string(&source)
        ))
    })?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| {
        LinkError(format!(
            "{} with programs linked in: {}",
            linkable.label,
            error.emit_to_string(&source)
        ))
    })?;
    Ok((source, module, info))
}

/// `shader`'s WGSL with `programs` linked in, parsed and validated.
pub fn link_module<'a>(
    shader: &NativeShader,
    programs: impl IntoIterator<Item = &'a Program>,
) -> Result<(naga::Module, naga::valid::ModuleInfo), LinkError> {
    let linkable = Linkable::native(shader)
        .ok_or_else(|| LinkError(format!("{} does not read the paint table", shader.label)))?;
    let (_, module, info) = link(&linkable, programs)?;
    Ok((module, info))
}

/// Metal Shading Language for `shader` with `programs` linked in.
pub fn link_msl<'a>(
    shader: &NativeShader,
    programs: impl IntoIterator<Item = &'a Program>,
) -> Result<String, LinkError> {
    let (module, info) = link_module(shader, programs)?;
    crate::msl::write_msl(&module, &info, shader.label).map_err(LinkError)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::NATIVE_SHADERS;
    use gpui::shader::{self, Paint};

    fn linkable() -> impl Iterator<Item = &'static NativeShader> {
        NATIVE_SHADERS
            .iter()
            .filter(|shader| shader.linkable_wgsl.is_some())
    }

    fn grain() -> Paint {
        shader::paint(|px| {
            let value = shader::noise::value(px.position() * 0.5);
            shader::rgba(&value, &value, &value, 1.0).opacity(px.uv().x().fwidth() * 0.0 + 1.0)
        })
    }

    fn rings() -> Paint {
        shader::paint(|px| {
            let distance = shader::shape::circle(px.centered(), 30.0).abs() - 4.0;
            shader::color(gpui::rgb(0xff8800)).clip(distance)
        })
    }

    #[test]
    fn the_shaders_that_read_paints_are_linkable() {
        let labels: Vec<_> = linkable().map(|shader| shader.label).collect();
        assert_eq!(
            labels,
            [
                "quads",
                "smoothed_quads",
                "shadows",
                "smoothed_shadows",
                "path_rasterization"
            ]
        );
        for shader in linkable() {
            let wgsl = shader.linkable_wgsl.unwrap();
            assert!(!wgsl.contains("fn program_color("), "{}", shader.label);
            assert!(wgsl.contains("program_color("), "{}", shader.label);
        }
    }

    #[test]
    fn shaders_link_with_no_programs_and_with_several() {
        let programs = [grain(), rings()].map(|paint| paint.compile().unwrap().program);
        for shader in linkable() {
            let msl = link_msl(shader, []).unwrap();
            assert!(msl.contains(shader.fragment_entry));
            let msl = link_msl(shader, &programs).unwrap();
            for program in &programs {
                assert!(msl.contains(program.entry_point()), "{}", shader.label);
            }
        }
    }

    /// The downlevel module, programs linked in, translates to GLSL ES 3.00
    /// for WebGL2, as wgpu's GL backend does.
    #[test]
    fn the_downlevel_module_with_programs_writes_gles() {
        let programs = [grain(), rings()].map(|paint| paint.compile().unwrap().program);
        let (_, module, info) = link(&Linkable::base(Dialect::Downlevel), &programs).unwrap();
        let mut binding_map = naga::back::glsl::BindingMap::default();
        for (_, variable) in module.global_variables.iter() {
            if let Some(binding) = variable.binding {
                binding_map.insert(binding, (binding.group * 8 + binding.binding) as u8);
            }
        }
        let options = naga::back::glsl::Options {
            version: naga::back::glsl::Version::Embedded {
                version: 300,
                is_webgl: true,
            },
            writer_flags: naga::back::glsl::WriterFlags::ADJUST_COORDINATE_SPACE,
            binding_map,
            zero_initialize_workgroup_memory: true,
        };
        for pipeline in [
            crate::shaders::interface::QUADS,
            crate::shaders::interface::SHADOWS,
            crate::shaders::interface::PATH_RASTERIZATION,
        ] {
            let mut output = String::new();
            naga::back::glsl::Writer::new(
                &mut output,
                &module,
                &info,
                &options,
                &naga::back::glsl::PipelineOptions {
                    shader_stage: naga::ShaderStage::Fragment,
                    entry_point: pipeline.fragment_entry.into(),
                    multiview: None,
                },
                naga::proc::BoundsCheckPolicies::default(),
            )
            .and_then(|mut writer| writer.write())
            .unwrap_or_else(|error| panic!("{}: {error}", pipeline.label));
            assert!(output.contains("switch"), "{}", pipeline.label);
        }
    }

    /// The wgpu renderers' whole module links, in both dialects.
    #[test]
    fn the_base_module_links_in_both_dialects() {
        let programs = [grain(), rings()].map(|paint| paint.compile().unwrap().program);
        for dialect in [Dialect::Modern, Dialect::Downlevel] {
            let linkable = Linkable::base(dialect);
            assert!(!linkable.wgsl.contains("fn program_color("));
            link(&linkable, []).unwrap();
            let (source, module, _) = link(&linkable, &programs).unwrap();
            for program in &programs {
                assert!(source.contains(&format!("case {}u", program.id())));
            }
            for pipeline in [
                crate::shaders::interface::QUADS,
                crate::shaders::interface::SHADOWS,
                crate::shaders::interface::PATH_RASTERIZATION,
            ] {
                assert!(
                    module
                        .entry_points
                        .iter()
                        .any(|point| point.name == pipeline.fragment_entry),
                    "{dialect:?} lacks {}",
                    pipeline.fragment_entry
                );
            }
        }
    }
}
