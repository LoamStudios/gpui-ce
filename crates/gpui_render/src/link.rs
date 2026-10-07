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
///
/// Programs that fill paints run through `program_color`, in every module
/// that reads the paint table; those that read a filter's input fill
/// nothing, and are left out of it. Every program can filter a group, and
/// runs through `program_filter_color`, in the module that runs group
/// filters, where `program_input` reads the group's picture. Elsewhere,
/// `program_input` reads nothing.
pub fn link_source<'a>(
    linkable: &Linkable,
    programs: impl IntoIterator<Item = &'a Program>,
) -> String {
    let programs: Vec<&Program> = programs.into_iter().collect();
    let fills = linkable.wgsl.contains("program_color(");
    let filters = linkable.wgsl.contains("program_filter_color(");
    let linked: Vec<&Program> = programs
        .iter()
        .copied()
        .filter(|program| filters || (fills && !program.reads_input()))
        .collect();
    let prelude = prelude_source();
    let mut source = String::with_capacity(
        linkable.wgsl.len()
            + prelude.len()
            + linked
                .iter()
                .map(|program| program.source().len() + 256)
                .sum::<usize>()
            + 2048,
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
    source.push_str(if linkable.wgsl.contains("fn group_filter_input(") {
        "
fn program_input(fragment: Fragment, offset: vec2<f32>) -> vec4<f32> {
    return group_filter_input(fragment.position + offset);
}
"
    } else {
        "
fn program_input(fragment: Fragment, offset: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(0.0);
}
"
    });
    if fills {
        source.push_str(
            "
fn program_color(id: u32, uv: vec2<f32>, position: vec2<f32>, size: vec2<f32>, origin: vec2<f32>, scale: f32, stroke: vec2<f32>, base: u32, fallback: vec4<f32>) -> vec4<f32> {
    let fragment = Fragment(uv, position, size, origin, scale, stroke);
    switch id {
",
        );
        for program in linked.iter().filter(|program| !program.reads_input()) {
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
    }
    if filters {
        source.push_str(
            "
fn program_filter_color(id: u32, uv: vec2<f32>, position: vec2<f32>, size: vec2<f32>, origin: vec2<f32>, scale: f32, stroke: vec2<f32>, base: u32, fallback: vec4<f32>) -> vec4<f32> {
    let fragment = Fragment(uv, position, size, origin, scale, stroke);
    switch id {
",
        );
        for program in &linked {
            source.push_str(&format!(
                "        case {}u: {{ return {}(fragment, base); }}\n",
                program.id(),
                program.entry_point(),
            ));
        }
        source.push_str(
            "        default: { return vec4<f32>(fallback.xyz * fallback.w, fallback.w); }
    }
}
",
        );
    }
    source.push('\n');
    for program in &linked {
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

/// Shader-model 5.0 HLSL for `shader` with `programs` linked in, for
/// Direct3D 11 to compile: linked and validated, then translated by the
/// floor Naga with the build's registers and draw constants (see
/// [`crate::hlsl`]), as the standard shaders' HLSL is.
#[cfg(any(windows, test))]
pub fn link_hlsl<'a>(
    shader: &NativeShader,
    programs: impl IntoIterator<Item = &'a Program>,
) -> Result<String, LinkError> {
    let linkable = Linkable::native(shader)
        .ok_or_else(|| LinkError(format!("{} does not read the paint table", shader.label)))?;
    let source = link_source(&linkable, programs);
    let module = naga_old::front::wgsl::parse_str(&source).map_err(|error| {
        LinkError(format!(
            "{} with programs linked in, for HLSL: {}",
            shader.label,
            error.emit_to_string(&source)
        ))
    })?;
    let info = naga_old::valid::Validator::new(
        naga_old::valid::ValidationFlags::all(),
        naga_old::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| {
        LinkError(format!(
            "{} with programs linked in, for HLSL: {}",
            shader.label,
            error.emit_to_string(&source)
        ))
    })?;
    crate::hlsl::write_hlsl(&module, &info, &source, shader.pipeline).map_err(LinkError)
}

/// Metal Shading Language for `shader` with `programs` linked in.
///
/// The programs, and the prelude functions they call, compute what they do
/// on the CPU: their bodies opt out of fast math, and call the precise forms
/// of the math functions. The renderer's own code, around them, is compiled
/// with fast math, as the standard shaders are.
pub fn link_msl<'a>(
    shader: &NativeShader,
    programs: impl IntoIterator<Item = &'a Program>,
) -> Result<String, LinkError> {
    let programs: Vec<&Program> = programs.into_iter().collect();
    let (module, info) = link_module(shader, programs.iter().copied())?;
    let msl = crate::msl::write_msl(&module, &info, shader.label).map_err(LinkError)?;
    let names = std::iter::once(prelude_source())
        .chain(programs.iter().map(|program| program.source()))
        .flat_map(wgsl_function_names)
        .collect::<std::collections::HashSet<_>>();
    Ok(crate::msl::precise_functions(&msl, &names))
}

/// The names of the functions `wgsl` defines.
fn wgsl_function_names(wgsl: &str) -> impl Iterator<Item = &str> {
    wgsl.split("fn ").skip(1).filter_map(|rest| {
        let name = &rest[..rest.find('(')?];
        name.chars()
            .all(|character| character.is_alphanumeric() || character == '_')
            .then_some(name)
    })
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
                "path_rasterization",
                "meshes",
                "monochrome_sprites",
                "group_filter"
            ]
        );
        for shader in linkable() {
            let wgsl = shader.linkable_wgsl.unwrap();
            assert!(!wgsl.contains("fn program_color("), "{}", shader.label);
            assert!(
                !wgsl.contains("fn program_filter_color("),
                "{}",
                shader.label
            );
            assert!(
                wgsl.contains("program_color(") || wgsl.contains("program_filter_color("),
                "{}",
                shader.label
            );
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

    /// Each shader that reads paints, programs linked in, translates to
    /// shader-model 5.0 HLSL the way the build translates it without: its
    /// entry points, the programs, and the draw constants as a `cbuffer`.
    #[test]
    fn shaders_link_into_hlsl() {
        let programs = [grain(), rings()].map(|paint| paint.compile().unwrap().program);
        for shader in linkable() {
            let stock = link_hlsl(shader, []).unwrap();
            let hlsl = link_hlsl(shader, &programs).unwrap();
            for source in [&stock, &hlsl] {
                assert!(source.contains(shader.vertex_entry), "{}", shader.label);
                assert!(source.contains(shader.fragment_entry), "{}", shader.label);
            }
            for program in &programs {
                assert!(hlsl.contains(program.entry_point()), "{}", shader.label);
            }
            if crate::hlsl::dx11_draw_constants_register(shader.pipeline).is_some() {
                assert!(
                    hlsl.contains("cbuffer DrawConstants : register(b")
                        && !hlsl.contains("ConstantBuffer<NagaConstants>"),
                    "{}",
                    shader.label
                );
            }
        }
    }

    /// On Metal, programs and the prelude compute precisely, whatever the
    /// library is compiled with; the renderer's own functions do not.
    #[test]
    fn linked_programs_compute_precisely_on_metal() {
        let wave = shader::paint(|px| {
            let value = (px.position().x() * 0.37).sin() * 0.5 + 0.5;
            shader::rgba(&value, &value, &value, 1.0)
        })
        .compile()
        .unwrap()
        .program;
        let shader = linkable().find(|shader| shader.label == "quads").unwrap();
        let msl = link_msl(shader, [&wave]).unwrap();
        let body = |name: &str| {
            let start = msl
                .lines()
                .position(|line| !line.starts_with(' ') && line.contains(&format!(" {name}")))
                .unwrap_or_else(|| panic!("{name} is defined"));
            msl.lines()
                .skip(start)
                .take_while(|line| *line != "}")
                .collect::<Vec<_>>()
                .join("\n")
        };
        let program = body(wave.entry_point());
        assert!(
            program.contains("#pragma METAL fp math_mode(safe)"),
            "{program}"
        );
        assert!(program.contains("metal::precise::sin("), "{program}");
        assert!(!program.contains("metal::sin("), "{program}");
        let quad = body("fragment_quad(");
        assert!(!quad.contains("#pragma"), "{quad}");
        assert!(!quad.contains("metal::precise::"), "{quad}");
    }

    /// A program reading a filter's input links into the group-filter
    /// shader, where `program_input` reads the group's picture, on Metal and
    /// Direct3D; the shaders that fill paints leave it out.
    #[test]
    fn filter_programs_link_only_where_groups_are_filtered() {
        let sharpen = shader::paint(|px| {
            let around = px.input_at(shader::vec2(1.0, 0.0)).rgba()
                + px.input_at(shader::vec2(-1.0, 0.0)).rgba();
            Paint::premultiplied(px.input().rgba() * 3.0 - around)
        })
        .compile()
        .unwrap()
        .program;
        assert!(sharpen.reads_input());
        let grain = grain().compile().unwrap().program;
        assert!(!grain.reads_input());
        let programs = [sharpen.clone(), grain.clone()];
        for shader in linkable() {
            let source = linked_wgsl(shader, &programs).unwrap();
            let filters = shader.label == "group_filter";
            assert_eq!(
                source.contains(sharpen.entry_point()),
                filters,
                "{}",
                shader.label
            );
            assert!(source.contains(grain.entry_point()), "{}", shader.label);
            assert_eq!(
                source.contains("group_filter_input(fragment.position + offset)"),
                filters,
                "{}",
                shader.label
            );
            let msl = link_msl(shader, &programs).unwrap();
            let hlsl = link_hlsl(shader, &programs).unwrap();
            if filters {
                assert!(msl.contains(sharpen.entry_point()));
                assert!(hlsl.contains(sharpen.entry_point()));
            }
        }
        // The wgpu module runs both, in both dialects.
        for dialect in [Dialect::Modern, Dialect::Downlevel] {
            let (source, _, _) = link(&Linkable::base(dialect), &programs).unwrap();
            assert!(source.contains(&format!(
                "case {}u: {{ return {}(",
                sharpen.id(),
                sharpen.entry_point()
            )));
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
            crate::shaders::interface::MESHES,
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
                crate::shaders::interface::MESHES,
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
