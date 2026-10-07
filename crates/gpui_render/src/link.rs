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

/// The WGSL of `shader` with `programs` linked in, or `None` if it does not
/// read the paint table. Programs run unchanged wherever a paint names
/// their id; any other program id draws its paint's fallback colour.
pub fn linked_wgsl<'a>(
    shader: &NativeShader,
    programs: impl IntoIterator<Item = &'a Program>,
) -> Option<String> {
    let linkable = shader.linkable_wgsl?;
    let programs: Vec<&Program> = programs.into_iter().collect();
    let prelude = prelude_source();
    let mut source = String::with_capacity(
        linkable.len()
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
    source.push_str(linkable);
    source.push('\n');
    source.push_str(prelude);
    source.push_str(
        "
fn paint_param(index: u32) -> vec4<f32> {
    return PAINTS[index].value;
}

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
    Some(source)
}

/// `shader`'s WGSL with `programs` linked in, parsed and validated.
pub fn link_module<'a>(
    shader: &NativeShader,
    programs: impl IntoIterator<Item = &'a Program>,
) -> Result<(naga::Module, naga::valid::ModuleInfo), LinkError> {
    let source = linked_wgsl(shader, programs)
        .ok_or_else(|| LinkError(format!("{} does not read the paint table", shader.label)))?;
    let module = naga::front::wgsl::parse_str(&source).map_err(|error| {
        LinkError(format!(
            "{} with programs linked in: {}",
            shader.label,
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
            shader.label,
            error.emit_to_string(&source)
        ))
    })?;
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
}
