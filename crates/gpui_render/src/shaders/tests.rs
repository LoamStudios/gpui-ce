use super::*;

#[test]
fn validates_subpixel_shader() {
    let generated = subpixel_sprite::WGSL_SOURCE.wgsl_source().unwrap();
    wgsl_rs::validate_wgsl_source(&format!("enable dual_source_blending;\n{generated}")).unwrap();
}

#[test]
fn build_generated_shaders_match_rust_sources() {
    assert_eq!(
        base::WGSL_SOURCE.wgsl_source().unwrap(),
        crate::artifacts::BASE_WGSL
    );
    let subpixel = subpixel_sprite::WGSL_SOURCE.wgsl_source().unwrap();
    assert_eq!(
        format!("enable dual_source_blending;\n{subpixel}"),
        crate::artifacts::SUBPIXEL_DUAL_SOURCE_WGSL
    );
}

#[test]
fn shader_interface_matches_generated_sources() {
    let standard_source = base::WGSL_SOURCE.wgsl_source().unwrap();
    let subpixel_source = subpixel_sprite::WGSL_SOURCE.wgsl_source().unwrap();
    for pipeline in interface::ALL {
        let source = if pipeline.label == interface::SUBPIXEL_SPRITES.label {
            &subpixel_source
        } else {
            &standard_source
        };
        assert!(
            source.contains(&format!("fn {}(", pipeline.vertex_entry)),
            "missing vertex entry point {}",
            pipeline.vertex_entry,
        );
        assert!(
            source.contains(&format!("fn {}(", pipeline.fragment_entry)),
            "missing fragment entry point {}",
            pipeline.fragment_entry,
        );
    }

    assert_eq!(common::GLOBALS.group, interface::GLOBAL_BIND_GROUP);
    assert_eq!(common::GLOBALS.binding, interface::GLOBAL_UNIFORMS_BINDING);
    assert_eq!(
        common::FONT_RASTERIZATION.group,
        interface::GLOBAL_BIND_GROUP
    );
    assert_eq!(
        common::FONT_RASTERIZATION.binding,
        interface::FONT_RASTERIZATION_BINDING
    );
    assert_eq!(quad::QUADS.group(), interface::DATA_BIND_GROUP);
    assert_eq!(quad::QUADS.binding(), interface::DATA_BUFFER_BINDING);
    assert_eq!(common::TRANSFORMS.group(), interface::GLOBAL_BIND_GROUP);
    assert_eq!(common::TRANSFORMS.binding(), interface::TRANSFORMS_BINDING);
    assert_eq!(common::CLIPS.group(), interface::GLOBAL_BIND_GROUP);
    assert_eq!(common::CLIPS.binding(), interface::CLIPS_BINDING);
    assert_eq!(common::PAINTS.binding(), interface::PAINTS_BINDING);
}

#[test]
fn native_slots_give_every_binding_its_own_slot() {
    let module = naga::front::wgsl::parse_str(&base::WGSL_SOURCE.wgsl_source().unwrap()).unwrap();
    let global_bindings = module
        .global_variables
        .iter()
        .filter_map(|(_, variable)| variable.binding.as_ref())
        .filter(|binding| binding.group == interface::GLOBAL_BIND_GROUP)
        .map(|binding| binding.binding + 1)
        .max();
    assert_eq!(global_bindings, Some(interface::GLOBAL_BINDING_COUNT));

    let mut slots = std::collections::BTreeSet::new();
    for group in [interface::GLOBAL_BIND_GROUP, interface::DATA_BIND_GROUP] {
        for binding in 0..interface::GLOBAL_BINDING_COUNT {
            assert!(
                slots.insert(interface::native_slot(group, binding)),
                "binding {group}:{binding} shares a native slot"
            );
        }
    }
    assert!(
        interface::MSL_BUFFER_SIZES_SLOT
            > interface::native_slot(interface::DATA_BIND_GROUP, interface::DATA_BUFFER_BINDING)
    );
    assert_ne!(
        interface::DX11_DRAW_CONSTANTS_REGISTER,
        interface::native_slot(
            interface::GLOBAL_BIND_GROUP,
            interface::GLOBAL_UNIFORMS_BINDING
        )
    );
    assert_ne!(
        interface::DX11_DRAW_CONSTANTS_REGISTER,
        interface::native_slot(
            interface::GLOBAL_BIND_GROUP,
            interface::FONT_RASTERIZATION_BINDING
        )
    );
    assert_ne!(
        interface::DX11_DRAW_CONSTANTS_REGISTER,
        interface::native_slot(interface::DATA_BIND_GROUP, interface::DATA_BUFFER_BINDING)
    );
}

fn vertex_output_shape(module: &naga::Module, entry_name: &str) -> (usize, u32) {
    let entry = module
        .entry_points
        .iter()
        .find(|entry| entry.name == entry_name)
        .unwrap_or_else(|| panic!("missing shader entry point {entry_name}"));
    let result = entry
        .function
        .result
        .as_ref()
        .unwrap_or_else(|| panic!("shader entry point {entry_name} has no output"));
    let naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner else {
        panic!("shader entry point {entry_name} must return a struct");
    };

    members
        .iter()
        .filter(|member| matches!(member.binding, Some(naga::Binding::Location { .. })))
        .fold((0, 0), |(locations, components), member| {
            let member_components = match module.types[member.ty].inner {
                naga::TypeInner::Scalar(_) => 1,
                naga::TypeInner::Vector { size, .. } => u32::from(size),
                ref ty => panic!("unsupported stage output type {ty:?} in {entry_name}"),
            };
            (locations + 1, components + member_components)
        })
}

#[test]
fn ordinary_vertex_interfaces_stay_within_compact_budgets() {
    let module = naga::front::wgsl::parse_str(&base::WGSL_SOURCE.wgsl_source().unwrap()).unwrap();
    for (entry, maximum_shape) in [
        ("vertex_quad", (8, 29)),
        ("vertex_shadow", (5, 17)),
        ("vertex_polychrome_sprite", (3, 7)),
        ("vertex_blur_composite", (2, 6)),
    ] {
        let actual_shape = vertex_output_shape(&module, entry);
        assert!(
            actual_shape.0 <= maximum_shape.0 && actual_shape.1 <= maximum_shape.1,
            "{entry} uses {actual_shape:?}, exceeding {maximum_shape:?} (locations, components)"
        );
    }
}

#[test]
fn scene_instance_storage_sizes_are_bounded() {
    for (name, actual, maximum) in [
        ("Quad", std::mem::size_of::<gpui::Quad>(), 256),
        ("Shadow", std::mem::size_of::<gpui::Shadow>(), 192),
        (
            "PolychromeSprite",
            std::mem::size_of::<gpui::PolychromeSprite>(),
            120,
        ),
    ] {
        assert!(
            actual <= maximum,
            "{name} is {actual} bytes, exceeding its {maximum}-byte storage budget"
        );
    }
}

#[test]
fn shader_buffer_layouts_match_host_layouts() {
    let modules = [
        crate::artifacts::BASE_WGSL,
        crate::artifacts::SUBPIXEL_DUAL_SOURCE_WGSL,
    ]
    .map(|source| naga::front::wgsl::parse_str(source).unwrap());

    for layouts in [gpui::SCENE_BUFFER_LAYOUTS, interface::RENDER_BUFFER_LAYOUTS] {
        for layout in layouts {
            let ty = modules
                .iter()
                .find_map(|module| {
                    module
                        .types
                        .iter()
                        .find_map(|(_, ty)| (ty.name.as_deref() == Some(layout.name)).then_some(ty))
                })
                .unwrap_or_else(|| panic!("missing shader ABI type {}", layout.name));
            let naga::TypeInner::Struct { members, span } = &ty.inner else {
                panic!("shader ABI type {} must be a struct", layout.name);
            };

            assert_eq!(
                *span as usize, layout.size,
                "shader ABI size: {}",
                layout.name
            );
            assert_eq!(
                members.len(),
                layout.fields.len(),
                "shader ABI fields: {}",
                layout.name
            );
            for (member, (name, offset)) in members.iter().zip(layout.fields) {
                assert_eq!(
                    member.name.as_deref(),
                    Some(*name),
                    "shader ABI field: {}",
                    layout.name
                );
                assert_eq!(
                    member.offset as usize, *offset,
                    "shader ABI offset: {}.{name}",
                    layout.name
                );
            }
        }
    }
}

#[test]
fn shader_discriminants_match_scene_types() {
    assert_eq!(common::ShaderBool::Disabled as u32, 0);
    assert_eq!(common::ShaderBool::Enabled as u32, 1);
    assert_eq!(
        common::BackgroundTag::Solid as u32,
        gpui::BackgroundTag::Solid as u32
    );
    assert_eq!(
        common::BackgroundTag::LinearGradient as u32,
        gpui::BackgroundTag::LinearGradient as u32
    );
    assert_eq!(
        common::BackgroundTag::PatternSlash as u32,
        gpui::BackgroundTag::PatternSlash as u32
    );
    assert_eq!(
        common::BackgroundTag::Checkerboard as u32,
        gpui::BackgroundTag::Checkerboard as u32
    );
    assert_eq!(
        common::BackgroundTag::Paint as u32,
        gpui::BackgroundTag::Paint as u32
    );
    assert_eq!(
        common::ColorSpace::Srgb as u32,
        gpui::ColorSpace::Srgb as u32
    );
    assert_eq!(
        common::ColorSpace::Oklab as u32,
        gpui::ColorSpace::Oklab as u32
    );
    assert_eq!(
        common::BorderStyle::Solid as u32,
        gpui::BorderStyle::Solid as u32
    );
    assert_eq!(
        common::BorderStyle::Dashed as u32,
        gpui::BorderStyle::Dashed as u32
    );
}

#[test]
fn gradients_are_dithered() {
    use common::*;
    use wgsl_rs::std::*;

    assert_ne!(
        gradient_dither(vec2f(10.0, 10.0)).w,
        gradient_dither(vec2f(11.0, 10.0)).w
    );
}

/// A paint-table gradient of `kind` with `geometry` and `radii`.
fn gradient(kind: common::PaintKind, geometry: [f32; 4], radii: [f32; 2]) -> common::ScenePaint {
    use common::*;
    use wgsl_rs::std::*;

    ScenePaint {
        transformation: TransformationMatrix {
            rotation_scale: mat2x2f(vec2f(1.0, 0.0), vec2f(0.0, 1.0)),
            translation: vec2f(0.0, 0.0),
        },
        kind,
        extend: PaintExtend::Pad,
        color_space: PaintColorSpace::Srgb,
        first_stop: 0,
        stop_count: 0,
        y_extend: PaintExtend::Pad,
        geometry: vec4f(geometry[0], geometry[1], geometry[2], geometry[3]),
        radii: vec4f(radii[0], radii[1], 0.0, 0.0),
    }
}

#[test]
fn gradient_offsets_follow_their_geometry() {
    use common::*;
    use wgsl_rs::std::*;

    let close = |actual: Vec2f, offset: f32| {
        assert_eq!(actual.y, 1.0, "defined");
        assert!((actual.x - offset).abs() < 1e-4, "{} vs {offset}", actual.x);
    };
    let linear = gradient(PaintKind::Linear, [10.0, 0.0, 110.0, 0.0], [0.0, 0.0]);
    close(gradient_offset(linear, vec2f(10.0, 50.0)), 0.0);
    close(gradient_offset(linear, vec2f(60.0, -5.0)), 0.5);
    close(gradient_offset(linear, vec2f(210.0, 0.0)), 2.0);

    let radial = gradient(PaintKind::Radial, [0.0, 0.0, 0.0, 0.0], [0.0, 100.0]);
    close(gradient_offset(radial, vec2f(30.0, 40.0)), 0.5);
    close(gradient_offset(radial, vec2f(0.0, 200.0)), 2.0);

    // A cone from a point to a circle beside it is undefined behind the point.
    let cone = gradient(PaintKind::Radial, [0.0, 0.0, 100.0, 0.0], [0.0, 10.0]);
    assert_eq!(gradient_offset(cone, vec2f(-50.0, 0.0)).y, 0.0);
    // Of the two circles through a point, the later one paints it.
    close(gradient_offset(cone, vec2f(100.0, 10.0)), 20200.0 / 19800.0);

    let sweep = gradient(PaintKind::Sweep, [0.0, 0.0, 0.0, 2.0 * PI], [0.0, 0.0]);
    close(gradient_offset(sweep, vec2f(0.0, 10.0)), 0.25);
    close(gradient_offset(sweep, vec2f(-10.0, 0.0)), 0.5);
}

#[test]
fn gradients_extend_past_their_ends() {
    use common::*;

    assert_eq!(extend_offset(PaintExtend::Pad, 1.5), 1.0);
    assert_eq!(extend_offset(PaintExtend::Pad, -0.5), 0.0);
    assert!((extend_offset(PaintExtend::Repeat, 1.25) - 0.25).abs() < 1e-6);
    assert!((extend_offset(PaintExtend::Repeat, -0.25) - 0.75).abs() < 1e-6);
    assert!((extend_offset(PaintExtend::Reflect, 1.25) - 0.75).abs() < 1e-6);
    assert!((extend_offset(PaintExtend::Reflect, -0.25) - 0.25).abs() < 1e-6);
}
