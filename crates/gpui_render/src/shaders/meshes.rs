/// Meshes: triangles from a mesh's retained vertices, placed by its instance
/// and filled with its paint. See `gpui::Mesh` for the design.
#[wgsl_rs::wgsl]
pub mod mesh {
    use super::super::common::*;
    use wgsl_rs::std::*;

    /// How a mesh is drawn: its vertices are at `origin + position * scale`
    /// in the space of transform-table entry `transform`.
    #[derive(Clone, Copy, Wgsl)]
    pub struct MeshInstance {
        pub order: u32,
        pub transform: u32,
        pub clip: u32,
        pub scale: f32,
        pub origin: Vec2f,
        pub bounds: Bounds,
        pub content_mask: ContentMask,
        pub paint: PaintRef,
        pub padding: u32,
    }

    /// A vertex of a mesh: its position in logical pixels, the stroke
    /// coordinates paints read, and its antialiasing fringe: pushed
    /// `extrude` device pixels along `normal`, at `coverage`.
    #[derive(Clone, Copy, Wgsl)]
    pub struct MeshVertex {
        pub position: Vec2f,
        pub normal: Vec2f,
        pub stroke: Vec2f,
        pub extrude: f32,
        pub coverage: f32,
    }

    storage!(group(1), binding(0), MESHES: RuntimeArray<MeshInstance>);
    storage!(group(1), binding(1), MESH_VERTICES: RuntimeArray<MeshVertex>);

    /// The fewest device pixels a mesh unit is taken to span, so a mesh
    /// scaled to nothing divides by no zero.
    pub const MIN_MESH_PIXELS: f32 = 0.000001;

    #[derive(Wgsl)]
    pub struct MeshVarying {
        #[builtin(position)]
        pub position: Vec4f,
        #[location(0)]
        #[interpolate(flat)]
        pub mesh_id: u32,
        #[location(1)]
        pub clip_distances: Vec4f,
        #[location(2)]
        #[interpolate(flat)]
        pub solid: Vec4f,
        #[location(3)]
        pub coverage: f32,
        #[location(4)]
        pub stroke: Vec2f,
    }

    #[vertex]
    pub fn vertex_mesh(
        #[builtin(vertex_index)] vertex_id: u32,
        #[builtin(instance_index)] instance_id: u32,
    ) -> MeshVarying {
        let mesh = get!(MESHES)[instance_id as usize];
        let vertex = get!(MESH_VERTICES)[vertex_id as usize];
        // Device pixels on screen per logical pixel of the mesh: its scale,
        // its transform's, and a chunk's placement's.
        let placement = get!(GLOBALS).placement;
        let placement_scale = sqrt(abs(placement.x * placement.w - placement.y * placement.z));
        let pixels = mesh.scale * pixels_per_unit(mesh.transform) * placement_scale;
        // The fringe is pushed by device pixels, whatever the scale.
        let local =
            vertex.position + vertex.normal * (vertex.extrude / max(pixels, MIN_MESH_PIXELS));
        let viewport_position = TransformationMatrix::transform_position(
            scene_transformation(mesh.transform),
            mesh.origin + local * mesh.scale,
        );
        MeshVarying {
            position: viewport_to_clip_position(viewport_position),
            mesh_id: instance_id,
            clip_distances: clip_distances(viewport_position, mesh.content_mask.bounds),
            solid: prepare_paint(mesh.paint),
            coverage: vertex.coverage,
            stroke: vertex.stroke,
        }
    }

    #[fragment]
    pub fn fragment_mesh(input: MeshVarying) -> Vec4f {
        if is_clipped(input.clip_distances) {
            return transparent();
        }
        let mesh = get!(MESHES)[input.mesh_id as usize];
        let position = scene_position(input.position.xy());
        let fade =
            ContentMask::alpha(mesh.content_mask, position) * clip_coverage(mesh.clip, position);
        let color = paint_color_at(mesh.paint, position, input.solid, input.stroke);
        blend_color(color, saturate(input.coverage) * fade)
    }
}
