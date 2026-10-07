//! Meshes: custom geometry, as triangles, drawn as a primitive of the scene.
//!
//! A [`Mesh`] is triangles in an element's own logical pixels, made once and
//! shared through an `Arc`. [`Window::paint_mesh`](crate::Window::paint_mesh)
//! draws one in scene order, like a quad: under the element's transform, inside
//! its clips and groups, and filled with any paint (a colour, a gradient, a
//! photo or a shader program). Pressure strokes, and later brush painting, are
//! drawn as meshes.
//!
//! # Design
//!
//! - **Vertices** ([`MeshVertex`], 32 bytes) carry a position in the element's
//!   logical pixels; stroke coordinates, which paints read (a shader program
//!   sees them as [`Pixel::stroke`](crate::shader::Pixel::stroke): distance
//!   along a stroke and across it, from -1 on its left to 1 on its right);
//!   and an antialiasing fringe: a coverage, and a direction and distance in
//!   device pixels to push the vertex along. Indices are `u32` triangles.
//! - **Antialiasing** needs no multisampling. The constructors give a mesh a
//!   fringe one device pixel wide around its outline: each vertex on the
//!   outline is pushed half a pixel inwards, at full coverage, and a copy of
//!   it half a pixel outwards, at none, with the outline's (mitred) normal.
//!   The vertex shader measures that push in device pixels under the mesh's
//!   transform, zoom and chunk placement, so the fringe stays a pixel wide
//!   at any scale, and coverage falls linearly across it.
//! - **Paint** is an entry of the scene's paint table, or a colour, as for
//!   quads: the mesh's fragments call `paint_color` with it, so every paint
//!   works, and programs also see the stroke coordinates.
//! - **Order**: a mesh is ordered by its bounds, transformed into the
//!   viewport and grown by its fringe, in the scene's bounds tree like any
//!   primitive. Meshes adjacent in draw order form one batch, drawn with
//!   one pipeline, a draw call each.
//! - **Transforms, clips, groups**: a mesh refers to the transform- and
//!   clip-table entries of the element that painted it, and has the
//!   viewport-aligned content mask, so it turns, clips and fades with its
//!   element; groups fold their opacity into its paint, or draw it into
//!   their targets.
//! - **Retention**: a mesh has a [`MeshId`], unique for its life. Renderers
//!   upload its vertices and indices once and keep them, keyed by that id,
//!   until it has not been drawn for a while. A frame that draws it again,
//!   or a chunk replayed under a new placement, uploads only the mesh's
//!   instance: its transform, origin, scale, mask and paint. Nothing is
//!   tessellated per frame.
//! - **Hit testing**: none. Meshes are geometry an element paints inside its
//!   own bounds, which is what its hitbox tests.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use lyon::tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers};

use crate::{Bounds, Pixels, Point, point, px, size};

/// A mesh's identity: unique among the meshes made in the process, so a
/// renderer can keep its buffers by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MeshId(pub u64);

impl MeshId {
    fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// A vertex of a [`Mesh`], as the GPU reads it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct MeshVertex {
    /// Where it is, in the element's logical pixels.
    pub position: [f32; 2],
    /// The direction it is pushed in, for its antialiasing fringe: unit
    /// length on a straight edge, longer at a mitred corner, so pushing
    /// it by a distance moves the edges beside it by that distance.
    pub normal: [f32; 2],
    /// Its stroke coordinates, which paints read: for a strip, its distance
    /// along the stroke in logical pixels, and across it, from -1 on the
    /// left to 1 on the right.
    pub stroke: [f32; 2],
    /// How far it is pushed along `normal`, in device pixels.
    pub extrude: f32,
    /// How much of a pixel it covers: 1 inside, 0 at the outer edge of the
    /// fringe.
    pub coverage: f32,
}

impl MeshVertex {
    /// A vertex inside the mesh, with no fringe.
    pub fn new(position: Point<Pixels>, stroke: Point<f32>) -> Self {
        Self {
            position: [position.x.0, position.y.0],
            normal: [0., 0.],
            stroke: [stroke.x, stroke.y],
            extrude: 0.,
            coverage: 1.,
        }
    }
}

/// Triangles in an element's logical pixels, with an antialiased fringe,
/// drawn with [`Window::paint_mesh`](crate::Window::paint_mesh). Immutable:
/// share it through an `Arc` and draw it every frame, and renderers upload it
/// once.
#[derive(Debug)]
pub struct Mesh {
    id: MeshId,
    vertices: Vec<MeshVertex>,
    indices: Vec<u32>,
    bounds: Bounds<Pixels>,
}

/// How far the constructors push an outline's vertices, in device pixels,
/// inwards at full coverage and outwards at none.
const HALF_FRINGE: f32 = 0.5;
/// The longest a fringe's mitre gets, in multiples of its width; sharper
/// corners are cut short.
const MITER_LIMIT: f32 = 4.;
/// Triangles with less area than this, in square logical pixels, have no
/// edges of their own on the outline.
const MIN_AREA: f32 = 1e-7;

impl Mesh {
    /// A mesh of `vertices` and `indices`, three to a triangle, as given:
    /// any antialiasing fringe is the caller's (see [`MeshVertex`]).
    pub fn from_vertices(vertices: Vec<MeshVertex>, indices: Vec<u32>) -> Self {
        debug_assert!(indices.len().is_multiple_of(3), "meshes are triangles");
        debug_assert!(
            indices
                .iter()
                .all(|&index| (index as usize) < vertices.len()),
            "a mesh index past its vertices"
        );
        let bounds = vertex_bounds(&vertices);
        Self {
            id: MeshId::next(),
            vertices,
            indices,
            bounds,
        }
    }

    /// Triangles over `positions`, with `stroke` coordinates for each
    /// position if given, and an antialiasing fringe one device pixel wide
    /// added around their outline: the edges that belong to one triangle.
    pub fn from_triangles(
        positions: &[Point<Pixels>],
        stroke: Option<&[Point<f32>]>,
        indices: &[u32],
    ) -> Self {
        let vertices = positions
            .iter()
            .enumerate()
            .map(|(index, position)| {
                let stroke = stroke
                    .and_then(|stroke| stroke.get(index))
                    .copied()
                    .unwrap_or_default();
                MeshVertex::new(*position, stroke)
            })
            .collect();
        let (vertices, indices) = with_fringe(vertices, indices);
        Self::from_vertices(vertices, indices)
    }

    /// The closed polygon through `points`, filled by `fill`, with an
    /// antialiased edge. Self-intersecting outlines, such as a pressure
    /// stroke's, are tessellated by `fill`'s rule; the fringe follows the
    /// outline of what is filled.
    pub fn from_polygon(points: &[Point<Pixels>], fill: peniko::Fill) -> Self {
        Self::from_contours(&[points], fill)
    }

    /// The closed polygons through each of `contours`, filled together by
    /// `fill`, so one inside another is a hole by the even-odd rule, or by
    /// the non-zero rule when it winds the other way.
    pub fn from_contours(contours: &[&[Point<Pixels>]], fill: peniko::Fill) -> Self {
        let mut builder = lyon::path::Path::builder();
        for contour in contours {
            let mut points = contour.iter();
            let Some(first) = points.next() else {
                continue;
            };
            builder.begin(lyon::math::point(first.x.0, first.y.0));
            for point in points {
                builder.line_to(lyon::math::point(point.x.0, point.y.0));
            }
            builder.end(true);
        }
        let path = builder.build();
        let options = FillOptions::default().with_fill_rule(match fill {
            peniko::Fill::NonZero => lyon::tessellation::FillRule::NonZero,
            peniko::Fill::EvenOdd => lyon::tessellation::FillRule::EvenOdd,
        });
        let mut buffers: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
        let tessellated = FillTessellator::new().tessellate_path(
            &path,
            &options,
            &mut BuffersBuilder::new(&mut buffers, |vertex: FillVertex| {
                vertex.position().to_array()
            }),
        );
        if let Err(error) = tessellated {
            log::error!("a mesh polygon did not tessellate: {error:?}");
            return Self::from_vertices(Vec::new(), Vec::new());
        }
        // The fringe follows edges that belong to one triangle, so vertices
        // the tessellator repeats at one position are made one.
        let (positions, indices) = weld(&buffers.vertices, &buffers.indices);
        let vertices = positions
            .into_iter()
            .map(|[x, y]| MeshVertex::new(point(px(x), px(y)), Point::default()))
            .collect();
        let (vertices, indices) = with_fringe(vertices, &indices);
        Self::from_vertices(vertices, indices)
    }

    /// A strip through `ribs`, each a stroke's left and right edge at one
    /// point along it, as pressure strokes are drawn: quads between
    /// successive ribs, antialiased all round. Its stroke coordinates are the
    /// distance along it, between the ribs' midpoints, in logical pixels, and
    /// -1 on its left edge to 1 on its right.
    ///
    /// A strip that crosses itself is drawn twice where it does: fill a
    /// stroke's outline with [`Self::from_polygon`] to draw it once.
    pub fn strip(ribs: &[(Point<Pixels>, Point<Pixels>)]) -> Self {
        let mut vertices = Vec::with_capacity(ribs.len() * 2);
        let mut along = 0.;
        let mut previous: Option<Point<Pixels>> = None;
        for (left, right) in ribs {
            let middle = point((left.x + right.x) / 2., (left.y + right.y) / 2.);
            if let Some(previous) = previous {
                along += (middle.x.0 - previous.x.0).hypot(middle.y.0 - previous.y.0);
            }
            previous = Some(middle);
            vertices.push(MeshVertex::new(*left, point(along, -1.)));
            vertices.push(MeshVertex::new(*right, point(along, 1.)));
        }
        let mut indices = Vec::with_capacity(ribs.len().saturating_sub(1) * 6);
        for rib in 1..ribs.len() as u32 {
            let (left, right) = (2 * (rib - 1), 2 * (rib - 1) + 1);
            let (next_left, next_right) = (2 * rib, 2 * rib + 1);
            indices.extend_from_slice(&[left, right, next_right, left, next_right, next_left]);
        }
        let (vertices, indices) = with_fringe(vertices, &indices);
        Self::from_vertices(vertices, indices)
    }

    /// The mesh's identity, which renderers keep its buffers by.
    pub fn id(&self) -> MeshId {
        self.id
    }

    /// Its vertices, fringe included.
    pub fn vertices(&self) -> &[MeshVertex] {
        &self.vertices
    }

    /// Its triangles, three indices each.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// The bounds of its vertices, in logical pixels, before any fringe
    /// is pushed out of them.
    pub fn bounds(&self) -> Bounds<Pixels> {
        self.bounds
    }

    /// How many bytes its vertices and indices take on the GPU.
    pub fn byte_len(&self) -> usize {
        std::mem::size_of_val(self.vertices.as_slice())
            + std::mem::size_of_val(self.indices.as_slice())
    }
}

fn vertex_bounds(vertices: &[MeshVertex]) -> Bounds<Pixels> {
    let mut iter = vertices.iter();
    let Some(first) = iter.next() else {
        return Bounds::default();
    };
    let (mut min, mut max) = (first.position, first.position);
    for vertex in iter {
        for axis in 0..2 {
            min[axis] = min[axis].min(vertex.position[axis]);
            max[axis] = max[axis].max(vertex.position[axis]);
        }
    }
    Bounds::new(
        point(px(min[0]), px(min[1])),
        size(px(max[0] - min[0]), px(max[1] - min[1])),
    )
}

/// `positions` with repeats made one, and `indices` into what is left.
fn weld(positions: &[[f32; 2]], indices: &[u32]) -> (Vec<[f32; 2]>, Vec<u32>) {
    let mut unique = collections::FxHashMap::<[u32; 2], u32>::default();
    let mut welded = Vec::with_capacity(positions.len());
    let remap: Vec<u32> = positions
        .iter()
        .map(|position| {
            *unique.entry(position.map(f32::to_bits)).or_insert_with(|| {
                welded.push(*position);
                (welded.len() - 1) as u32
            })
        })
        .collect();
    let indices = indices.iter().map(|&index| remap[index as usize]).collect();
    (welded, indices)
}

/// `vertices` and `indices` with an antialiasing fringe one device pixel
/// wide around the outline of their triangles: the edges only one
/// (non-degenerate) triangle has. Each vertex on the outline is pushed half a
/// pixel inwards, and a copy of it at no coverage half a pixel outwards; a
/// quad of two triangles joins each outline edge to its copy.
fn with_fringe(mut vertices: Vec<MeshVertex>, indices: &[u32]) -> (Vec<MeshVertex>, Vec<u32>) {
    let position = |vertices: &[MeshVertex], index: u32| vertices[index as usize].position;
    // Each undirected edge, by how many triangles have it, and the way the
    // last one goes round it, with the triangle's inside on its left.
    let mut edges = collections::FxHashMap::<(u32, u32), (u32, (u32, u32))>::default();
    for triangle in indices.chunks_exact(3) {
        let [a, mut b, mut c] = [triangle[0], triangle[1], triangle[2]];
        let (pa, pb, pc) = (
            position(&vertices, a),
            position(&vertices, b),
            position(&vertices, c),
        );
        let area = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]);
        if area.abs() < MIN_AREA {
            continue;
        }
        if area < 0. {
            std::mem::swap(&mut b, &mut c);
        }
        for (from, to) in [(a, b), (b, c), (c, a)] {
            let entry = edges
                .entry((from.min(to), from.max(to)))
                .or_insert((0, (from, to)));
            entry.0 += 1;
            entry.1 = (from, to);
        }
    }

    // The outline's edges, each with the inside on its left, and its
    // outward normal.
    let mut outline: Vec<(u32, u32, [f32; 2])> = edges
        .values()
        .filter(|(count, _)| *count == 1)
        .filter_map(|&(_, (from, to))| {
            let (p, q) = (position(&vertices, from), position(&vertices, to));
            let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
            let length = dx.hypot(dy);
            (length > 1e-6).then(|| (from, to, [dy / length, -dx / length]))
        })
        .collect();
    // In a fixed order, so a mesh is built the same way every time.
    outline.sort_unstable_by_key(|&(from, to, _)| (from, to));

    // The normals of the outline edges at each vertex on it.
    let mut normals = collections::FxHashMap::<u32, smallvec::SmallVec<[[f32; 2]; 2]>>::default();
    for &(from, to, normal) in &outline {
        normals.entry(from).or_default().push(normal);
        normals.entry(to).or_default().push(normal);
    }

    let mut indices = indices.to_vec();
    let mut outer = collections::FxHashMap::<u32, u32>::default();
    let mut on_outline: Vec<u32> = normals.keys().copied().collect();
    on_outline.sort_unstable();
    for index in on_outline {
        let normal = miter(&normals[&index]);
        let inner = &mut vertices[index as usize];
        inner.normal = normal;
        inner.extrude = -HALF_FRINGE;
        inner.coverage = 1.;
        let copy = MeshVertex {
            extrude: HALF_FRINGE,
            coverage: 0.,
            ..*inner
        };
        vertices.push(copy);
        outer.insert(index, (vertices.len() - 1) as u32);
    }
    indices.reserve(outline.len() * 6);
    for (from, to, _) in outline {
        let (outer_from, outer_to) = (outer[&from], outer[&to]);
        indices.extend_from_slice(&[from, to, outer_to, from, outer_to, outer_from]);
    }
    (vertices, indices)
}

/// The direction a vertex on an outline is pushed in, given the normals of
/// the outline edges that meet there: the mitre of two, so each edge moves
/// by the distance pushed, as long as [`MITER_LIMIT`] allows.
fn miter(normals: &[[f32; 2]]) -> [f32; 2] {
    let sum = normals.iter().fold([0., 0.], |sum, normal| {
        [sum[0] + normal[0], sum[1] + normal[1]]
    });
    match normals {
        [normal] => *normal,
        [first, second] => {
            let cosine = first[0] * second[0] + first[1] * second[1];
            let denominator = 1. + cosine;
            if denominator < 1. / (MITER_LIMIT * MITER_LIMIT) {
                // Nearly reversed: a spike, cut short along its axis.
                let length = sum[0].hypot(sum[1]);
                if length < 1e-6 {
                    return [0., 0.];
                }
                return [sum[0] / length * MITER_LIMIT, sum[1] / length * MITER_LIMIT];
            }
            let miter = [sum[0] / denominator, sum[1] / denominator];
            let length = miter[0].hypot(miter[1]);
            if length > MITER_LIMIT {
                [
                    miter[0] / length * MITER_LIMIT,
                    miter[1] / length * MITER_LIMIT,
                ]
            } else {
                miter
            }
        }
        // Where the outline touches itself: pushed the way its edges face
        // on the whole.
        _ => {
            let length = sum[0].hypot(sum[1]);
            if length < 1e-6 {
                [0., 0.]
            } else {
                [sum[0] / length, sum[1] / length]
            }
        }
    }
}

/// A mesh where a scene draws it: the shared mesh, and its instance.
#[derive(Clone, Debug)]
pub struct MeshPrimitive {
    /// What the GPU reads to place and paint it.
    pub instance: MeshInstance,
    /// Its vertices and indices.
    pub mesh: Arc<Mesh>,
}

/// How a scene draws a [`Mesh`], as the GPU reads it: a mesh vertex is at
/// `origin + position * scale` in the space of transform-table entry
/// `transform`.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct MeshInstance {
    /// Where it draws among the scene's primitives.
    pub order: super::DrawOrder,
    /// Its entry in the scene's transform table.
    pub transform: u32,
    /// Its entry in the scene's clip table.
    pub clip: u32,
    /// Device pixels in the transform's space per logical pixel of the mesh.
    pub scale: f32,
    /// Where the mesh's origin is, in the transform's space.
    pub origin: Point<crate::ScaledPixels>,
    /// The bounds of what it draws, fringe included, in the transform's
    /// space.
    pub bounds: Bounds<crate::ScaledPixels>,
    /// The viewport rectangle it is drawn within.
    pub content_mask: crate::ContentMask<crate::ScaledPixels>,
    /// What it is filled with.
    pub paint: super::ScenePaintRef,
    /// Aligns the instance as the shaders do.
    pub padding: u32,
}

impl From<MeshPrimitive> for super::Primitive {
    fn from(mesh: MeshPrimitive) -> Self {
        super::Primitive::Mesh(mesh)
    }
}

const _: () = {
    assert!(std::mem::size_of::<MeshVertex>() == 32);
    assert!(std::mem::size_of::<MeshInstance>() == 96);
};

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Vec<Point<Pixels>> {
        [(0., 0.), (10., 0.), (10., 10.), (0., 10.)]
            .map(|(x, y)| point(px(x), px(y)))
            .to_vec()
    }

    /// The fringe of `mesh`: its vertices at no coverage.
    fn fringe(mesh: &Mesh) -> Vec<&MeshVertex> {
        mesh.vertices()
            .iter()
            .filter(|vertex| vertex.coverage == 0.)
            .collect()
    }

    #[test]
    fn a_polygon_has_a_fringe_pushed_outwards_along_mitres() {
        let mesh = Mesh::from_polygon(&square(), peniko::Fill::NonZero);
        let fringe = fringe(&mesh);
        assert_eq!(fringe.len(), 4);
        for vertex in fringe {
            assert_eq!(vertex.extrude, HALF_FRINGE);
            // Outwards from the centre, along the diagonal, and long enough
            // that each side moves by the push.
            let outward = [vertex.position[0] - 5., vertex.position[1] - 5.];
            assert!(vertex.normal[0] * outward[0] > 0. && vertex.normal[1] * outward[1] > 0.);
            assert!((vertex.normal[0].abs() - 1.).abs() < 1e-5);
            assert!((vertex.normal[1].abs() - 1.).abs() < 1e-5);
        }
        // Two interior triangles, and a quad on each of four sides.
        assert_eq!(mesh.indices().len(), 3 * (2 + 4 * 2));
        assert_eq!(
            mesh.bounds(),
            Bounds::new(point(px(0.), px(0.)), size(px(10.), px(10.)))
        );
    }

    #[test]
    fn a_self_intersecting_outline_is_filled_by_its_rule() {
        // A bow tie: two triangles meeting at (5, 5).
        let bow_tie =
            [(0., 0.), (10., 10.), (10., 0.), (0., 10.)].map(|(x, y)| point(px(x), px(y)));
        for fill in [peniko::Fill::NonZero, peniko::Fill::EvenOdd] {
            let mesh = Mesh::from_polygon(&bow_tie, fill);
            let inside: f32 = mesh
                .indices()
                .chunks_exact(3)
                .map(|triangle| {
                    let [a, b, c] = [0, 1, 2].map(|i| mesh.vertices()[triangle[i] as usize]);
                    if [a, b, c].iter().any(|vertex| vertex.coverage < 1.) {
                        return 0.;
                    }
                    let (pa, pb, pc) = (a.position, b.position, c.position);
                    ((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])).abs()
                        / 2.
                })
                .sum();
            assert!((inside - 50.).abs() < 1e-3, "{fill:?}: area {inside}");
            // The crossing joins the two triangles' outlines.
            assert_eq!(fringe(&mesh).len(), 5, "{fill:?}");
        }
    }

    #[test]
    fn a_strip_measures_along_and_across() {
        let ribs: Vec<_> = (0..4)
            .map(|step| {
                let x = px(step as f32 * 10.);
                (point(x, px(0.)), point(x, px(4.)))
            })
            .collect();
        let mesh = Mesh::strip(&ribs);
        let inner: Vec<_> = mesh
            .vertices()
            .iter()
            .filter(|vertex| vertex.coverage == 1.)
            .collect();
        assert_eq!(inner.len(), 8);
        for vertex in &inner {
            assert_eq!(vertex.stroke[0], vertex.position[0]);
            assert_eq!(
                vertex.stroke[1],
                if vertex.position[1] == 0. { -1. } else { 1. }
            );
        }
        // Every vertex of a strip is on its outline.
        assert_eq!(fringe(&mesh).len(), 8);
        // Its fringe copies carry their stroke coordinates.
        for vertex in fringe(&mesh) {
            assert_eq!(vertex.stroke[0], vertex.position[0]);
        }
    }

    #[test]
    fn meshes_have_their_own_ids() {
        let first = Mesh::from_polygon(&square(), peniko::Fill::NonZero);
        let second = Mesh::from_polygon(&square(), peniko::Fill::NonZero);
        assert_ne!(first.id(), second.id());
        assert_eq!(
            first.byte_len(),
            first.vertices().len() * 32 + first.indices().len() * 4
        );
    }
}
