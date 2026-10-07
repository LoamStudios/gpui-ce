//! The meshes a renderer keeps on the GPU, for every backend.
//!
//! A [`gpui::Mesh`] is immutable and has an id unique for its life, so a
//! renderer uploads its vertices and indices the first time a frame draws it
//! and keeps them, by that id, while frames go on drawing it. A frame that
//! draws a mesh again uploads only its instance. A mesh not drawn for
//! [`MAX_IDLE_FRAMES`] frames is let go, and so are the least recently drawn
//! once the meshes kept take more than the budget, though never one the
//! frame being drawn uses.

use std::collections::HashMap;

use gpui::{Mesh, MeshId};

/// How many frames a mesh stays on the GPU without being drawn.
pub const MAX_IDLE_FRAMES: u64 = 600;
/// How many bytes of meshes a renderer keeps, at most, beyond those the
/// frame being drawn uses.
pub const DEFAULT_MESH_BUDGET: usize = 256 * 1024 * 1024;

/// What a renderer's meshes have cost: totals since it was made, and what it
/// keeps now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeshStats {
    /// Meshes uploaded.
    pub uploads: u64,
    /// Bytes of vertices and indices uploaded.
    pub uploaded_bytes: u64,
    /// Meshes let go.
    pub evictions: u64,
    /// Meshes kept on the GPU now.
    pub resident: usize,
    /// Bytes they take.
    pub resident_bytes: usize,
}

struct Entry<B> {
    buffers: B,
    bytes: usize,
    last_used: u64,
}

/// The meshes a renderer keeps, as its buffers `B`.
pub struct MeshCache<B> {
    entries: HashMap<MeshId, Entry<B>>,
    frame: u64,
    budget: usize,
    stats: MeshStats,
}

impl<B> Default for MeshCache<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B> MeshCache<B> {
    /// No meshes, and [`DEFAULT_MESH_BUDGET`].
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            frame: 0,
            budget: DEFAULT_MESH_BUDGET,
            stats: MeshStats::default(),
        }
    }

    /// Keeps at most `budget` bytes of meshes the frame being drawn does
    /// not use.
    pub fn set_budget(&mut self, budget: usize) {
        self.budget = budget;
    }

    /// Starts a frame: meshes drawn from here on are drawn in it.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }

    /// The buffers of `mesh`, uploaded by `upload` if they are not kept
    /// already: `None` if they could not be.
    pub fn buffers(&mut self, mesh: &Mesh, upload: impl FnOnce(&Mesh) -> Option<B>) -> Option<&B> {
        let frame = self.frame;
        if !self.entries.contains_key(&mesh.id()) {
            let buffers = upload(mesh)?;
            let bytes = mesh.byte_len();
            self.stats.uploads += 1;
            self.stats.uploaded_bytes += bytes as u64;
            self.stats.resident_bytes += bytes;
            self.entries.insert(
                mesh.id(),
                Entry {
                    buffers,
                    bytes,
                    last_used: frame,
                },
            );
        }
        let entry = self.entries.get_mut(&mesh.id())?;
        entry.last_used = frame;
        Some(&entry.buffers)
    }

    /// Ends a frame: lets go of meshes idle too long, then of the least
    /// recently drawn while over budget.
    pub fn end_frame(&mut self) {
        let frame = self.frame;
        let mut evicted = 0;
        let mut freed = 0;
        self.entries.retain(|_, entry| {
            let keep = frame.saturating_sub(entry.last_used) <= MAX_IDLE_FRAMES;
            if !keep {
                evicted += 1;
                freed += entry.bytes;
            }
            keep
        });
        self.stats.resident_bytes -= freed;
        if self.stats.resident_bytes > self.budget {
            let mut idle: Vec<(u64, MeshId, usize)> = self
                .entries
                .iter()
                .filter(|(_, entry)| entry.last_used < frame)
                .map(|(id, entry)| (entry.last_used, *id, entry.bytes))
                .collect();
            idle.sort_unstable();
            for (_, id, bytes) in idle {
                if self.stats.resident_bytes <= self.budget {
                    break;
                }
                self.entries.remove(&id);
                self.stats.resident_bytes -= bytes;
                evicted += 1;
            }
        }
        self.stats.evictions += evicted;
        self.stats.resident = self.entries.len();
    }

    /// Lets go of every mesh, as when the device they live on is lost.
    pub fn clear(&mut self) {
        self.stats.evictions += self.entries.len() as u64;
        self.entries.clear();
        self.stats.resident = 0;
        self.stats.resident_bytes = 0;
    }

    /// What the meshes have cost so far.
    pub fn stats(&self) -> MeshStats {
        MeshStats {
            resident: self.entries.len(),
            ..self.stats
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{point, px};

    fn triangle() -> Mesh {
        Mesh::from_polygon(
            &[(0., 0.), (10., 0.), (0., 10.)].map(|(x, y)| point(px(x), px(y))),
            gpui::peniko::Fill::NonZero,
        )
    }

    /// Uploads a mesh as a count of how many times it was uploaded.
    fn draw(cache: &mut MeshCache<u32>, meshes: &[&Mesh], uploads: &mut u32) {
        cache.begin_frame();
        for mesh in meshes {
            cache
                .buffers(mesh, |_| {
                    *uploads += 1;
                    Some(*uploads)
                })
                .unwrap();
        }
        cache.end_frame();
    }

    #[test]
    fn redrawing_an_unchanged_mesh_uploads_nothing() {
        let mesh = triangle();
        let mut cache = MeshCache::new();
        let mut uploads = 0;
        draw(&mut cache, &[&mesh], &mut uploads);
        let first = cache.stats();
        assert_eq!(first.uploads, 1);
        assert_eq!(first.uploaded_bytes, mesh.byte_len() as u64);
        for _ in 0..10 {
            draw(&mut cache, &[&mesh, &mesh], &mut uploads);
        }
        assert_eq!(uploads, 1);
        assert_eq!(cache.stats(), first);

        // A new mesh, even of the same shape, is uploaded once.
        let other = triangle();
        draw(&mut cache, &[&mesh, &other], &mut uploads);
        draw(&mut cache, &[&mesh, &other], &mut uploads);
        assert_eq!(uploads, 2);
        assert_eq!(cache.stats().resident, 2);
    }

    #[test]
    fn idle_meshes_are_let_go() {
        let (kept, idle) = (triangle(), triangle());
        let mut cache = MeshCache::new();
        let mut uploads = 0;
        draw(&mut cache, &[&kept, &idle], &mut uploads);
        for _ in 0..MAX_IDLE_FRAMES + 1 {
            draw(&mut cache, &[&kept], &mut uploads);
        }
        let stats = cache.stats();
        assert_eq!((stats.resident, stats.evictions), (1, 1));
        assert_eq!(stats.resident_bytes, kept.byte_len());
        // Drawn again, it is uploaded again.
        draw(&mut cache, &[&kept, &idle], &mut uploads);
        assert_eq!(uploads, 3);
    }

    #[test]
    fn over_budget_the_least_recently_drawn_go_first() {
        let meshes: Vec<Mesh> = (0..4).map(|_| triangle()).collect();
        let mut cache = MeshCache::new();
        cache.set_budget(2 * meshes[0].byte_len());
        let mut uploads = 0;
        for mesh in &meshes {
            draw(&mut cache, &[mesh], &mut uploads);
        }
        // The frame's own mesh is kept whatever the budget, and the most
        // recent idle one fits beside it.
        assert_eq!(cache.stats().resident, 2);
        draw(&mut cache, &[&meshes[2], &meshes[3]], &mut uploads);
        assert_eq!(uploads, 4);
        // Every mesh a frame draws is kept for it.
        draw(&mut cache, &meshes.iter().collect::<Vec<_>>(), &mut uploads);
        assert_eq!(cache.stats().resident, 4);
    }
}
