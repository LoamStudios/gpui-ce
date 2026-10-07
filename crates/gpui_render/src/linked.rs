//! Which shader programs a renderer's paint pipelines have linked in, and
//! linking them again when a scene needs others, for every backend.
//!
//! A scene whose paints run no programs draws with the standard pipelines.
//! One that does draws with pipelines whose `program_color` runs the
//! programs linked so far, and draws its paint's fallback colour for any
//! other. When a scene runs programs they lack, the renderer links a new
//! set, on a thread of its own, and draws with the pipelines it has, which
//! show those paints' fallback colours, until the new ones are ready.
//!
//! The linked set holds at most [`DEFAULT_PROGRAM_CAP`] programs. A new set
//! is the programs the scene runs, then those used most recently, by frame,
//! up to the cap; programs left out are forgotten until a scene runs them
//! again, when they draw their fallback colour until they are linked once
//! more. A scene whose programs are all linked, or that runs more than the
//! cap and has the cap's worth linked, links nothing, so a renderer links
//! again only when what it draws changes.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

use gpui::{Scene, shader::Program};

/// How many programs a renderer links into its pipelines at most.
pub const DEFAULT_PROGRAM_CAP: usize = 64;

/// What preparing a frame's programs found.
pub struct FramePrograms<P> {
    /// The pipelines to draw paints with: `None` for the standard ones.
    pub pipelines: Option<Arc<P>>,
    /// Whether the scene runs programs those pipelines lack, which are being
    /// linked: drawn with their fallback colours, the frame should be drawn
    /// again once they are ready.
    pub pending: bool,
}

struct Seen {
    program: Program,
    /// The frame that last ran it.
    last_used: u64,
}

struct PendingLink<P> {
    ids: BTreeSet<u32>,
    result: mpsc::Receiver<Result<(P, Duration), String>>,
}

/// The programs a renderer has linked into its pipelines `P`, and is
/// linking.
pub struct LinkedPrograms<P> {
    /// The programs that may be linked: those linked, and those run since.
    seen: BTreeMap<u32, Seen>,
    /// The programs `pipelines` run.
    linked: BTreeSet<u32>,
    pipelines: Option<Arc<P>>,
    pending: Option<PendingLink<P>>,
    /// Programs that failed to link, which are not tried again.
    failed: BTreeSet<u32>,
    /// Whether a frame waits for the programs it needs to be linked, rather
    /// than drawing their fallback colours meanwhile.
    synchronous: bool,
    cap: usize,
    frame: u64,
    /// How long the last link took.
    last_link_time: Option<Duration>,
    links: u64,
}

impl<P: Send + Sync + 'static> Default for LinkedPrograms<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Send + Sync + 'static> LinkedPrograms<P> {
    /// No programs linked. Frames wait for their programs with
    /// `GPUI_LINK_PROGRAMS_SYNCHRONOUSLY` set, and at most
    /// `GPUI_LINKED_PROGRAMS_CAP` programs are linked if it is set.
    pub fn new() -> Self {
        let cap = std::env::var("GPUI_LINKED_PROGRAMS_CAP")
            .ok()
            .and_then(|cap| cap.parse().ok())
            .filter(|&cap| cap > 0)
            .unwrap_or(DEFAULT_PROGRAM_CAP);
        Self {
            seen: BTreeMap::new(),
            linked: BTreeSet::new(),
            pipelines: None,
            pending: None,
            failed: BTreeSet::new(),
            // Without threads, a link runs where it is asked for.
            synchronous: cfg!(target_family = "wasm")
                || std::env::var_os("GPUI_LINK_PROGRAMS_SYNCHRONOUSLY").is_some(),
            cap,
            frame: 0,
            last_link_time: None,
            links: 0,
        }
    }

    /// Whether frames wait for the programs they run to be linked.
    pub fn set_synchronous(&mut self, synchronous: bool) {
        self.synchronous = synchronous || cfg!(target_family = "wasm");
    }

    /// Links at most `cap` programs, at least one.
    pub fn set_cap(&mut self, cap: usize) {
        self.cap = cap.max(1);
    }

    /// How long the last link took.
    pub fn last_link_time(&self) -> Option<Duration> {
        self.last_link_time
    }

    /// How many links have finished, successfully or not.
    pub fn link_count(&self) -> u64 {
        self.links
    }

    /// The ids of the programs the current pipelines run.
    pub fn linked(&self) -> &BTreeSet<u32> {
        &self.linked
    }

    /// The pipelines to draw `scene` with, starting to link the programs it
    /// runs that they lack, with a linker from `linker`, and in synchronous
    /// mode waiting for them. A linker builds `P` with the programs it is
    /// given linked in, on another thread unless frames are synchronous.
    pub fn prepare<F>(&mut self, scene: &Scene, linker: impl Fn() -> F) -> FramePrograms<P>
    where
        F: FnOnce(Vec<Program>) -> Result<P, String> + Send + 'static,
    {
        self.frame += 1;
        let mut needed = BTreeSet::new();
        self.collect(scene, &mut needed);
        if needed.is_empty() {
            return FramePrograms {
                pipelines: None,
                pending: false,
            };
        }
        self.poll(false);
        // A link in progress finishes before the next starts, so that
        // programs arriving every frame still get linked.
        if self.pending.is_none()
            && let Some(set) = self.set_to_link(&needed)
        {
            self.start_link(set, linker());
        }
        while self.synchronous && self.pending.is_some() {
            self.poll(true);
            if let Some(set) = self.set_to_link(&needed) {
                self.start_link(set, linker());
            }
        }
        FramePrograms {
            pipelines: self.pipelines.clone(),
            pending: self.pending.is_some() && !self.missing(&needed).is_empty(),
        }
    }

    /// Those of `needed` the linked pipelines lack and that have not failed.
    fn missing(&self, needed: &BTreeSet<u32>) -> BTreeSet<u32> {
        needed
            .iter()
            .filter(|id| !self.linked.contains(id) && !self.failed.contains(id))
            .copied()
            .collect()
    }

    /// The programs to link for a scene that runs `needed`, if they are not
    /// the ones linked and would link some it lacks: the cap's worth of
    /// `needed`, by id, then of the others seen, most recently used first.
    fn set_to_link(&self, needed: &BTreeSet<u32>) -> Option<BTreeSet<u32>> {
        if self.missing(needed).is_empty() {
            return None;
        }
        let mut others: Vec<(&u32, &Seen)> = self
            .seen
            .iter()
            .filter(|(id, _)| !needed.contains(id))
            .collect();
        others.sort_by(|(a_id, a), (b_id, b)| b.last_used.cmp(&a.last_used).then(a_id.cmp(b_id)));
        let set: BTreeSet<u32> = needed
            .iter()
            .filter(|id| self.seen.contains_key(id))
            .chain(others.into_iter().map(|(id, _)| id))
            .take(self.cap)
            .copied()
            .collect();
        (set != self.linked && !self.missing(&set).is_empty()).then_some(set)
    }

    /// Adds the programs `scene` and the chunks it draws run to those seen,
    /// as used this frame, and their ids, but for those that failed, to
    /// `needed`.
    fn collect(&mut self, scene: &Scene, needed: &mut BTreeSet<u32>) {
        for program in scene.programs() {
            let id = program.id();
            if self.failed.contains(&id) {
                continue;
            }
            self.seen
                .entry(id)
                .or_insert_with(|| Seen {
                    program: program.clone(),
                    last_used: 0,
                })
                .last_used = self.frame;
            needed.insert(id);
        }
        for placed in &scene.chunks {
            self.collect(&placed.chunk.scene, needed);
        }
    }

    /// Links the programs `ids` with `link`, on a thread of its own unless
    /// frames are synchronous, and forgets the others seen.
    fn start_link<F>(&mut self, ids: BTreeSet<u32>, link: F)
    where
        F: FnOnce(Vec<Program>) -> Result<P, String> + Send + 'static,
    {
        self.seen.retain(|id, _| ids.contains(id));
        let programs: Vec<Program> = self
            .seen
            .values()
            .map(|seen| seen.program.clone())
            .collect();
        let (sender, result) = mpsc::channel();
        let run = move || {
            let start = Instant::now();
            let pipelines = link(programs);
            sender
                .send(pipelines.map(|pipelines| (pipelines, start.elapsed())))
                .ok();
        };
        if self.synchronous {
            run();
        } else if let Err(error) = std::thread::Builder::new()
            .name("gpui-link-programs".into())
            .spawn(run)
        {
            log::error!("failed to start linking shader programs: {error}");
            return;
        }
        self.pending = Some(PendingLink { ids, result });
    }

    /// Takes the result of the link in progress, if it is done, or, if
    /// `wait`, once it is.
    fn poll(&mut self, wait: bool) {
        let Some(pending) = &self.pending else {
            return;
        };
        let result = if wait {
            pending.result.recv().map_err(|_| ())
        } else {
            match pending.result.try_recv() {
                Ok(result) => Ok(result),
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => Err(()),
            }
        };
        let pending = self.pending.take().expect("a link is in progress");
        self.links += 1;
        match result {
            Ok(Ok((pipelines, elapsed))) => {
                log::debug!(
                    "linked {} shader programs in {:.1} ms",
                    pending.ids.len(),
                    elapsed.as_secs_f64() * 1000.
                );
                self.last_link_time = Some(elapsed);
                self.pipelines = Some(Arc::new(pipelines));
                self.linked = pending.ids;
            }
            Ok(Err(error)) => {
                log::error!("failed to link shader programs: {error}");
                self.fail(&pending.ids);
            }
            Err(()) => {
                log::error!("linking shader programs stopped without a result");
                self.fail(&pending.ids);
            }
        }
    }

    /// Gives up on those of `ids` not already linked: their paints draw
    /// their fallback colours.
    fn fail(&mut self, ids: &BTreeSet<u32>) {
        for id in ids.difference(&self.linked) {
            self.failed.insert(*id);
            self.seen.remove(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{
        TransformationMatrix,
        shader::{self, CompiledPaint},
    };

    /// A paint of a program of its own for each `level`: a grey of
    /// `level`, multiplied by one `level` times.
    fn program(level: u8) -> CompiledPaint {
        let mut value = shader::Pixel.uv().x() * 0.0 + level as f32 / 255.;
        for _ in 0..level {
            value = value * 1.0;
        }
        shader::rgba(&value, &value, &value, 1.0).compile().unwrap()
    }

    fn scene(programs: &[&CompiledPaint]) -> Scene {
        let mut scene = Scene::default();
        for program in programs {
            scene.push_program(
                program,
                [1., 1.],
                TransformationMatrix::unit(),
                [0., 0., 0., 1.],
            );
        }
        scene.finish();
        scene
    }

    /// Pipelines that record the programs linked into them.
    type Ids = BTreeSet<u32>;

    fn linker() -> impl FnOnce(Vec<Program>) -> Result<Ids, String> + Send + 'static {
        |programs: Vec<Program>| Ok(programs.iter().map(Program::id).collect())
    }

    fn ids(programs: &[&CompiledPaint]) -> Ids {
        programs.iter().map(|paint| paint.program.id()).collect()
    }

    fn drawn_with(frame: &FramePrograms<Ids>) -> Ids {
        frame.pipelines.as_deref().cloned().unwrap_or_default()
    }

    #[test]
    fn a_scene_without_programs_keeps_the_standard_pipelines() {
        let mut linked = LinkedPrograms::<Ids>::new();
        linked.set_synchronous(true);
        let frame = linked.prepare(&scene(&[]), linker);
        assert!(frame.pipelines.is_none() && !frame.pending);
        assert_eq!(linked.link_count(), 0);
    }

    #[test]
    fn programs_link_once_while_the_scene_runs_the_same_ones() {
        let (a, b) = (program(1), program(2));
        let mut linked = LinkedPrograms::<Ids>::new();
        linked.set_synchronous(true);
        for _ in 0..5 {
            let frame = linked.prepare(&scene(&[&a, &b]), linker);
            assert_eq!(drawn_with(&frame), ids(&[&a, &b]));
            assert!(!frame.pending);
        }
        // Fewer of them: still linked.
        let frame = linked.prepare(&scene(&[&b]), linker);
        assert_eq!(drawn_with(&frame), ids(&[&a, &b]));
        assert_eq!(linked.link_count(), 1);
    }

    #[test]
    fn the_least_recently_used_program_is_evicted_at_the_cap() {
        let programs: Vec<CompiledPaint> = (0..6).map(program).collect();
        let mut linked = LinkedPrograms::<Ids>::new();
        linked.set_synchronous(true);
        linked.set_cap(3);
        for program in &programs[..3] {
            linked.prepare(&scene(&[program]), linker);
        }
        assert_eq!(
            linked.linked(),
            &ids(&[&programs[0], &programs[1], &programs[2]])
        );
        assert_eq!(linked.link_count(), 3);
        // The fourth evicts the first, the least recently used.
        let frame = linked.prepare(&scene(&[&programs[3]]), linker);
        assert_eq!(
            drawn_with(&frame),
            ids(&[&programs[1], &programs[2], &programs[3]])
        );
        // Using the second keeps it; the fifth evicts the third.
        linked.prepare(&scene(&[&programs[1]]), linker);
        linked.prepare(&scene(&[&programs[4]]), linker);
        assert_eq!(
            linked.linked(),
            &ids(&[&programs[1], &programs[3], &programs[4]])
        );
        assert_eq!(linked.link_count(), 5);
        // Cycling through all six, every frame draws its program.
        for round in 0..3 {
            for program in &programs {
                let frame = linked.prepare(&scene(&[program]), linker);
                assert!(
                    drawn_with(&frame).contains(&program.program.id()),
                    "round {round}"
                );
                assert!(drawn_with(&frame).len() <= 3);
            }
        }
    }

    #[test]
    fn a_scene_running_more_than_the_cap_links_the_cap_once() {
        let programs: Vec<CompiledPaint> = (0..5).map(program).collect();
        let all: Vec<&CompiledPaint> = programs.iter().collect();
        let mut linked = LinkedPrograms::<Ids>::new();
        linked.set_synchronous(true);
        linked.set_cap(3);
        for _ in 0..4 {
            let frame = linked.prepare(&scene(&all), linker);
            assert_eq!(drawn_with(&frame).len(), 3);
            assert!(!frame.pending, "the rest draw their fallback colour");
        }
        assert_eq!(linked.link_count(), 1);
    }

    #[test]
    fn an_evicted_program_draws_its_fallback_then_is_linked_again() {
        let (a, b) = (program(3), program(4));
        let mut linked = LinkedPrograms::<Ids>::new();
        linked.set_cap(1);
        let wait = |linked: &mut LinkedPrograms<Ids>, scene: &Scene| loop {
            let frame = linked.prepare(scene, linker);
            if !frame.pending {
                return frame;
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        let frame = wait(&mut linked, &scene(&[&a]));
        assert_eq!(drawn_with(&frame), ids(&[&a]));
        let frame = wait(&mut linked, &scene(&[&b]));
        assert_eq!(drawn_with(&frame), ids(&[&b]));
        // `a` was evicted: its first frame draws with the pipelines that
        // lack it, so its fallback, and a later one runs it.
        let frame = linked.prepare(&scene(&[&a]), linker);
        assert!(frame.pending);
        assert_eq!(drawn_with(&frame), ids(&[&b]));
        let frame = wait(&mut linked, &scene(&[&a]));
        assert_eq!(drawn_with(&frame), ids(&[&a]));
        assert_eq!(linked.link_count(), 3);
    }

    #[test]
    fn a_program_that_fails_to_link_draws_its_fallback_without_relinking() {
        let a = program(5);
        let mut linked = LinkedPrograms::<Ids>::new();
        linked.set_synchronous(true);
        for _ in 0..3 {
            let frame = linked.prepare(&scene(&[&a]), || {
                |_: Vec<Program>| Err::<Ids, _>("no".to_string())
            });
            assert!(frame.pipelines.is_none() && !frame.pending);
        }
        assert_eq!(linked.link_count(), 1);
    }
}
