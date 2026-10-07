//! The states elements keep across frames, held in one map for as long as
//! their elements are drawn in consecutive frames.
//!
//! A state belongs to the subframe it was last used in: the innermost cached
//! view being drawn then, or none. A frame that reuses a cached view uses
//! none of the states inside it, yet keeps them, since its subframe is still
//! drawn. So rather than move each state used into the next frame, as a
//! frame of flat lists would, the end of each frame lets go of what stopped
//! being drawn, looking only at what could have:
//!
//! - the states used in the frame before, which this frame did not use and
//!   whose subframe it does not draw;
//! - the subframes drawn in the frame before, which this frame does not
//!   draw, with the states in them and in the subframes inside them.
//!
//! A frame that only reuses a cached view looks at none of its states.

use super::{ElementStateBox, GlobalElementId, SubframeId};
use collections::{FxHashMap, FxHashSet};
use std::{any::TypeId, mem};

pub(crate) type ElementStateKey = (GlobalElementId, TypeId);

struct Entry {
    state: ElementStateBox,
    /// The subframe the state was last used in.
    subframe: Option<SubframeId>,
    /// The frame the state was last used in.
    frame: usize,
}

#[derive(Default)]
pub(crate) struct ElementStates {
    states: FxHashMap<ElementStateKey, Entry>,
    /// The states last used in each subframe.
    members: FxHashMap<SubframeId, FxHashSet<ElementStateKey>>,
    /// The subframe each subframe was last drawn in, if any.
    parents: FxHashMap<SubframeId, Option<SubframeId>>,
    /// The subframes each subframe has had drawn in it.
    children: FxHashMap<SubframeId, Vec<SubframeId>>,
    /// The last frame each subframe was recorded or reused in, as itself
    /// rather than inside another.
    drawn: FxHashMap<SubframeId, usize>,
    /// The states used in the frame being drawn, and in the one before.
    used: Vec<ElementStateKey>,
    used_before: Vec<ElementStateKey>,
    /// The subframes drawn in the frame being drawn, and in the one before.
    drawn_now: Vec<SubframeId>,
    drawn_before: Vec<SubframeId>,
}

impl ElementStates {
    /// Takes the state at `key` out, to be used in `frame`, if it is still
    /// alive: used in `frame` or `last_frame`, or inside a subframe drawn in
    /// one of them.
    pub(crate) fn take(
        &mut self,
        key: &ElementStateKey,
        last_frame: usize,
        frame: usize,
    ) -> Option<ElementStateBox> {
        let entry = self.states.remove(key)?;
        if let Some(subframe) = entry.subframe
            && let Some(members) = self.members.get_mut(&subframe)
        {
            members.remove(key);
        }
        let alive = entry.frame == frame
            || entry.frame == last_frame
            || entry.subframe.is_some_and(|subframe| {
                let mut alive = FxHashMap::default();
                [last_frame, frame].into_iter().any(|drawn_in| {
                    alive.clear();
                    is_drawn(subframe, drawn_in, &self.parents, &self.drawn, &mut alive)
                })
            });
        alive.then_some(entry.state)
    }

    /// Puts `state` back at `key`, used in `frame`, in `subframe`.
    pub(crate) fn put(
        &mut self,
        key: ElementStateKey,
        state: ElementStateBox,
        subframe: Option<SubframeId>,
        frame: usize,
    ) {
        if let Some(subframe) = subframe {
            self.members
                .entry(subframe)
                .or_default()
                .insert(key.clone());
        }
        self.used.push(key.clone());
        self.states.insert(
            key,
            Entry {
                state,
                subframe,
                frame,
            },
        );
    }

    /// Notes that `subframe` is drawn in `frame`, inside `parent`: recorded
    /// there, or reused there whole.
    pub(crate) fn drawn(&mut self, subframe: SubframeId, parent: Option<SubframeId>, frame: usize) {
        self.parents.insert(subframe, parent);
        if let Some(parent) = parent {
            self.children.entry(parent).or_default().push(subframe);
        }
        self.drawn.insert(subframe, frame);
        self.drawn_now.push(subframe);
    }

    /// Lets go of the states that `frame`, just drawn, stopped drawing: those
    /// used in the frame before and not since, outside a subframe `frame`
    /// draws, and those in a subframe drawn before and not in `frame`.
    pub(crate) fn sweep(&mut self, frame: usize) {
        let mut alive: FxHashMap<SubframeId, bool> = FxHashMap::default();

        for key in mem::take(&mut self.used_before) {
            let Some(entry) = self.states.get(&key) else {
                continue;
            };
            let kept = entry.frame == frame
                || entry.subframe.is_some_and(|subframe| {
                    is_drawn(subframe, frame, &self.parents, &self.drawn, &mut alive)
                });
            if !kept {
                let subframe = entry.subframe;
                self.states.remove(&key);
                if let Some(members) = subframe.and_then(|subframe| self.members.get_mut(&subframe))
                {
                    members.remove(&key);
                }
            }
        }

        let mut gone = Vec::new();
        for subframe in mem::take(&mut self.drawn_before) {
            if !is_drawn(subframe, frame, &self.parents, &self.drawn, &mut alive) {
                gone.push(subframe);
            }
        }
        while let Some(subframe) = gone.pop() {
            // A subframe drawn in another since is still drawn.
            if self.drawn.get(&subframe) == Some(&frame) {
                continue;
            }
            for key in self.members.remove(&subframe).unwrap_or_default() {
                if self
                    .states
                    .get(&key)
                    .is_some_and(|entry| entry.subframe == Some(subframe))
                {
                    self.states.remove(&key);
                }
            }
            self.parents.remove(&subframe);
            self.drawn.remove(&subframe);
            gone.extend(self.children.remove(&subframe).unwrap_or_default());
        }

        self.used_before = mem::take(&mut self.used);
        self.drawn_before = mem::take(&mut self.drawn_now);
    }
}

/// Whether `subframe` is drawn in `frame`: drawn there itself, or inside a
/// subframe that is, which it was last drawn in. Memoized in `alive`.
fn is_drawn(
    subframe: SubframeId,
    frame: usize,
    parents: &FxHashMap<SubframeId, Option<SubframeId>>,
    drawn: &FxHashMap<SubframeId, usize>,
    alive: &mut FxHashMap<SubframeId, bool>,
) -> bool {
    if let Some(&known) = alive.get(&subframe) {
        return known;
    }
    let answer = drawn.get(&subframe) == Some(&frame)
        || parents
            .get(&subframe)
            .copied()
            .flatten()
            .is_some_and(|parent| is_drawn(parent, frame, parents, drawn, alive));
    alive.insert(subframe, answer);
    answer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ElementId;

    fn key(n: usize) -> ElementStateKey {
        (
            GlobalElementId(std::sync::Arc::from([ElementId::Integer(n as u64)])),
            TypeId::of::<u32>(),
        )
    }

    fn state() -> ElementStateBox {
        ElementStateBox {
            inner: Box::new(Some(0u32)),
            #[cfg(debug_assertions)]
            type_name: "u32",
        }
    }

    /// Uses the state at `n` in `frame`, inside `subframe`, as an element
    /// drawn there does.
    fn use_state(states: &mut ElementStates, n: usize, subframe: Option<SubframeId>, frame: usize) {
        let state = states.take(&key(n), frame - 1, frame).unwrap_or_else(state);
        states.put(key(n), state, subframe, frame);
    }

    fn held(states: &ElementStates, n: usize) -> bool {
        states.states.contains_key(&key(n))
    }

    #[test]
    fn states_inside_a_reused_subframe_are_kept_until_it_is_dropped() {
        let mut states = ElementStates::default();
        let (page, item) = (SubframeId::next(), SubframeId::next());
        // Frame 1 records a page with an item in it.
        states.drawn(page, None, 1);
        use_state(&mut states, 0, Some(page), 1);
        states.drawn(item, Some(page), 1);
        use_state(&mut states, 1, Some(item), 1);
        states.sweep(1);

        // Frames 2 and 3 reuse the page whole.
        for frame in 2..=3 {
            states.drawn(page, None, frame);
            states.sweep(frame);
            assert!(held(&states, 0) && held(&states, 1), "frame {frame}");
        }

        // Frame 4 draws neither.
        states.sweep(4);
        assert!(
            states.states.is_empty(),
            "the page and the item inside it are let go"
        );
        assert!(states.drawn.is_empty() && states.parents.is_empty());
    }

    #[test]
    fn a_state_whose_element_skipped_a_frame_is_dropped_at_its_end() {
        let mut states = ElementStates::default();
        use_state(&mut states, 0, None, 1);
        use_state(&mut states, 1, None, 1);
        states.sweep(1);

        use_state(&mut states, 1, None, 2);
        states.sweep(2);
        assert!(
            !held(&states, 0),
            "an element missing from frame 2 is let go"
        );
        assert!(held(&states, 1), "an element drawn again is kept");
    }

    #[test]
    fn a_view_rendered_again_keeps_the_states_it_uses_and_its_reused_children() {
        let mut states = ElementStates::default();
        let (outer, inner) = (SubframeId::next(), SubframeId::next());
        states.drawn(outer, None, 1);
        use_state(&mut states, 0, Some(outer), 1);
        use_state(&mut states, 1, Some(outer), 1);
        states.drawn(inner, Some(outer), 1);
        use_state(&mut states, 2, Some(inner), 1);
        states.sweep(1);

        // Frame 2 renders the outer view again, as a new subframe, using one
        // of its states and reusing the inner view whole.
        let outer_again = SubframeId::next();
        states.drawn(outer_again, None, 2);
        use_state(&mut states, 0, Some(outer_again), 2);
        states.drawn(inner, Some(outer_again), 2);
        states.sweep(2);

        assert!(held(&states, 0), "a state the view used again is kept");
        assert!(!held(&states, 1), "a state it stopped using is let go");
        assert!(held(&states, 2), "the reused inner view keeps its state");
    }

    #[test]
    fn a_stale_state_is_not_handed_back() {
        let mut states = ElementStates::default();
        states.put(key(0), state(), None, 1);
        assert!(states.take(&key(0), 2, 3).is_none());
    }
}
