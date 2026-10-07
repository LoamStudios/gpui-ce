//! The states elements keep across frames, held in one map for as long as
//! their elements are drawn.
//!
//! A state belongs to the subframe it was last used in: the innermost cached
//! view being drawn then, or none. A frame that reuses a cached view uses
//! none of the states inside it, yet keeps them, since its subframe is still
//! drawn. So rather than move each state used into the next frame, as a
//! frame of flat lists would, the map is swept now and then: a state stays
//! while it was used in the last frame drawn or its subframe is still drawn.

use super::{ElementStateBox, GlobalElementId, SubframeId};
use collections::FxHashMap;
use std::any::TypeId;

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
    /// The subframe each subframe was last drawn in, if any.
    parents: FxHashMap<SubframeId, Option<SubframeId>>,
    /// The last frame each subframe was recorded or reused in, as itself
    /// rather than inside another.
    drawn: FxHashMap<SubframeId, usize>,
    /// How many states the last sweep kept.
    kept: usize,
}

impl ElementStates {
    /// Takes the state at `key` out, to be used in `frame`, if it is still
    /// alive: used in `frame` or `last_frame`, or inside a subframe drawn in
    /// one of them. A state whose element went undrawn for a frame is dropped
    /// here, though the lazy sweep has not yet let it go, so a state lasts
    /// only while its element is drawn in consecutive frames.
    pub(crate) fn take(
        &mut self,
        key: &ElementStateKey,
        last_frame: usize,
        frame: usize,
    ) -> Option<ElementStateBox> {
        let entry = self.states.remove(key)?;
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
        self.drawn.insert(subframe, frame);
    }

    /// Lets go of the states neither used in `frame` nor in a subframe drawn
    /// in it, once there are twice as many as the last sweep kept, so a sweep
    /// costs no more than the frames since the last one.
    pub(crate) fn sweep(&mut self, frame: usize) {
        if self.states.len() < 2 * self.kept + 1024 {
            return;
        }
        let mut alive: FxHashMap<SubframeId, bool> = FxHashMap::default();
        let (parents, drawn) = (&self.parents, &self.drawn);
        self.states.retain(|_, entry| {
            entry.frame == frame
                || entry
                    .subframe
                    .is_some_and(|subframe| is_drawn(subframe, frame, parents, drawn, &mut alive))
        });
        self.parents
            .retain(|subframe, _| alive.get(subframe).copied().unwrap_or(false));
        self.drawn
            .retain(|subframe, _| alive.get(subframe).copied().unwrap_or(false));
        self.kept = self.states.len();
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

    #[test]
    fn states_inside_a_reused_subframe_outlive_a_sweep() {
        let mut states = ElementStates::default();
        let (page, item, dropped) = (SubframeId::next(), SubframeId::next(), SubframeId::next());
        // Frame 1 records a page with an item in it, and another view.
        states.drawn(page, None, 1);
        states.drawn(item, Some(page), 1);
        states.drawn(dropped, None, 1);
        states.put(key(0), state(), Some(item), 1);
        states.put(key(1), state(), Some(dropped), 1);
        states.put(key(2), state(), None, 1);
        // Frame 2 reuses the page whole, and draws nothing else.
        states.drawn(page, None, 2);
        for n in 3..1100 {
            states.put(key(n), state(), None, 2);
        }
        states.sweep(2);

        assert!(
            states.take(&key(0), 2, 3).is_some(),
            "the item's state is kept"
        );
        assert!(
            states.take(&key(1), 2, 3).is_none(),
            "the dropped view's state goes"
        );
        assert!(
            states.take(&key(2), 2, 3).is_none(),
            "an unused state at the root goes"
        );
        assert!(
            states.take(&key(3), 2, 3).is_some(),
            "a state used this frame is kept"
        );
    }

    #[test]
    fn a_state_whose_element_skipped_a_frame_is_not_handed_back() {
        let mut states = ElementStates::default();
        let (page, popup) = (SubframeId::next(), SubframeId::next());
        // Frame 1 draws a popup at the root and a page with an item in it.
        states.drawn(page, None, 1);
        states.drawn(popup, None, 1);
        states.put(key(0), state(), None, 1);
        states.put(key(1), state(), Some(popup), 1);
        states.put(key(2), state(), Some(page), 1);
        // Frame 2 reuses the page whole and draws nothing else; no sweep runs.
        states.drawn(page, None, 2);

        // Frame 3 draws everything again.
        assert!(
            states.take(&key(0), 2, 3).is_none(),
            "a root element missing from frame 2 starts over"
        );
        assert!(
            states.take(&key(1), 2, 3).is_none(),
            "a view missing from frame 2 starts over"
        );
        assert!(
            states.take(&key(2), 2, 3).is_some(),
            "a view reused whole in frame 2 keeps its states"
        );
    }
}
