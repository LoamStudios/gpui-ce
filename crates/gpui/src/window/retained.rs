//! Frame lists that keep each cached view's part as a subframe of its own,
//! which a later frame that reuses the view takes whole: the reuse costs the
//! same however much the view holds.
//!
//! A list is the root's entries, in the order they were recorded, with each
//! subframe nested where it was recorded. While a frame is built, the
//! subframes being recorded are open, innermost last, and entries go into the
//! innermost. Once the frame is drawn, the next frame takes the subframes it
//! reuses from it: those at its root, and those inside a view that is
//! rendered again, set aside when the view is.

use collections::FxHashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Identifies a subframe: a cached view's part of a frame, kept in its
/// element state so the next frame can find it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SubframeId(u64);

impl SubframeId {
    /// A new subframe's identity.
    pub(crate) fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// An entry of a retained list.
pub(crate) enum Entry<T, P> {
    Item(T),
    Subframe(Box<Subframe<T, P>>),
}

/// A subframe of a retained list: its entries, and `placement`, which places
/// them in the coordinates of the list it sits in.
pub(crate) struct Subframe<T, P> {
    pub(crate) id: SubframeId,
    pub(crate) placement: P,
    pub(crate) entries: Vec<Entry<T, P>>,
    /// How many items it holds, its subframes' included.
    pub(crate) len: usize,
}

/// Where a retained list is, for rolling back to: how many subframes are
/// open, and how many entries the innermost holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ListIndex {
    depth: usize,
    len: usize,
}

/// A frame list whose cached views' parts are subframes.
pub(crate) struct RetainedList<T, P> {
    root: Vec<Entry<T, P>>,
    /// The subframes being recorded, innermost last, and the entries recorded
    /// into each so far.
    open: Vec<(SubframeId, Vec<Entry<T, P>>)>,
    /// Subframes of this frame, once drawn, that the next frame may take:
    /// those at its root, and the children of views rendered again.
    available: FxHashMap<SubframeId, Box<Subframe<T, P>>>,
}

impl<T, P> Default for RetainedList<T, P> {
    fn default() -> Self {
        Self {
            root: Vec::new(),
            open: Vec::new(),
            available: FxHashMap::default(),
        }
    }
}

impl<T, P> RetainedList<T, P> {
    /// The root's entries.
    pub(crate) fn entries(&self) -> &[Entry<T, P>] {
        &self.root
    }

    fn current(&mut self) -> &mut Vec<Entry<T, P>> {
        match self.open.last_mut() {
            Some((_, entries)) => entries,
            None => &mut self.root,
        }
    }

    /// Records `item`, and returns it.
    pub(crate) fn push(&mut self, item: T) -> &mut T {
        let entries = self.current();
        entries.push(Entry::Item(item));
        match entries.last_mut() {
            Some(Entry::Item(item)) => item,
            _ => unreachable!("an item was just pushed"),
        }
    }

    /// Where the list is now.
    pub(crate) fn index(&self) -> ListIndex {
        ListIndex {
            depth: self.open.len(),
            len: match self.open.last() {
                Some((_, entries)) => entries.len(),
                None => self.root.len(),
            },
        }
    }

    /// Drops what was recorded since `index`, in the subframe that was
    /// innermost then, which must be innermost again.
    pub(crate) fn truncate(&mut self, index: ListIndex) {
        debug_assert_eq!(
            index.depth,
            self.open.len(),
            "rolled back across a subframe"
        );
        self.current().truncate(index.len);
    }

    /// Starts recording subframe `id`: entries go into it until
    /// [`Self::end`].
    pub(crate) fn begin(&mut self, id: SubframeId) {
        self.open.push((id, Vec::new()));
    }

    /// Ends the innermost subframe, placed in its parent by what `placement`
    /// makes of its entries.
    pub(crate) fn end(&mut self, placement: impl FnOnce(&[Entry<T, P>]) -> P) {
        let (id, entries) = self.open.pop().expect("ended a subframe that wasn't begun");
        let len = entries
            .iter()
            .map(|entry| match entry {
                Entry::Item(_) => 1,
                Entry::Subframe(subframe) => subframe.len,
            })
            .sum();
        self.current().push(Entry::Subframe(Box::new(Subframe {
            id,
            placement: placement(&entries),
            entries,
            len,
        })));
    }

    /// Makes the subframes at this frame's root available to the next frame.
    /// The root keeps its items, which nothing reuses.
    pub(crate) fn make_available(&mut self) {
        debug_assert!(self.open.is_empty());
        let root = std::mem::take(&mut self.root);
        self.make_children_available(root);
    }

    fn make_children_available(&mut self, entries: Vec<Entry<T, P>>) {
        for entry in entries {
            if let Entry::Subframe(subframe) = entry {
                self.available.insert(subframe.id, subframe);
            }
        }
    }

    /// Whether subframe `id` is available to the next frame.
    pub(crate) fn has(&self, id: SubframeId) -> bool {
        self.available.contains_key(&id)
    }

    /// Lets go of subframe `id`, for a view rendered again, and makes its
    /// children available, for those of them it reuses.
    pub(crate) fn open(&mut self, id: SubframeId) {
        if let Some(subframe) = self.available.remove(&id) {
            self.make_children_available(subframe.entries);
        }
    }

    /// Records subframe `id` of `previous`, a frame drawn before this one, as
    /// it was, placed by what `place` makes of its placement there. Returns
    /// whether `previous` had it.
    pub(crate) fn reuse(
        &mut self,
        previous: &mut Self,
        id: SubframeId,
        place: impl FnOnce(&P) -> P,
    ) -> bool {
        let Some(mut subframe) = previous.available.remove(&id) else {
            return false;
        };
        subframe.placement = place(&subframe.placement);
        self.current().push(Entry::Subframe(subframe));
        true
    }

    /// The first item, depth first, that `predicate` accepts.
    #[cfg(any(feature = "inspector", debug_assertions))]
    pub(crate) fn find(&self, mut predicate: impl FnMut(&T) -> bool) -> Option<&T> {
        fn find<'a, T, P>(
            entries: &'a [Entry<T, P>],
            predicate: &mut impl FnMut(&T) -> bool,
        ) -> Option<&'a T> {
            entries.iter().find_map(|entry| match entry {
                Entry::Item(item) => predicate(item).then_some(item),
                Entry::Subframe(subframe) => find(&subframe.entries, predicate),
            })
        }
        find(&self.root, &mut predicate)
    }

    pub(crate) fn clear(&mut self) {
        self.root.clear();
        self.open.clear();
        self.available.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items(entries: &[Entry<u32, i32>], offset: i32, out: &mut Vec<i32>) {
        for entry in entries {
            match entry {
                Entry::Item(item) => out.push(*item as i32 + offset),
                Entry::Subframe(subframe) => {
                    items(&subframe.entries, offset + subframe.placement, out)
                }
            }
        }
    }

    fn flattened(list: &RetainedList<u32, i32>) -> Vec<i32> {
        let mut out = Vec::new();
        items(list.entries(), 0, &mut out);
        out
    }

    #[test]
    fn a_reused_subframe_keeps_its_items_and_their_order() {
        let mut previous = RetainedList::<u32, i32>::default();
        previous.push(1);
        let view = SubframeId::next();
        previous.begin(view);
        previous.push(10);
        previous.push(11);
        previous.end(|_| 0);
        previous.push(2);
        previous.make_available();

        let mut next = RetainedList::default();
        next.push(1);
        assert!(next.reuse(&mut previous, view, |placement| placement + 100));
        next.push(3);
        assert_eq!(flattened(&next), [1, 110, 111, 3]);
        // A subframe is taken once.
        assert!(!next.reuse(&mut previous, view, |placement| *placement));
    }

    #[test]
    fn a_view_rendered_again_sets_its_children_aside_for_reuse() {
        let (outer, inner) = (SubframeId::next(), SubframeId::next());
        let mut previous = RetainedList::<u32, i32>::default();
        previous.begin(outer);
        previous.push(5);
        previous.begin(inner);
        previous.push(20);
        previous.end(|_| 0);
        previous.end(|_| 0);
        previous.make_available();

        let mut next = RetainedList::default();
        previous.open(outer);
        assert!(!previous.has(outer));
        let outer_again = SubframeId::next();
        next.begin(outer_again);
        next.push(6);
        assert!(next.reuse(&mut previous, inner, |_| 1));
        next.end(|_| 0);
        assert_eq!(flattened(&next), [6, 21]);
    }

    #[test]
    fn rolling_back_drops_what_was_recorded_since() {
        let mut list = RetainedList::<u32, i32>::default();
        list.push(1);
        let index = list.index();
        list.begin(SubframeId::next());
        list.push(2);
        list.end(|_| 0);
        list.push(3);
        list.truncate(index);
        assert_eq!(flattened(&list), [1]);
    }
}
