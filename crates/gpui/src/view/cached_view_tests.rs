//! Cached views reused where they move to.
//!
//! Ported from huacnlee's zed-industries/zed#64218.

use super::*;
use crate::{
    AnyWindowHandle, InteractiveElement as _, ListAlignment, ListState, Modifiers, MouseButton,
    ParentElement as _, ScaledPixels, ScrollHandle, StatefulInteractiveElement as _, Styled as _,
    TestAppContext, VisualTestContext, canvas, div, hsla, list, point, px, red,
};
use std::{cell::Cell, rc::Rc};

/// A fixed-size card whose renders are counted, with a mouse listener so a
/// hitbox gets recorded (a hover style would refresh the whole window on
/// hover, hiding what the pointer checks below test) and solid backgrounds so
/// quads do.
struct Card {
    renders: Rc<Cell<usize>>,
}

impl Render for Card {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        div()
            .id("card")
            .size(px(100.))
            .bg(red())
            .on_mouse_down(MouseButton::Left, |_, _, _| {})
            .child(div().id("inner").size(px(20.)).bg(hsla(0.6, 1., 0.5, 1.)))
    }
}

fn card_style() -> StyleRefinement {
    let mut style = StyleRefinement::default();
    style.size.width = Some(px(100.).into());
    style.size.height = Some(px(100.).into());
    style
}

/// A clipping viewport, `height` tall, with the card placed `scroll` pixels
/// from its top, like an item in a scrolling list.
struct Viewport {
    card: Entity<Card>,
    scroll: Rc<Cell<f32>>,
    height: Rc<Cell<f32>>,
}

impl Render for Viewport {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .w(px(200.))
            .h(px(self.height.get()))
            .overflow_hidden()
            .child(
                div()
                    .mt(px(self.scroll.get()))
                    .child(self.card.clone().cached(card_style())),
            )
    }
}

/// A 400px-tall scrolling viewport over 1000px of content with the card 300px
/// down, scrolled by `scroll`. Layout snaps to device pixels, but a scroll
/// offset does not, so this is how a card lands on a fraction of a pixel.
struct ScrollingViewport {
    card: Entity<Card>,
    scroll: Rc<Cell<f32>>,
    scroll_handle: ScrollHandle,
}

impl Render for ScrollingViewport {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.scroll_handle
            .set_offset(point(px(0.), px(-self.scroll.get())));
        div()
            .id("viewport")
            .size(px(400.))
            .overflow_y_scroll()
            .track_scroll(&self.scroll_handle)
            .child(
                div().h(px(1000.)).child(
                    div()
                        .mt(px(300.))
                        .child(self.card.clone().cached(card_style())),
                ),
            )
    }
}

fn quads(cx: &mut TestAppContext, window: AnyWindowHandle) -> Vec<(f32, f32, f32, f32)> {
    cx.update_window(window, |_, window, _| {
        let scale = window.scale_factor();
        let unscale = |v: ScaledPixels| v.0 / scale;
        let mut quads: Vec<_> = window
            .rendered_frame
            .scene
            .quads
            .iter()
            .map(|quad| {
                let clipped = quad.bounds.intersect(&quad.content_mask.bounds);
                (
                    unscale(clipped.origin.x),
                    unscale(clipped.origin.y),
                    unscale(clipped.size.width),
                    unscale(clipped.size.height),
                )
            })
            .collect();
        quads.sort_by(|a, b| a.partial_cmp(b).unwrap());
        quads
    })
    .unwrap()
}

fn hitboxes(cx: &mut TestAppContext, window: AnyWindowHandle) -> Vec<(f32, f32)> {
    cx.update_window(window, |_, window, _| {
        let mut hitboxes: Vec<_> = window
            .rendered_frame
            .hitboxes
            .iter()
            .map(|hitbox| {
                (
                    f32::from(hitbox.bounds.origin.y),
                    f32::from(hitbox.size.height),
                )
            })
            .collect();
        hitboxes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        hitboxes
    })
    .unwrap()
}

/// A card of `items` columns of `depth` nested stateful divs.
struct DeepCard {
    items: usize,
    depth: usize,
    renders: Rc<Cell<usize>>,
}

impl Render for DeepCard {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        let depth = self.depth;
        div()
            .flex()
            .flex_wrap()
            .size_full()
            .children((0..self.items).map(move |item| {
                let mut element = div().id(("leaf", item as u64)).size(px(2.));
                for level in (0..depth).rev() {
                    element = div()
                        .id(("some-container-element", level as u64))
                        .child(element);
                }
                element.id(("item", item as u64)).size(px(4.))
            }))
    }
}

/// A list of `CARD_HEIGHT`-tall cached cards in an 800px viewport, rendering
/// only the cards that intersect it, scrolled by `scroll`.
struct ScrollingList {
    cards: Vec<Entity<DeepCard>>,
    scroll: Rc<Cell<f32>>,
}

const CARD_HEIGHT: f32 = 100.;
const VIEWPORT_HEIGHT: f32 = 800.;

impl Render for ScrollingList {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let scroll = self.scroll.get();
        div()
            .w(px(400.))
            .h(px(VIEWPORT_HEIGHT))
            .overflow_hidden()
            .children(self.cards.iter().enumerate().filter_map(|(ix, card)| {
                let top = ix as f32 * CARD_HEIGHT - scroll;
                (top + CARD_HEIGHT > 0. && top < VIEWPORT_HEIGHT).then(|| {
                    let mut style = StyleRefinement::default();
                    style.size.width = Some(px(400.).into());
                    style.size.height = Some(px(CARD_HEIGHT).into());
                    style.position = Some(crate::Position::Absolute);
                    style.inset.top = Some(px(top).into());
                    card.clone().cached(style)
                })
            }))
    }
}

/// Frame cost of scrolling a list of cached views by a few pixels per frame.
/// Before cached views were reused where they moved to, every card missed its
/// cache on every frame, since the key included where it was.
///
/// `cargo test -p gpui-ce --release --lib cached_view_tests::scrolling_frame_cost -- --ignored --nocapture`
#[crate::test]
#[ignore = "prints timings; run by hand"]
fn scrolling_frame_cost(cx: &mut TestAppContext) {
    let scroll = Rc::new(Cell::new(0.));
    let renders = Rc::new(Cell::new(0));
    let window = cx.add_window({
        let scroll = scroll.clone();
        let renders = renders.clone();
        move |_, cx| ScrollingList {
            cards: (0..100)
                .map(|_| {
                    cx.new(|_| DeepCard {
                        items: 40,
                        depth: 8,
                        renders: renders.clone(),
                    })
                })
                .collect(),
            scroll,
        }
    });
    let window = AnyWindowHandle::from(window);
    let mut frame = |cx: &mut TestAppContext| {
        scroll.set(scroll.get() + 3.);
        cx.update_window(window, |root, window, cx| {
            root.downcast::<ScrollingList>()
                .unwrap()
                .update(cx, |_, cx| cx.notify());
            let started = std::time::Instant::now();
            window.draw(cx).clear(cx);
            started.elapsed()
        })
        .unwrap()
    };
    for _ in 0..5 {
        frame(cx);
    }
    renders.set(0);
    let mut samples: Vec<_> = (0..60).map(|_| frame(cx)).collect();
    samples.sort();
    let median = samples[samples.len() / 2];
    eprintln!(
        "8 visible cards x (40 items x 8 deep) scrolled 3px per frame: median frame {:.2} ms, {} card renders over 60 frames",
        median.as_secs_f64() * 1e3,
        renders.get(),
    );
}

#[crate::test]
fn cached_view_is_reused_where_it_moves_to(cx: &mut TestAppContext) {
    let renders = Rc::new(Cell::new(0));
    let scroll = Rc::new(Cell::new(0.));
    let height = Rc::new(Cell::new(200.));
    let window = cx.add_window({
        let renders = renders.clone();
        let scroll = scroll.clone();
        let height = height.clone();
        move |_, cx| Viewport {
            card: cx.new(|_| Card { renders }),
            scroll,
            height,
        }
    });
    let window = AnyWindowHandle::from(window);
    // Opening the window drew it once.
    assert_eq!(renders.get(), 1);
    // The viewport re-renders (it moved the card); the card itself is only
    // re-rendered when it is notified.
    let draw = |cx: &mut TestAppContext| {
        cx.update_window(window, |root, window, cx| {
            root.downcast::<Viewport>()
                .unwrap()
                .update(cx, |_, cx| cx.notify());
            window.draw(cx).clear(cx)
        })
        .unwrap();
    };

    draw(cx);
    assert_eq!(renders.get(), 1);
    let at_top = quads(cx, window);
    assert_eq!(
        at_top,
        vec![(0., 0., 20., 20.), (0., 0., 100., 100.)],
        "the card and its inner square, at the top"
    );
    let hitboxes_at_top = hitboxes(cx, window);
    assert!(!hitboxes_at_top.is_empty());

    // Scrolling moves the card; the view is not dirty, so the previous
    // frame's records are reused, moved down.
    scroll.set(50.);
    draw(cx);
    assert_eq!(renders.get(), 1, "the card was not rendered again");
    assert_eq!(
        quads(cx, window),
        vec![(0., 50., 20., 20.), (0., 50., 100., 100.)],
        "the reused quads moved with the card"
    );
    assert_eq!(
        hitboxes(cx, window),
        hitboxes_at_top
            .iter()
            .map(|(y, h)| (y + 50., *h))
            .collect::<Vec<_>>(),
        "the reused hitboxes moved with the card"
    );

    // Partly out of the viewport: still reused, and clipped by the viewport
    // at the new position.
    scroll.set(150.);
    draw(cx);
    assert_eq!(renders.get(), 1);
    assert_eq!(
        quads(cx, window),
        vec![(0., 150., 20., 20.), (0., 150., 100., 50.)],
        "the reused quads are clipped by the viewport where it now cuts them"
    );

    // A recording made while clipped is missing what fell outside, so it is
    // not reused anywhere else: moving back renders the card again.
    scroll.set(0.);
    draw(cx);
    assert_eq!(renders.get(), 2, "a clipped recording is not moved");
    assert_eq!(quads(cx, window), at_top);

    // Notifying the entity always renders it again, wherever it is.
    cx.update_window(window, |root, _, cx| {
        let card = root.downcast::<Viewport>().unwrap().read(cx).card.clone();
        card.update(cx, |_, cx| cx.notify());
    })
    .unwrap();
    draw(cx);
    assert_eq!(renders.get(), 3);

    // The mask can change while the card stays put: a shorter viewport cuts
    // it at the same position. The records are reused in place, clipped by
    // the new mask.
    height.set(60.);
    draw(cx);
    assert_eq!(renders.get(), 3, "reused in place under a smaller mask");
    assert_eq!(
        quads(cx, window),
        vec![(0., 0., 20., 20.), (0., 0., 100., 60.)],
        "the reused quads are clipped by the shorter viewport"
    );
    let hitbox_mask_bottoms = cx
        .update_window(window, |_, window, _| {
            window
                .rendered_frame
                .hitboxes
                .iter()
                .map(|hitbox| f32::from(hitbox.content_mask.bounds.bottom()))
                .collect::<Vec<_>>()
        })
        .unwrap();
    assert!(
        hitbox_mask_bottoms.iter().all(|bottom| *bottom <= 60.),
        "the reused hitboxes are clipped by the shorter viewport: {hitbox_mask_bottoms:?}"
    );

    // Growing the viewport back does not restore what that recording lost:
    // it is clipped, so the card is rendered again.
    height.set(200.);
    draw(cx);
    assert_eq!(
        renders.get(),
        4,
        "a clipped recording is not reused under a larger mask"
    );

    // Closures the card registered during paint answer mouse events with the
    // coordinates it was rendered at, so the card under the pointer is
    // rendered again where it now is rather than moved.
    let mut visual_cx = VisualTestContext::from_window(window, cx);
    visual_cx.simulate_mouse_move(point(px(50.), px(50.)), None, Modifiers::default());
    let before = renders.get();
    scroll.set(50.);
    draw(cx);
    assert_eq!(
        renders.get(),
        before + 1,
        "the card under the pointer is rendered again"
    );

    // With the pointer elsewhere the card moves without being rendered, and
    // is rendered again once the pointer reaches it.
    visual_cx.simulate_mouse_move(point(px(150.), px(150.)), None, Modifiers::default());
    let before = renders.get();
    scroll.set(100.);
    draw(cx);
    assert_eq!(renders.get(), before, "moved without the pointer over it");
    visual_cx.simulate_mouse_move(point(px(50.), px(150.)), None, Modifiers::default());
    assert_eq!(
        renders.get(),
        before + 1,
        "a moved card is rendered again when the pointer reaches it"
    );
    draw(cx);
    assert_eq!(renders.get(), before + 1, "and is then reused in place");
}

#[crate::test]
fn cached_view_moved_by_a_fraction_of_a_pixel_is_snapped(cx: &mut TestAppContext) {
    let renders = Rc::new(Cell::new(0));
    let scroll = Rc::new(Cell::new(100.));
    let window = cx.add_window({
        let renders = renders.clone();
        let scroll = scroll.clone();
        move |_, cx| ScrollingViewport {
            card: cx.new(|_| Card { renders }),
            scroll,
            scroll_handle: ScrollHandle::new(),
        }
    });
    let window = AnyWindowHandle::from(window);
    let draw = |cx: &mut TestAppContext| {
        cx.update_window(window, |root, window, cx| {
            root.downcast::<ScrollingViewport>()
                .unwrap()
                .update(cx, |_, cx| cx.notify());
            window.draw(cx).clear(cx)
        })
        .unwrap();
    };
    assert_eq!(renders.get(), 1);
    assert_eq!(
        quads(cx, window),
        vec![(0., 200., 20., 20.), (0., 200., 100., 100.)]
    );

    // The test window has a scale factor of 2, so 0.3px up is 0.6 of a device
    // pixel: rounded to a whole one, the records land half a logical pixel
    // up, where glyph sprites stay on the pixel they were rasterized for.
    scroll.set(100.3);
    draw(cx);
    assert_eq!(renders.get(), 1, "moved without being rendered");
    assert_eq!(
        quads(cx, window),
        vec![(0., 199.5, 20., 20.), (0., 199.5, 100., 100.)],
        "the move was rounded to a whole device pixel"
    );

    // Another 0.3px up is 0.2 of a device pixel from where the records now
    // are: rounded away, without the earlier rounding being lost.
    scroll.set(100.6);
    draw(cx);
    assert_eq!(renders.get(), 1);
    assert_eq!(
        quads(cx, window),
        vec![(0., 199.5, 20., 20.), (0., 199.5, 100., 100.)],
        "too small a move from where the records are to reach the next device pixel"
    );

    // 0.7px more is 1.6 device pixels from the records: two of them.
    scroll.set(101.3);
    draw(cx);
    assert_eq!(renders.get(), 1);
    assert_eq!(
        quads(cx, window),
        vec![(0., 198.5, 20., 20.), (0., 198.5, 100., 100.)],
        "the rounding tracks the card's true position rather than accumulating"
    );
}

/// A 150px-tall `list` of the cached card over a 100px square that asks to be
/// scrolled into view when told to, below an optional 10px stateful div whose
/// hitbox shifts every record index in the frame by one.
struct RollbackList {
    card: Entity<Card>,
    state: ListState,
    request: Rc<Cell<bool>>,
    extra: Rc<Cell<bool>>,
}

impl Render for RollbackList {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let card = self.card.clone();
        let request = self.request.clone();
        // Inset from the pointer at the window's origin, which would keep the
        // card from being moved (see `ViewElementState::reuse_offset`).
        div()
            .flex()
            .flex_col()
            .pl(px(50.))
            .w(px(250.))
            .h(px(160.))
            .children(self.extra.get().then(|| {
                div()
                    .id("extra")
                    .size(px(10.))
                    .on_mouse_down(MouseButton::Left, |_, _, _| {})
            }))
            .child(
                list(self.state.clone(), move |ix, _, _| match ix {
                    0 => card.clone().cached(card_style()).into_any_element(),
                    _ => {
                        let request = request.clone();
                        canvas(
                            move |bounds, window, _| {
                                if request.replace(false) {
                                    window.request_autoscroll(bounds);
                                }
                            },
                            |_, _, _, _| {},
                        )
                        .size(px(100.))
                        .into_any_element()
                    }
                })
                .w(px(200.))
                .h(px(150.)),
            )
    }
}

/// A `list` prepaints its items, and if one of them asks to be scrolled into
/// view it rolls the prepaint back (`Window::transact`) and prepaints them
/// again, scrolled. A cached view reused in the first attempt recorded where
/// its records lie in *this* frame, and the rollback discarded them; reusing
/// that record in the second attempt would copy whatever the rendered frame
/// holds at those indices.
///
/// Ported from zed-industries/zed#64236.
#[crate::test]
fn cached_view_is_rendered_again_after_its_list_rolls_a_prepaint_back(cx: &mut TestAppContext) {
    let renders = Rc::new(Cell::new(0));
    let request = Rc::new(Cell::new(false));
    let extra = Rc::new(Cell::new(false));
    let state = ListState::new(2, ListAlignment::Top, px(0.));
    let window = cx.add_window({
        let (renders, request, extra) = (renders.clone(), request.clone(), extra.clone());
        move |_, cx| RollbackList {
            card: cx.new(|_| Card { renders }),
            state,
            request,
            extra,
        }
    });
    let window = AnyWindowHandle::from(window);
    let draw = |cx: &mut TestAppContext| {
        cx.update_window(window, |root, window, cx| {
            root.downcast::<RollbackList>()
                .unwrap()
                .update(cx, |_, cx| cx.notify());
            window.draw(cx).clear(cx)
        })
        .unwrap();
    };
    assert_eq!(renders.get(), 1);
    assert_eq!(
        hitboxes(cx, window),
        vec![(0., 100.), (0., 150.)],
        "the card's hitbox and the list's"
    );

    // The extra div moves the list down by 10px and puts a hitbox before
    // everything else in the frame; the square asks to be scrolled into
    // view, which scrolls the list by 50px. The card is reused where it moved
    // to in the first prepaint, and must be rendered again in the second:
    // the first prepaint's records are gone.
    extra.set(true);
    request.set(true);
    draw(cx);
    assert_eq!(renders.get(), 2, "the card was rendered again");
    assert_eq!(
        hitboxes(cx, window),
        vec![(-40., 100.), (0., 10.), (10., 150.)],
        "the card's hitbox, scrolled, the extra div's and the list's"
    );
    assert_eq!(
        quads(cx, window),
        vec![(50., 10., 100., 50.)],
        "the visible part of the card"
    );

    // With the records made in the second attempt, the card is reused again
    // from the next frame on.
    draw(cx);
    assert_eq!(renders.get(), 2);
    assert_eq!(quads(cx, window), vec![(50., 10., 100., 50.)]);
}
