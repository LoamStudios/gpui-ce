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
            .placed_hitboxes()
            .into_iter()
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

    // Rendered again while the viewport cuts it off: the card is recorded
    // whole all the same, and only drawn clipped.
    cx.update_window(window, |root, _, cx| {
        let card = root.downcast::<Viewport>().unwrap().read(cx).card.clone();
        card.update(cx, |_, cx| cx.notify());
    })
    .unwrap();
    draw(cx);
    assert_eq!(renders.get(), 2, "notifying the card renders it again");
    assert_eq!(
        quads(cx, window),
        vec![(0., 150., 20., 20.), (0., 150., 100., 50.)],
        "drawn clipped by the viewport"
    );

    // So it can be moved back into view without being rendered, whole.
    scroll.set(20.);
    draw(cx);
    assert_eq!(renders.get(), 2, "a recording made while clipped is moved");
    assert_eq!(
        quads(cx, window),
        vec![(0., 20., 20., 20.), (0., 20., 100., 100.)],
        "what the viewport cut off when it was recorded is there"
    );
    assert_eq!(
        hitboxes(cx, window),
        hitboxes_at_top
            .iter()
            .map(|(y, h)| (y + 20., *h))
            .collect::<Vec<_>>(),
    );

    // The mask can change while the card stays put: a shorter viewport cuts
    // it at the same position. The records are reused in place, clipped by
    // the new mask, hitboxes included.
    height.set(60.);
    draw(cx);
    assert_eq!(renders.get(), 2, "reused in place under a smaller mask");
    assert_eq!(
        quads(cx, window),
        vec![(0., 20., 20., 20.), (0., 20., 100., 40.)],
        "the reused quads are clipped by the shorter viewport"
    );
    let hitbox_mask_bottoms = cx
        .update_window(window, |_, window, _| {
            window
                .rendered_frame
                .placed_hitboxes()
                .into_iter()
                .map(|hitbox| f32::from(hitbox.content_mask.bounds.bottom()))
                .collect::<Vec<_>>()
        })
        .unwrap();
    assert!(
        hitbox_mask_bottoms.iter().all(|bottom| *bottom <= 60.),
        "the reused hitboxes are clipped by the shorter viewport: {hitbox_mask_bottoms:?}"
    );

    // Growing the viewport back shows the whole card again, still without
    // rendering it.
    height.set(200.);
    draw(cx);
    assert_eq!(renders.get(), 2, "reused under a larger mask");
    assert_eq!(
        quads(cx, window),
        vec![(0., 20., 20., 20.), (0., 20., 100., 100.)],
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

/// A canvas item: a bordered, rounded card with a few swatches and a label,
/// so that it paints a dozen or so primitives as a real one would.
struct CanvasItem {
    index: usize,
    renders: Rc<Cell<usize>>,
}

impl Render for CanvasItem {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        let hue = (self.index * 37 % 360) as f32 / 360.0;
        div()
            .id("item")
            .size_full()
            .p(px(2.))
            .flex()
            .flex_wrap()
            .gap(px(2.))
            .bg(hsla(hue, 0.55, 0.62, 1.))
            .border_1()
            .border_color(hsla(hue, 0.6, 0.32, 1.))
            .rounded(px(4.))
            .on_mouse_down(MouseButton::Left, |_, _, _| {})
            .text_size(px(8.))
            .child(format!("{}", self.index))
            .children((0..8).map(|swatch| {
                div()
                    .size(px(5.))
                    .bg(hsla((hue + swatch as f32 / 8.) % 1., 0.6, 0.5, 1.))
            }))
    }
}

const CANVAS_COLUMNS: usize = 40;
/// Rows that fit the test display's 1080px height.
const CANVAS_ROWS: usize = 26;
const CANVAS_PITCH: f32 = 38.;

/// A 40×26 grid of cached items over a full-window background, shifted right
/// by `pan`, which stays small enough that no item is clipped.
struct PannedCanvas {
    items: Vec<Entity<CanvasItem>>,
    pan: Rc<Cell<f32>>,
}

impl Render for PannedCanvas {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let pan = self.pan.get();
        div()
            .size(px(1600.))
            .relative()
            .overflow_hidden()
            .bg(hsla(0.12, 0.1, 0.96, 1.))
            .children(self.items.iter().enumerate().map(|(ix, item)| {
                let column = (ix % CANVAS_COLUMNS) as f32;
                let row = (ix / CANVAS_COLUMNS) as f32;
                let mut style = StyleRefinement::default();
                style.position = Some(crate::Position::Absolute);
                style.inset.left = Some(px(20. + column * CANVAS_PITCH + pan).into());
                style.inset.top = Some(px(20. + row * CANVAS_PITCH).into());
                style.size.width = Some(px(34.).into());
                style.size.height = Some(px(34.).into());
                item.clone().cached(style)
            }))
    }
}

/// Frame cost of panning a canvas of cached items back and forth: every item
/// is reused where it moves to, so a frame replays their records.
///
/// `cargo test -p gpui-ce --release --lib cached_view_tests::panning_canvas_frame_cost -- --ignored --nocapture`
#[crate::test]
#[ignore = "prints timings; run by hand"]
fn panning_canvas_frame_cost(cx: &mut TestAppContext) {
    let pan = Rc::new(Cell::new(0.));
    let renders = Rc::new(Cell::new(0));
    let window = cx.add_window({
        let (pan, renders) = (pan.clone(), renders.clone());
        move |_, cx| PannedCanvas {
            items: (0..CANVAS_COLUMNS * CANVAS_ROWS)
                .map(|index| {
                    let renders = renders.clone();
                    cx.new(|_| CanvasItem { index, renders })
                })
                .collect(),
            pan,
        }
    });
    let window = AnyWindowHandle::from(window);
    let mut frame = |cx: &mut TestAppContext| {
        pan.set((pan.get() + 3.) % 12.);
        cx.update_window(window, |root, window, cx| {
            root.downcast::<PannedCanvas>()
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
    let frames = std::env::var("FRAMES").map_or(100, |frames| frames.parse().unwrap());
    let mut samples: Vec<_> = (0..frames).map(|_| frame(cx)).collect();
    samples.sort();
    let operations = cx
        .update_window(window, |_, window, _| window.rendered_frame.scene.len())
        .unwrap();
    eprintln!(
        "{} cached canvas items panned: median frame {:.2} ms, {operations} paint operations, {} item renders over {frames} frames",
        CANVAS_COLUMNS * CANVAS_ROWS,
        samples[samples.len() / 2].as_secs_f64() * 1e3,
        renders.get(),
    );
}

/// A cached panel holding the cached card in a 60px-tall clipping well, 20px
/// down: the well cuts the card's bottom 60px off.
struct Panel {
    card: Entity<Card>,
}

impl Render for Panel {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size(px(100.)).child(
            div()
                .mt(px(20.))
                .h(px(60.))
                .overflow_hidden()
                .child(self.card.clone().cached(card_style())),
        )
    }
}

/// The panel, cached, in a 200px viewport scrolled by `scroll`, 50px in from
/// the left, away from the test pointer at the window's origin.
struct PanelViewport {
    panel: Entity<Panel>,
    scroll: Rc<Cell<f32>>,
}

impl Render for PanelViewport {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size(px(200.)).overflow_hidden().child(
            div()
                .mt(px(self.scroll.get()))
                .ml(px(50.))
                .child(self.panel.clone().cached(card_style())),
        )
    }
}

/// The outermost cached view is recorded whole, whatever clips it from
/// outside; what clips a cached view inside it, from inside it, stays in the
/// recording and moves with it.
#[crate::test]
fn cached_view_keeps_the_clips_inside_it_when_moved(cx: &mut TestAppContext) {
    let renders = Rc::new(Cell::new(0));
    let scroll = Rc::new(Cell::new(30.));
    let window = cx.add_window({
        let (renders, scroll) = (renders.clone(), scroll.clone());
        move |_, cx| PanelViewport {
            panel: cx.new(|cx| Panel {
                card: cx.new(|_| Card { renders }),
            }),
            scroll,
        }
    });
    let window = AnyWindowHandle::from(window);
    let draw = |cx: &mut TestAppContext| {
        cx.update_window(window, |root, window, cx| {
            root.downcast::<PanelViewport>()
                .unwrap()
                .update(cx, |_, cx| cx.notify());
            window.draw(cx).clear(cx)
        })
        .unwrap();
    };
    // The well shows the card's top 60px, 20px into the panel.
    assert_eq!(
        quads(cx, window),
        vec![(50., 50., 20., 20.), (50., 50., 100., 60.)]
    );

    // Most of the panel scrolled out of the viewport: the well is cut by
    // the viewport too.
    scroll.set(-60.);
    draw(cx);
    assert_eq!(
        quads(cx, window),
        vec![(50., 0., 100., 20.)],
        "only the bottom of the well's view of the card is in the viewport"
    );

    // Back in view, the panel's recording still has the well cutting the
    // card, and nothing the viewport cut.
    scroll.set(40.);
    draw(cx);
    assert_eq!(renders.get(), 1, "the card was never rendered again");
    assert_eq!(
        quads(cx, window),
        vec![(50., 60., 20., 20.), (50., 60., 100., 60.)],
        "the well still cuts the card, where the panel now is"
    );
}

/// A cached view inside one that was reused for a frame, then rendered
/// again, is drawn where it should be. It is rendered again too: a view that
/// renders renders the cached views inside it (`Window::refreshing`), since
/// their records index the frame they were made in, which may be older than
/// the one before. Reusing them needs every frame list kept in subframes
/// (decision 006 of workroom-canvas-gpui); then this counts one render.
#[crate::test]
fn cached_view_inside_a_reused_one_is_drawn_right_after_it_renders_again(cx: &mut TestAppContext) {
    let renders = Rc::new(Cell::new(0));
    let scroll = Rc::new(Cell::new(30.));
    let panel = Rc::new(std::cell::RefCell::new(None));
    let window = cx.add_window({
        let (renders, scroll, panel) = (renders.clone(), scroll.clone(), panel.clone());
        move |_, cx| {
            let entity = cx.new(|cx| Panel {
                card: cx.new(|_| Card { renders }),
            });
            *panel.borrow_mut() = Some(entity.clone());
            PanelViewport {
                panel: entity,
                scroll,
            }
        }
    });
    let window = AnyWindowHandle::from(window);
    let panel = panel.borrow().clone().unwrap();
    let draw = |cx: &mut TestAppContext, notify_panel: bool| {
        cx.update_window(window, |root, window, cx| {
            root.downcast::<PanelViewport>()
                .unwrap()
                .update(cx, |_, cx| cx.notify());
            if notify_panel {
                panel.update(cx, |_, cx| cx.notify());
            }
            window.draw(cx).clear(cx)
        })
        .unwrap();
    };

    // The panel moves: it is reused, with the card inside it untouched.
    scroll.set(40.);
    draw(cx, false);
    // The panel renders again back where the card was recorded, which the
    // panel's well clips, so the card is reused only where it was: from
    // records made two frames ago.
    scroll.set(30.);
    draw(cx, true);
    // And once more, with both reused.
    draw(cx, false);

    assert_eq!(renders.get(), 2, "the card renders again with the panel");
    assert_eq!(
        quads(cx, window),
        vec![(50., 50., 20., 20.), (50., 50., 100., 60.)],
        "the card is drawn where the panel is"
    );
    assert_eq!(
        hitboxes(cx, window).len(),
        1,
        "the card's hitbox is kept once"
    );
}
