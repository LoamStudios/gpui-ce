use super::*;
use crate::{
    AnyWindowHandle, InteractiveElement as _, IntoElement, Modifiers, MouseButton,
    ParentElement as _, Render, Styled as _, TestAppContext, VisualTestContext, canvas, div, hsla,
    red,
};
use std::{cell::Cell, f32::consts::FRAC_PI_2, f32::consts::FRAC_PI_4, rc::Rc};

/// A 400px stage with one 100×50 red element at (100, 100), transformed by
/// `transform` about its center.
struct Stage {
    transform: Rc<Cell<kurbo::Affine>>,
}

impl Render for Stage {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size(px(400.)).relative().child(
            div()
                .absolute()
                .left(px(100.))
                .top(px(100.))
                .w(px(100.))
                .h(px(50.))
                .bg(red())
                .rounded(px(5.))
                .transform(self.transform.get()),
        )
    }
}

fn stage(cx: &mut TestAppContext, transform: kurbo::Affine) -> AnyWindowHandle {
    let transform = Rc::new(Cell::new(transform));
    cx.add_window(move |_, _| Stage { transform }).into()
}

fn quads(cx: &mut TestAppContext, window: AnyWindowHandle) -> Vec<Quad> {
    cx.update_window(window, |_, window, _| {
        window.rendered_frame.scene.quads.clone()
    })
    .unwrap()
}

fn transform_entry(cx: &mut TestAppContext, window: AnyWindowHandle, index: u32) -> SceneTransform {
    cx.update_window(window, |_, window, _| {
        window.rendered_frame.scene.transforms()[index as usize]
    })
    .unwrap()
}

fn close(actual: Point<Pixels>, expected: Point<Pixels>) -> bool {
    (actual.x - expected.x).abs() < px(0.01) && (actual.y - expected.y).abs() < px(0.01)
}

#[crate::test]
fn a_uniform_scale_is_folded_into_window_coordinates(cx: &mut TestAppContext) {
    let window = stage(cx, kurbo::Affine::scale(2.));
    let quads = quads(cx, window);
    assert_eq!(quads.len(), 1);
    let quad = quads[0];
    // About the center (150, 125), the element covers (50, 75)-(250, 175) in
    // the window: at the test window's scale factor of 2, device pixels
    // (100, 150) by (400, 200). It needs no scene transform.
    assert_eq!(
        quad.transform, 0,
        "a uniform scale is not placed on the GPU"
    );
    assert_eq!(
        quad.bounds,
        Bounds::new(
            point(ScaledPixels(100.), ScaledPixels(150.)),
            size(ScaledPixels(400.), ScaledPixels(200.))
        )
    );
    assert_eq!(
        quad.corner_radii.top_left,
        ScaledPixels(20.),
        "corner radii scale with the element"
    );
}

#[crate::test]
fn a_rotation_places_the_element_by_a_scene_transform(cx: &mut TestAppContext) {
    let window = stage(cx, kurbo::Affine::rotate(f64::from(FRAC_PI_2)));
    let quads = quads(cx, window);
    assert_eq!(quads.len(), 1);
    let quad = quads[0];
    assert_ne!(quad.transform, 0, "a rotation is placed on the GPU");
    // The quad keeps its own shape, in device pixels...
    assert_eq!(
        quad.bounds,
        Bounds::new(
            point(ScaledPixels(200.), ScaledPixels(200.)),
            size(ScaledPixels(200.), ScaledPixels(100.))
        )
    );
    // ...and its transform turns it a quarter about its center (150, 125):
    // the top-left corner (100, 100) lands at (175, 75).
    let entry = transform_entry(cx, window, quad.transform);
    let top_left = entry.transformation.apply(point(px(200.), px(200.)));
    assert!(
        close(top_left, point(px(350.), px(150.))),
        "top-left corner lands at {top_left:?}"
    );
    let back = entry.inverse.apply(top_left);
    assert!(
        close(back, point(px(200.), px(200.))),
        "the inverse undoes it"
    );
}

/// A rotated, clipping 100×50 frame at (100, 100) holding a larger child.
struct ClippingFrame;

impl Render for ClippingFrame {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size(px(400.)).relative().child(
            div()
                .absolute()
                .left(px(100.))
                .top(px(100.))
                .w(px(100.))
                .h(px(50.))
                .overflow_hidden()
                .rotate(crate::radians(FRAC_PI_4))
                .child(div().size(px(200.)).bg(red())),
        )
    }
}

#[crate::test]
fn a_clip_under_a_rotation_goes_into_the_clip_table(cx: &mut TestAppContext) {
    let window: AnyWindowHandle = cx.add_window(|_, _| ClippingFrame).into();
    let (quad, clips) = cx
        .update_window(window, |_, window, _| {
            let scene = &window.rendered_frame.scene;
            (scene.quads[0], scene.clips().to_vec())
        })
        .unwrap();
    assert_ne!(quad.clip, 0, "the child is clipped by its rotated frame");
    let clip = clips[quad.clip as usize];
    assert_eq!(
        clip.bounds,
        Bounds::new(
            point(ScaledPixels(200.), ScaledPixels(200.)),
            size(ScaledPixels(200.), ScaledPixels(100.))
        ),
        "the frame's bounds, in its own space"
    );
    assert_eq!(
        clip.transform, quad.transform,
        "in the space the child is in"
    );
    assert_eq!(clip.parent, 0);
}

/// A 100×100 square at (100, 100), rotated a quarter turn's half, that
/// records where it hears mouse downs and the mouse position it paints with.
struct Diamond {
    downs: Rc<Cell<Option<Point<Pixels>>>>,
    painted_mouse: Rc<Cell<Option<Point<Pixels>>>>,
}

impl Render for Diamond {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let downs = self.downs.clone();
        let painted_mouse = self.painted_mouse.clone();
        div().size(px(400.)).relative().child(
            div()
                .id("diamond")
                .absolute()
                .left(px(100.))
                .top(px(100.))
                .size(px(100.))
                .bg(hsla(0.6, 1., 0.5, 1.))
                .rotate(crate::radians(FRAC_PI_4))
                .on_mouse_down(MouseButton::Left, move |event, _, _| {
                    downs.set(Some(event.position))
                })
                .child(
                    canvas(
                        |_, _, _| {},
                        move |_, _, window, _| painted_mouse.set(Some(window.mouse_position())),
                    )
                    .size_full(),
                ),
        )
    }
}

#[crate::test]
fn hit_testing_and_mouse_events_follow_the_transform(cx: &mut TestAppContext) {
    let downs = Rc::new(Cell::new(None));
    let painted_mouse = Rc::new(Cell::new(None));
    let window: AnyWindowHandle = cx
        .add_window({
            let (downs, painted_mouse) = (downs.clone(), painted_mouse.clone());
            move |_, _| Diamond {
                downs,
                painted_mouse,
            }
        })
        .into();
    let mut cx = VisualTestContext::from_window(window, cx);

    // Above the square's unrotated bounds, but inside the diamond: a hit,
    // heard in the square's own coordinates, 65px from its center along the
    // turned axis.
    cx.simulate_click(point(px(150.), px(85.)), Modifiers::default());
    let heard = downs.take().expect("the diamond heard the click");
    let along = 65. * std::f32::consts::FRAC_1_SQRT_2;
    assert!(
        close(heard, point(px(150. - along), px(150. - along))),
        "heard at {heard:?}"
    );

    // Inside the unrotated bounds, near a corner the rotation turned away: a miss.
    cx.simulate_click(point(px(105.), px(105.)), Modifiers::default());
    assert_eq!(downs.take(), None, "the turned-away corner is not hit");

    // While the square paints, the mouse position is in its coordinates too.
    cx.simulate_mouse_move(point(px(150.), px(85.)), None, Modifiers::default());
    cx.update_window(window, |_, window, cx| {
        window.refresh();
        window.draw(cx).clear(cx)
    })
    .unwrap();
    let painted = painted_mouse.get().expect("the canvas painted");
    assert!(
        close(painted, point(px(150. - along), px(150. - along))),
        "painted with {painted:?}"
    );
}

/// A cached card inside an element rotated by `angle`.
struct RotatedCard {
    card: Entity<Card>,
    angle: Rc<Cell<f32>>,
}

struct Card {
    renders: Rc<Cell<usize>>,
}

impl Render for Card {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        div().size_full().bg(red())
    }
}

impl Render for RotatedCard {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let mut style = crate::StyleRefinement::default();
        style.size.width = Some(px(50.).into());
        style.size.height = Some(px(50.).into());
        div()
            .size(px(100.))
            .rotate(crate::radians(self.angle.get()))
            .child(self.card.clone().cached(style))
    }
}

#[crate::test]
fn a_cached_view_is_reused_under_the_same_transform_only(cx: &mut TestAppContext) {
    let renders = Rc::new(Cell::new(0));
    let angle = Rc::new(Cell::new(0.3));
    let window: AnyWindowHandle = cx
        .add_window({
            let (renders, angle) = (renders.clone(), angle.clone());
            move |_, cx| RotatedCard {
                card: cx.new(|_| Card { renders }),
                angle,
            }
        })
        .into();
    let draw = |cx: &mut TestAppContext| {
        cx.update_window(window, |root, window, cx| {
            root.downcast::<RotatedCard>()
                .unwrap()
                .update(cx, |_, cx| cx.notify());
            window.draw(cx).clear(cx)
        })
        .unwrap();
    };
    assert_eq!(renders.get(), 1);
    draw(cx);
    assert_eq!(renders.get(), 1, "reused under the same rotation");
    let (quad, transform_count) = cx
        .update_window(window, |_, window, _| {
            let scene = &window.rendered_frame.scene;
            (scene.quads[0], scene.transforms().len())
        })
        .unwrap();
    assert_ne!(quad.transform, 0);
    assert!(
        (quad.transform as usize) < transform_count,
        "the reused quad refers to this frame's transform table"
    );

    angle.set(0.6);
    draw(cx);
    assert_eq!(renders.get(), 2, "rendered again under another rotation");
}
