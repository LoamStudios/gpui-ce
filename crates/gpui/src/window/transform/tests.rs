use super::*;
use crate::{
    AnyWindowHandle, Entity, FocusHandle, InteractiveElement as _, IntoElement, Modifiers,
    MouseButton, ParentElement as _, Render, Styled as _, TestAppContext, VisualTestContext,
    canvas, div, hsla, red,
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

/// A cached card inside an element under `transform`.
struct TransformedCard {
    card: Entity<Card>,
    transform: Rc<Cell<kurbo::Affine>>,
}

struct Card {
    renders: Rc<Cell<usize>>,
}

impl Render for Card {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        div()
            .size_full()
            .bg(red())
            .border_2()
            .border_color(hsla(0.6, 1., 0.5, 1.))
            .rounded(px(6.))
    }
}

impl Render for TransformedCard {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let mut style = crate::StyleRefinement::default();
        style.size.width = Some(px(50.).into());
        style.size.height = Some(px(40.).into());
        div().size(px(400.)).relative().child(
            div()
                .absolute()
                .left(px(100.))
                .top(px(100.))
                .size(px(100.))
                .transform(self.transform.get())
                .child(self.card.clone().cached(style)),
        )
    }
}

/// The card's quad, with its corners in the viewport, after drawing the
/// window, with the card notified first when `fresh`.
fn card_quad(
    cx: &mut TestAppContext,
    window: AnyWindowHandle,
    fresh: bool,
) -> (Quad, [Point<Pixels>; 2]) {
    cx.update_window(window, |root, window, cx| {
        let root = root.downcast::<TransformedCard>().unwrap();
        if fresh {
            let card = root.read(cx).card.clone();
            card.update(cx, |_, cx| cx.notify());
        }
        root.update(cx, |_, cx| cx.notify());
        window.draw(cx).clear(cx);
        let scene = &window.rendered_frame.scene;
        let quad = scene.quads[0];
        let transformation = scene.transforms()[quad.transform as usize].transformation;
        let corner =
            |point: Point<ScaledPixels>| transformation.apply(point.map(|value| px(value.0)));
        (
            quad,
            [
                corner(quad.bounds.origin),
                corner(quad.bounds.bottom_right()),
            ],
        )
    })
    .unwrap()
}

fn transformed_card(
    cx: &mut TestAppContext,
    transform: kurbo::Affine,
) -> (AnyWindowHandle, Rc<Cell<usize>>, Rc<Cell<kurbo::Affine>>) {
    let renders = Rc::new(Cell::new(0));
    let transform = Rc::new(Cell::new(transform));
    let window = cx
        .add_window({
            let (renders, transform) = (renders.clone(), transform.clone());
            move |_, cx| TransformedCard {
                card: cx.new(|_| Card { renders }),
                transform,
            }
        })
        .into();
    (window, renders, transform)
}

#[crate::test]
fn a_cached_view_is_placed_under_a_new_zoom_as_if_painted_there(cx: &mut TestAppContext) {
    let (window, renders, transform) = transformed_card(cx, kurbo::Affine::scale(1.));
    card_quad(cx, window, false);
    assert_eq!(renders.get(), 1);

    transform.set(kurbo::Affine::scale(1.5));
    let (reused, _) = card_quad(cx, window, false);
    assert_eq!(renders.get(), 1, "zooming reuses the card");
    assert_eq!(reused.transform, 0, "a zoom keeps the card aligned");

    let (fresh, _) = card_quad(cx, window, true);
    assert_eq!(renders.get(), 2);
    let near = |a: f32, b: f32| (a - b).abs() < 0.51;
    assert!(
        near(reused.bounds.origin.x.0, fresh.bounds.origin.x.0)
            && near(reused.bounds.origin.y.0, fresh.bounds.origin.y.0)
            && near(reused.bounds.size.width.0, fresh.bounds.size.width.0)
            && near(reused.bounds.size.height.0, fresh.bounds.size.height.0),
        "placed where painting it there puts it: {:?} against {:?}",
        reused.bounds,
        fresh.bounds
    );
    assert_eq!(
        reused.corner_radii, fresh.corner_radii,
        "radii scale with it"
    );
    assert!(
        near(reused.border_widths.top.0, fresh.border_widths.top.0),
        "borders scale with it"
    );
}

#[crate::test]
fn a_cached_view_is_placed_under_a_new_rotation_by_the_gpu(cx: &mut TestAppContext) {
    let (window, renders, transform) = transformed_card(cx, kurbo::Affine::rotate(0.3));
    card_quad(cx, window, false);
    assert_eq!(renders.get(), 1);

    transform.set(kurbo::Affine::rotate(0.9));
    let (reused, reused_corners) = card_quad(cx, window, false);
    assert_eq!(renders.get(), 1, "turning reuses the card");
    assert_ne!(reused.transform, 0);

    let (_, fresh_corners) = card_quad(cx, window, true);
    assert_eq!(renders.get(), 2);
    for (reused, fresh) in reused_corners.into_iter().zip(fresh_corners) {
        assert!(
            (reused.x - fresh.x).abs() < px(0.01) && (reused.y - fresh.y).abs() < px(0.01),
            "a corner lands at {reused:?}, where painting it there puts it at {fresh:?}"
        );
    }

    // A recording made under a rotation is not reused under a change that
    // scales it: its text would be stretched.
    transform.set(kurbo::Affine::rotate(0.9) * kurbo::Affine::scale(2.));
    card_quad(cx, window, false);
    assert_eq!(renders.get(), 3, "rendered again under a scaling change");
}

struct Icon;

impl Render for Icon {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        crate::svg()
            .size(px(20.))
            .text_color(red())
            .data(br#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10"/></svg>"#)
    }
}

struct ZoomedIcon {
    icon: Entity<Icon>,
    transform: Rc<Cell<kurbo::Affine>>,
}

impl Render for ZoomedIcon {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let mut style = crate::StyleRefinement::default();
        style.size.width = Some(px(20.).into());
        style.size.height = Some(px(20.).into());
        div().size(px(400.)).relative().child(
            div()
                .absolute()
                .left(px(100.))
                .top(px(100.))
                .size(px(20.))
                .transform(self.transform.get())
                .child(self.icon.clone().cached(style)),
        )
    }
}

#[crate::test]
fn a_cached_svg_is_rasterized_again_under_a_new_zoom(cx: &mut TestAppContext) {
    let transform = Rc::new(Cell::new(kurbo::Affine::scale(1.)));
    let window: AnyWindowHandle = cx
        .add_window({
            let transform = transform.clone();
            move |_, cx| ZoomedIcon {
                icon: cx.new(|_| Icon),
                transform,
            }
        })
        .into();
    let sprite = |cx: &mut TestAppContext, fresh: bool| {
        cx.update_window(window, |root, window, cx| {
            let root = root.downcast::<ZoomedIcon>().unwrap();
            if fresh {
                let icon = root.read(cx).icon.clone();
                icon.update(cx, |_, cx| cx.notify());
            }
            root.update(cx, |_, cx| cx.notify());
            window.draw(cx).clear(cx);
            window.rendered_frame.scene.monochrome_sprites[0]
        })
        .unwrap()
    };
    let painted = sprite(cx, false);

    transform.set(kurbo::Affine::scale(2.));
    let reused = sprite(cx, false);
    assert_eq!(reused.transform, 0, "a zoom keeps the icon aligned");
    assert_ne!(
        reused.tile.tile_id, painted.tile.tile_id,
        "from a new raster"
    );
    assert_eq!(
        reused.tile.bounds.size.width.0,
        painted.tile.bounds.size.width.0 * 2,
        "at the size it now appears"
    );

    let fresh = sprite(cx, true);
    assert_eq!(
        reused.tile.tile_id, fresh.tile.tile_id,
        "the raster painting it there uses"
    );
    assert_eq!(
        reused.bounds, fresh.bounds,
        "where painting it there puts it"
    );
}

/// A square in three groups, each made by an element of no height that
/// the square does not overlap.
struct NestedFades;
impl Render for NestedFades {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size(px(400.)).relative().child(
            div().group_opacity(0.5).child(
                div().group_opacity(0.5).child(
                    div().group_opacity(0.5).child(
                        div()
                            .absolute()
                            .left(px(50.))
                            .top(px(30.))
                            .size(px(100.))
                            .bg(red()),
                    ),
                ),
            ),
        )
    }
}

/// Content that does not overlap the elements that make its groups still
/// falls inside every one of them. The innermost group fades one square,
/// so it is drawn in place, the square faded.
#[crate::test]
fn content_falls_inside_the_groups_that_enclose_it(cx: &mut TestAppContext) {
    let window: AnyWindowHandle = cx.add_window(|_, _| NestedFades).into();
    cx.update_window(window, |_, window, _| {
        let commands: Vec<&str> = window
            .rendered_frame
            .scene
            .render_commands()
            .iter()
            .map(|command| match command {
                crate::RenderCommand::BeginGroup {
                    target: crate::GroupTarget::Isolated { .. },
                    ..
                } => "begin",
                crate::RenderCommand::EndGroup {
                    target: crate::GroupTarget::Isolated { .. },
                    ..
                } => "end",
                crate::RenderCommand::Batch(_) => "quad",
                _ => "inline",
            })
            .collect();
        assert_eq!(
            commands,
            ["begin", "begin", "inline", "quad", "inline", "end", "end"]
        );
    })
    .unwrap();
}

/// A one-line text field whose characters are 10px wide and 20px tall,
/// left to right from its element's origin, with the caret after the second.
struct Field {
    focus: FocusHandle,
    /// Its element's bounds where it last painted, in its own coordinates.
    bounds: Rc<Cell<Bounds<Pixels>>>,
    renders: Rc<Cell<usize>>,
}

impl crate::EntityInputHandler for Field {
    fn text_for_range(
        &mut self,
        _: std::ops::Range<usize>,
        _: &mut Option<std::ops::Range<usize>>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<String> {
        None
    }

    fn selected_text_range(
        &mut self,
        _: bool,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<crate::UTF16Selection> {
        Some(crate::UTF16Selection {
            range: 2..2,
            reversed: false,
        })
    }

    fn marked_text_range(
        &self,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<std::ops::Range<usize>> {
        None
    }

    fn unmark_text(&mut self, _: &mut Window, _: &mut Context<Self>) {}

    fn replace_text_in_range(
        &mut self,
        _: Option<std::ops::Range<usize>>,
        _: &str,
        _: &mut Window,
        _: &mut Context<Self>,
    ) {
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        _: Option<std::ops::Range<usize>>,
        _: &str,
        _: Option<std::ops::Range<usize>>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) {
    }

    fn bounds_for_range(
        &mut self,
        range: std::ops::Range<usize>,
        element_bounds: Bounds<Pixels>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<Bounds<Pixels>> {
        Some(Bounds::new(
            element_bounds.origin + point(px(10. * range.start as f32), px(0.)),
            size(px(10. * range.len().max(1) as f32), px(20.)),
        ))
    }

    fn character_index_for_point(
        &mut self,
        point: Point<Pixels>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<usize> {
        let x = point.x - self.bounds.get().origin.x;
        Some((x / px(10.)).floor().max(0.) as usize)
    }
}

impl Render for Field {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        let (field, focus, last_bounds) = (cx.entity(), self.focus.clone(), self.bounds.clone());
        canvas(
            |_, _, _| (),
            move |bounds, _, window, cx| {
                last_bounds.set(bounds);
                window.handle_input(&focus, crate::ElementInputHandler::new(bounds, field), cx);
            },
        )
        .size_full()
    }
}

/// A 100×20 field at (100, 100), cached, transformed about its center
/// (150, 110).
struct FieldStage {
    field: Entity<Field>,
    transform: Rc<Cell<kurbo::Affine>>,
}

impl Render for FieldStage {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let mut style = crate::StyleRefinement::default();
        style.size.width = Some(px(100.).into());
        style.size.height = Some(px(20.).into());
        div().size(px(400.)).relative().child(
            div()
                .absolute()
                .left(px(100.))
                .top(px(100.))
                .w(px(100.))
                .h(px(20.))
                .transform(self.transform.get())
                .child(self.field.clone().cached(style)),
        )
    }
}

fn field_stage(
    cx: &mut TestAppContext,
    transform: kurbo::Affine,
) -> (AnyWindowHandle, Rc<Cell<kurbo::Affine>>, Rc<Cell<usize>>) {
    let transform = Rc::new(Cell::new(transform));
    let renders = Rc::new(Cell::new(0));
    let window: AnyWindowHandle = cx
        .add_window({
            let (transform, renders) = (transform.clone(), renders.clone());
            move |window, cx| {
                let focus = cx.focus_handle();
                window.focus(&focus, cx);
                FieldStage {
                    field: cx.new(|_| Field {
                        focus,
                        bounds: Rc::default(),
                        renders,
                    }),
                    transform,
                }
            }
        })
        .into();
    (window, transform, renders)
}

/// Draws the window and returns the input handler it gives the platform.
fn platform_input_handler(
    cx: &mut TestAppContext,
    window: AnyWindowHandle,
) -> PlatformInputHandler {
    cx.update_window(window, |root, window, cx| {
        root.downcast::<FieldStage>()
            .unwrap()
            .update(cx, |_, cx| cx.notify());
        window.draw(cx).clear(cx);
        window
            .platform_window
            .take_input_handler()
            .expect("the focused field registers an input handler")
    })
    .unwrap()
}

/// `transform` about the field's center, from its coordinates to the window's.
fn field_to_window(transform: kurbo::Affine) -> kurbo::Affine {
    kurbo::Affine::translate((150., 110.)) * transform * kurbo::Affine::translate((-150., -110.))
}

/// The window bounds of the field's element `rect` under `transform`.
fn window_rect(transform: kurbo::Affine, rect: kurbo::Rect) -> Bounds<Pixels> {
    let rect = field_to_window(transform).transform_rect_bbox(rect);
    Bounds::from_corners(
        point(px(rect.x0 as f32), px(rect.y0 as f32)),
        point(px(rect.x1 as f32), px(rect.y1 as f32)),
    )
}

fn close_bounds(actual: Bounds<Pixels>, expected: Bounds<Pixels>) -> bool {
    close(actual.origin, expected.origin) && close(actual.bottom_right(), expected.bottom_right())
}

#[crate::test]
fn an_input_handler_reports_and_hears_window_coordinates_under_a_transform(
    cx: &mut TestAppContext,
) {
    for transform in [
        kurbo::Affine::rotate(f64::from(FRAC_PI_4)),
        kurbo::Affine::scale_non_uniform(2., 3.),
        kurbo::Affine::scale(2.),
    ] {
        let (window, _, _) = field_stage(cx, transform);
        let mut handler = platform_input_handler(cx, window);

        // The third character, (120, 100)-(130, 120) in the field.
        let expected = window_rect(transform, kurbo::Rect::new(120., 100., 130., 120.));
        let bounds = handler.bounds_for_range(2..3).unwrap();
        assert!(
            close_bounds(bounds, expected),
            "under {transform:?}, the third character is at {bounds:?}, not {expected:?}"
        );
        // The IME is placed at the caret, before the third character.
        let caret = window_rect(transform, kurbo::Rect::new(120., 100., 130., 120.));
        let ime = handler.ime_candidate_bounds().unwrap();
        assert!(
            close_bounds(ime, caret),
            "under {transform:?}, the IME is placed at {ime:?}, not {caret:?}"
        );
        let element = window_rect(transform, kurbo::Rect::new(100., 100., 200., 120.));
        assert!(close_bounds(handler.element_bounds().unwrap(), element));

        // The middle of the fourth character, (135, 110) in the field, is
        // where the window says it is.
        let middle = field_to_window(transform) * kurbo::Point::new(135., 110.);
        assert_eq!(
            handler.character_index_for_point(point(px(middle.x as f32), px(middle.y as f32))),
            Some(3),
            "under {transform:?}"
        );
    }
}

#[crate::test]
fn a_reused_input_handler_follows_its_view_to_a_new_transform(cx: &mut TestAppContext) {
    let (window, transform, renders) = field_stage(cx, kurbo::Affine::IDENTITY);
    let handler = platform_input_handler(cx, window);
    cx.update_window(window, |_, window, _| {
        window.platform_window.set_input_handler(handler)
    })
    .unwrap();
    assert_eq!(renders.get(), 1);

    let rotation = kurbo::Affine::rotate(f64::from(FRAC_PI_2));
    transform.set(rotation);
    let mut handler = platform_input_handler(cx, window);
    assert_eq!(renders.get(), 1, "the field is reused, not rendered again");
    let expected = window_rect(rotation, kurbo::Rect::new(120., 100., 130., 120.));
    let bounds = handler.bounds_for_range(2..3).unwrap();
    assert!(
        close_bounds(bounds, expected),
        "the third character is at {bounds:?}, not {expected:?}"
    );
}
