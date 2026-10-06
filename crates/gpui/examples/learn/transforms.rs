//! Transforms Example
//!
//! Elements drawn under transforms with `.rotate()`, `.scale()` and
//! `.transform()`, which turn an element and its children about its center
//! without changing layout:
//!
//! 1. Rotated cards that stay interactive: hover and click land where the
//!    card is drawn, and click positions arrive in the card's coordinates.
//! 2. A rotated frame that clips what overflows it along its own edges.
//! 3. A scaled group, whose text is rasterized at the size it appears.
//!
//! The cards turn continuously; click one to count.

use std::f32::consts::PI;
use std::time::Instant;

use gpui::{
    App, Bounds, Context, MouseButton, Render, SharedString, Window, WindowBounds, WindowOptions,
    div, hsla, kurbo, prelude::*, px, radians, rgb, size,
};
use gpui_platform::application;

struct Transforms {
    started: Instant,
    clicks: [usize; 3],
    last_click: Option<SharedString>,
}

impl Render for Transforms {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        window.request_animation_frame();
        let turn = self.started.elapsed().as_secs_f32() * 0.4;

        let card = |index: usize, angle: f32, cx: &mut Context<Self>| {
            let hue = index as f32 * 0.27;
            div()
                .id(("card", index))
                .w(px(180.))
                .h(px(110.))
                .p_3()
                .flex()
                .flex_col()
                .justify_between()
                .rounded_lg()
                .border_2()
                .border_color(hsla(hue, 0.6, 0.35, 1.))
                .bg(hsla(hue, 0.6, 0.75, 1.))
                .hover(move |style| style.bg(hsla(hue, 0.7, 0.85, 1.)))
                .text_color(rgb(0x111111))
                .rotate(radians(angle))
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, event: &gpui::MouseDownEvent, _, cx| {
                        this.clicks[index] += 1;
                        this.last_click = Some(
                            format!(
                                "card {index} at ({:.0}, {:.0}) in its own coordinates",
                                f32::from(event.position.x),
                                f32::from(event.position.y)
                            )
                            .into(),
                        );
                        cx.notify();
                    }),
                )
                .child(div().text_lg().child(format!("Card {index}")))
                .child(
                    div()
                        .text_sm()
                        .child(format!("{} clicks", self.clicks[index])),
                )
        };

        let rotated_frame = div()
            .w(px(220.))
            .h(px(140.))
            .overflow_hidden()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x334155))
            .bg(rgb(0xe2e8f0))
            .rotate(radians(-turn * 0.5))
            .child(
                div()
                    .w(px(320.))
                    .p_2()
                    .text_color(rgb(0x0f172a))
                    .child(
                        "This frame is rotated and clips its content along its own edges, \
                         not along the window's. The text runs on past the frame's width.",
                    )
                    .child(div().mt_2().size(px(260.)).rounded_full().bg(rgb(0x38bdf8))),
            );

        let scaled_group = div()
            .flex()
            .gap_2()
            .p_2()
            .rounded_md()
            .bg(rgb(0xfef3c7))
            .text_color(rgb(0x451a03))
            .transform(kurbo::Affine::scale(
                1.0 + 0.5 * (turn * 2.).sin().abs() as f64,
            ))
            .child("Scaled text stays crisp");

        div()
            .size_full()
            .bg(rgb(0xf8fafc))
            .flex()
            .flex_col()
            .gap_12()
            .p_12()
            .child(
                div()
                    .flex()
                    .gap_16()
                    .child(card(0, turn, cx))
                    .child(card(1, -turn * 1.3 + PI / 6., cx))
                    .child(card(2, PI / 2., cx)),
            )
            .child(
                div()
                    .flex()
                    .gap_24()
                    .child(rotated_frame)
                    .child(scaled_group),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x475569))
                    .child(self.last_click.clone().unwrap_or("Click a card.".into())),
            )
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(980.), px(640.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|_| Transforms {
                    started: Instant::now(),
                    clicks: [0; 3],
                    last_click: None,
                })
            },
        )
        .unwrap();
        cx.activate(true);
    });
}

fn main() {
    run_example();
}
