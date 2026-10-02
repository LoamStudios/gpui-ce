use gpui::BackgroundExecutor;
use smol::process::Command;

type PaletteCommand<'a> = (&'a str, &'a [&'a str]);

fn palette_commands(desktop: &str) -> [PaletteCommand<'static>; 5] {
    let plasma = ("plasma-emojier", &[] as &[_]);
    let legacy_plasma = ("ibus-ui-emojier-plasma", &[] as &[_]);
    let ibus = ("ibus", &["emoji"] as &[_]);
    let characters = ("gnome-characters", &[] as &[_]);
    let kcharselect = ("kcharselect", &[] as &[_]);
    if desktop
        .split(':')
        .any(|name| name.eq_ignore_ascii_case("KDE"))
    {
        [plasma, legacy_plasma, ibus, characters, kcharselect]
    } else {
        [characters, ibus, plasma, legacy_plasma, kcharselect]
    }
}

async fn launch_palette(commands: &[PaletteCommand<'_>], activation_token: Option<&str>) -> bool {
    for (program, arguments) in commands {
        let mut command = Command::new(program);
        command.args(*arguments);
        if let Some(token) = activation_token {
            command.env("XDG_ACTIVATION_TOKEN", token);
        }
        match command.status().await {
            Ok(status) if status.success() => return true,
            Ok(status) => log::warn!("Character palette {program} exited with {status}"),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => log::warn!("Failed to open character palette {program}: {error}"),
        }
    }
    false
}

pub(super) fn show_character_palette(
    executor: BackgroundExecutor,
    activation_token: Option<String>,
) {
    let desktop = std::env::var("XDG_CURRENT_DESKTOP").unwrap_or_default();
    executor
        .spawn(async move {
            if !launch_palette(&palette_commands(&desktop), activation_token.as_deref()).await {
                log::warn!(
                    "No character palette is available; install Plasma's Emoji Selector, IBus, \
                     GNOME Characters, or KCharSelect"
                );
            }
        })
        .detach();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_the_desktop_picker() {
        assert_eq!(
            palette_commands("ubuntu:GNOME")[0],
            ("gnome-characters", &[][..])
        );
        assert_eq!(palette_commands("KDE")[0].0, "plasma-emojier");
        assert_eq!(palette_commands("other:kde")[0].0, "plasma-emojier");
    }

    #[test]
    fn falls_back_after_missing_or_failing_commands() {
        assert!(smol::block_on(launch_palette(
            &[
                ("/nonexistent/gpui-character-palette", &[]),
                ("/bin/sh", &["-c", "exit 1"]),
                ("/bin/sh", &["-c", "exit 0"]),
            ],
            None,
        )));
    }

    #[test]
    fn reports_unavailable_pickers() {
        assert!(!smol::block_on(launch_palette(
            &[
                ("/nonexistent/gpui-character-palette", &[]),
                ("/bin/sh", &["-c", "exit 1"]),
            ],
            None,
        )));
    }

    #[test]
    fn passes_the_wayland_activation_token() {
        assert!(smol::block_on(launch_palette(
            &[(
                "/bin/sh",
                &["-c", "test \"$XDG_ACTIVATION_TOKEN\" = gpui-palette-test"]
            )],
            Some("gpui-palette-test"),
        )));
    }
}
