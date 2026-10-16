//! rope-gui: browse and organize the rope image library.
//!
//!     rope-gui [--library DIR] [--foreground] [ID]
//!
//! ID opens on that image. The library is the one `rope` uses: --library, else $ROPE_LIBRARY, else
//! ~/.local/share/rope.
//!
//! The window runs on its own, in a session of its own, and the command returns at once: the terminal is free,
//! and closing it or pressing Ctrl-C there leaves the window open. What the window prints goes to
//! .cache/rope-gui.log in the library. --foreground keeps it attached to the terminal instead.

mod app;
mod decode;
mod fonts;
mod model;
mod player;
mod thumbs;

use eframe::egui;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Set for the detached copy, so that it runs the window instead of detaching again.
const DETACHED: &str = "ROPE_GUI_DETACHED";
/// The log starts over once it is larger than this.
const LOG_LIMIT: u64 = 1 << 20;

const USAGE: &str = "usage: rope-gui [--library DIR] [--foreground] [ID]

Browse and organize the rope image library; ID opens on that image.

The window runs on its own and the command returns at once; what it prints goes to .cache/rope-gui.log in the
library. --foreground keeps it attached to the terminal instead.";

fn main() -> eframe::Result {
    let mut library: Option<PathBuf> = None;
    let mut foreground = std::env::var_os(DETACHED).is_some();
    let mut open = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--library" => library = args.next().map(PathBuf::from),
            s if s.starts_with("--library=") => library = Some(PathBuf::from(&s["--library=".len()..])),
            "--foreground" => foreground = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            _ => open = Some(a),
        }
    }
    // the library is opened here first, so that a missing one is reported in the terminal
    let lib = match rope_core::Library::locate(library.as_deref()) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("rope-gui: {e:#}");
            std::process::exit(1);
        }
    };
    if !foreground {
        if let Err(e) = detach(&lib) {
            eprintln!("rope-gui: cannot start in the background ({e}); run it with --foreground");
            std::process::exit(1);
        }
        return Ok(());
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("rope")
            .with_app_id("rope")
            .with_inner_size([1400.0, 900.0]),
        ..Default::default()
    };
    eframe::run_native("rope", options, Box::new(move |cc| Ok(Box::new(app::App::new(&cc.egui_ctx, lib, open)))))
}

/// Starts this program again in a new session, with the same arguments, its output going to the log, and
/// returns without waiting for it.
fn detach(lib: &rope_core::Library) -> std::io::Result<()> {
    std::fs::create_dir_all(lib.cache_dir())?;
    let path = lib.cache_dir().join("rope-gui.log");
    let too_big = std::fs::metadata(&path).is_ok_and(|m| m.len() > LOG_LIMIT);
    let log = std::fs::OpenOptions::new().create(true).append(!too_big).write(true).truncate(too_big).open(&path)?;
    let mut cmd = Command::new(std::env::current_exe()?);
    cmd.args(std::env::args_os().skip(1)).env(DETACHED, "1").stdin(Stdio::null()).stdout(log.try_clone()?).stderr(log);
    // SAFETY: setsid is async-signal-safe, so it may run between fork and exec
    unsafe {
        cmd.pre_exec(|| if libc::setsid() == -1 { Err(std::io::Error::last_os_error()) } else { Ok(()) });
    }
    cmd.spawn().map(drop)
}
