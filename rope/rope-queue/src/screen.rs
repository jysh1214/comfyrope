//! The daemon's screen: the queue by group, the running job's progress, the last messages. ratatui over
//! crossterm. When stdout is not a terminal the daemon prints lines instead and never opens this.

use crate::daemon::Msg;
use anyhow::Result;
use ratatui::crossterm::event::{self, Event, KeyEventKind};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table, TableState, Wrap};
use ratatui::{DefaultTerminal, Frame};
use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    Group(u64),
    Job(u64),
}

pub struct RowData {
    pub kind: RowKind,
    /// One per column: marker, id, jobs or state, tool, into, progress, now.
    pub cells: Vec<String>,
    /// Finished, nothing left to do.
    pub dim: bool,
    /// Failed, or otherwise wanting a look.
    pub warn: bool,
}

pub struct View {
    pub header: String,
    /// Paused, with the reason.
    pub status: Option<String>,
    pub rows: Vec<RowData>,
    pub detail: Vec<String>,
    pub notes: Vec<String>,
}

pub const COLUMNS: [&str; 7] = ["", "id", "jobs", "tool", "into", "progress", "now"];

pub struct Screen {
    terminal: DefaultTerminal,
    state: TableState,
    kinds: Vec<RowKind>,
    suspended: Arc<AtomicBool>,
    pub expanded: BTreeSet<u64>,
}

impl Screen {
    /// Takes the terminal over and starts the thread that reads keys into `tx`.
    pub fn open(tx: Sender<Msg>) -> Result<Screen> {
        let terminal = ratatui::init();
        let suspended = Arc::new(AtomicBool::new(false));
        spawn_input(tx, suspended.clone());
        Ok(Screen {
            terminal,
            state: TableState::default().with_selected(0),
            kinds: Vec::new(),
            suspended,
            expanded: BTreeSet::new(),
        })
    }

    pub fn close(self) {
        ratatui::restore();
    }

    pub fn selected(&self) -> Option<RowKind> {
        self.state.selected().and_then(|i| self.kinds.get(i).copied())
    }

    pub fn move_by(&mut self, delta: i32) {
        if self.kinds.is_empty() {
            return;
        }
        let at = self.state.selected().unwrap_or(0) as i32 + delta;
        self.state.select(Some(at.clamp(0, self.kinds.len() as i32 - 1) as usize));
    }

    /// Expands or collapses the selected group.
    pub fn toggle(&mut self) {
        if let Some(RowKind::Group(g)) = self.selected()
            && !self.expanded.remove(&g)
        {
            self.expanded.insert(g);
        }
    }

    pub fn draw(&mut self, view: &View) -> Result<()> {
        self.kinds = view.rows.iter().map(|r| r.kind).collect();
        if self.kinds.is_empty() {
            self.state.select(None);
        } else {
            let at = self.state.selected().unwrap_or(0).min(self.kinds.len() - 1);
            self.state.select(Some(at));
        }
        let state = &mut self.state;
        self.terminal.draw(|f| render(f, view, state))?;
        Ok(())
    }

    /// Shows `path` in `$PAGER` (less) with the screen set aside.
    pub fn page(&mut self, path: &Path) -> Result<()> {
        self.suspended.store(true, Ordering::SeqCst);
        ratatui::restore();
        let pager = std::env::var("PAGER").ok().filter(|p| !p.trim().is_empty()).unwrap_or_else(|| "less".to_owned());
        let status = Command::new("sh").arg("-c").arg(format!("{pager} \"$1\"")).arg("sh").arg(path).status();
        self.terminal = ratatui::init();
        self.suspended.store(false, Ordering::SeqCst);
        status.map(|_| ()).map_err(Into::into)
    }
}

fn render(f: &mut Frame, view: &View, state: &mut TableState) {
    let [head, table, detail, notes, foot] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(4),
        Constraint::Length(4),
        Constraint::Length(7),
        Constraint::Length(1),
    ])
    .areas(f.area());

    let mut spans = vec![Span::raw("rope-queue").bold(), Span::raw("   "), Span::raw(view.header.as_str())];
    if let Some(s) = &view.status {
        spans.push(Span::raw("   "));
        spans.push(Span::raw(s.as_str()).yellow().bold());
    }
    f.render_widget(Paragraph::new(Line::from(spans)), head);

    let widths = [
        Constraint::Length(2),
        Constraint::Length(6),
        Constraint::Length(12),
        Constraint::Length(10),
        Constraint::Min(12),
        Constraint::Length(22),
        Constraint::Min(24),
    ];
    let rows = view.rows.iter().map(|r| {
        let row = Row::new(r.cells.iter().map(String::as_str));
        if r.warn {
            row.red()
        } else if r.dim {
            row.dim()
        } else {
            row
        }
    });
    let table_widget = Table::new(rows, widths)
        .header(Row::new(COLUMNS).bold().underlined())
        .row_highlight_style(Style::new().reversed())
        .column_spacing(1);
    f.render_stateful_widget(table_widget, table, state);

    let lines: Vec<Line> = view.detail.iter().map(|s| Line::from(s.as_str())).collect();
    f.render_widget(Paragraph::new(lines).block(Block::new().borders(Borders::TOP)).wrap(Wrap { trim: false }), detail);

    let lines: Vec<Line> = view.notes.iter().map(|s| Line::from(s.as_str()).dim()).collect();
    f.render_widget(Paragraph::new(lines).block(Block::new().borders(Borders::TOP)).wrap(Wrap { trim: false }), notes);

    let keys =
        " ↑↓ select   Enter expand   c cancel   r retry   l log   p pause/resume   q stop (interrupts the running job)";
    f.render_widget(Paragraph::new(keys).dim(), foot);
}

/// Reads keys on a thread and hands them to the loop. While `suspended` (a pager runs), it reads nothing.
fn spawn_input(tx: Sender<Msg>, suspended: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        loop {
            if suspended.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
            match event::poll(Duration::from_millis(100)) {
                Ok(true) => {
                    if let Ok(Event::Key(k)) = event::read()
                        && k.kind == KeyEventKind::Press
                        && tx.send(Msg::Key(k)).is_err()
                    {
                        return;
                    }
                }
                Ok(false) => {}
                Err(_) => return,
            }
        }
    });
}
