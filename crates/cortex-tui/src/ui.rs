//! All rendering. Reads App and Supervisor; never mutates them.
use crate::app::{App, ChatLine, Tab, Who, CORTICES, TRAIN_TARGETS};
use crate::supervisor::{Kind, Status, Supervisor};
use cortex_core::{Frame as CamFrame, CONNECTOME};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, BorderType, Gauge, List, ListItem, ListState, Paragraph, Sparkline, Tabs},
    Frame,
};

const DIM: Color = Color::DarkGray;

fn cortex_color(name: &str) -> Color {
    match name {
        "visual" => Color::LightCyan,
        "perceptual" => Color::LightGreen,
        "memory" => Color::LightMagenta,
        "associative" => Color::LightYellow,
        "language" => Color::LightBlue,
        "workspace" => Color::LightRed,
        _ => Color::Gray,
    }
}

fn heat(rate: f32) -> Color {
    if rate > 15.0 { Color::LightRed } else if rate > 7.0 { Color::LightYellow } else if rate > 2.0 { Color::LightGreen } else if rate > 0.0 { Color::Cyan } else { DIM }
}

fn panel(title: impl Into<Line<'static>>) -> Block<'static> {
    Block::bordered().border_type(BorderType::Rounded).border_style(Style::new().fg(DIM)).title(title)
}

fn title(s: &str) -> Line<'static> {
    Line::from(vec![Span::raw(" "), Span::styled(s.to_string(), Style::new().bold().fg(Color::White)), Span::raw(" ")])
}

pub fn draw(f: &mut Frame, app: &App, sup: &Supervisor) {
    let rows = Layout::vertical([Constraint::Length(3), Constraint::Min(0), Constraint::Length(1)]).split(f.area());
    header(f, rows[0], app, sup);
    match app.tab {
        Tab::Brain => brain(f, rows[1], app),
        Tab::Chat => chat(f, rows[1], app),
        Tab::Camera => camera(f, rows[1], app, sup),
        Tab::Teach => teach(f, rows[1], app, sup),
        Tab::Processes => processes(f, rows[1], app, sup),
        Tab::Train => train(f, rows[1], app, sup),
    }
    footer(f, rows[2], app);
}

fn header(f: &mut Frame, area: Rect, app: &App, sup: &Supervisor) {
    let nats_up = sup.procs[0].is_up();
    let status = Line::from(vec![
        Span::styled(if nats_up { " ● nats " } else { " ○ nats " }, Style::new().fg(if nats_up { Color::LightGreen } else { Color::LightRed })),
        Span::styled(format!("{} msg/s ", app.msg_rate()), Style::new().fg(Color::Cyan)),
        Span::styled(format!("{}/{} nodes ", sup.running().saturating_sub(nats_up as usize), crate::supervisor::NODE_ROLES.len()), Style::new().fg(Color::Gray)),
        Span::styled(format!("up {}s ", app.started.elapsed().as_secs()), Style::new().fg(DIM)),
    ]);
    let tabs = Tabs::new(Tab::ALL.iter().enumerate().map(|(i, t)| format!("{} {}", i + 1, t.title())))
        .select(app.tab.index())
        .style(Style::new().fg(Color::Gray))
        .highlight_style(Style::new().fg(Color::Black).bg(Color::LightCyan).bold())
        .divider("│")
        .block(panel(Line::from(vec![Span::raw(" "), Span::styled("CORTEX", Style::new().bold().fg(Color::LightCyan)), Span::styled(" seed-0 ", Style::new().fg(DIM))])).title_top(status.right_aligned()));
    f.render_widget(tabs, area);
}

fn footer(f: &mut Frame, area: Rect, app: &App) {
    let hint = match app.tab {
        Tab::Brain => "1-6/Tab switch  R restart nodes  q quit",
        Tab::Chat => "type + Enter to speak to the brain   F1-F6/Tab switch   Esc brain   Ctrl-C quit",
        Tab::Camera => "1-6/Tab switch  R restart nodes  q quit",
        Tab::Teach => "t start/stop teacher  g synthetic/camera mode  s speed  R restart nodes  q quit",
        Tab::Processes => "j/k select  Enter (re)start  x stop  R restart nodes  q quit",
        Tab::Train => "j/k choose  Enter train  R restart nodes (load new weights)  q quit",
    };
    let text = if app.quit_armed { " press q again to quit (stops all processes) ".to_string() } else { format!(" {hint} ") };
    f.render_widget(Paragraph::new(text).style(Style::new().fg(if app.quit_armed { Color::LightYellow } else { DIM })), area);
}

// ───────────────────────────── Brain ─────────────────────────────

fn brain(f: &mut Frame, area: Rect, app: &App) {
    let cols = Layout::horizontal([Constraint::Percentage(52), Constraint::Min(40)]).split(area);
    let left = Layout::vertical([Constraint::Length(11), Constraint::Min(5)]).split(cols[0]);
    diagram(f, left[0], app);
    connections(f, left[1], app);

    let mut c = vec![Constraint::Length(4); CORTICES.len()];
    c.push(Constraint::Min(3));
    let right = Layout::vertical(c).split(cols[1]);
    for (i, (name, inp, hid, out)) in CORTICES.iter().enumerate() {
        cortex_card(f, right[i], app, name, *inp, *hid, *out);
    }
    thoughts(f, right[CORTICES.len()], app);
}

fn node<'a>(app: &App, name: &'a str) -> Span<'a> {
    let col = cortex_color(name);
    let style = if app.is_active(name) { Style::new().fg(col).bold() } else { Style::new().fg(col).add_modifier(Modifier::DIM) };
    Span::styled(name.to_uppercase(), style)
}

fn diagram(f: &mut Frame, area: Rect, app: &App) {
    let d = Style::new().fg(DIM);
    let lines = vec![
        Line::from(Span::styled("                         ┌──────────────┐", d)),
        Line::from(vec![Span::raw("  CAMERA ─────► "), node(app, "visual"), Span::raw(" ─────► "), node(app, "perceptual"), Span::raw(" ◄─────┐")]),
        Line::from("                                      │        │"),
        Line::from("                         ┌────────────┼────────┐"),
        Line::from("                         ▼            ▼        │"),
        Line::from(vec![Span::raw("                      "), node(app, "memory"), Span::raw(" ◄────► "), node(app, "associative"), Span::raw(" ─┘")]),
        Line::from("                         ▲             ▲  │"),
        Line::from("                         │             │  ▼"),
        Line::from(vec![Span::raw("                      "), node(app, "workspace"), Span::raw("       "), node(app, "language")]),
    ];
    f.render_widget(Paragraph::new(lines).block(panel(title("sparse connectome"))), area);
}

fn connections(f: &mut Frame, area: Rect, app: &App) {
    let mut lines = vec![Line::from(Span::styled(" projection                  weight  traffic    activity", Style::new().bold().fg(Color::Gray)))];
    for c in CONNECTOME {
        let rate = app.edge_rate(c.source, c.target);
        let w = app.edges.get(&(c.source.to_string(), c.target.to_string())).map_or(c.weight, |e| e.last_weight);
        let n = ((rate / 2.0).ceil() as usize).min(12);
        lines.push(Line::from(vec![
            Span::styled(format!(" {:<11} → {:<11}", c.source, c.target), Style::new().fg(heat(rate))),
            Span::raw(format!("  {:>4.2}  {:>6.1}/s   ", w, rate)),
            Span::styled("█".repeat(n), Style::new().fg(heat(rate))),
            Span::styled("·".repeat(12 - n), Style::new().fg(DIM)),
        ]));
    }
    f.render_widget(Paragraph::new(lines).block(panel(title("projections · 3s window"))), area);
}

fn neuron_strip(emb: &[f32]) -> Vec<Span<'static>> {
    const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if emb.is_empty() {
        return vec![Span::styled("· ".repeat(16), Style::new().fg(DIM))];
    }
    emb.iter()
        .flat_map(|v| {
            let m = v.abs().min(1.0);
            let k = (60.0 + 195.0 * m) as u8;
            let col = if *v >= 0.0 { Color::Rgb(k, k / 2, 0) } else { Color::Rgb(0, k / 2, k) };
            [Span::styled(BLOCKS[(m * 7.0) as usize].to_string(), Style::new().fg(col)), Span::raw(" ")]
        })
        .collect()
}

fn cortex_card(f: &mut Frame, area: Rect, app: &App, name: &str, inp: usize, hid: usize, out: usize) {
    let col = cortex_color(name);
    let st = app.cortex.get(name);
    let active = app.is_active(name);
    let border = if active { col } else { DIM };
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(border))
        .title(Line::from(vec![Span::raw(" "), Span::styled(name.to_string(), Style::new().fg(col).bold()), Span::styled(format!("  {inp}→{hid}→{out} "), Style::new().fg(DIM))]));
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height < 2 {
        return;
    }
    let l1 = Line::from(vec![
        Span::styled(format!("{:<8}", st.map_or("—", |s| s.symbol.as_str())), Style::new().fg(Color::White)),
        Span::styled(format!(" act {:.2}  n={}", st.map_or(0.0, |s| s.activation), st.map_or(0, |s| s.count)), Style::new().fg(DIM)),
    ]);
    let mut l2 = vec![Span::styled(format!("{inp:>2} ▶ {hid:>2} ▶ "), Style::new().fg(DIM))];
    l2.extend(neuron_strip(st.map_or(&[][..], |s| &s.embedding)));
    f.render_widget(Paragraph::new(vec![l1, Line::from(l2)]), inner);
}

fn thoughts(f: &mut Frame, area: Rect, app: &App) {
    let h = area.height.saturating_sub(2) as usize;
    let lines: Vec<Line> = app.thoughts.iter().take(h).map(|s| {
        Line::from(vec![
            Span::styled(format!("{:<9}", s.symbol), Style::new().fg(Color::LightRed).bold()),
            Span::styled(format!(" {:.2} ", s.activation), Style::new().fg(DIM)),
            Span::raw(s.context.clone().unwrap_or_default()),
        ])
    }).collect();
    f.render_widget(Paragraph::new(lines).block(panel(title("workspace · what the brain is attending to"))), area);
}

// ───────────────────────────── Chat ─────────────────────────────

fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(8);
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() { return vec![String::new()]; }
    chars.chunks(width).map(|c| c.iter().collect()).collect()
}

fn say_color(kind: &str) -> Color {
    match kind {
        "question" => Color::LightYellow,
        "answer" => Color::LightGreen,
        "notice" => Color::LightCyan,
        "memory" => Color::LightMagenta,
        "prediction" => Color::LightBlue,
        "info" => Color::Gray,
        other => cortex_color(other),
    }
}

fn chat(f: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::vertical([Constraint::Min(3), Constraint::Length(3)]).split(area);
    let cols = Layout::horizontal([Constraint::Min(30), Constraint::Length(38)]).split(rows[0]);

    let inner_w = cols[0].width.saturating_sub(4) as usize;
    let h = cols[0].height.saturating_sub(2) as usize;
    let mut lines: Vec<Line> = vec![];
    for ChatLine { who, text } in &app.chat {
        let (tag, style) = match who {
            Who::You => ("you ▸ ".to_string(), Style::new().fg(Color::LightCyan).bold()),
            Who::Brain(s) => (format!("{s} ▸ "), Style::new().fg(say_color(s)).bold()),
            Who::System => ("· ".to_string(), Style::new().fg(DIM)),
        };
        let pad = tag.chars().count();
        for (i, part) in wrap(text, inner_w.saturating_sub(pad)).into_iter().enumerate() {
            let lead = if i == 0 { Span::styled(tag.clone(), style) } else { Span::raw(" ".repeat(pad)) };
            let body = if matches!(who, Who::System) { Span::styled(part, Style::new().fg(DIM)) } else { Span::raw(part) };
            lines.push(Line::from(vec![lead, body]));
        }
    }
    let skip = lines.len().saturating_sub(h);
    f.render_widget(Paragraph::new(lines.into_iter().skip(skip).collect::<Vec<_>>()).block(panel(title("conversation"))), cols[0]);

    let lang = app.cortex.get("language");
    let side = vec![
        Line::from(Span::styled("language cortex", Style::new().fg(Color::LightBlue).bold())),
        Line::from(neuron_strip(lang.map_or(&[][..], |s| &s.embedding))),
        Line::from(Span::styled(format!("symbol {}", lang.map_or("—", |s| s.symbol.as_str())), Style::new().fg(DIM))),
        answer_line(app),
        Line::from(""),
        Line::from(Span::styled("last associations", Style::new().fg(Color::LightYellow).bold())),
    ];
    let mut side = side;
    side.extend(app.signals.iter().filter(|s| s.kind == "association" || s.kind == "recall").take(6).map(|s| {
        Line::from(Span::styled(format!("{} {}", s.source, s.symbol), Style::new().fg(Color::Gray)))
    }));
    f.render_widget(Paragraph::new(side).block(panel(title("mind"))), cols[1]);

    let input = Paragraph::new(Line::from(vec![Span::styled("▸ ", Style::new().fg(Color::LightCyan).bold()), Span::raw(app.input.clone())]))
        .block(Block::bordered().border_type(BorderType::Rounded).border_style(Style::new().fg(if app.tab == Tab::Chat { Color::LightCyan } else { DIM })).title(title("say something")));
    f.render_widget(input, rows[1]);
    f.set_cursor_position((rows[1].x + 3 + app.input.chars().count() as u16, rows[1].y + 1));
}

// ───────────────────────────── Camera ─────────────────────────────

/// Render a grayscale frame with half-block characters (two pixels per cell), nearest-neighbour scaled.
pub fn frame_lines(frame: &CamFrame, w: usize, h: usize) -> Vec<Line<'static>> {
    if frame.width == 0 || frame.height == 0 || frame.gray.len() < frame.width * frame.height || w == 0 || h == 0 {
        return vec![];
    }
    // Keep aspect: a cell is ~1x2 pixels, so rows*2 pixel rows map onto the frame height.
    let (mut cw, mut ch) = (w, h);
    if cw * frame.height > ch * 2 * frame.width { cw = ch * 2 * frame.width / frame.height; } else { ch = cw * frame.height / (2 * frame.width); }
    let (cw, ch) = (cw.max(1), ch.max(1));
    let px = |x: usize, y: usize| frame.gray[(y * frame.height / (ch * 2)).min(frame.height - 1) * frame.width + (x * frame.width / cw).min(frame.width - 1)];
    (0..ch).map(|cy| {
        Line::from((0..cw).map(|cx| {
            let (t, b) = (px(cx, cy * 2), px(cx, cy * 2 + 1));
            Span::styled("▀", Style::new().fg(Color::Rgb(t, t, t)).bg(Color::Rgb(b, b, b)))
        }).collect::<Vec<_>>())
    }).collect()
}

fn camera(f: &mut Frame, area: Rect, app: &App, sup: &Supervisor) {
    let cols = Layout::horizontal([Constraint::Min(30), Constraint::Length(38)]).split(area);
    let block = panel(title("visual cortex · camera"));
    let inner = block.inner(cols[0]);
    f.render_widget(block, cols[0]);
    let live = app.last_frame.is_some_and(|t| t.elapsed().as_secs() < 2);
    match (&app.frame, live) {
        (Some(fr), true) => {
            let lines = frame_lines(fr, inner.width as usize, inner.height as usize);
            let w = lines.first().map_or(0, |l| l.width() as u16);
            let r = Rect { x: inner.x + (inner.width.saturating_sub(w)) / 2, y: inner.y, width: w.min(inner.width), height: (lines.len() as u16).min(inner.height) };
            f.render_widget(Paragraph::new(lines), r);
        }
        _ => {
            let why = match &sup.procs[1].status {
                Status::Failed(e) => format!("visual node failed: {e}\nCheck camera permission for your terminal (System Settings → Privacy → Camera)\nor set CORTEX_CAMERA=<index>. See Processes tab for its log."),
                Status::Running => "waiting for frames from the visual node…".into(),
                s => format!("visual node: {s:?}"),
            };
            f.render_widget(Paragraph::new(why).style(Style::new().fg(DIM)).wrap(ratatui::widgets::Wrap { trim: true }), inner);
        }
    }
    let v = app.cortex.get("visual");
    let side = vec![
        Line::from(vec![Span::styled(format!("{:.1} fps", app.fps()), Style::new().fg(if live { Color::LightGreen } else { Color::LightRed }).bold())]),
        Line::from(Span::styled(format!("frame {}x{}", app.frame.as_ref().map_or(0, |x| x.width), app.frame.as_ref().map_or(0, |x| x.height)), Style::new().fg(DIM))),
        Line::from(""),
        Line::from(Span::styled("visual embedding (16)", Style::new().fg(Color::LightCyan).bold())),
        Line::from(neuron_strip(v.map_or(&[][..], |s| &s.embedding))),
        Line::from(Span::styled(format!("symbol {}", v.map_or("—", |s| s.symbol.as_str())), Style::new().fg(DIM))),
        Line::from(""),
        answer_line(app),
        Line::from(Span::styled(format!("{}", v.and_then(|s| app.signals.iter().find(|x| x.source == "visual").and_then(|x| x.context.clone()).filter(|_| s.count > 0)).unwrap_or_default()), Style::new().fg(DIM))),
    ];
    f.render_widget(Paragraph::new(side).block(panel(title("signal"))), cols[1]);
}

/// "I see: circle (82%)" for the most recent answer, fading out after a few seconds.
fn answer_line(app: &App) -> Line<'static> {
    match &app.last_answer {
        Some((w, c, t)) if t.elapsed().as_secs() < 8 => Line::from(vec![
            Span::styled("I see: ", Style::new().fg(DIM)),
            Span::styled(w.clone(), Style::new().fg(Color::LightGreen).bold()),
            Span::styled(format!(" ({:.0}%)", c * 100.0), Style::new().fg(DIM)),
        ]),
        _ => Line::from(Span::styled("I see: —", Style::new().fg(DIM))),
    }
}

// ───────────────────────────── Teach ─────────────────────────────

fn teach(f: &mut Frame, area: Rect, app: &App, sup: &Supervisor) {
    let cols = Layout::horizontal([Constraint::Min(40), Constraint::Length(46)]).split(area);
    let left = Layout::vertical([Constraint::Min(6), Constraint::Length(6)]).split(cols[0]);
    let right = Layout::vertical([Constraint::Length(5), Constraint::Length(6), Constraint::Min(8)]).split(cols[1]);
    let running = sup.procs[sup.teacher_index()].is_up();
    let st = app.teacher.clone().unwrap_or_default();

    let block = panel(title("what the brain is shown"));
    let inner = block.inner(left[0]);
    f.render_widget(block, left[0]);
    if let (Some(fr), true) = (&app.frame, running) {
        let lines = frame_lines(fr, inner.width as usize, inner.height as usize);
        let w = lines.first().map_or(0, |l| l.width() as u16);
        let r = Rect { x: inner.x + inner.width.saturating_sub(w) / 2, y: inner.y, width: w.min(inner.width), height: (lines.len() as u16).min(inner.height) };
        f.render_widget(Paragraph::new(lines), r);
    } else {
        f.render_widget(Paragraph::new("Press t to start the automatic teacher.\nIt shows generated shapes with their names, then tests the brain on unseen ones.").style(Style::new().fg(DIM)).wrap(ratatui::widgets::Wrap { trim: true }), inner);
    }

    let verdict = if st.phase == "test" && !st.last_answer.is_empty() {
        let ok = st.last_answer == st.class;
        Span::styled(if ok { "✔ correct" } else { "✘ wrong" }, Style::new().fg(if ok { Color::LightGreen } else { Color::LightRed }).bold())
    } else {
        Span::raw("")
    };
    let info = vec![
        Line::from(vec![Span::styled(format!("{:<8}", st.phase), Style::new().fg(Color::LightCyan).bold()), Span::raw(format!(" class: ")), Span::styled(st.class.clone(), Style::new().bold())]),
        Line::from(vec![Span::raw("brain says: "), Span::styled(if st.last_answer.is_empty() { "—".to_string() } else { st.last_answer.clone() }, Style::new().fg(Color::LightGreen).bold()), Span::styled(format!(" ({:.0}%)  ", st.last_confidence * 100.0), Style::new().fg(DIM)), verdict]),
        Line::from(Span::styled(format!("episode {}   speed x{}   next mode: {}", st.episode, app.teach_speed, if app.teach_camera { format!("camera [{}]", app.teach_words) } else { "synthetic".into() }), Style::new().fg(DIM))),
    ];
    let mut info = info;
    if !st.prompt.is_empty() {
        info.push(Line::from(vec![Span::styled(format!("▶ {}  ", st.prompt), Style::new().fg(Color::Black).bg(Color::LightYellow).bold()), Span::styled(format!(" {:.0}s", st.countdown.ceil()), Style::new().fg(Color::LightYellow).bold())]));
    }
    f.render_widget(Paragraph::new(info).block(panel(title("teacher"))), left[1]);

    let acc = st.accuracy.unwrap_or(0.0);
    f.render_widget(
        Gauge::default().block(panel(title(if running { "accuracy · last test round" } else { "accuracy · teacher stopped" })))
            .gauge_style(Style::new().fg(if acc >= 0.8 { Color::LightGreen } else if acc >= 0.5 { Color::LightYellow } else { Color::LightRed }))
            .ratio(acc as f64).label(st.accuracy.map_or("no test yet".to_string(), |a| format!("{:.0}%", a * 100.0))),
        right[0],
    );
    let hist: Vec<u64> = st.history.iter().map(|a| (a * 100.0) as u64).collect();
    f.render_widget(Sparkline::default().block(panel(title("accuracy per test round"))).data(&hist).max(100).style(Style::new().fg(Color::LightCyan)), right[1]);

    let mut lines = vec![];
    for (name, right_n, total) in &st.per_class {
        let frac = if *total == 0 { 0.0 } else { *right_n as f32 / *total as f32 };
        let n = (frac * 16.0).round() as usize;
        lines.push(Line::from(vec![
            Span::styled(format!(" {name:<9}"), Style::new().fg(Color::Gray)),
            Span::styled("█".repeat(n), Style::new().fg(if frac >= 0.8 { Color::LightGreen } else { Color::LightYellow })),
            Span::styled("·".repeat(16 - n), Style::new().fg(DIM)),
            Span::styled(format!(" {right_n}/{total}"), Style::new().fg(DIM)),
        ]));
    }
    if lines.is_empty() { lines.push(Line::from(Span::styled(" waiting for the first test round…", Style::new().fg(DIM)))); }
    f.render_widget(Paragraph::new(lines).block(panel(title("per class (all tests)"))), right[2]);
}

// ───────────────────────────── Processes / Train ─────────────────────────────

fn status_span(s: &Status) -> Span<'static> {
    match s {
        Status::Running => Span::styled("● running", Style::new().fg(Color::LightGreen)),
        Status::External => Span::styled("● external", Style::new().fg(Color::LightCyan)),
        Status::Stopped => Span::styled("○ stopped", Style::new().fg(DIM)),
        Status::Exited(c) => Span::styled(format!("○ exited {c}"), Style::new().fg(Color::Gray)),
        Status::Failed(_) => Span::styled("✗ failed", Style::new().fg(Color::LightRed)),
    }
}

fn log_view(f: &mut Frame, area: Rect, p: &crate::supervisor::Proc, name: &str) {
    let h = area.height.saturating_sub(2) as usize;
    let skip = p.log.len().saturating_sub(h);
    let mut lines: Vec<Line> = p.log.iter().skip(skip).map(|l| Line::from(l.clone())).collect();
    if let Status::Failed(e) = &p.status {
        lines.push(Line::from(Span::styled(e.clone(), Style::new().fg(Color::LightRed))));
    }
    f.render_widget(Paragraph::new(lines).block(panel(title(&format!("{name} · log")))), area);
}

fn processes(f: &mut Frame, area: Rect, app: &App, sup: &Supervisor) {
    let cols = Layout::horizontal([Constraint::Length(36), Constraint::Min(30)]).split(area);
    let items: Vec<ListItem> = sup.procs.iter().filter(|p| !matches!(p.kind, Kind::Train | Kind::Teacher)).map(|p| {
        ListItem::new(Line::from(vec![Span::styled(format!("{:<13}", p.name), Style::new().fg(if let Kind::Node(n) = p.kind { cortex_color(n) } else { Color::White })), status_span(&p.status)]))
    }).collect();
    let mut state = ListState::default().with_selected(Some(app.selected_proc));
    f.render_stateful_widget(List::new(items).block(panel(title("processes"))).highlight_style(Style::new().bg(Color::Rgb(40, 44, 60)).bold()).highlight_symbol("▸ "), cols[0], &mut state);
    if let Some(p) = sup.procs.get(app.selected_proc) {
        log_view(f, cols[1], p, &p.name);
    }
}

fn train(f: &mut Frame, area: Rect, app: &App, sup: &Supervisor) {
    let cols = Layout::horizontal([Constraint::Length(36), Constraint::Min(30)]).split(area);
    let left = Layout::vertical([Constraint::Length(TRAIN_TARGETS.len() as u16 + 2), Constraint::Min(3)]).split(cols[0]);
    let items: Vec<ListItem> = TRAIN_TARGETS.iter().map(|t| ListItem::new(Line::from(Span::styled(*t, Style::new().fg(cortex_color(t)))))).collect();
    let mut state = ListState::default().with_selected(Some(app.train_sel));
    f.render_stateful_widget(List::new(items).block(panel(title("train cortex"))).highlight_style(Style::new().bg(Color::Rgb(40, 44, 60)).bold()).highlight_symbol("▸ "), left[0], &mut state);
    let p = &sup.procs[sup.train_index()];
    let steps = std::env::var("CORTEX_TRAIN_STEPS").unwrap_or_else(|_| "20000".into());
    let note = vec![
        Line::from(vec![Span::raw("status "), status_span(&p.status)]),
        Line::from(Span::styled(format!("steps {steps} (CORTEX_TRAIN_STEPS)"), Style::new().fg(DIM))),
        Line::from(Span::styled("weights → weights/cortex-0/", Style::new().fg(DIM))),
        Line::from(Span::styled("press R after training to load", Style::new().fg(DIM))),
    ];
    f.render_widget(Paragraph::new(note).block(panel(title("run"))), left[1]);
    log_view(f, cols[1], p, "train");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_renders_half_blocks_with_aspect() {
        let fr = CamFrame { width: 64, height: 32, gray: vec![128; 64 * 32] };
        let lines = frame_lines(&fr, 100, 40);
        assert_eq!((lines.len(), lines[0].width()), (25, 100));
        let small = frame_lines(&fr, 20, 40);
        assert_eq!((small.len(), small[0].width()), (5, 20));
        assert!(frame_lines(&CamFrame { width: 4, height: 4, gray: vec![] }, 10, 10).is_empty());
    }
}
