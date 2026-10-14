mod app;
mod snapshot;
mod supervisor;
mod ui;

use anyhow::{Context, Result};
use app::{App, Tab, Who, TRAIN_TARGETS};
use cortex_core::{intent::Intent, Answer, BrainState, Frame, Metrics, Percept, Prediction, Say, Signal, TeacherStatus, ANSWER_SUBJECT, BRAIN_STATE_SUBJECT, FRAME_SUBJECT, INPUT_TEXT, INTENT_SUBJECT, METRICS_SUBJECT, PERCEPT_SUBJECT, PREDICTION_SUBJECT, SAY_SUBJECT, TEACHER_STATUS_SUBJECT};
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use futures::StreamExt;
use std::time::Duration;
use supervisor::{Kind, Supervisor};

#[tokio::main]
async fn main() -> Result<()> {
    let mut sup = Supervisor::new();
    eprintln!("cortex: starting nats-server and cortex nodes…");
    sup.start_all().await;
    let url = std::env::var("NATS_URL").unwrap_or_else(|_| "127.0.0.1:4222".into());
    // The connectome loops generate far more signals than the UI can paint; a deep queue plus a
    // dedicated frame subscription keeps rare camera frames from being dropped as slow-consumer overflow.
    let nc = match async_nats::ConnectOptions::new().subscription_capacity(65536).connect(&url).await {
        Ok(nc) => nc,
        Err(e) => {
            sup.shutdown().await;
            return Err(e).context(format!("cannot reach NATS at {url} (is nats-server installed? brew install nats-server)"));
        }
    };
    let mut sub = nc.subscribe("cortex.>").await?;
    let mut frames = nc.subscribe(FRAME_SUBJECT).await?;
    let mut answers = nc.subscribe(ANSWER_SUBJECT).await?;
    let mut teacher = nc.subscribe(TEACHER_STATUS_SUBJECT).await?;
    let mut says = nc.subscribe(SAY_SUBJECT).await?;
    let mut percepts = nc.subscribe(PERCEPT_SUBJECT).await?;
    let mut metrics = nc.subscribe(METRICS_SUBJECT).await?;
    let mut brain = nc.subscribe(BRAIN_STATE_SUBJECT).await?;
    let mut predictions = nc.subscribe(PREDICTION_SUBJECT).await?;

    // ratatui::init installs a panic hook that restores the terminal.
    let mut term = ratatui::init();
    let result = run(&mut term, &mut sup, &nc, Feeds { signals: &mut sub, frames: &mut frames, answers: &mut answers, teacher: &mut teacher, says: &mut says, percepts: &mut percepts, predictions: &mut predictions, metrics: &mut metrics, brain: &mut brain }).await;
    ratatui::restore();
    sup.shutdown().await;
    result
}

struct Feeds<'a> {
    signals: &'a mut async_nats::Subscriber,
    frames: &'a mut async_nats::Subscriber,
    answers: &'a mut async_nats::Subscriber,
    teacher: &'a mut async_nats::Subscriber,
    says: &'a mut async_nats::Subscriber,
    percepts: &'a mut async_nats::Subscriber,
    predictions: &'a mut async_nats::Subscriber,
    metrics: &'a mut async_nats::Subscriber,
    brain: &'a mut async_nats::Subscriber,
}

async fn run(term: &mut ratatui::DefaultTerminal, sup: &mut Supervisor, nc: &async_nats::Client, feeds: Feeds<'_>) -> Result<()> {
    let Feeds { signals: sub, frames, answers, teacher, says, percepts, predictions, metrics, brain } = feeds;
    let mut app = App::new();
    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(50));
    loop {
        tokio::select! {
            _ = tick.tick() => {
                sup.poll();
                app.tick();
                term.draw(|f| ui::draw(f, &app, sup))?;
            }
            msg = sub.next() => {
                let Some(msg) = msg else { break };
                if msg.subject.as_str() == FRAME_SUBJECT { continue }
                if let Ok(s) = serde_json::from_slice::<Signal>(&msg.payload) {
                    app.on_signal(s);
                }
            }
            msg = frames.next() => {
                let Some(msg) = msg else { break };
                if let Ok(fr) = serde_json::from_slice::<Frame>(&msg.payload) { app.on_frame(fr); }
            }
            msg = answers.next() => {
                let Some(msg) = msg else { break };
                if let Ok(a) = serde_json::from_slice::<Answer>(&msg.payload) { app.on_answer(a); }
            }
            msg = metrics.next() => {
                let Some(msg) = msg else { break };
                if let Ok(x) = serde_json::from_slice::<Metrics>(&msg.payload) { app.on_metrics(x); }
            }
            msg = brain.next() => {
                let Some(msg) = msg else { break };
                if let Ok(x) = serde_json::from_slice::<BrainState>(&msg.payload) { app.on_brain_state(x); }
            }
            msg = percepts.next() => {
                let Some(msg) = msg else { break };
                if let Ok(p) = serde_json::from_slice::<Percept>(&msg.payload) { app.on_percept(p); }
            }
            msg = predictions.next() => {
                let Some(msg) = msg else { break };
                if let Ok(p) = serde_json::from_slice::<Prediction>(&msg.payload) { app.on_prediction(p); }
            }
            msg = says.next() => {
                let Some(msg) = msg else { break };
                if let Ok(t) = serde_json::from_slice::<Say>(&msg.payload) { app.on_say(t); }
            }
            msg = teacher.next() => {
                let Some(msg) = msg else { break };
                if let Ok(t) = serde_json::from_slice::<TeacherStatus>(&msg.payload) { app.on_teacher(t); }
            }
            ev = events.next() => {
                let Some(ev) = ev else { break };
                if let Event::Key(k) = ev? {
                    if k.kind != KeyEventKind::Release && handle_key(&mut app, sup, nc, k).await { break; }
                }
            }
        }
    }
    Ok(())
}

/// Returns true when the dashboard should exit.
async fn handle_key(app: &mut App, sup: &mut Supervisor, nc: &async_nats::Client, k: KeyEvent) -> bool {
    if k.modifiers.contains(KeyModifiers::CONTROL) && k.code == KeyCode::Char('c') {
        return true;
    }
    match k.code {
        KeyCode::Tab => { app.tab = app.tab.step(1); return false; }
        KeyCode::BackTab => { app.tab = app.tab.step(-1); return false; }
        KeyCode::F(n @ 1..=7) => { app.tab = Tab::ALL[n as usize - 1]; return false; }
        _ => {}
    }
    if app.tab == Tab::Chat {
        match k.code {
            KeyCode::Esc => app.tab = Tab::Brain,
            KeyCode::Backspace => { app.input.pop(); }
            KeyCode::Char(c) => app.input.push(c),
            KeyCode::Enter => {
                let text = std::mem::take(&mut app.input);
                let text = text.trim().to_string();
                if !text.is_empty() {
                    if !sup.procs.iter().any(|p| p.kind == Kind::Node("language") && p.is_up()) {
                        app.say(Who::System, "language node is not running (Processes tab) — nobody is listening".into());
                    }
                    app.say(Who::You, text.clone());
                    if nc.publish(INPUT_TEXT, text.into()).await.is_err() {
                        app.say(Who::System, "publish failed: NATS disconnected".into());
                    }
                }
            }
            _ => {}
        }
        return false;
    }
    let was_armed = std::mem::take(&mut app.quit_armed);
    match k.code {
        KeyCode::Char('q') => {
            if was_armed { return true; }
            app.quit_armed = true;
        }
        KeyCode::Char(c @ '1'..='7') => app.tab = Tab::ALL[c as usize - '1' as usize],
        KeyCode::Char('R') => sup.restart_nodes().await,
        KeyCode::Char('j') | KeyCode::Down => match app.tab {
            Tab::Processes => app.selected_proc = (app.selected_proc + 1).min(sup.train_index() - 1),
            Tab::Train => app.train_sel = (app.train_sel + 1).min(TRAIN_TARGETS.len() - 1),
            Tab::Learn => app.learn_sel += 1,
            _ => {}
        },
        KeyCode::Char('k') | KeyCode::Up => match app.tab {
            Tab::Processes => app.selected_proc = app.selected_proc.saturating_sub(1),
            Tab::Train => app.train_sel = app.train_sel.saturating_sub(1),
            Tab::Learn => app.learn_sel = app.learn_sel.saturating_sub(1),
            _ => {}
        },
        KeyCode::Enter => match app.tab {
            Tab::Processes => sup.start(app.selected_proc, vec![]).await,
            Tab::Train => {
                let i = sup.train_index();
                sup.start(i, vec![TRAIN_TARGETS[app.train_sel].to_string()]).await;
            }
            _ => {}
        },
        KeyCode::Char('x') if app.tab == Tab::Processes => sup.stop(app.selected_proc),
        KeyCode::Char('t') if app.tab == Tab::Teach => {
            let i = sup.teacher_index();
            if sup.procs[i].is_up() {
                sup.stop(i);
            } else {
                let args = if app.teach_camera {
                    vec!["--mode".into(), "camera".into(), "--words".into(), app.teach_words.clone()]
                } else {
                    vec!["--speed".into(), app.teach_speed.to_string()]
                };
                sup.start(i, args).await;
            }
        }
        KeyCode::Char('d') if app.tab == Tab::Learn => {
            if let Some(w) = app.selected_word().map(str::to_string) {
                let _ = nc.publish(INTENT_SUBJECT, serde_json::to_vec(&Intent::Forget(w.clone())).unwrap_or_default().into()).await;
                app.learn_note = format!("forgot {w}");
            }
        }
        KeyCode::Char('e') if app.tab == Tab::Learn => app.learn_note = match snapshot::export(sup.root()) {
            Ok(p) => format!("exported to {}", p.display()),
            Err(e) => format!("export failed: {e}"),
        },
        KeyCode::Char('i') if app.tab == Tab::Learn => match snapshot::import_latest(sup.root()) {
            Ok(p) => { sup.restart_nodes().await; app.learn_note = format!("imported {} and restarted nodes", p.display()); }
            Err(e) => app.learn_note = format!("import failed: {e}"),
        },
        KeyCode::Char('g') if app.tab == Tab::Teach => app.teach_camera = !app.teach_camera,
        KeyCode::Char('s') if app.tab == Tab::Teach => app.teach_speed = match app.teach_speed { 1 => 2, 2 => 4, _ => 1 },
        _ => {}
    }
    false
}
