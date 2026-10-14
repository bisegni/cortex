//! Pure dashboard state, fed by NATS signals and camera frames.
use cortex_core::{Frame, Signal};
use std::{collections::{HashMap, VecDeque}, time::{Duration, Instant}};

/// (name, input, hidden, output) of every cortex; matches the geometry in cortex-node.
pub const CORTICES: &[(&str, usize, usize, usize)] = &[
    ("visual", 64, 48, 16),
    ("perceptual", 16, 24, 16),
    ("memory", 16, 32, 16),
    ("associative", 32, 32, 16),
    ("language", 32, 24, 16),
    ("workspace", 16, 16, 16),
];
pub const TRAIN_TARGETS: &[&str] = &["all", "visual", "perceptual", "memory", "associative", "language"];
const RATE_WINDOW: Duration = Duration::from_secs(3);

#[derive(Clone, Copy, PartialEq)]
pub enum Tab { Brain, Chat, Camera, Processes, Train }
impl Tab {
    pub const ALL: [Tab; 5] = [Tab::Brain, Tab::Chat, Tab::Camera, Tab::Processes, Tab::Train];
    pub fn title(self) -> &'static str {
        match self { Tab::Brain => "Brain", Tab::Chat => "Chat", Tab::Camera => "Camera", Tab::Processes => "Processes", Tab::Train => "Train" }
    }
    pub fn index(self) -> usize { Tab::ALL.iter().position(|t| *t == self).unwrap_or(0) }
    pub fn step(self, d: isize) -> Tab { Tab::ALL[(self.index() as isize + d).rem_euclid(5) as usize] }
}

pub enum Who { You, Brain(String), System }
pub struct ChatLine { pub who: Who, pub text: String }

#[derive(Default)]
pub struct CortexState {
    pub embedding: Vec<f32>,
    pub symbol: String,
    pub activation: f32,
    pub last: Option<Instant>,
    pub count: u64,
}

pub struct EdgeActivity { pub events: VecDeque<Instant>, pub last_weight: f32 }

pub struct App {
    pub tab: Tab,
    pub edges: HashMap<(String, String), EdgeActivity>,
    pub cortex: HashMap<String, CortexState>,
    pub signals: VecDeque<Signal>,
    pub thoughts: VecDeque<Signal>,
    pub chat: Vec<ChatLine>,
    pub input: String,
    pub frame: Option<Frame>,
    pub frame_times: VecDeque<Instant>,
    pub last_frame: Option<Instant>,
    pub msg_times: VecDeque<Instant>,
    pub selected_proc: usize,
    pub train_sel: usize,
    pub started: Instant,
    pub quit_armed: bool,
    last_chat: HashMap<String, (String, Instant)>,
}

impl App {
    pub fn new() -> Self {
        Self {
            tab: Tab::Brain, edges: HashMap::new(), cortex: HashMap::new(), signals: VecDeque::new(), thoughts: VecDeque::new(),
            chat: vec![ChatLine { who: Who::System, text: "Type a sentence and press Enter. The language cortex hears it; replies are what the brain broadcasts, recalls or associates.".into() }],
            input: String::new(), frame: None, frame_times: VecDeque::new(), last_frame: None, msg_times: VecDeque::new(),
            selected_proc: 0, train_sel: 0, started: Instant::now(), quit_armed: false, last_chat: HashMap::new(),
        }
    }

    pub fn on_signal(&mut self, s: Signal) {
        let now = Instant::now();
        self.msg_times.push_back(now);
        let e = self.edges.entry((s.source.clone(), s.target.clone())).or_insert(EdgeActivity { events: VecDeque::new(), last_weight: s.connection_weight });
        e.events.push_back(now);
        e.last_weight = s.connection_weight;
        let c = self.cortex.entry(s.source.clone()).or_default();
        c.embedding = s.embedding.clone();
        c.symbol = s.symbol.clone();
        c.activation = s.activation;
        c.last = Some(now);
        c.count += 1;
        if s.kind == "broadcast" {
            push_cap(&mut self.thoughts, s.clone(), 60);
        }
        if matches!(s.kind.as_str(), "broadcast" | "recall" | "association") {
            self.maybe_reply(&s, now);
        }
        push_cap(&mut self.signals, s, 300);
    }

    /// Surface brain output in the chat transcript without flooding it.
    fn maybe_reply(&mut self, s: &Signal, now: Instant) {
        let prev = self.last_chat.get(&s.source);
        let stale = prev.map_or(true, |(sym, t)| (sym != &s.symbol && now.duration_since(*t) > Duration::from_millis(800)) || now.duration_since(*t) > Duration::from_secs(8));
        if !stale {
            return;
        }
        self.last_chat.insert(s.source.clone(), (s.symbol.clone(), now));
        let ctx = s.context.as_deref().unwrap_or("");
        self.chat.push(ChatLine { who: Who::Brain(s.source.clone()), text: format!("{} [{}]  {}", s.symbol, s.kind, ctx) });
        if self.chat.len() > 500 {
            self.chat.remove(0);
        }
    }

    pub fn on_frame(&mut self, f: Frame) {
        let now = Instant::now();
        self.frame = Some(f);
        self.last_frame = Some(now);
        self.frame_times.push_back(now);
    }

    pub fn tick(&mut self) {
        let now = Instant::now();
        for e in self.edges.values_mut() {
            while e.events.front().is_some_and(|t| now.duration_since(*t) > RATE_WINDOW) { e.events.pop_front(); }
        }
        while self.msg_times.front().is_some_and(|t| now.duration_since(*t) > Duration::from_secs(1)) { self.msg_times.pop_front(); }
        while self.frame_times.front().is_some_and(|t| now.duration_since(*t) > Duration::from_secs(2)) { self.frame_times.pop_front(); }
    }

    pub fn edge_rate(&self, src: &str, dst: &str) -> f32 {
        self.edges.get(&(src.to_string(), dst.to_string())).map_or(0.0, |e| e.events.len() as f32 / RATE_WINDOW.as_secs_f32())
    }
    pub fn msg_rate(&self) -> usize { self.msg_times.len() }
    pub fn fps(&self) -> f32 { self.frame_times.len() as f32 / 2.0 }
    pub fn is_active(&self, cortex: &str) -> bool {
        self.cortex.get(cortex).and_then(|c| c.last).is_some_and(|t| t.elapsed() < Duration::from_millis(1500))
    }
    pub fn say(&mut self, who: Who, text: String) { self.chat.push(ChatLine { who, text }); }
}

fn push_cap<T>(q: &mut VecDeque<T>, v: T, cap: usize) {
    q.push_front(v);
    if q.len() > cap { q.pop_back(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(src: &str, dst: &str, kind: &str, sym: &str) -> Signal {
        Signal::new(src, dst, kind, sym.into(), 1.0, 1.0, vec![0.5; 16], Some("ctx".into()))
    }

    #[test]
    fn edge_rate_counts_recent_events() {
        let mut a = App::new();
        for _ in 0..6 { a.on_signal(sig("visual", "perceptual", "representation", "V1")); }
        assert_eq!(a.edge_rate("visual", "perceptual"), 2.0);
        assert_eq!(a.edge_rate("memory", "workspace"), 0.0);
        assert!(a.is_active("visual"));
    }

    #[test]
    fn chat_reply_is_throttled_per_source_symbol() {
        let mut a = App::new();
        let base = a.chat.len();
        a.on_signal(sig("workspace", "memory", "broadcast", "P1"));
        a.on_signal(sig("workspace", "memory", "broadcast", "P1"));
        a.on_signal(sig("workspace", "memory", "broadcast", "P2"));
        assert_eq!(a.chat.len(), base + 1);
        a.on_signal(sig("visual", "perceptual", "representation", "V1"));
        assert_eq!(a.chat.len(), base + 1);
    }

    #[test]
    fn tabs_wrap() {
        assert!(Tab::Brain.step(-1) == Tab::Train);
        assert!(Tab::Train.step(1) == Tab::Brain);
    }
}
