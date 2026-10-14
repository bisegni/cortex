//! Pure dashboard state, fed by NATS signals and camera frames.
use cortex_core::{Answer, Frame, Percept, Prediction, Say, Signal, TeacherStatus};
use std::{collections::{HashMap, VecDeque}, time::{Duration, Instant}};

/// (name, input, hidden, output) of every cortex; matches the geometry in cortex-node.
pub const CORTICES: &[(&str, usize, usize, usize)] = &[
    ("visual", 128, 64, 16),
    ("perceptual", 16, 24, 16),
    ("memory", 16, 32, 16),
    ("associative", 32, 32, 16),
    ("language", 32, 24, 16),
    ("workspace", 16, 16, 16),
];
pub const TRAIN_TARGETS: &[&str] = &["curriculum", "all", "visual", "perceptual", "memory", "associative", "language"];
const RATE_WINDOW: Duration = Duration::from_secs(3);

#[derive(Clone, Copy, PartialEq)]
pub enum Tab { Brain, Chat, Camera, Teach, Processes, Train }
impl Tab {
    pub const ALL: [Tab; 6] = [Tab::Brain, Tab::Chat, Tab::Camera, Tab::Teach, Tab::Processes, Tab::Train];
    pub fn title(self) -> &'static str {
        match self { Tab::Brain => "Brain", Tab::Chat => "Chat", Tab::Camera => "Camera", Tab::Teach => "Teach", Tab::Processes => "Processes", Tab::Train => "Train" }
    }
    pub fn index(self) -> usize { Tab::ALL.iter().position(|t| *t == self).unwrap_or(0) }
    pub fn step(self, d: isize) -> Tab { Tab::ALL[(self.index() as isize + d).rem_euclid(6) as usize] }
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
    pub teacher: Option<TeacherStatus>,
    pub teacher_seen: Option<Instant>,
    pub last_answer: Option<(String, f32, Instant)>,
    pub teach_speed: u32,
    pub said: VecDeque<Say>,
    pub percept: Option<(String, f32, Instant)>,
    /// (from, expected next, confidence, when) — cleared by a surprise.
    pub expecting: Option<(String, String, f32, Instant)>,
    /// Teach tab: false = generated shapes, true = guided camera objects.
    pub teach_camera: bool,
    pub teach_words: String,
}

impl App {
    pub fn new() -> Self {
        Self {
            tab: Tab::Brain, edges: HashMap::new(), cortex: HashMap::new(), signals: VecDeque::new(), thoughts: VecDeque::new(),
            chat: vec![ChatLine { who: Who::System, text: "Show the camera something and say what it is ('this is a cup'). Ask 'what is this?', 'what did you see?' or 'what comes next?'. Answer yes/no when it asks.".into() }],
            input: String::new(), frame: None, frame_times: VecDeque::new(), last_frame: None, msg_times: VecDeque::new(),
            selected_proc: 0, train_sel: 0, started: Instant::now(), quit_armed: false, teacher: None, teacher_seen: None, last_answer: None, teach_speed: 1, said: VecDeque::new(), percept: None, expecting: None, teach_camera: false, teach_words: std::env::var("CORTEX_WORDS").unwrap_or_else(|_| "hand,cup".into()),
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
        push_cap(&mut self.signals, s, 300);
    }

    /// Spontaneous recognition; chat only shows what the brain chooses to say (`on_say`).
    pub fn on_answer(&mut self, a: Answer) {
        let now = Instant::now();
        self.last_answer = Some((a.word, a.confidence, now));
    }

    /// Something the brain said on its own initiative or in reply to a question.
    pub fn on_say(&mut self, s: Say) {
        self.chat.push(ChatLine { who: Who::Brain(s.kind.clone()), text: s.text.clone() });
        if self.chat.len() > 500 { self.chat.remove(0); }
        push_cap(&mut self.said, s, 20);
    }

    pub fn on_percept(&mut self, p: Percept) { self.percept = Some((p.word, p.confidence, Instant::now())); }

    pub fn on_prediction(&mut self, p: Prediction) {
        self.expecting = if p.got.is_some() { None } else { Some((p.from, p.next, p.confidence, Instant::now())) };
    }

    pub fn on_teacher(&mut self, t: TeacherStatus) {
        self.teacher = Some(t);
        self.teacher_seen = Some(Instant::now());
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
        Signal::new(src, dst, kind, sym.into(), 1.0, 1.0, vec![0.5; 16], Some("ctx".into()), 3)
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
    fn signals_and_answers_stay_out_of_the_chat() {
        let mut a = App::new();
        let base = a.chat.len();
        a.on_signal(sig("workspace", "memory", "broadcast", "P1"));
        a.on_answer(Answer { word: "square".into(), confidence: 0.5 });
        assert_eq!(a.chat.len(), base);
        assert_eq!(a.last_answer.as_ref().map(|x| x.0.as_str()), Some("square"));
        assert_eq!(a.thoughts.len(), 1);
    }

    #[test]
    fn say_lands_in_chat_and_inner_voice() {
        let mut a = App::new();
        let base = a.chat.len();
        a.on_say(Say::new("What is this?", "question"));
        assert_eq!(a.chat.len(), base + 1);
        assert_eq!(a.said.front().map(|s| s.kind.as_str()), Some("question"));
    }

    #[test]
    fn percept_and_expectation_track_latest() {
        let mut a = App::new();
        a.on_percept(Percept { word: "cup".into(), confidence: 0.9, since_ms: 10 });
        assert_eq!(a.percept.as_ref().map(|p| p.0.as_str()), Some("cup"));
        a.on_prediction(Prediction { from: "a".into(), next: "b".into(), confidence: 0.8, got: None });
        assert!(a.expecting.is_some());
        a.on_prediction(Prediction { from: "a".into(), next: "b".into(), confidence: 0.8, got: Some("c".into()) });
        assert!(a.expecting.is_none());
    }

    #[test]
    fn tabs_wrap() {
        assert!(Tab::Brain.step(-1) == Tab::Train);
        assert!(Tab::Train.step(1) == Tab::Brain);
    }
}
