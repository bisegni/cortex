use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

pub mod curriculum;
pub mod intent;

pub const SYMBOL_DIMS: usize = 16;

#[derive(Debug, Clone, Copy)]
pub struct Connection { pub source: &'static str, pub target: &'static str, pub weight: f32 }

pub const CONNECTOME: &[Connection] = &[
    Connection{source:"visual",target:"perceptual",weight:1.0},
    Connection{source:"perceptual",target:"memory",weight:1.0},
    Connection{source:"perceptual",target:"associative",weight:1.0},
    Connection{source:"perceptual",target:"workspace",weight:1.0},
    Connection{source:"memory",target:"perceptual",weight:1.0},
    Connection{source:"memory",target:"associative",weight:1.0},
    Connection{source:"memory",target:"workspace",weight:1.0},
    Connection{source:"associative",target:"perceptual",weight:1.0},
    Connection{source:"associative",target:"memory",weight:1.0},
    Connection{source:"associative",target:"language",weight:1.0},
    Connection{source:"associative",target:"workspace",weight:1.0},
    Connection{source:"language",target:"associative",weight:1.0},
    Connection{source:"workspace",target:"perceptual",weight:1.0},
    Connection{source:"workspace",target:"memory",weight:1.0},
    Connection{source:"workspace",target:"associative",weight:1.0},
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub id: Uuid, pub source: String, pub target: String, pub kind: String,
    pub symbol: String, pub activation: f32, pub connection_weight: f32,
    pub embedding: Vec<f32>, pub context: Option<String>, pub timestamp_ms: u128,
    /// Remaining relay hops: a node that receives ttl 0 must not emit further signals.
    #[serde(default)] pub ttl: u8,
}
impl Signal {
    pub fn new(source:&str,target:&str,kind:&str,symbol:String,activation:f32,connection_weight:f32,embedding:Vec<f32>,context:Option<String>,ttl:u8)->Self{
        Self{id:Uuid::new_v4(),source:source.into(),target:target.into(),kind:kind.into(),symbol,activation,connection_weight,embedding,context,
        timestamp_ms:SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis(),ttl}
    }
}
pub const INPUT_TEXT: &str = "cortex.input.text";
pub const FRAME_SUBJECT: &str = "cortex.visual.frame";
/// Frames published here replace the camera (used by the teacher).
pub const INJECT_SUBJECT: &str = "cortex.visual.inject";
pub const ANSWER_SUBJECT: &str = "cortex.answer";
pub const TEACHER_STATUS_SUBJECT: &str = "cortex.teacher.status";
/// Parsed non-label user intents (`intent::Intent` as JSON).
pub const INTENT_SUBJECT: &str = "cortex.intent";
/// Anything the brain says out loud (replies and initiative).
pub const SAY_SUBJECT: &str = "cortex.say";

/// An utterance. `kind`: answer, question, notice, memory, prediction, info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Say {
    pub text: String, pub kind: String,
    /// The word a question is about ("Is this a cup?" -> "cup"), so a following yes/no can be applied.
    #[serde(default)] pub word: Option<String>,
}
impl Say {
    pub fn new(text: impl Into<String>, kind: &str) -> Self { Self { text: text.into(), kind: kind.into(), word: None } }
    pub fn about(mut self, word: &str) -> Self { self.word = Some(word.into()); self }
}

/// Continuous recognition result from the associative cortex ("unknown" for novel images).
pub const PERCEPT_SUBJECT: &str = "cortex.percept";
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Percept { pub word: String, pub confidence: f32, /// How long this word has been the stable percept.
    pub since_ms: u64 }

/// Sequence expectation or a broken expectation ("surprise").
pub const PREDICTION_SUBJECT: &str = "cortex.prediction";
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// The word this prediction follows from.
    pub from: String,
    /// What is expected next.
    pub next: String,
    pub confidence: f32,
    /// Set when the expectation was broken: what actually appeared.
    pub got: Option<String>,
}

/// What the brain says it sees, published by the language cortex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer { pub word: String, pub confidence: f32 }

/// Teacher progress, published for the dashboard.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TeacherStatus {
    pub phase: String, pub class: String, pub episode: u32,
    /// Accuracy of the latest completed test round (0..1), if any.
    pub accuracy: Option<f32>,
    /// (class, right, total) accumulated over all test rounds.
    pub per_class: Vec<(String, u32, u32)>,
    pub last_answer: String, pub last_confidence: f32,
    /// Accuracy of every completed test round.
    pub history: Vec<f32>,
    /// (rejected, total) for never-taught images the brain should call unknown.
    pub novel: (u32, u32),
    /// What the user should do right now in guided camera mode, e.g. "Show me: cup".
    pub prompt: String,
    /// Seconds left in the current prompt window.
    pub countdown: f32,
    /// Teaching mode: synthetic, camera, images.
    pub mode: String,
}

/// Downsampled grayscale camera frame published by the visual cortex for observers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame { pub width: usize, pub height: usize, pub gray: Vec<u8> }

pub fn subject(source:&str,target:&str)->String { format!("cortex.{source}.to.{target}") }
pub fn incoming_subject(target:&str)->String { format!("cortex.*.to.{target}") }
pub fn outgoing(source:&str)->impl Iterator<Item=&'static Connection>+'_{CONNECTOME.iter().filter(move|c|c.source==source)}
pub fn cosine(a:&[f32],b:&[f32])->f32{
    let n=a.len().min(b.len()); if n==0{return 0.0}
    let (mut dot,mut aa,mut bb)=(0.0,0.0,0.0);
    for i in 0..n{dot+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i];}
    let d=aa.sqrt()*bb.sqrt(); if d<1e-8{0.0}else{(dot/d).clamp(-1.0,1.0)}
}
