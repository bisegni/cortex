use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

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
}
impl Signal {
    pub fn new(source:&str,target:&str,kind:&str,symbol:String,activation:f32,connection_weight:f32,embedding:Vec<f32>,context:Option<String>)->Self{
        Self{id:Uuid::new_v4(),source:source.into(),target:target.into(),kind:kind.into(),symbol,activation,connection_weight,embedding,context,
        timestamp_ms:SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis()}
    }
}
pub fn subject(source:&str,target:&str)->String { format!("cortex.{source}.to.{target}") }
pub fn incoming_subject(target:&str)->String { format!("cortex.*.to.{target}") }
pub fn outgoing(source:&str)->impl Iterator<Item=&'static Connection>{CONNECTOME.iter().filter(move|c|c.source==source)}
pub fn cosine(a:&[f32],b:&[f32])->f32{
    let n=a.len().min(b.len()); if n==0{return 0.0}
    let (mut dot,mut aa,mut bb)=(0.0,0.0,0.0);
    for i in 0..n{dot+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i];}
    let d=aa.sqrt()*bb.sqrt(); if d<1e-8{0.0}else{(dot/d).clamp(-1.0,1.0)}
}
