use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub id: Uuid,
    pub source: String,
    pub kind: String,
    pub symbol: String,
    pub activation: f32,
    pub embedding: Vec<f32>,
    pub context: Option<String>,
    pub timestamp_ms: u128,
}

impl Signal {
    pub fn new(source: &str, kind: &str, symbol: String, activation: f32, embedding: Vec<f32>, context: Option<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            source: source.into(),
            kind: kind.into(),
            symbol,
            activation,
            embedding,
            context,
            timestamp_ms: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis(),
        }
    }
}

pub fn subject(cortex: &str) -> String { format!("cortex.{cortex}.signal") }

pub fn hash_embedding(bytes: &[u8], dims: usize) -> Vec<f32> {
    let mut out = vec![0.0; dims];
    for (i, b) in bytes.iter().enumerate() {
        let j = i % dims;
        out[j] += (*b as f32 / 127.5) - 1.0;
    }
    let n = out.iter().map(|x| x*x).sum::<f32>().sqrt().max(1e-6);
    out.iter_mut().for_each(|x| *x /= n);
    out
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 { return 0.0; }
    a[..n].iter().zip(&b[..n]).map(|(x,y)| x*y).sum()
}
