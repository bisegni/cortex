//! Word-level sequence learning: first-order transition counts, expectation and surprise.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
pub enum SeqEvent {
    /// The stable percept changed.
    Changed { from: Option<String>, to: String },
    /// After `from`, `next` is expected with this probability.
    Expect { from: String, next: String, confidence: f32 },
    /// The expectation was wrong.
    Surprise { from: String, expected: String, got: String, confidence: f32 },
}

#[derive(Default, Serialize, Deserialize, Clone)]
pub struct Sequence {
    transitions: HashMap<String, HashMap<String, f32>>,
    #[serde(skip)] current: Option<String>,
    #[serde(skip)] pending: Option<(String, u32)>,
    #[serde(skip)] expected: Option<(String, String, f32)>,
}

/// Consecutive observations needed to accept a new percept (rejects flicker).
const CONFIRM: u32 = 2;
const MIN_CONFIDENCE: f32 = 0.5;
const MIN_COUNT: f32 = 2.0;

impl Sequence {
    /// Most likely successor of `word` and its probability, if enough evidence exists.
    pub fn expect(&self, word: &str) -> Option<(String, f32)> {
        let next = self.transitions.get(word)?;
        let total: f32 = next.values().sum();
        let (w, c) = next.iter().max_by(|a, b| a.1.total_cmp(b.1))?;
        (total >= MIN_COUNT && c / total >= MIN_CONFIDENCE).then(|| (w.clone(), c / total))
    }

    pub fn current(&self) -> Option<&str> { self.current.as_deref() }

    pub fn forget(&mut self, word: &str) {
        self.transitions.remove(word);
        for next in self.transitions.values_mut() { next.remove(word); }
        if self.current.as_deref() == Some(word) { self.current = None; }
    }

    /// Strongest transitions, for display: (from, to, count).
    pub fn top(&self, n: usize) -> Vec<(String, String, f32)> {
        let mut all: Vec<_> = self.transitions.iter().flat_map(|(f, m)| m.iter().map(move |(t, c)| (f.clone(), t.clone(), *c))).collect();
        all.sort_by(|a, b| b.2.total_cmp(&a.2));
        all.truncate(n);
        all
    }

    /// Feed one recognition per frame; "unknown" never changes the sequence state.
    pub fn observe(&mut self, word: &str) -> Vec<SeqEvent> {
        if word == "unknown" { return vec![]; }
        if self.current.as_deref() == Some(word) { self.pending = None; return vec![]; }
        let count = match &mut self.pending {
            Some((w, n)) if w == word => { *n += 1; *n }
            _ => { self.pending = Some((word.to_string(), 1)); 1 }
        };
        if count < CONFIRM { return vec![]; }
        self.pending = None;
        let from = self.current.replace(word.to_string());
        let mut events = vec![SeqEvent::Changed { from: from.clone(), to: word.to_string() }];
        if let Some(f) = &from { *self.transitions.entry(f.clone()).or_default().entry(word.to_string()).or_default() += 1.0; }
        if let Some((from_w, exp, conf)) = self.expected.take() {
            if exp != word { events.push(SeqEvent::Surprise { from: from_w, expected: exp, got: word.to_string(), confidence: conf }); }
        }
        if let Some((next, confidence)) = self.expect(word) {
            self.expected = Some((word.to_string(), next.clone(), confidence));
            events.push(SeqEvent::Expect { from: word.to_string(), next, confidence });
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(s: &mut Sequence, words: &[&str]) -> Vec<SeqEvent> {
        words.iter().flat_map(|w| s.observe(w)).collect()
    }

    #[test]
    fn needs_two_consecutive_frames_to_change() {
        let mut s = Sequence::default();
        assert!(feed(&mut s, &["a"]).is_empty());
        assert_eq!(s.observe("a"), vec![SeqEvent::Changed { from: None, to: "a".into() }]);
        assert!(s.observe("b").is_empty());
        assert!(s.observe("a").is_empty(), "flicker back to the current word cancels");
        assert_eq!(s.current(), Some("a"));
    }

    #[test]
    fn learns_cycle_expects_and_is_surprised() {
        let mut s = Sequence::default();
        for _ in 0..4 { feed(&mut s, &["a", "a", "b", "b", "c", "c"]); }
        assert_eq!(s.expect("a").map(|x| x.0), Some("b".into()));
        assert_eq!(s.expect("c").map(|x| x.0), Some("a".into()));
        let ev = feed(&mut s, &["a", "a"]);
        assert!(ev.iter().any(|e| matches!(e, SeqEvent::Expect { next, .. } if next == "b")));
        let ev = feed(&mut s, &["c", "c"]);
        assert!(ev.iter().any(|e| matches!(e, SeqEvent::Surprise { expected, got, .. } if expected == "b" && got == "c")), "{ev:?}");
    }

    #[test]
    fn unknown_is_ignored_and_forget_cleans() {
        let mut s = Sequence::default();
        feed(&mut s, &["a", "a", "unknown", "unknown", "b", "b"]);
        assert_eq!(s.current(), Some("b"));
        s.forget("b");
        assert!(s.top(5).iter().all(|(f, t, _)| f != "b" && t != "b"));
    }
}
