//! Initiative: turns what the brain perceives, expects and is unsure about into things it says.
use cortex_core::{Percept, Prediction, Say};
use std::{collections::HashMap, time::{Duration, Instant}};

pub struct Drive {
    pub proactive: bool,
    cooldowns: HashMap<String, Instant>,
}

impl Drive {
    pub fn new(proactive: bool) -> Self { Self { proactive, cooldowns: HashMap::new() } }

    /// True (and starts the cooldown) if `key` has not fired within `secs`.
    fn ready(&mut self, key: &str, secs: u64, now: Instant) -> bool {
        if self.cooldowns.get(key).is_some_and(|t| now.duration_since(*t) < Duration::from_secs(secs)) { return false; }
        self.cooldowns.insert(key.to_string(), now);
        true
    }

    pub fn on_percept(&mut self, p: &Percept, now: Instant) -> Vec<Say> {
        if !self.proactive { return vec![]; }
        let stable = Duration::from_millis(p.since_ms);
        if p.word == "unknown" {
            // Curiosity: something novel stays in view.
            if stable >= Duration::from_secs(2) && self.ready("curious", 30, now) { return vec![Say::new("What is this?", "question")]; }
        } else if p.confidence >= 0.6 {
            // Reaction: a known thing appears.
            if stable >= Duration::from_millis(500) && self.ready(&format!("notice:{}", p.word), 20, now) {
                return vec![Say::new(format!("I see {}.", p.word), "notice").about(&p.word)];
            }
        } else if stable >= Duration::from_millis(1500) && self.ready("confirm", 20, now) {
            // Unsure: ask for confirmation so a yes/no can sharpen the memory.
            return vec![Say::new(format!("Is this {}?", p.word), "question").about(&p.word)];
        }
        vec![]
    }

    pub fn on_prediction(&mut self, pr: &Prediction, now: Instant) -> Vec<Say> {
        if !self.proactive { return vec![]; }
        if let Some(got) = &pr.got {
            if self.ready("surprise", 5, now) { return vec![Say::new(format!("I expected {} but saw {}.", pr.next, got), "prediction")]; }
        } else if pr.confidence >= 0.7 && self.ready(&format!("expect:{}", pr.from), 30, now) {
            return vec![Say::new(format!("After {} I expect {}.", pr.from, pr.next), "prediction")];
        }
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn percept(word: &str, confidence: f32, since_ms: u64) -> Percept { Percept { word: word.into(), confidence, since_ms } }

    #[test]
    fn asks_about_novelty_once_after_two_seconds() {
        let (mut d, t) = (Drive::new(true), Instant::now());
        assert!(d.on_percept(&percept("unknown", 0.9, 1000), t).is_empty());
        let said = d.on_percept(&percept("unknown", 0.9, 2100), t);
        assert_eq!((said[0].kind.as_str(), said[0].text.as_str()), ("question", "What is this?"));
        assert!(d.on_percept(&percept("unknown", 0.9, 3000), t).is_empty(), "cooldown");
    }

    #[test]
    fn notices_known_words_with_cooldown_per_word() {
        let (mut d, t) = (Drive::new(true), Instant::now());
        assert_eq!(d.on_percept(&percept("cup", 0.8, 600), t)[0].text, "I see cup.");
        assert!(d.on_percept(&percept("cup", 0.8, 900), t).is_empty());
        assert_eq!(d.on_percept(&percept("hand", 0.8, 600), t).len(), 1, "other words are independent");
    }

    #[test]
    fn unsure_percept_asks_for_confirmation_about_that_word() {
        let (mut d, t) = (Drive::new(true), Instant::now());
        let said = d.on_percept(&percept("cup", 0.5, 1600), t);
        assert_eq!((said[0].text.as_str(), said[0].word.as_deref()), ("Is this cup?", Some("cup")));
    }

    #[test]
    fn surprise_and_expectation() {
        let (mut d, t) = (Drive::new(true), Instant::now());
        let broken = Prediction { from: "a".into(), next: "b".into(), confidence: 0.9, got: Some("c".into()) };
        assert_eq!(d.on_prediction(&broken, t)[0].text, "I expected b but saw c.");
        let expect = Prediction { from: "a".into(), next: "b".into(), confidence: 0.9, got: None };
        assert_eq!(d.on_prediction(&expect, t)[0].text, "After a I expect b.");
        assert!(d.on_prediction(&Prediction { confidence: 0.4, ..expect }, t).is_empty());
    }

    #[test]
    fn proactive_off_is_silent() {
        let (mut d, t) = (Drive::new(false), Instant::now());
        assert!(d.on_percept(&percept("unknown", 1.0, 5000), t).is_empty());
        assert!(d.on_percept(&percept("cup", 1.0, 5000), t).is_empty());
    }
}
