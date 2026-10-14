//! Tiny rule-based reader for what the user typed to the brain.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Intent {
    /// "this is a cup" / "cup": teach this name for what the camera sees.
    Label(String),
    WhatIsThis,
    WhatSeen,
    WhatNext,
    Yes,
    No,
    Forget(String),
    Chat(String),
}

const LEAD_INS: [&str; 14] = ["this is ", "that is ", "it is ", "it's ", "its ", "thats ", "that's ", "this looks like ", "i see ", "it is called ", "called ", "a ", "an ", "the "];
const GREETINGS: [&str; 9] = ["hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "good morning", "bye"];

fn clean(text: &str) -> String {
    text.trim().trim_matches(|c: char| matches!(c, '.' | '!' | ',' | ';' | ':')).trim().to_lowercase()
}

fn label_of(rest: &str) -> Option<String> {
    let mut s = clean(rest);
    'strip: loop {
        for p in LEAD_INS {
            if let Some(r) = s.strip_prefix(p) { s = r.trim().to_string(); continue 'strip; }
        }
        break;
    }
    let words = s.split_whitespace().count();
    (words >= 1 && words <= 3 && s.chars().all(|c| c.is_alphanumeric() || c == ' ' || c == '-')).then_some(s)
}

pub fn parse(text: &str) -> Intent {
    let t = clean(text.trim_end_matches('?'));
    let question = text.trim_end().ends_with('?') || ["what", "which", "who", "do you", "can you"].iter().any(|q| t.starts_with(q));
    if t.is_empty() { return Intent::Chat(String::new()); }
    if matches!(t.as_str(), "yes" | "y" | "yeah" | "yep" | "correct" | "right" | "yes it is") { return Intent::Yes; }
    if matches!(t.as_str(), "no" | "n" | "nope" | "wrong" | "no it is not") { return Intent::No; }
    if let Some(w) = t.strip_prefix("forget ") { if let Some(l) = label_of(w) { return Intent::Forget(l); } }
    if question {
        let has = |ks: &[&str]| ks.iter().any(|k| t.contains(k));
        return if has(&["did you see", "have you seen", "what was", "remember"]) { Intent::WhatSeen }
        else if has(&["next", "predict", "expect"]) { Intent::WhatNext }
        else if has(&["this", "that", "it", "you see", "holding", "know"]) { Intent::WhatIsThis }
        else { Intent::Chat(text.trim().to_string()) };
    }
    if GREETINGS.contains(&t.as_str()) { return Intent::Chat(text.trim().to_string()); }
    // "no, it's a cup" corrects the previous guess with a new label.
    let body = t.strip_prefix("no").filter(|r| r.starts_with([',', ' '])).map(|r| r.trim_start_matches([',', ' '])).unwrap_or(&t);
    match label_of(body) { Some(l) => Intent::Label(l), None => Intent::Chat(text.trim().to_string()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Intent::*;

    #[test]
    fn labels() {
        assert_eq!(parse("cup"), Label("cup".into()));
        assert_eq!(parse("This is a Coffee Cup."), Label("coffee cup".into()));
        assert_eq!(parse("it's the circle"), Label("circle".into()));
        assert_eq!(parse("no, it's a square"), Label("square".into()));
    }

    #[test]
    fn questions_and_feedback() {
        assert_eq!(parse("what is this?"), WhatIsThis);
        assert_eq!(parse("What do you see"), WhatIsThis);
        assert_eq!(parse("what did you see?"), WhatSeen);
        assert_eq!(parse("what comes next?"), WhatNext);
        assert_eq!(parse("Yes"), Yes);
        assert_eq!(parse("nope"), No);
        assert_eq!(parse("forget cup"), Forget("cup".into()));
    }

    #[test]
    fn small_talk_is_not_a_label() {
        assert!(matches!(parse("hello"), Chat(_)));
        assert!(matches!(parse("please tell me about your day and everything"), Chat(_)));
        assert!(matches!(parse("how are you?"), Chat(_)));
        assert!(matches!(parse(""), Chat(_)));
    }
}
