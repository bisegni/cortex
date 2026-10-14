//! Automatic teacher: shows generated images paired with their names, then tests the brain.
//!
//! usage: cortex-teacher [--headless] [--episodes N] [--target 0.8] [--speed 1.0] [--test-every 12]
use anyhow::Result;
use cortex_core::{curriculum, Answer, TeacherStatus, ANSWER_SUBJECT, INJECT_SUBJECT, INPUT_TEXT, TEACHER_STATUS_SUBJECT};
use futures::StreamExt;
use std::time::Duration;

struct Args { headless: bool, episodes: u32, target: f32, speed: f32, test_every: u32, rounds: u32 }

fn parse() -> Args {
    let mut a = Args { headless: false, episodes: 0, target: 0.8, speed: 1.0, test_every: 12, rounds: 5 };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let val = argv.get(i + 1).cloned().unwrap_or_default();
        match argv[i].as_str() {
            "--headless" => a.headless = true,
            "--episodes" => { a.episodes = val.parse().unwrap_or(0); i += 1 }
            "--target" => { a.target = val.parse().unwrap_or(0.8); i += 1 }
            "--speed" => { a.speed = val.parse().unwrap_or(1.0f32).max(0.1); i += 1 }
            "--test-every" => { a.test_every = val.parse().unwrap_or(12).max(1); i += 1 }
            "--rounds" => { a.rounds = val.parse().unwrap_or(5).max(1); i += 1 }
            other => eprintln!("ignoring unknown argument {other}"),
        }
        i += 1;
    }
    a
}

struct Teacher {
    nc: async_nats::Client,
    answers: async_nats::Subscriber,
    status: TeacherStatus,
    rng: u64,
    speed: f32,
    test_seed: u64,
}

impl Teacher {
    fn rand(&mut self) -> u64 {
        self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.rng >> 33
    }
    async fn publish_status(&self) -> Result<()> {
        self.nc.publish(TEACHER_STATUS_SUBJECT, serde_json::to_vec(&self.status)?.into()).await?;
        Ok(())
    }
    async fn show(&self, class: usize, seed: u64) -> Result<()> {
        self.nc.publish(INJECT_SUBJECT, serde_json::to_vec(&curriculum::render(class, seed))?.into()).await?;
        Ok(())
    }
    fn pause(&self, secs: f32) -> tokio::time::Sleep { tokio::time::sleep(Duration::from_secs_f32(secs / self.speed)) }

    /// Say the name, then show several variations of the class.
    async fn teach(&mut self, class: usize) -> Result<()> {
        self.status.phase = "teach".into();
        self.status.class = curriculum::CLASSES[class].into();
        self.publish_status().await?;
        self.nc.publish(INPUT_TEXT, curriculum::CLASSES[class].into()).await?;
        self.pause(0.1).await;
        for _ in 0..3 {
            let seed = self.rand() % curriculum::TEST_SEED_BASE;
            self.show(class, seed).await?;
            self.pause(0.2).await;
        }
        Ok(())
    }

    /// One held-out image per class, no text. Returns right answers.
    async fn test_round(&mut self) -> Result<u32> {
        // Let the nodes' word/image pairing window (1 s) expire so the test is not taken as teaching.
        tokio::time::sleep(Duration::from_millis(1300)).await;
        let mut right = 0;
        for class in 0..curriculum::CLASSES.len() {
            self.status.phase = "test".into();
            self.status.class = curriculum::CLASSES[class].into();
            self.publish_status().await?;
            while tokio::time::timeout(Duration::from_millis(20), self.answers.next()).await.is_ok() {}
            self.test_seed += 1;
            self.show(class, curriculum::TEST_SEED_BASE + self.test_seed).await?;
            let answer = tokio::time::timeout(Duration::from_millis(1500), self.answers.next()).await.ok().flatten()
                .and_then(|m| serde_json::from_slice::<Answer>(&m.payload).ok());
            let ok = answer.as_ref().is_some_and(|a| a.word == curriculum::CLASSES[class]);
            self.status.last_answer = answer.as_ref().map_or("(no answer)".into(), |a| a.word.clone());
            self.status.last_confidence = answer.as_ref().map_or(0.0, |a| a.confidence);
            if !ok { println!("  miss: {} -> {}", curriculum::CLASSES[class], self.status.last_answer); }
            let row = &mut self.status.per_class[class];
            row.2 += 1;
            if ok { row.1 += 1; right += 1; }
            self.publish_status().await?;
            tokio::time::sleep(Duration::from_millis(1300)).await;
        }
        let acc = right as f32 / curriculum::CLASSES.len() as f32;
        self.status.accuracy = Some(acc);
        self.status.history.push(acc);
        if self.status.history.len() > 200 { self.status.history.remove(0); }
        self.publish_status().await?;
        Ok(right)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse();
    let nc = async_nats::connect(std::env::var("NATS_URL").unwrap_or_else(|_| "127.0.0.1:4222".into())).await?;
    let answers = nc.subscribe(ANSWER_SUBJECT).await?;
    let mut t = Teacher {
        nc, answers, rng: 0x7EAC4E5, speed: args.speed, test_seed: 0,
        status: TeacherStatus { per_class: curriculum::CLASSES.iter().map(|c| (c.to_string(), 0, 0)).collect(), ..Default::default() },
    };
    println!("teacher: {} classes, test every {} episodes", curriculum::CLASSES.len(), args.test_every);
    let mut episode = 0u32;
    while args.episodes == 0 || episode < args.episodes {
        let class = (t.rand() % curriculum::CLASSES.len() as u64) as usize;
        t.teach(class).await?;
        episode += 1;
        t.status.episode = episode;
        if episode % args.test_every == 0 {
            let right = t.test_round().await?;
            println!("episode {episode}: {right}/{} correct", curriculum::CLASSES.len());
        }
    }
    if !args.headless { return Ok(()); }

    // Final evaluation on fresh held-out images.
    for row in &mut t.status.per_class { row.1 = 0; row.2 = 0; }
    let mut right = 0;
    for _ in 0..args.rounds { right += t.test_round().await?; }
    let total = args.rounds * curriculum::CLASSES.len() as u32;
    let acc = right as f32 / total as f32;
    println!("\nfinal accuracy {right}/{total} = {:.0}%  (target {:.0}%)", acc * 100.0, args.target * 100.0);
    for (c, r, n) in &t.status.per_class { println!("  {c:<10} {r}/{n}"); }
    t.status.phase = "done".into();
    t.publish_status().await?;
    t.nc.flush().await?;
    if acc < args.target { std::process::exit(1); }
    Ok(())
}
