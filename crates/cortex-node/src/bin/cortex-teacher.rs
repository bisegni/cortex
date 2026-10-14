//! Automatic teacher: shows images paired with their names, then tests the brain on unseen ones.
//!
//! usage: cortex-teacher [--mode synthetic|camera|images] [--images DIR] [--words a,b,c]
//!        [--headless] [--episodes N] [--target 0.8] [--speed 1.0] [--test-every 12] [--rounds 5]
//!
//! synthetic: generated shapes are injected into the visual cortex (default).
//! images:    like synthetic, but from DIR/<label>/*.png|jpg (every 5th image is held out for tests).
//! camera:    guided; asks you to show real objects (`--words`) to the camera and tests you afterwards.
use anyhow::{bail, Result};
use cortex_core::{curriculum, Answer, Frame, TeacherStatus, ANSWER_SUBJECT, INJECT_SUBJECT, INPUT_TEXT, TEACHER_STATUS_SUBJECT};
use futures::StreamExt;
use std::{collections::HashMap, path::{Path, PathBuf}, time::Duration};

#[derive(Clone, Copy, PartialEq)]
enum Mode { Synthetic, Camera, Images }

struct Args { mode: Mode, images: Option<PathBuf>, words: Vec<String>, headless: bool, episodes: u32, target: f32, speed: f32, test_every: u32, rounds: u32 }

fn parse() -> Result<Args> {
    let mut a = Args { mode: Mode::Synthetic, images: None, words: vec![], headless: false, episodes: 0, target: 0.8, speed: 1.0, test_every: 12, rounds: 5 };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let val = argv.get(i + 1).cloned().unwrap_or_default();
        match argv[i].as_str() {
            "--headless" => a.headless = true,
            "--mode" => { a.mode = match val.as_str() { "synthetic" => Mode::Synthetic, "camera" => Mode::Camera, "images" => Mode::Images, o => bail!("unknown mode {o}") }; i += 1 }
            "--images" => { a.images = Some(PathBuf::from(&val)); a.mode = Mode::Images; i += 1 }
            "--words" => { a.words = val.split(',').map(|w| w.trim().to_lowercase()).filter(|w| !w.is_empty()).collect(); i += 1 }
            "--episodes" => { a.episodes = val.parse().unwrap_or(0); i += 1 }
            "--target" => { a.target = val.parse().unwrap_or(0.8); i += 1 }
            "--speed" => { a.speed = val.parse().unwrap_or(1.0f32).max(0.1); i += 1 }
            "--test-every" => { a.test_every = val.parse().unwrap_or(12).max(1); i += 1 }
            "--rounds" => { a.rounds = val.parse().unwrap_or(5).max(1); i += 1 }
            other => eprintln!("ignoring unknown argument {other}"),
        }
        i += 1;
    }
    if a.mode == Mode::Camera && a.words.is_empty() { bail!("camera mode needs --words cup,phone,..."); }
    if a.mode == Mode::Images && a.images.is_none() { bail!("images mode needs --images DIR"); }
    Ok(a)
}

/// Letterbox any picture into the 64x32 grayscale frame the visual cortex expects.
fn frame_from_image(img: &image::DynamicImage) -> Frame {
    let (w, h) = (curriculum::WIDTH as u32, curriculum::HEIGHT as u32);
    let gray = img.to_luma8();
    let scale = (w as f32 / gray.width() as f32).min(h as f32 / gray.height() as f32);
    let (nw, nh) = (((gray.width() as f32 * scale).round() as u32).max(1), ((gray.height() as f32 * scale).round() as u32).max(1));
    let small = image::imageops::resize(&gray, nw, nh, image::imageops::FilterType::Triangle);
    let mut canvas = image::GrayImage::from_pixel(w, h, image::Luma([10]));
    image::imageops::overlay(&mut canvas, &small, ((w - nw) / 2) as i64, ((h - nh) / 2) as i64);
    Frame { width: w as usize, height: h as usize, gray: canvas.into_raw() }
}

type Split = (Vec<String>, Vec<Vec<Frame>>, Vec<Vec<Frame>>);

/// DIR/<label>/*.png|jpg -> (labels, train frames, held-out frames); every 5th image is held out.
fn load_folder(dir: &Path) -> Result<Split> {
    let mut labels: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).filter(|e| e.path().is_dir()).collect();
    labels.sort_by_key(|e| e.file_name());
    let (mut names, mut train, mut test) = (vec![], vec![], vec![]);
    for l in labels {
        let mut files: Vec<_> = std::fs::read_dir(l.path())?.filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|e| e.to_str()).is_some_and(|e| matches!(e.to_lowercase().as_str(), "png" | "jpg" | "jpeg"))).collect();
        files.sort();
        let (mut tr, mut te) = (vec![], vec![]);
        for (i, f) in files.iter().enumerate() {
            let Ok(img) = image::open(f) else { eprintln!("skipping unreadable {}", f.display()); continue };
            if i % 5 == 4 { te.push(frame_from_image(&img)) } else { tr.push(frame_from_image(&img)) }
        }
        if tr.is_empty() || te.is_empty() { eprintln!("skipping label {:?}: needs at least 5 images", l.file_name()); continue; }
        names.push(l.file_name().to_string_lossy().to_lowercase());
        train.push(tr);
        test.push(te);
    }
    if names.is_empty() { bail!("no usable label folders in {}", dir.display()); }
    Ok((names, train, test))
}

enum Source { Synthetic, Folder { train: Vec<Vec<Frame>>, test: Vec<Vec<Frame>> }, Camera }

struct Teacher {
    nc: async_nats::Client,
    answers: async_nats::Subscriber,
    status: TeacherStatus,
    classes: Vec<String>,
    source: Source,
    rng: u64,
    speed: f32,
    test_seed: u64,
    teach_secs: f32,
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
    async fn inject(&self, frame: &Frame) -> Result<()> {
        self.nc.publish(INJECT_SUBJECT, serde_json::to_vec(frame)?.into()).await?;
        Ok(())
    }
    async fn say_word(&self, class: usize) -> Result<()> {
        self.nc.publish(INPUT_TEXT, self.classes[class].clone().into()).await?;
        Ok(())
    }
    fn pause(&self, secs: f32) -> tokio::time::Sleep { tokio::time::sleep(Duration::from_secs_f32(secs / self.speed)) }
    fn image(&mut self, class: usize, test: bool) -> Frame {
        let r = self.rand();
        match &self.source {
            Source::Folder { train, test: held } => { let v = if test { &held[class] } else { &train[class] }; v[r as usize % v.len()].clone() }
            _ if test => { self.test_seed += 1; curriculum::render(class, curriculum::TEST_SEED_BASE + self.test_seed) }
            _ => curriculum::render(class, r % curriculum::TEST_SEED_BASE),
        }
    }
    async fn drain_answers(&mut self) { while tokio::time::timeout(Duration::from_millis(20), self.answers.next()).await.is_ok() {} }
    async fn next_answer(&mut self, wait: Duration) -> Option<Answer> {
        tokio::time::timeout(wait, self.answers.next()).await.ok().flatten().and_then(|m| serde_json::from_slice::<Answer>(&m.payload).ok())
    }
    fn record(&mut self, class: usize, ok: bool, answer: &str, confidence: f32) {
        self.status.last_answer = answer.to_string();
        self.status.last_confidence = confidence;
        if !ok { println!("  miss: {} -> {}", self.classes[class], answer); }
        let row = &mut self.status.per_class[class];
        row.2 += 1;
        if ok { row.1 += 1; }
    }
    async fn finish_round(&mut self, right: u32) -> Result<u32> {
        let acc = right as f32 / self.classes.len() as f32;
        self.status.accuracy = Some(acc);
        self.status.history.push(acc);
        if self.status.history.len() > 200 { self.status.history.remove(0); }
        self.publish_status().await?;
        Ok(right)
    }

    /// Say the name, then show several variations of the class.
    async fn teach(&mut self, class: usize) -> Result<()> {
        self.status.phase = "teach".into();
        self.status.class = self.classes[class].clone();
        self.publish_status().await?;
        self.say_word(class).await?;
        self.pause(0.1).await;
        for _ in 0..3 {
            let f = self.image(class, false);
            self.inject(&f).await?;
            self.pause(0.2).await;
        }
        Ok(())
    }

    /// One held-out image per class, no text. Returns right answers.
    async fn test_round(&mut self) -> Result<u32> {
        // Let the nodes' word/image pairing window (1 s) expire so the test is not taken as teaching.
        tokio::time::sleep(Duration::from_millis(1300)).await;
        let mut right = 0;
        for class in 0..self.classes.len() {
            self.status.phase = "test".into();
            self.status.class = self.classes[class].clone();
            self.publish_status().await?;
            self.drain_answers().await;
            let f = self.image(class, true);
            self.inject(&f).await?;
            let answer = self.next_answer(Duration::from_millis(1500)).await;
            let ok = answer.as_ref().is_some_and(|a| a.word == self.classes[class]);
            right += ok as u32;
            self.record(class, ok, answer.as_ref().map_or("(no answer)", |a| &a.word), answer.as_ref().map_or(0.0, |a| a.confidence));
            self.publish_status().await?;
            tokio::time::sleep(Duration::from_millis(1300)).await;
        }
        if matches!(self.source, Source::Synthetic) { self.novel_trial().await?; }
        self.finish_round(right).await
    }

    /// One never-taught shape: the right answer is "unknown" (or silence).
    async fn novel_trial(&mut self) -> Result<()> {
        self.status.class = curriculum::NOVEL[0].into();
        self.publish_status().await?;
        self.drain_answers().await;
        self.test_seed += 1;
        let f = curriculum::render(curriculum::CLASSES.len(), curriculum::TEST_SEED_BASE + self.test_seed);
        self.inject(&f).await?;
        let answer = self.next_answer(Duration::from_millis(1500)).await;
        let rejected = answer.as_ref().map_or(true, |a| a.word == "unknown");
        self.status.last_answer = answer.as_ref().map_or("(no answer)".into(), |a| a.word.clone());
        if !rejected { println!("  novel ring called {}", self.status.last_answer); }
        self.status.novel.1 += 1;
        if rejected { self.status.novel.0 += 1; }
        tokio::time::sleep(Duration::from_millis(1300)).await;
        Ok(())
    }

    async fn countdown(&mut self, prompt: String, secs: f32) -> Result<()> {
        let mut left = secs;
        while left > 0.0 {
            self.status.prompt = prompt.clone();
            self.status.countdown = left;
            self.publish_status().await?;
            tokio::time::sleep(Duration::from_millis(250)).await;
            left -= 0.25;
        }
        self.status.countdown = 0.0;
        Ok(())
    }

    /// Guided: ask the user to hold the object up, repeat its name while the camera streams.
    async fn teach_camera(&mut self, class: usize) -> Result<()> {
        let word = self.classes[class].clone();
        self.status.class = word.clone();
        self.status.phase = "get ready".into();
        self.countdown(format!("Get ready to show: {word}"), 3.0).await?;
        self.status.phase = "teach".into();
        let mut left = self.teach_secs;
        while left > 0.0 {
            self.status.prompt = format!("Show me: {word}");
            self.status.countdown = left;
            self.publish_status().await?;
            self.say_word(class).await?;
            tokio::time::sleep(Duration::from_millis(700)).await;
            left -= 0.7;
        }
        Ok(())
    }

    /// Guided test: show each object again without being told; majority answer wins.
    async fn test_camera(&mut self) -> Result<u32> {
        let mut right = 0;
        for class in 0..self.classes.len() {
            let word = self.classes[class].clone();
            self.status.class = word.clone();
            self.status.phase = "get ready".into();
            tokio::time::sleep(Duration::from_millis(1300)).await;
            self.countdown(format!("Show me again: {word}"), 3.0).await?;
            self.status.phase = "test".into();
            self.drain_answers().await;
            let mut votes: HashMap<String, (u32, f32)> = HashMap::new();
            let end = tokio::time::Instant::now() + Duration::from_secs(3);
            while let Some(left) = end.checked_duration_since(tokio::time::Instant::now()) {
                self.status.prompt = format!("Keep showing: {word}");
                self.status.countdown = left.as_secs_f32();
                self.publish_status().await?;
                if let Some(a) = self.next_answer(left.min(Duration::from_millis(500))).await {
                    let e = votes.entry(a.word).or_default();
                    e.0 += 1;
                    e.1 = a.confidence;
                }
            }
            let best = votes.into_iter().max_by_key(|(_, (n, _))| *n);
            let ok = best.as_ref().is_some_and(|(w, _)| *w == word);
            right += ok as u32;
            let (w, c) = best.map_or(("(no answer)".to_string(), 0.0), |(w, (_, c))| (w, c));
            self.record(class, ok, &w, c);
        }
        self.status.prompt.clear();
        self.finish_round(right).await
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse()?;
    let nc = async_nats::connect(std::env::var("NATS_URL").unwrap_or_else(|_| "127.0.0.1:4222".into())).await?;
    let answers = nc.subscribe(ANSWER_SUBJECT).await?;
    let (classes, source, mode_name) = match args.mode {
        Mode::Synthetic => (curriculum::CLASSES.iter().map(|c| c.to_string()).collect(), Source::Synthetic, "synthetic"),
        Mode::Camera => (args.words.clone(), Source::Camera, "camera"),
        Mode::Images => {
            let (names, train, test) = load_folder(args.images.as_ref().expect("checked in parse"))?;
            (names, Source::Folder { train, test }, "images")
        }
    };
    let mut t = Teacher {
        nc, answers, rng: 0x7EAC4E5, speed: args.speed, test_seed: 0, teach_secs: 5.0 / args.speed,
        status: TeacherStatus { per_class: classes.iter().map(|c| (c.clone(), 0, 0)).collect(), mode: mode_name.into(), ..Default::default() },
        classes, source,
    };
    println!("teacher ({mode_name}): {} classes: {}", t.classes.len(), t.classes.join(", "));
    let camera = args.mode == Mode::Camera;
    let mut episode = 0u32;
    while args.episodes == 0 || episode < args.episodes {
        if camera {
            for class in 0..t.classes.len() { t.teach_camera(class).await?; }
            episode += 1;
            t.status.episode = episode;
            let right = t.test_camera().await?;
            println!("cycle {episode}: {right}/{} correct", t.classes.len());
        } else {
            let class = (t.rand() % t.classes.len() as u64) as usize;
            t.teach(class).await?;
            episode += 1;
            t.status.episode = episode;
            if episode % args.test_every == 0 {
                let right = t.test_round().await?;
                println!("episode {episode}: {right}/{} correct", t.classes.len());
            }
        }
    }
    if !args.headless { return Ok(()); }

    // Final evaluation on fresh held-out images.
    for row in &mut t.status.per_class { row.1 = 0; row.2 = 0; }
    t.status.novel = (0, 0);
    let mut right = 0;
    for _ in 0..args.rounds { right += if camera { t.test_camera().await? } else { t.test_round().await? }; }
    let total = args.rounds * t.classes.len() as u32;
    let acc = right as f32 / total as f32;
    println!("\nfinal accuracy {right}/{total} = {:.0}%  (target {:.0}%)", acc * 100.0, args.target * 100.0);
    for (c, r, n) in &t.status.per_class { println!("  {c:<10} {r}/{n}"); }
    if t.status.novel.1 > 0 { println!("  unknown-rejection of never-taught shapes: {}/{}", t.status.novel.0, t.status.novel.1); }
    t.status.phase = "done".into();
    t.publish_status().await?;
    t.nc.flush().await?;
    if acc < args.target { std::process::exit(1); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_class(dir: &Path, label: &str, class: usize, n: u64) {
        std::fs::create_dir_all(dir.join(label)).unwrap();
        for i in 0..n {
            let f = curriculum::render(class, i);
            image::GrayImage::from_raw(f.width as u32, f.height as u32, f.gray).unwrap().save(dir.join(label).join(format!("{i}.png"))).unwrap();
        }
    }

    #[test]
    fn folder_loader_splits_every_fifth_image_and_skips_small_labels() {
        let dir = std::env::temp_dir().join(format!("cortex-teacher-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_class(&dir, "circle", 0, 10);
        write_class(&dir, "square", 1, 10);
        write_class(&dir, "tiny", 2, 2);
        let (names, train, test) = load_folder(&dir).unwrap();
        assert_eq!(names, vec!["circle", "square"]);
        assert_eq!((train[0].len(), test[0].len()), (8, 2));
        assert_eq!(train[1][0].gray.len(), curriculum::WIDTH * curriculum::HEIGHT);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn letterbox_keeps_aspect_and_pads() {
        let tall = image::DynamicImage::ImageLuma8(image::GrayImage::from_pixel(10, 40, image::Luma([250])));
        let f = frame_from_image(&tall);
        assert_eq!((f.width, f.height), (64, 32));
        assert_eq!(f.gray[0], 10, "left padding stays dark");
        assert_eq!(f.gray[16 * 64 + 32], 250, "image content in the centre");
    }
}
