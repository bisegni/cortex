//! Child-process supervision: replaces the old tmux session.
use std::{collections::VecDeque, path::PathBuf, process::Stdio, time::{Duration, Instant}};
use tokio::{
    io::{AsyncBufReadExt, AsyncRead, BufReader},
    net::TcpStream,
    process::Command,
    sync::{mpsc, oneshot},
    task::JoinHandle,
};

const LOG_CAP: usize = 500;
pub const NODE_ROLES: [&str; 6] = ["visual", "perceptual", "memory", "associative", "language", "workspace"];

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Stopped,
    Running,
    Exited(i32),
    Failed(String),
    /// Server already listening on the NATS port; we reuse it and do not own it.
    External,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Nats,
    Node(&'static str),
    Train,
    Teacher,
}

enum Ev {
    Line(usize, u64, String),
    Exit(usize, u64, i32),
}

pub struct Proc {
    pub name: String,
    pub kind: Kind,
    pub status: Status,
    pub log: VecDeque<String>,
    pub since: Option<Instant>,
    gen: u64,
    kill: Option<oneshot::Sender<()>>,
    waiter: Option<JoinHandle<()>>,
}

impl Proc {
    fn new(name: &str, kind: Kind) -> Self {
        Self { name: name.into(), kind, status: Status::Stopped, log: VecDeque::new(), since: None, gen: 0, kill: None, waiter: None }
    }
    fn push_log(&mut self, line: String) {
        if self.log.len() == LOG_CAP {
            self.log.pop_front();
        }
        self.log.push_back(line);
    }
    pub fn is_up(&self) -> bool {
        matches!(self.status, Status::Running | Status::External)
    }
}

pub struct Supervisor {
    pub procs: Vec<Proc>,
    tx: mpsc::UnboundedSender<Ev>,
    rx: mpsc::UnboundedReceiver<Ev>,
    bin_dir: PathBuf,
    root: PathBuf,
    nats_addr: String,
}

impl Supervisor {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        // target/<profile>/cortex-tui -> repo root is two levels above the binary directory.
        let exe = std::env::current_exe().unwrap_or_default();
        let bin_dir = exe.parent().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
        let root = std::env::var("CORTEX_ROOT").map(PathBuf::from).unwrap_or_else(|_| {
            bin_dir.parent().and_then(|p| p.parent()).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
        });
        let mut procs = vec![Proc::new("nats-server", Kind::Nats)];
        procs.extend(NODE_ROLES.iter().map(|r| Proc::new(r, Kind::Node(r))));
        procs.push(Proc::new("train", Kind::Train));
        procs.push(Proc::new("teacher", Kind::Teacher));
        let nats_addr = std::env::var("NATS_URL").unwrap_or_else(|_| "127.0.0.1:4222".into());
        let nats_addr = nats_addr.trim_start_matches("nats://").to_string();
        Self { procs, tx, rx, bin_dir, root, nats_addr }
    }

    pub fn train_index(&self) -> usize {
        self.procs.len() - 2
    }

    pub fn teacher_index(&self) -> usize {
        self.procs.len() - 1
    }

    async fn nats_listening(&self) -> bool {
        matches!(tokio::time::timeout(Duration::from_millis(250), TcpStream::connect(&self.nats_addr)).await, Ok(Ok(_)))
    }

    /// Start NATS, wait for it, then start every cortex node.
    pub async fn start_all(&mut self) {
        self.start(0, vec![]).await;
        for _ in 0..25 {
            if self.nats_listening().await {
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        for i in 1..=NODE_ROLES.len() {
            self.start(i, vec![]).await;
        }
    }

    pub async fn restart_nodes(&mut self) {
        for i in 1..=NODE_ROLES.len() {
            self.stop(i);
            self.start(i, vec![]).await;
        }
    }

    pub async fn start(&mut self, i: usize, args: Vec<String>) {
        let kind = self.procs[i].kind;
        if self.procs[i].kill.is_some() {
            self.stop(i);
        }
        if kind == Kind::Nats && self.nats_listening().await {
            let p = &mut self.procs[i];
            p.status = Status::External;
            p.push_log(format!("{} already accepting connections; reusing it", self.nats_addr));
            return;
        }
        let (program, args) = match kind {
            Kind::Nats => (PathBuf::from("nats-server"), vec![]),
            Kind::Node(role) => (self.bin_dir.join("cortex-node"), vec![role.to_string()]),
            Kind::Train => (self.bin_dir.join("cortex-train"), args),
            Kind::Teacher => (self.bin_dir.join("cortex-teacher"), args),
        };
        let mut cmd = Command::new(&program);
        cmd.args(&args).current_dir(&self.root).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
        let p = &mut self.procs[i];
        p.gen += 1;
        let gen = p.gen;
        match cmd.spawn() {
            Err(e) => {
                let hint = if kind == Kind::Nats { " (brew install nats-server)" } else { " (run: cargo build --workspace)" };
                p.status = Status::Failed(format!("{e}{hint}"));
                p.push_log(format!("spawn {} failed: {e}{hint}", program.display()));
            }
            Ok(mut child) => {
                if let Some(out) = child.stdout.take() {
                    forward(out, self.tx.clone(), i, gen);
                }
                if let Some(err) = child.stderr.take() {
                    forward(err, self.tx.clone(), i, gen);
                }
                let (kill_tx, kill_rx) = oneshot::channel();
                let tx = self.tx.clone();
                p.waiter = Some(tokio::spawn(async move {
                    let code = tokio::select! {
                        st = child.wait() => st.ok().and_then(|s| s.code()).unwrap_or(-1),
                        _ = kill_rx => { let _ = child.kill().await; -1 }
                    };
                    let _ = tx.send(Ev::Exit(i, gen, code));
                }));
                p.kill = Some(kill_tx);
                p.status = Status::Running;
                p.since = Some(Instant::now());
                p.push_log(format!("$ {} {}", program.file_name().and_then(|s| s.to_str()).unwrap_or("?"), args.join(" ")));
            }
        }
    }

    pub fn stop(&mut self, i: usize) {
        let p = &mut self.procs[i];
        if let Some(k) = p.kill.take() {
            let _ = k.send(());
        }
        if p.status == Status::Running || p.status == Status::External {
            p.status = Status::Stopped;
        }
        p.since = None;
    }

    /// Apply pending process output and exit events.
    pub fn poll(&mut self) {
        while let Ok(ev) = self.rx.try_recv() {
            match ev {
                Ev::Line(i, gen, line) if self.procs[i].gen == gen => self.procs[i].push_log(line),
                Ev::Exit(i, gen, code) if self.procs[i].gen == gen => {
                    let p = &mut self.procs[i];
                    p.kill = None;
                    if p.status == Status::Running {
                        p.status = if code == 0 { Status::Exited(0) } else { Status::Failed(format!("exit code {code}")) };
                        p.push_log(format!("process ended ({code})"));
                    }
                }
                _ => {}
            }
        }
    }

    /// Kill every child and wait briefly for the waiter tasks.
    pub async fn shutdown(&mut self) {
        for i in 0..self.procs.len() {
            self.stop(i);
        }
        for p in &mut self.procs {
            if let Some(w) = p.waiter.take() {
                let _ = tokio::time::timeout(Duration::from_secs(2), w).await;
            }
        }
    }

    pub fn running(&self) -> usize {
        self.procs.iter().filter(|p| p.is_up() && !matches!(p.kind, Kind::Train | Kind::Teacher)).count()
    }
}

fn forward<R: AsyncRead + Unpin + Send + 'static>(r: R, tx: mpsc::UnboundedSender<Ev>, i: usize, gen: u64) {
    tokio::spawn(async move {
        let mut lines = BufReader::new(r).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if tx.send(Ev::Line(i, gen, line)).is_err() {
                break;
            }
        }
    });
}
