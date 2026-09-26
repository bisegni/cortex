use anyhow::{bail, Result};
use cortex_core::{cosine, subject, Signal};
mod neural;
use neural::{fit, sensory_vector, NeuralNet};
use futures::StreamExt;
use std::collections::HashMap;
use tokio::io::{self, AsyncBufReadExt};
use nokhwa::{Camera, utils::{CameraIndex, RequestedFormat, RequestedFormatType}};

#[tokio::main]
async fn main() -> Result<()> {
    let role = std::env::args().nth(1).unwrap_or_else(|| "help".into());
    let nc = async_nats::connect(std::env::var("NATS_URL").unwrap_or_else(|_| "127.0.0.1:4222".into())).await?;
    match role.as_str() {
        "language" => language(nc).await?,
        "visual" => visual(nc).await?,
        "perceptual" => perceptual(nc).await?,
        "memory" => memory(nc).await?,
        "associative" => associative(nc).await?,
        "workspace" => workspace(nc).await?,
        _ => bail!("usage: cortex-node <visual|language|perceptual|memory|associative|workspace>"),
    }
    Ok(())
}

async fn emit(nc: &async_nats::Client, cortex: &str, s: &Signal) -> Result<()> {
    nc.publish(subject(cortex), serde_json::to_vec(s)?.into()).await?;
    Ok(())
}

async fn language(nc: async_nats::Client) -> Result<()> {
    println!("Language cortex: type words/phrases. Ctrl-C exits.");
    let mut lines = io::BufReader::new(io::stdin()).lines();
    while let Some(line) = lines.next_line().await? {
        let text = line.trim();
        if text.is_empty() { continue; }
        let x = sensory_vector(text.as_bytes(), 32);
        let mut net = NeuralNet::new(32, 24, 16, 0x1A6E, 0.015);
        for _ in 0..4 { net.train(&x, &fit(&x,16)); }
        let (_, emb) = net.forward(&x);
        let id = stable_symbol("L", &emb);
        emit(&nc, "language", &Signal::new("language","activation",id,1.0,emb,Some(text.into()))).await?;
    }
    Ok(())
}

async fn visual(nc: async_nats::Client) -> Result<()> {
    let index: u32 = std::env::var("CORTEX_CAMERA").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    println!("Visual cortex: opening macOS camera index {index}. Set CORTEX_CAMERA=N to select another device.");
    let requested = RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = Camera::new(CameraIndex::Index(index), requested)?;
    camera.open_stream()?;
    println!("Visual cortex: camera active. macOS may ask Terminal/tmux for Camera permission.");
    let mut net = NeuralNet::new(64, 48, 16, 0x715A1, 0.006);
    loop {
        let frame = camera.frame()?;
        let decoded = frame.decode_image::<nokhwa::pixel_format::RgbFormat>()?;
        let raw = decoded.as_raw();
        if raw.is_empty() { continue; }
        // 64 luminance samples distributed across the frame: pixels, not filenames/labels.
        let mut x=Vec::with_capacity(64);
        let stride=(raw.len()/3/64).max(1);
        for p in (0..raw.len()/3).step_by(stride).take(64) {
            let i=p*3; let y=(raw[i] as f32*.299+raw[i+1] as f32*.587+raw[i+2] as f32*.114)/127.5-1.0; x.push(y);
        }
        while x.len()<64 { x.push(0.0); }
        let target=fit(&x,16);
        net.train(&x,&target);
        let (_,emb)=net.forward(&x);
        let id=stable_symbol("V",&emb);
        emit(&nc,"visual",&Signal::new("visual","camera",id,1.0,emb,Some(format!("camera:{index} {}x{}",decoded.width(),decoded.height())))).await?;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

