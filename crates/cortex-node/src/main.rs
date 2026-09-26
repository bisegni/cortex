use anyhow::{bail, Result};
use cortex_core::{cosine, subject, Signal};
mod neural;
use neural::{fit, sensory_vector, NeuralNet};
use futures::StreamExt;
use std::collections::HashMap;
use tokio::io::{self, AsyncBufReadExt};

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


async fn perceptual(nc: async_nats::Client) -> Result<()> {
    let mut sub = nc.subscribe("cortex.*.signal").await?;
    let mut prototypes: Vec<(String, Vec<f32>, u32)> = vec![];
    let mut net=NeuralNet::new(16,24,16,0x9E2C,0.008);
    while let Some(msg) = sub.next().await {
        let s: Signal = serde_json::from_slice(&msg.payload)?;
        net.train(&s.embedding,&s.embedding);
        let (_,latent)=net.forward(&s.embedding);
        let best = prototypes.iter().enumerate().map(|(i,(_,e,_))|(i,cosine(e,&latent))).max_by(|a,b|a.1.total_cmp(&b.1));
        let (symbol, idx) = match best {
            Some((i,score)) if score > 0.86 => (prototypes[i].0.clone(), Some(i)),
            _ => (format!("P{}", prototypes.len()+1), None),
        };
        if let Some(i)=idx {
            let (_, e, count)=&mut prototypes[i];
            let c=*count as f32;
            for (x,y) in e.iter_mut().zip(&latent) { *x=(*x*c+*y)/(c+1.0); }
            *count+=1;
        } else { prototypes.push((symbol.clone(),latent.clone(),1)); }
        emit(&nc,"perceptual",&Signal::new("perceptual","stable",symbol.clone(),1.0,latent.clone(),Some(s.symbol))).await?;
        if prediction.is_some() {
            emit(&nc,"perceptual",&Signal::new("perceptual","prediction_error",format!("E:{symbol}"),surprise,latent,Some(format!("local surprise={surprise:.3}; plasticity x{modulation:.2}")))).await?;
            prediction=None;
        }
    }
    Ok(())
}

async fn memory(nc: async_nats::Client) -> Result<()> {
    let mut sub = nc.subscribe("cortex.*.signal").await?;
    let mut recent: Vec<Signal> = vec![];
    let mut net=NeuralNet::new(16,32,16,0x4D454D,0.006);
    while let Some(msg)=sub.next().await {
        let s: Signal=serde_json::from_slice(&msg.payload)?;
        if s.source=="memory" { continue; }
        let target=fit(&s.embedding,16); net.train(&target,&target);
        let (_,mem)=net.forward(&target);
        let mut stored=s.clone(); stored.embedding=mem;
        recent.push(stored);
        if recent.len()>64 { recent.remove(0); }
        let related=recent.iter().rev().skip(1).find(|x| cosine(&x.embedding,&s.embedding)>0.90);
        if let Some(r)=related {
            emit(&nc,"memory",&Signal::new("memory","recall",format!("M:{}",r.symbol),0.65,r.embedding.clone(),Some(format!("recalled by {}",s.symbol)))).await?;
        }
    }
    Ok(())
}

async fn associative(nc: async_nats::Client) -> Result<()> {
    let mut sub=nc.subscribe("cortex.*.signal").await?;
    let mut last: HashMap<String,Signal>=HashMap::new();
    let mut links: HashMap<(String,String),u32>=HashMap::new();
    let mut net=NeuralNet::new(32,32,16,0xA550C,0.01);
    while let Some(msg)=sub.next().await {
        let s:Signal=serde_json::from_slice(&msg.payload)?;
        if s.source=="associative" || s.source=="workspace" { continue; }
        for other in last.values() {
            let dt=s.timestamp_ms.abs_diff(other.timestamp_ms);
            if other.source!=s.source && dt<2500 {
                let key=(other.symbol.clone(),s.symbol.clone());
                let count=links.entry(key.clone()).or_default(); *count+=1;
                if *count>=3 {
                    let mut pair=fit(&other.embedding,16); pair.extend(fit(&s.embedding,16));
                    let target:Vec<f32>=(0..16).map(|i|(pair[i]+pair[i+16])*.5).collect(); net.train(&pair,&target);
                    let (_,emb)=net.forward(&pair);
                    emit(&nc,"associative",&Signal::new("associative","association",format!("C:{}<->{}",key.0,key.1),(*count as f32/6.0).min(1.0),emb.clone(),Some(format!("co-activated {} times",count)))).await?;
                    if other.source=="perceptual" || s.source=="perceptual" {
                        emit(&nc,"associative",&Signal::new("associative","prediction",format!("PR:{}+{}",key.0,key.1),(*count as f32/8.0).min(1.0),emb,Some("top-down teaching signal for perceptual cortex; no gradient transport".into()))).await?;
                    }
                }
            }
        }
        last.insert(s.source.clone(),s);
    }
    Ok(())
}

async fn workspace(nc: async_nats::Client) -> Result<()> {
    let mut sub=nc.subscribe("cortex.*.signal").await?;
    while let Some(msg)=sub.next().await {
        let s:Signal=serde_json::from_slice(&msg.payload)?;
        if s.source=="workspace" || s.activation<0.70 { continue; }
        emit(&nc,"workspace",&Signal::new("workspace","broadcast",s.symbol,s.activation,s.embedding,Some(format!("selected from {}",s.source)))).await?;
    }
    Ok(())
}

fn stable_symbol(prefix:&str, emb:&[f32])->String {
    let mut h:u64=1469598103934665603;
    for x in emb { for b in x.to_le_bytes() { h^=b as u64; h=h.wrapping_mul(1099511628211); } }
    format!("{prefix}{:04}",h%10000)
}
