use anyhow::{bail, Result};
use cortex_core::{cosine, hash_embedding, subject, Signal};
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
        let emb = hash_embedding(text.as_bytes(), 16);
        let id = stable_symbol("L", &emb);
        emit(&nc, "language", &Signal::new("language","activation",id,1.0,emb,Some(text.into()))).await?;
    }
    Ok(())
}

async fn visual(nc: async_nats::Client) -> Result<()> {
    println!("Visual cortex Seed-0 sensor: enter a visual observation token (e.g. cup-red-left).");
    println!("This deliberately keeps semantics out: text is only converted to a local latent signature.");
    let mut lines = io::BufReader::new(io::stdin()).lines();
    while let Some(line) = lines.next_line().await? {
        let raw = line.trim();
        if raw.is_empty() { continue; }
        let emb = hash_embedding(raw.as_bytes(), 16);
        let id = stable_symbol("V", &emb);
        emit(&nc, "visual", &Signal::new("visual","primitive",id,1.0,emb,Some(raw.into()))).await?;
    }
    Ok(())
}

async fn perceptual(nc: async_nats::Client) -> Result<()> {
    let mut sub = nc.subscribe("cortex.visual.signal").await?;
    let mut prototypes: Vec<(String, Vec<f32>, u32)> = vec![];
    while let Some(msg) = sub.next().await {
        let s: Signal = serde_json::from_slice(&msg.payload)?;
        let best = prototypes.iter().enumerate().map(|(i,(_,e,_))|(i,cosine(e,&s.embedding))).max_by(|a,b|a.1.total_cmp(&b.1));
        let (symbol, idx) = match best {
            Some((i,score)) if score > 0.86 => (prototypes[i].0.clone(), Some(i)),
            _ => (format!("P{}", prototypes.len()+1), None),
        };
        if let Some(i)=idx {
            let (_, e, count)=&mut prototypes[i];
            let c=*count as f32;
            for (x,y) in e.iter_mut().zip(&s.embedding) { *x=(*x*c+*y)/(c+1.0); }
            *count+=1;
        } else { prototypes.push((symbol.clone(),s.embedding.clone(),1)); }
        emit(&nc,"perceptual",&Signal::new("perceptual","stable",symbol,1.0,s.embedding,Some(s.symbol))).await?;
    }
    Ok(())
}

async fn memory(nc: async_nats::Client) -> Result<()> {
    let mut sub = nc.subscribe("cortex.*.signal").await?;
    let mut recent: Vec<Signal> = vec![];
    while let Some(msg)=sub.next().await {
        let s: Signal=serde_json::from_slice(&msg.payload)?;
        if s.source=="memory" { continue; }
        recent.push(s.clone());
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
    while let Some(msg)=sub.next().await {
        let s:Signal=serde_json::from_slice(&msg.payload)?;
        if s.source=="associative" || s.source=="workspace" { continue; }
        for other in last.values() {
            let dt=s.timestamp_ms.abs_diff(other.timestamp_ms);
            if other.source!=s.source && dt<2500 {
                let key=(other.symbol.clone(),s.symbol.clone());
                let count=links.entry(key.clone()).or_default(); *count+=1;
                if *count>=3 {
                    let emb=s.embedding.iter().zip(&other.embedding).map(|(a,b)|(a+b)/2.0).collect();
                    emit(&nc,"associative",&Signal::new("associative","association",format!("C:{}<->{}",key.0,key.1),(*count as f32/6.0).min(1.0),emb,Some(format!("co-activated {} times",count)))).await?;
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
