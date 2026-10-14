use anyhow::{bail, Result};
use cortex_core::{cosine, incoming_subject, outgoing, subject, Frame, Signal, FRAME_SUBJECT, INPUT_TEXT};
use futures::StreamExt;
use nokhwa::{Camera, utils::{CameraIndex, RequestedFormat, RequestedFormatType}};
use std::collections::HashMap;
use tokio::io::{self, AsyncBufReadExt};
mod neural;
use neural::{fit, sensory_vector, NeuralNet};

#[tokio::main]
async fn main()->Result<()>{
    let role=std::env::args().nth(1).unwrap_or_else(||"help".into());
    let nc=async_nats::connect(std::env::var("NATS_URL").unwrap_or_else(|_|"127.0.0.1:4222".into())).await?;
    match role.as_str(){
        "language"=>language(nc).await?,"visual"=>visual(nc).await?,"perceptual"=>perceptual(nc).await?,
        "memory"=>memory(nc).await?,"associative"=>associative(nc).await?,"workspace"=>workspace(nc).await?,
        _=>bail!("usage: cortex-node <visual|language|perceptual|memory|associative|workspace>"),
    } Ok(())
}
async fn emit(nc:&async_nats::Client,source:&str,kind:&str,symbol:String,activation:f32,embedding:Vec<f32>,context:Option<String>)->Result<()>{
    for edge in outgoing(source){
        let s=Signal::new(source,edge.target,kind,symbol.clone(),activation,edge.weight,embedding.clone(),context.clone());
        nc.publish(subject(source,edge.target),serde_json::to_vec(&s)?.into()).await?;
    } Ok(())
}
async fn language(nc:async_nats::Client)->Result<()>{
    let mut sub=nc.subscribe(incoming_subject("language")).await?;
    let mut text_in=nc.subscribe(INPUT_TEXT).await?;
    let mut lines=io::BufReader::new(io::stdin()).lines();let mut stdin_open=true;
    let mut net=NeuralNet::load_or_new("weights/cortex-0/language.bin",32,24,16,0x1A6E,0.015);
    println!("Language cortex: text via stdin or {INPUT_TEXT}; associative feedback is also connected.");
    loop{tokio::select!{
        line=lines.next_line(),if stdin_open=>{match line?{Some(line)=>hear(&nc,&mut net,&line).await?,None=>stdin_open=false}},
        msg=text_in.next()=>{if let Some(msg)=msg{hear(&nc,&mut net,&String::from_utf8_lossy(&msg.payload)).await?;}},
        msg=sub.next()=>{if let Some(msg)=msg{let s:Signal=serde_json::from_slice(&msg.payload)?;let x=fit(&s.embedding,32);net.train_modulated(&x,&fit(&x,16),s.connection_weight);}}
    }}
}
async fn hear(nc:&async_nats::Client,net:&mut NeuralNet,line:&str)->Result<()>{
    let text=line.trim();if text.is_empty(){return Ok(())}
    let x=sensory_vector(text.as_bytes(),32);for _ in 0..4{net.train(&x,&fit(&x,16));}
    let(_,emb)=net.forward(&x);let id=stable_symbol("L",&emb);
    emit(nc,"language","representation",id,1.0,emb,Some(text.into())).await
}
async fn visual(nc:async_nats::Client)->Result<()>{
    let index=std::env::var("CORTEX_CAMERA").ok().and_then(|v|v.parse::<u32>().ok()).unwrap_or(0);
    let requested=RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera=Camera::new(CameraIndex::Index(index),requested)?;camera.open_stream()?;
    let mut net=NeuralNet::load_or_new("weights/cortex-0/visual.bin",64,48,16,0x715A1,0.006);
    loop{let frame=camera.frame()?;let decoded=frame.decode_image::<nokhwa::pixel_format::RgbFormat>()?;let raw=decoded.as_raw();if raw.is_empty(){continue}
        let mut x=Vec::with_capacity(64);let stride=(raw.len()/3/64).max(1);
        for p in(0..raw.len()/3).step_by(stride).take(64){let i=p*3;x.push((raw[i]as f32*0.299+raw[i+1]as f32*0.587+raw[i+2]as f32*0.114)/127.5-1.0);}
        publish_frame(&nc,raw,decoded.width()as usize,decoded.height()as usize).await?;
        while x.len()<64{x.push(0.0)}net.train(&x,&fit(&x,16));let(_,emb)=net.forward(&x);let id=stable_symbol("V",&emb);
        emit(&nc,"visual","representation",id,1.0,emb,Some(format!("camera:{index} {}x{}",decoded.width(),decoded.height()))).await?;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;}
}
async fn publish_frame(nc:&async_nats::Client,rgb:&[u8],w:usize,h:usize)->Result<()>{
    if w==0||h==0||rgb.len()<w*h*3{return Ok(())}
    let(ow,oh)=(64usize,32usize);let mut gray=Vec::with_capacity(ow*oh);
    for oy in 0..oh{for ox in 0..ow{let i=((oy*h/oh)*w+ox*w/ow)*3;gray.push((rgb[i]as f32*0.299+rgb[i+1]as f32*0.587+rgb[i+2]as f32*0.114)as u8);}}
    nc.publish(FRAME_SUBJECT,serde_json::to_vec(&Frame{width:ow,height:oh,gray})?.into()).await?;Ok(())
}
async fn perceptual(nc:async_nats::Client)->Result<()>{
    let mut sub=nc.subscribe(incoming_subject("perceptual")).await?;let mut prototypes:Vec<(String,Vec<f32>,u32)>=vec![];
    let mut net=NeuralNet::load_or_new("weights/cortex-0/perceptual.bin",16,24,16,0x9E2C,0.008);let mut prediction:Option<Vec<f32>>=None;
    while let Some(msg)=sub.next().await{let s:Signal=serde_json::from_slice(&msg.payload)?;
        if s.source=="associative"{prediction=Some(fit(&s.embedding,16));continue} if s.source=="workspace"{continue}
        let actual=fit(&s.embedding,16);let surprise=prediction.as_ref().map(|p|(1.0-cosine(p,&actual)).clamp(0.0,2.0)).unwrap_or(0.0);
        net.train_modulated(&actual,&actual,(s.connection_weight*(0.5+surprise)).clamp(0.1,2.0));let(_,latent)=net.forward(&actual);
        let best=prototypes.iter().enumerate().map(|(i,(_,e,_))|(i,cosine(e,&latent))).max_by(|a,b|a.1.total_cmp(&b.1));
        let(symbol,idx)=match best{Some((i,score))if score>0.86=>(prototypes[i].0.clone(),Some(i)),_=>(format!("P{}",prototypes.len()+1),None)};
        if let Some(i)=idx{let(_,e,count)=&mut prototypes[i];let c=*count as f32;for(x,y)in e.iter_mut().zip(&latent){*x=(*x*c+*y)/(c+1.0)}*count+=1}else{prototypes.push((symbol.clone(),latent.clone(),1))}
        emit(&nc,"perceptual","representation",symbol,1.0,latent,Some(format!("from {}",s.source))).await?;prediction=None;
    }Ok(())
}
async fn memory(nc:async_nats::Client)->Result<()>{
    let mut sub=nc.subscribe(incoming_subject("memory")).await?;let mut recent:Vec<Signal>=vec![];let mut net=NeuralNet::load_or_new("weights/cortex-0/memory.bin",16,32,16,0x4D454D,0.006);
    while let Some(msg)=sub.next().await{let s:Signal=serde_json::from_slice(&msg.payload)?;if s.source=="workspace"{continue}
        let x=fit(&s.embedding,16);net.train_modulated(&x,&x,s.connection_weight);let(_,mem)=net.forward(&x);
        let related=recent.iter().rev().find(|r|cosine(&r.embedding,&mem)>0.90).cloned();let mut stored=s.clone();stored.embedding=mem.clone();recent.push(stored);if recent.len()>64{recent.remove(0);}
        if let Some(r)=related{emit(&nc,"memory","recall",format!("M:{}",r.symbol),0.65,mem,Some(format!("recalled by {}",s.symbol))).await?;}
    }Ok(())
}
async fn associative(nc:async_nats::Client)->Result<()>{
    let mut sub=nc.subscribe(incoming_subject("associative")).await?;let mut last:HashMap<String,Signal>=HashMap::new();let mut links:HashMap<(String,String),u32>=HashMap::new();
    let mut net=NeuralNet::load_or_new("weights/cortex-0/associative.bin",32,32,16,0xA550C,0.01);
    while let Some(msg)=sub.next().await{let s:Signal=serde_json::from_slice(&msg.payload)?;if s.source=="workspace"{continue}
        for other in last.values(){if other.source!=s.source&&s.timestamp_ms.abs_diff(other.timestamp_ms)<2500{let key=(other.symbol.clone(),s.symbol.clone());let count=links.entry(key.clone()).or_default();*count+=1;
            if *count>=3{let mut pair=fit(&other.embedding,16);pair.extend(fit(&s.embedding,16));let target=(0..16).map(|i|(pair[i]+pair[i+16])*0.5).collect::<Vec<_>>();
                net.train_modulated(&pair,&target,s.connection_weight);let(_,emb)=net.forward(&pair);emit(&nc,"associative","association",format!("C:{}<->{}",key.0,key.1),(*count as f32/6.0).min(1.0),emb,Some(format!("co-activated {} times",count))).await?;}}}
        last.insert(s.source.clone(),s);
    }Ok(())
}
async fn workspace(nc:async_nats::Client)->Result<()>{
    let mut sub=nc.subscribe(incoming_subject("workspace")).await?;
    while let Some(msg)=sub.next().await{let s:Signal=serde_json::from_slice(&msg.payload)?;let salience=s.activation*s.connection_weight;if salience<0.70{continue}
        emit(&nc,"workspace","broadcast",s.symbol,salience,s.embedding,Some(format!("selected from {}",s.source))).await?;}Ok(())
}
fn stable_symbol(prefix:&str,emb:&[f32])->String{let mut h:u64=1469598103934665603;for x in emb{for b in x.to_le_bytes(){h^=b as u64;h=h.wrapping_mul(1099511628211);}}format!("{prefix}{:04}",h%10000)}
