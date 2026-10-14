use anyhow::{bail, Result};
use cortex_core::{cosine, curriculum, incoming_subject, outgoing, subject, Answer, Frame, Signal, ANSWER_SUBJECT, FRAME_SUBJECT, INJECT_SUBJECT, INPUT_TEXT};
use futures::StreamExt;
use nokhwa::{Camera, utils::{CameraIndex, RequestedFormat, RequestedFormatType}};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::{Duration, Instant}};
use tokio::io::{self, AsyncBufReadExt};
mod neural;
use neural::{autosave, fit, sensory_vector, NeuralNet, Prototypes};

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
async fn emit(nc:&async_nats::Client,source:&str,kind:&str,symbol:String,activation:f32,embedding:Vec<f32>,context:Option<String>,ttl:u8)->Result<()>{
    for edge in outgoing(source){
        let s=Signal::new(source,edge.target,kind,symbol.clone(),activation,edge.weight,embedding.clone(),context.clone(),ttl);
        nc.publish(subject(source,edge.target),serde_json::to_vec(&s)?.into()).await?;
    } Ok(())
}
async fn language(nc:async_nats::Client)->Result<()>{
    let mut sub=nc.subscribe(incoming_subject("language")).await?;
    let mut text_in=nc.subscribe(INPUT_TEXT).await?;
    let mut lines=io::BufReader::new(io::stdin()).lines();let mut stdin_open=true;
    let mut net=NeuralNet::load_or_new("weights/cortex-0/language.bin",32,24,16,0x1A6E,0.015);let mut last_save=Instant::now();
    println!("Language cortex: text via stdin or {INPUT_TEXT}; answers are published on {ANSWER_SUBJECT}.");
    loop{
        tokio::select!{
            line=lines.next_line(),if stdin_open=>{match line?{Some(line)=>hear(&nc,&mut net,&line).await?,None=>stdin_open=false}},
            msg=text_in.next()=>{if let Some(msg)=msg{hear(&nc,&mut net,&String::from_utf8_lossy(&msg.payload)).await?;}},
            msg=sub.next()=>{if let Some(msg)=msg{let s:Signal=serde_json::from_slice(&msg.payload)?;let x=fit(&s.embedding,32);net.train_modulated(&x,&fit(&x,16),s.connection_weight);
                if s.kind=="recall"{let word=s.symbol.trim_start_matches("W:").to_string();println!("answer: {word} ({:.2})",s.activation);
                    nc.publish(ANSWER_SUBJECT,serde_json::to_vec(&Answer{word,confidence:s.activation})?.into()).await?;}}}
        }
        autosave(&net,"weights/cortex-0/language.bin",&mut last_save);
    }
}
async fn hear(nc:&async_nats::Client,net:&mut NeuralNet,line:&str)->Result<()>{
    let word=line.trim().to_lowercase();if word.is_empty(){return Ok(())}
    let x=sensory_vector(word.as_bytes(),32);for _ in 0..4{net.train(&x,&fit(&x,16));}
    let(_,emb)=net.forward(&x);
    emit(nc,"language","representation",format!("W:{word}"),1.0,emb,Some(word),3).await
}
async fn visual(nc:async_nats::Client)->Result<()>{
    let index=std::env::var("CORTEX_CAMERA").ok().and_then(|v|v.parse::<u32>().ok()).unwrap_or(0);
    let(tx,mut rx)=tokio::sync::mpsc::channel::<Frame>(2);
    if std::env::var("CORTEX_CAMERA").as_deref()!=Ok("off"){std::thread::spawn(move||camera_loop(index,tx));}
    let mut inject=nc.subscribe(INJECT_SUBJECT).await?;
    let mut net=NeuralNet::load_or_new("weights/cortex-0/visual.bin",64,48,16,0x715A1,0.006);
    let mut protos=Prototypes::default();let mut last_inject:Option<Instant>=None;let mut cam_open=true;let mut last_save=Instant::now();
    loop{
        tokio::select!{
            msg=inject.next()=>{let Some(msg)=msg else{break};if let Ok(frame)=serde_json::from_slice::<Frame>(&msg.payload){last_inject=Some(Instant::now());see(&nc,&mut net,&mut protos,frame,"inject").await?;}},
            frame=rx.recv(),if cam_open=>match frame{
                Some(frame)=>if !last_inject.is_some_and(|t|t.elapsed()<Duration::from_secs(3)){see(&nc,&mut net,&mut protos,frame,&format!("camera:{index}")).await?;},
                None=>{cam_open=false;eprintln!("camera unavailable; visual cortex now follows injected frames only");}
            }
        }
        autosave(&net,"weights/cortex-0/visual.bin",&mut last_save);
    }
    Ok(())
}
async fn see(nc:&async_nats::Client,net:&mut NeuralNet,protos:&mut Prototypes,frame:Frame,source:&str)->Result<()>{
    nc.publish(FRAME_SUBJECT,serde_json::to_vec(&frame)?.into()).await?;
    let x=curriculum::retina(&frame);net.train_modulated(&x,&curriculum::summary(&x),0.3);
    let(_,emb)=net.forward(&x);let id=format!("V{}",protos.assign(&emb));
    emit(nc,"visual","representation",id,1.0,emb,Some(format!("{source} {}x{}",frame.width,frame.height)),3).await
}
/// Runs on its own thread: nokhwa cameras are blocking and not Send.
fn camera_loop(index:u32,tx:tokio::sync::mpsc::Sender<Frame>){
    let run=||->Result<()>{
        let requested=RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        let mut camera=Camera::new(CameraIndex::Index(index),requested)?;camera.open_stream()?;
        loop{let decoded=camera.frame()?.decode_image::<nokhwa::pixel_format::RgbFormat>()?;
            if let Some(f)=to_gray(decoded.as_raw(),decoded.width()as usize,decoded.height()as usize){if tx.blocking_send(f).is_err(){return Ok(())}}
            std::thread::sleep(Duration::from_millis(100));}
    };
    if let Err(e)=run(){eprintln!("camera unavailable: {e}")}
}
fn to_gray(rgb:&[u8],w:usize,h:usize)->Option<Frame>{
    if w==0||h==0||rgb.len()<w*h*3{return None}
    let(ow,oh)=(curriculum::WIDTH,curriculum::HEIGHT);let mut gray=Vec::with_capacity(ow*oh);
    for oy in 0..oh{for ox in 0..ow{let i=((oy*h/oh)*w+ox*w/ow)*3;gray.push((rgb[i]as f32*0.299+rgb[i+1]as f32*0.587+rgb[i+2]as f32*0.114)as u8);}}
    Some(Frame{width:ow,height:oh,gray})
}
async fn perceptual(nc:async_nats::Client)->Result<()>{
    let mut sub=nc.subscribe(incoming_subject("perceptual")).await?;let mut prototypes:Vec<(String,Vec<f32>,u32)>=vec![];
    let mut net=NeuralNet::load_or_new("weights/cortex-0/perceptual.bin",16,24,16,0x9E2C,0.008);let mut prediction:Option<Vec<f32>>=None;let mut last_save=Instant::now();
    while let Some(msg)=sub.next().await{let s:Signal=serde_json::from_slice(&msg.payload)?;
        if s.source=="associative"{prediction=Some(fit(&s.embedding,16));continue} if s.source=="workspace"{continue}
        let actual=fit(&s.embedding,16);let surprise=prediction.as_ref().map(|p|(1.0-cosine(p,&actual)).clamp(0.0,2.0)).unwrap_or(0.0);
        net.train_modulated(&actual,&actual,(s.connection_weight*(0.5+surprise)).clamp(0.1,2.0));let(_,latent)=net.forward(&actual);
        let best=prototypes.iter().enumerate().map(|(i,(_,e,_))|(i,cosine(e,&latent))).max_by(|a,b|a.1.total_cmp(&b.1));
        let(symbol,idx)=match best{Some((i,score))if score>0.86=>(prototypes[i].0.clone(),Some(i)),_=>(format!("P{}",prototypes.len()+1),None)};
        if let Some(i)=idx{let(_,e,count)=&mut prototypes[i];let c=*count as f32;for(x,y)in e.iter_mut().zip(&latent){*x=(*x*c+*y)/(c+1.0)}*count+=1}else{prototypes.push((symbol.clone(),latent.clone(),1))}
        if s.ttl>0{emit(&nc,"perceptual","representation",symbol,1.0,latent,Some(format!("from {}",s.source)),s.ttl-1).await?;}prediction=None;autosave(&net,"weights/cortex-0/perceptual.bin",&mut last_save);
    }Ok(())
}
async fn memory(nc:async_nats::Client)->Result<()>{
    let mut sub=nc.subscribe(incoming_subject("memory")).await?;let mut recent:Vec<Signal>=vec![];let mut net=NeuralNet::load_or_new("weights/cortex-0/memory.bin",16,32,16,0x4D454D,0.006);let mut last_save=Instant::now();
    while let Some(msg)=sub.next().await{let s:Signal=serde_json::from_slice(&msg.payload)?;if s.source=="workspace"{continue}
        let x=fit(&s.embedding,16);net.train_modulated(&x,&x,s.connection_weight);let(_,mem)=net.forward(&x);
        let related=recent.iter().rev().find(|r|cosine(&r.embedding,&mem)>0.90).cloned();let mut stored=s.clone();stored.embedding=mem.clone();recent.push(stored);if recent.len()>64{recent.remove(0);}
        if let(Some(r),true)=(related,s.ttl>0){emit(&nc,"memory","recall",format!("M:{}",r.symbol),0.65,mem,Some(format!("recalled by {}",s.symbol)),s.ttl-1).await?;}autosave(&net,"weights/cortex-0/memory.bin",&mut last_save);
    }Ok(())
}
/// One remembered image latent with the words it was taught alongside.
#[derive(Clone,Serialize,Deserialize)]
struct Proto{emb:Vec<f32>,words:HashMap<String,f32>}
const LINKS_PATH:&str="weights/cortex-0/links.json";
const PAIR_WINDOW:Duration=Duration::from_millis(1000);
const MAX_PROTOS:usize=800;
fn dist(a:&[f32],b:&[f32])->f32{a.iter().zip(b).map(|(x,y)|(x-y)*(x-y)).sum::<f32>().sqrt()}
/// Hebbian-style linking: strengthen word weight on the nearest image prototype, or create one.
fn teach(protos:&mut Vec<Proto>,emb:&[f32],word:&str)->usize{
    let best=protos.iter().enumerate().map(|(i,p)|(i,dist(&p.emb,emb))).min_by(|a,b|a.1.total_cmp(&b.1));
    let idx=match best{
        Some((i,d))if d<0.03=>{let p=&mut protos[i];let n:f32=p.words.values().sum();for(x,y)in p.emb.iter_mut().zip(emb){*x=(*x*n+*y)/(n+1.0)}i},
        _=>{protos.push(Proto{emb:emb.to_vec(),words:HashMap::new()});protos.len()-1}
    };
    *protos[idx].words.entry(word.to_string()).or_default()+=1.0;
    if protos.len()>MAX_PROTOS{let weakest=protos.iter().enumerate().min_by(|a,b|a.1.words.values().sum::<f32>().total_cmp(&b.1.words.values().sum::<f32>())).map(|(i,_)|i).unwrap_or(0);protos.remove(weakest);return idx.min(protos.len().saturating_sub(1))}
    idx
}
/// k-nearest-prototype vote: returns the best word and its share of the vote.
fn classify(protos:&[Proto],emb:&[f32])->Option<(String,f32)>{
    let mut near:Vec<(f32,&Proto)>=protos.iter().map(|p|(dist(&p.emb,emb),p)).collect();
    near.sort_by(|a,b|a.0.total_cmp(&b.0));
    let mut votes:HashMap<&str,f32>=HashMap::new();let mut total=0.0;
    for(d,p)in near.iter().take(7){let w=1.0/(d+0.02);let n:f32=p.words.values().sum();for(word,c)in &p.words{let v=w*c/n;*votes.entry(word).or_default()+=v;total+=v;}}
    votes.into_iter().max_by(|a,b|a.1.total_cmp(&b.1)).map(|(w,v)|(w.to_string(),v/total.max(1e-6)))
}
async fn associative(nc:async_nats::Client)->Result<()>{
    let mut sub=nc.subscribe(incoming_subject("associative")).await?;
    let mut net=NeuralNet::load_or_new("weights/cortex-0/associative.bin",32,32,16,0xA550C,0.01);
    let mut protos:Vec<Proto>=std::fs::read(LINKS_PATH).ok().and_then(|b|serde_json::from_slice(&b).ok()).unwrap_or_default();
    println!("Associative cortex: {} image prototypes loaded",protos.len());
    let mut last_word:Option<(String,Vec<f32>,Instant)>=None;let mut last_image:Option<(Vec<f32>,Instant,bool)>=None;
    let mut last_recall:Option<(String,Instant)>=None;let mut dirty=false;let mut last_save=Instant::now();
    while let Some(msg)=sub.next().await{let s:Signal=serde_json::from_slice(&msg.payload)?;if s.source=="workspace"{continue}
        let now=Instant::now();let out=s.ttl.saturating_sub(1);
        if s.source=="language"&&s.kind=="representation"{
            let word=s.symbol.trim_start_matches("W:").to_string();
            if let Some((emb,t,taught))=&last_image{if !*taught&&now.duration_since(*t)<PAIR_WINDOW{let i=teach(&mut protos,emb,&word);dirty=true;
                if s.ttl>0{emit(&nc,"associative","association",format!("C:P{i}<->{word}"),1.0,s.embedding.clone(),Some(format!("taught {word}")),out).await?;}}}
            last_word=Some((word,s.embedding.clone(),now));
        }else if s.source=="perceptual"&&s.context.as_deref()==Some("from visual"){
            let fresh_word=last_word.as_ref().is_some_and(|(_,_,t)|now.duration_since(*t)<PAIR_WINDOW);last_image=Some((s.embedding.clone(),now,fresh_word));
            if let Some((word,wemb,t))=&last_word{if now.duration_since(*t)<PAIR_WINDOW{
                let i=teach(&mut protos,&s.embedding,word);dirty=true;
                let mut pair=fit(&s.embedding,16);pair.extend(fit(wemb,16));let target=(0..16).map(|k|(pair[k]+pair[k+16])*0.5).collect::<Vec<_>>();net.train_modulated(&pair,&target,s.connection_weight);
                if s.ttl>0{emit(&nc,"associative","association",format!("C:P{i}<->{word}"),1.0,s.embedding.clone(),Some(format!("taught {word}")),out).await?;}
                continue}}
            if s.ttl>0{if let Some((word,conf))=classify(&protos,&s.embedding){
                if last_recall.as_ref().map_or(true,|(w,t)|w!=&word||now.duration_since(*t)>Duration::from_millis(500)){
                    last_recall=Some((word.clone(),now));
                    emit(&nc,"associative","recall",format!("W:{word}"),conf,s.embedding.clone(),Some(format!("{:.0}% sure",conf*100.0)),out).await?;}}}
        }
        if dirty&&last_save.elapsed()>=Duration::from_secs(5){let _=std::fs::create_dir_all("weights/cortex-0");let _=std::fs::write(LINKS_PATH,serde_json::to_vec(&protos)?);dirty=false;last_save=Instant::now();}
    }Ok(())
}
async fn workspace(nc:async_nats::Client)->Result<()>{
    let mut sub=nc.subscribe(incoming_subject("workspace")).await?;let mut last:Option<Instant>=None;
    while let Some(msg)=sub.next().await{let s:Signal=serde_json::from_slice(&msg.payload)?;let salience=s.activation*s.connection_weight;
        if s.ttl==0||salience<0.70||last.is_some_and(|t|t.elapsed()<Duration::from_millis(300)){continue}
        last=Some(Instant::now());
        emit(&nc,"workspace","broadcast",s.symbol,salience,s.embedding,Some(format!("selected from {}",s.source)),s.ttl-1).await?;}Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teach_then_classify_votes_for_nearest_word() {
        let mut protos=vec![];
        for i in 0..5{let d=i as f32*0.01;teach(&mut protos,&[1.0+d,0.0,0.0],"circle");teach(&mut protos,&[0.0,1.0+d,0.0],"square");}
        let (w,conf)=classify(&protos,&[0.97,0.05,0.0]).unwrap();
        assert_eq!(w,"circle");assert!(conf>0.5,"conf {conf}");
        assert_eq!(classify(&protos,&[0.05,1.02,0.0]).unwrap().0,"square");
        assert!(classify(&[],&[1.0]).is_none());
    }

    #[test]
    fn teach_merges_identical_latents_and_counts_words() {
        let mut protos=vec![];
        teach(&mut protos,&[0.5,0.5],"a");teach(&mut protos,&[0.5,0.5],"a");teach(&mut protos,&[0.5,0.5],"b");
        assert_eq!(protos.len(),1);assert_eq!(protos[0].words["a"],2.0);assert_eq!(protos[0].words["b"],1.0);
    }

    #[test]
    fn gray_conversion_downsamples_to_curriculum_size() {
        let f=to_gray(&vec![200u8;128*64*3],128,64).unwrap();
        assert_eq!((f.width,f.height,f.gray.len()),(curriculum::WIDTH,curriculum::HEIGHT,curriculum::WIDTH*curriculum::HEIGHT));
        assert!(to_gray(&[0u8;3],10,10).is_none());
    }
}
