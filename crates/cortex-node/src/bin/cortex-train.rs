use anyhow::{bail, Result};
#[path = "../neural.rs"]\nmod neural;
use neural::{fit, sensory_vector, NeuralNet};

fn sample(seed:u64,n:usize)->Vec<f32>{
    let mut s=seed; let mut v=Vec::with_capacity(n);
    for _ in 0..n { s=s.wrapping_mul(6364136223846793005).wrapping_add(1); v.push((((s>>32) as u32) as f32/u32::MAX as f32)*2.0-1.0); }
    let norm=v.iter().map(|x|x*x).sum::<f32>().sqrt().max(1e-6); v.iter_mut().for_each(|x|*x/=norm); v
}
fn train_auto(net:&mut NeuralNet,input:usize,steps:usize,seed:u64)->f32{
    let mut loss=0.; for i in 0..steps { let x=sample(seed+i as u64,input); loss=net.train(&x,&fit(&x,16)); } loss
}
fn main()->Result<()>{
    let cortex=std::env::args().nth(1).unwrap_or_else(||"all".into());
    let steps:usize=std::env::var("CORTEX_TRAIN_STEPS").ok().and_then(|v|v.parse().ok()).unwrap_or(20_000);
    std::fs::create_dir_all("weights/cortex-0")?;
    let mut one=|name:&str,input,hidden,seed,lr|->Result<()>{
        println!("training {name}: {input}->{hidden}->16, {steps} steps");
        let mut net=NeuralNet::new(input,hidden,16,seed,lr);
        let loss=train_auto(&mut net,input,steps,seed^0xC07E_0000);
        let path=format!("weights/cortex-0/{name}.bin"); net.save(&path)?;
        println!("saved {path}; final local loss {loss:.6}"); Ok(())
    };
    match cortex.as_str(){
      "visual"=>one("visual",64,48,0x715A1,0.006)?,
      "perceptual"=>one("perceptual",16,24,0x9E2C,0.008)?,
      "memory"=>one("memory",16,32,0x4D454D,0.006)?,
      "associative"=>{
          println!("training associative: paired cortical symbols");
          let mut net=NeuralNet::new(32,32,16,0xA550C,0.01); let mut loss=0.;
          for i in 0..steps { let a=sample(0xA55+i as u64,16); let b=sample(0xB66+(i/4) as u64,16); let mut pair=a.clone(); pair.extend(&b); let target=(0..16).map(|j|(a[j]+b[j])*.5).collect::<Vec<_>>(); loss=net.train(&pair,&target); }
          net.save("weights/cortex-0/associative.bin")?; println!("saved weights/cortex-0/associative.bin; loss {loss:.6}");
      },
      "language"=>{
          println!("training language: byte-pattern encoder");
          let mut net=NeuralNet::new(32,24,16,0x1A6E,0.015); let corpus=["the quick brown fox","hello world","memory predicts context","symbols emerge from experience","visual auditory language association"];
          let mut loss=0.; for i in 0..steps {let x=sensory_vector(corpus[i%corpus.len()].as_bytes(),32);loss=net.train(&x,&fit(&x,16));}
          net.save("weights/cortex-0/language.bin")?; println!("saved weights/cortex-0/language.bin; loss {loss:.6}");
      },
      "all"=>{ for (name,input,hidden,seed,lr) in [("visual",64,48,0x715A1,0.006),("perceptual",16,24,0x9E2C,0.008),("memory",16,32,0x4D454D,0.006)] {one(name,input,hidden,seed,lr)?;} 
          std::process::Command::new(std::env::current_exe()?).arg("associative").status()?; std::process::Command::new(std::env::current_exe()?).arg("language").status()?;
      },
      _=>bail!("usage: cortex-train <visual|perceptual|memory|associative|language|all>")
    } Ok(())
}
