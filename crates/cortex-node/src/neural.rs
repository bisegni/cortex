// Tiny continuously-plastic neural primitives for Seed-0.
// Deliberately dependency-free: this is a real trainable network, not a hash encoder.
#[derive(Clone)]
pub struct NeuralNet {
    input: usize, hidden: usize, output: usize,
    w1: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>,
    lr: f32,
}
impl NeuralNet {
    pub fn new(input:usize, hidden:usize, output:usize, seed:u64, lr:f32)->Self {
        let mut s=seed;
        let mut rnd=|| { s=s.wrapping_mul(6364136223846793005).wrapping_add(1); (((s>>32) as u32) as f32/u32::MAX as f32-.5)*.25 };
        Self{input,hidden,output,w1:(0..input*hidden).map(|_|rnd()).collect(),b1:vec![0.;hidden],w2:(0..hidden*output).map(|_|rnd()).collect(),b2:vec![0.;output],lr}
    }
    pub fn forward(&self,x:&[f32])->(Vec<f32>,Vec<f32>){
        let h=(0..self.hidden).map(|j| (self.b1[j]+(0..self.input).map(|i|x[i]*self.w1[j*self.input+i]).sum::<f32>()).tanh()).collect::<Vec<_>>();
        let y=(0..self.output).map(|k| (self.b2[k]+(0..self.hidden).map(|j|h[j]*self.w2[k*self.hidden+j]).sum::<f32>()).tanh()).collect();
        (h,y)
    }
    // Local self-supervision: reconstruct a target latent. Each cortex learns only from signals it sees.
    pub fn train(&mut self,x:&[f32],target:&[f32])->f32 { self.train_modulated(x,target,1.0) }
    pub fn train_modulated(&mut self,x:&[f32],target:&[f32],modulation:f32)->f32 {
        let (h,y)=self.forward(x); let mut dy=vec![0.;self.output]; let mut loss=0.;
        for k in 0..self.output { let e=y[k]-target[k]; loss+=e*e; dy[k]=2.*e/self.output as f32*(1.-y[k]*y[k]); }
        let mut dh=vec![0.;self.hidden];
        for j in 0..self.hidden { dh[j]=(0..self.output).map(|k|dy[k]*self.w2[k*self.hidden+j]).sum::<f32>()*(1.-h[j]*h[j]); }
        for k in 0..self.output { for j in 0..self.hidden { self.w2[k*self.hidden+j]-=self.lr*modulation*dy[k]*h[j]; } self.b2[k]-=self.lr*modulation*dy[k]; }
        for j in 0..self.hidden { for i in 0..self.input { self.w1[j*self.input+i]-=self.lr*modulation*dh[j]*x[i]; } self.b1[j]-=self.lr*modulation*dh[j]; }
        loss/self.output as f32
    }
}
pub fn sensory_vector(bytes:&[u8], dims:usize)->Vec<f32>{
    let mut x=vec![0.;dims];
    for (i,b) in bytes.iter().enumerate(){ x[i%dims]+=(*b as f32/127.5)-1.; }
    let n=x.iter().map(|v|v*v).sum::<f32>().sqrt().max(1e-6); for v in &mut x{*v/=n;} x
}
pub fn fit(v:&[f32],n:usize)->Vec<f32>{(0..n).map(|i|v.get(i%v.len()).copied().unwrap_or(0.)).collect()}
