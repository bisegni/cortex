//! Synthetic visual curriculum shared by the teacher and the offline trainer.
use crate::Frame;

pub const WIDTH: usize = 64;
pub const HEIGHT: usize = 32;
pub const CLASSES: [&str; 6] = ["circle", "square", "triangle", "cross", "hbars", "vbars"];

/// Seeds at or above this value are reserved for held-out test images.
pub const TEST_SEED_BASE: u64 = 1_000_000;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 { lo + (self.next() % (hi - lo + 1) as u64) as i32 }
}

fn inside(class: usize, dx: i32, dy: i32, r: i32) -> bool {
    match class {
        0 => dx * dx + dy * dy <= r * r,
        1 => dx.abs() <= r && dy.abs() <= r,
        2 => dy >= -r && dy <= r && dx.abs() <= (dy + r) / 2,
        3 => (dx.abs() <= r / 3 && dy.abs() <= r) || (dy.abs() <= r / 3 && dx.abs() <= r),
        4 => dx.abs() <= r * 3 / 2 && dy.abs() <= r && (dy + r).rem_euclid(4) < 2,
        _ => dx.abs() <= r * 3 / 2 && dy.abs() <= r && (dx + r * 2).rem_euclid(4) < 2,
    }
}

/// Deterministic image of `class` with position, size, brightness and noise drawn from `seed`.
pub fn render(class: usize, seed: u64) -> Frame {
    let mut rng = Rng(seed ^ ((class as u64 + 1) << 40));
    let r = rng.range(7, 12);
    let cx = rng.range(r + 2, WIDTH as i32 - r - 3);
    let cy = rng.range(r + 1, HEIGHT as i32 - r - 2);
    let fg = rng.range(150, 245) as u8;
    let mut gray = vec![0u8; WIDTH * HEIGHT];
    for y in 0..HEIGHT as i32 {
        for x in 0..WIDTH as i32 {
            let noise = rng.range(0, 24) as u8;
            let on = inside(class, x - cx, y - cy, r);
            gray[y as usize * WIDTH + x as usize] = if on { fg.saturating_sub(noise) } else { 10 + noise };
        }
    }
    Frame { width: WIDTH, height: HEIGHT, gray }
}

/// Innate "retina": a fixed feature extractor turning a frame into the 64-dim visual cortex input.
/// Features are measured on the bright mask relative to its bounding box, so they are largely
/// invariant to position, size and brightness (shape summary, width/height profiles, stripe texture).
pub fn retina(frame: &Frame) -> Vec<f32> {
    let (w, h) = (frame.width, frame.height);
    let mut out = vec![-1.0f32; 64];
    if w == 0 || h == 0 || frame.gray.len() < w * h { return out; }
    let (mn, mx) = frame.gray.iter().fold((255u8, 0u8), |(a, b), v| (a.min(*v), b.max(*v)));
    if mx.saturating_sub(mn) < 40 { return out; }
    let thr = ((mn as u16 + mx as u16) / 2) as u8;
    let on = |x: usize, y: usize| frame.gray[y * w + x] > thr;
    let (mut x0, mut x1, mut y0, mut y1, mut area) = (w, 0, h, 0, 0usize);
    let (mut sx, mut sy) = (0f32, 0f32);
    for y in 0..h { for x in 0..w { if on(x, y) { x0 = x0.min(x); x1 = x1.max(x); y0 = y0.min(y); y1 = y1.max(y); area += 1; sx += x as f32; sy += y as f32; } } }
    if area < 8 { return out; }
    let (bw, bh) = (x1 - x0 + 1, y1 - y0 + 1);
    let at = |fx: f32, fy: f32| on(x0 + ((fx * (bw - 1) as f32).round() as usize), y0 + ((fy * (bh - 1) as f32).round() as usize));
    let unit = |v: f32| (v * 2.0 - 1.0).clamp(-1.0, 1.0);
    // Transitions along rows / columns inside the bounding box.
    let row_tx: Vec<f32> = (y0..=y1).map(|y| (x0 + 1..=x1).filter(|&x| on(x, y) != on(x - 1, y)).count() as f32).collect();
    let col_ty: Vec<f32> = (x0..=x1).map(|x| (y0 + 1..=y1).filter(|&y| on(x, y) != on(x, y - 1)).count() as f32).collect();
    let row_w: Vec<f32> = (y0..=y1).map(|y| (x0..=x1).filter(|&x| on(x, y)).count() as f32 / bw as f32).collect();
    let col_h: Vec<f32> = (x0..=x1).map(|x| (y0..=y1).filter(|&y| on(x, y)).count() as f32 / bh as f32).collect();
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    let fill = area as f32 / (bw * bh) as f32;
    out[0] = unit(area as f32 / (w * h) as f32 * 4.0);
    out[1] = unit(bw as f32 / w as f32);
    out[2] = unit(bh as f32 / h as f32);
    out[3] = unit(fill);
    out[4] = unit(mean(&row_tx) / 8.0);
    out[5] = unit(mean(&col_ty) / 8.0);
    out[6] = unit(bw as f32 / (bw + bh) as f32);
    out[7] = ((sx / area as f32 - x0 as f32) / bw as f32 * 2.0 - 1.0).clamp(-1.0, 1.0);
    out[8] = ((sy / area as f32 - y0 as f32) / bh as f32 * 2.0 - 1.0).clamp(-1.0, 1.0);
    let (top, bottom) = (mean(&row_w[..row_w.len() / 2]), mean(&row_w[row_w.len() / 2..]));
    out[9] = ((top - bottom) / (top + bottom).max(1e-3)).clamp(-1.0, 1.0);
    let cells = |pts: &[(f32, f32)]| pts.iter().filter(|(a, b)| at(*a, *b)).count() as f32 / pts.len() as f32;
    out[10] = unit(cells(&[(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]));
    out[11] = unit(cells(&[(0.5, 0.5)]));
    out[12] = unit(cells(&[(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]));
    out[13] = unit(cells(&[(0.5, 0.05), (0.5, 0.95)]));
    out[14] = unit(cells(&[(0.05, 0.5), (0.95, 0.5)]));
    out[15] = 1.0;
    for k in 0..16 {
        out[16 + k] = unit(row_w[(k * (row_w.len() - 1)) / 15]);
        out[32 + k] = unit(col_h[(k * (col_h.len() - 1)) / 15]);
    }
    for k in 0..8 {
        let band = |v: &[f32]| { let (a, b) = (k * v.len() / 8, ((k + 1) * v.len() / 8).max(k * v.len() / 8 + 1)); mean(&v[a..b.min(v.len())]) };
        out[48 + k] = unit(band(&row_tx) / 8.0);
        out[56 + k] = unit(band(&col_ty) / 8.0);
    }
    out
}

/// What the visual cortex learns to predict from its input: the 16 global shape features.
pub fn summary(retina_out: &[f32]) -> Vec<f32> { retina_out[..16].to_vec() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_is_deterministic_and_varied() {
        assert_eq!(render(0, 7).gray, render(0, 7).gray);
        assert_ne!(render(0, 7).gray, render(0, 8).gray);
        assert_ne!(render(0, 7).gray, render(1, 7).gray);
    }

    #[test]
    fn classes_have_separable_features() {
        let mean = |class: usize| {
            let mut acc = vec![0f32; 16];
            for s in 0..40 { for (a, v) in acc.iter_mut().zip(retina(&render(class, s))) { *a += v / 40.0; } }
            acc
        };
        let means: Vec<_> = (0..CLASSES.len()).map(mean).collect();
        for i in 0..means.len() {
            for j in i + 1..means.len() {
                let d: f32 = means[i].iter().zip(&means[j]).map(|(a, b)| (a - b).abs()).sum();
                assert!(d > 0.15, "{} vs {} too similar ({d})", CLASSES[i], CLASSES[j]);
            }
        }
    }

    #[test]
    fn retina_has_64_dims() {
        assert_eq!(retina(&render(2, 1)).len(), 64);
        assert_eq!(summary(&retina(&render(2, 1))).len(), 16);
    }
}

#[cfg(test)]
mod knn_check {
    use super::*;
    fn acc(feat: impl Fn(&Frame) -> Vec<f32>, train: u64, k: usize) -> f32 {
        let mut bank = vec![];
        for c in 0..CLASSES.len() { for s in 0..train { bank.push((c, feat(&render(c, s)))); } }
        let (mut ok, mut n) = (0, 0);
        for c in 0..CLASSES.len() { for s in 0..20 {
            let f = feat(&render(c, TEST_SEED_BASE + s));
            let mut d: Vec<(f32, usize)> = bank.iter().map(|(bc, bf)| (bf.iter().zip(&f).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt(), *bc)).collect();
            d.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut v = [0f32; 6]; for (dd, bc) in d.iter().take(k) { v[*bc] += 1.0 / (dd + 0.02); }
            ok += (v.iter().cloned().enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0 == c) as u32; n += 1;
        } }
        ok as f32 / n as f32
    }
    #[test]
    fn retina_knn_accuracy() {
        println!("summary knn {:.2}", acc(|f| summary(&retina(f)), 30, 5));
        let a = acc(retina, 30, 5);
        println!("retina knn {a:.2}");
        assert!(a > 0.9, "retina features must be linearly-ish separable by knn, got {a}");
    }
}
