#[derive(Debug, Clone)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    Manhattan,
    Minkowski(f32),
}

pub(crate) fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for idx in 0..a.len() {
        dot += a[idx] * b[idx];
        na += a[idx] * a[idx];
        nb += b[idx] * b[idx];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 { 1.0 } else { 1.0 - (dot / denom) }
}

pub(crate) fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for idx in 0..a.len() {
        let d = a[idx] - b[idx];
        acc += d * d;
    }
    acc.sqrt()
}

pub(crate) fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for idx in 0..a.len() {
        acc += (a[idx] - b[idx]).abs();
    }
    acc
}

pub(crate) fn minkowski_distance(a: &[f32], b: &[f32], p: f32) -> f32 {
    let mut acc = 0.0f32;
    for idx in 0..a.len() {
        acc += (a[idx] - b[idx]).abs().powf(p);
    }
    acc.powf(1.0 / p)
}
