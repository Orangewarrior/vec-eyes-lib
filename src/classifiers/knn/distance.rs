#[derive(Debug, Clone)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    Manhattan,
    Minkowski(f32),
}

pub(crate) fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum();
    let nb: f32 = b.iter().map(|x| x * x).sum();
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 { 1.0 } else { 1.0 - (dot / denom) }
}

pub(crate) fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

pub(crate) fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

pub(crate) fn minkowski_distance(a: &[f32], b: &[f32], p: f32) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs().powf(p))
        .sum::<f32>()
        .powf(1.0 / p)
}
