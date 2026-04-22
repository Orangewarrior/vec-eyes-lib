#[derive(Debug, Clone)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    Manhattan,
    Minkowski(f32),
}

pub(crate) fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

#[inline(always)]
pub(crate) fn euclidean_distance_squared(a: &[f32], b: &[f32], a_norm_sq: f32) -> f32 {
    // ||a-b||² = ||a||² - 2·a·b + ||b||²
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let b_norm_sq: f32 = b.iter().map(|v| v * v).sum();
    (a_norm_sq - 2.0 * dot + b_norm_sq).max(0.0)
}

pub(crate) fn minkowski_distance(a: &[f32], b: &[f32], p: f32) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs().powf(p))
        .sum::<f32>()
        .powf(1.0 / p)
}
