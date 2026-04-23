#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    Manhattan,
    Minkowski(f32),
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
