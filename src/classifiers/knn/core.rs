use std::collections::HashMap;

use crate::math::softmax_scores;
use crate::classifiers::knn::{euclidean_distance_squared, manhattan_distance, minkowski_distance, DenseFeatureModel, DistanceMetric, KnnClassifier};
use crate::dataset::TrainingSample;
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::nlp::{dense_matrix_from_texts, DenseMatrix, FastTextConfigBuilder, NlpOption, WordEmbeddingModel};
use crate::parallel::install_pool;

/// Compute column-wise mean and std, normalise every row in place.
/// Returns `(mean, std)` for use at inference time.
pub(crate) fn apply_feature_normalization(matrix: &mut DenseMatrix) -> (Vec<f32>, Vec<f32>) {
    let rows = matrix.shape()[0].max(1) as f32;
    let cols = matrix.shape()[1];

    // BLAS/SIMD-backed column-wise mean via ndarray reduction.
    let mean = matrix
        .mean_axis(ndarray::Axis(0))
        .unwrap_or_else(|| ndarray::Array1::zeros(cols));

    // Column-wise variance: accumulate (x - mean)^2 row-by-row with Zip
    // so each inner loop is contiguous (row-major) and LLVM-vectorisable.
    let mut var = ndarray::Array1::<f32>::zeros(cols);
    ndarray::Zip::from(matrix.rows()).for_each(|row| {
        ndarray::Zip::from(&mut var)
            .and(&row)
            .and(&mean)
            .for_each(|v, &x, &m| *v += (x - m) * (x - m));
    });
    let std_vals = var.mapv(|v| (v / rows).sqrt().max(1e-6));

    // Normalize rows in-place.
    ndarray::Zip::from(matrix.rows_mut()).for_each(|mut row| {
        ndarray::Zip::from(&mut row)
            .and(&mean)
            .and(&std_vals)
            .for_each(|v, &m, &s| *v = (*v - m) / s);
    });

    (mean.to_vec(), std_vals.to_vec())
}

pub(crate) fn train(
    samples: &[TrainingSample],
    nlp: NlpOption,
    metric: DistanceMetric,
    dims: usize,
    k: usize,
    threads: Option<usize>,
    normalize_features: bool,
) -> Result<KnnClassifier, VecEyesError> {
    let texts: Vec<&str> = samples.iter().map(|s| s.text.as_str()).collect();
    let labels: Vec<ClassificationLabel> = samples.iter().map(|s| s.label.clone()).collect();

    let model = match nlp {
        NlpOption::Word2Vec => DenseFeatureModel::Word2Vec(WordEmbeddingModel::train_word2vec(&texts, dims)),
        NlpOption::FastText => {
            let config = FastTextConfigBuilder::new().build().expect("default FastTextConfigBuilder must be valid");
            DenseFeatureModel::FastText(WordEmbeddingModel::train_fasttext(&texts, dims, config))
        }
        _ => return Err(VecEyesError::invalid_config("classifier::KnnClassifier::train", "KNN requires Word2Vec or FastText")),
    };

    let mut matrix = match &model {
        DenseFeatureModel::Word2Vec(inner) => dense_matrix_from_texts(inner, &texts),
        DenseFeatureModel::FastText(inner) => dense_matrix_from_texts(inner, &texts),
        // External-embedding variants are never constructed by this path.
        DenseFeatureModel::ExternalFastText { .. } => unreachable!("ExternalFastText not reachable from core::train"),
        DenseFeatureModel::ExternalEmbeddings { .. } => unreachable!("ExternalEmbeddings not reachable from core::train"),
    };

    let (feature_mean, feature_std) = if normalize_features {
        let (mean, std) = apply_feature_normalization(&mut matrix);
        (Some(mean), Some(std))
    } else {
        (None, None)
    };

    Ok(KnnClassifier::from_parts(metric, threads, labels, matrix, model, k, normalize_features, feature_mean, feature_std))
}

pub(crate) fn score_neighbors(model: &KnnClassifier, text: &str) -> Vec<(ClassificationLabel, f32)> {
    let probe = model.matrix_for_text(text);
    let probe_row = probe.row(0);
    let probe_vec: Vec<f32> = probe_row.iter().copied().collect();

    // Precompute probe norm once — reused for every cosine / euclidean comparison.
    let probe_norm_sq: f32 = probe_vec.iter().map(|v| v * v).sum();
    let probe_norm = probe_norm_sq.sqrt();

    let mut ranked: Vec<(f32, ClassificationLabel)> = install_pool(model.threads(), || {
        use rayon::prelude::*;
        (0..model.matrix().shape()[0])
            .into_par_iter()
            .map(|row_idx| {
                let candidate = model.matrix().row(row_idx);
                let cand: &[f32] = candidate.as_slice().unwrap_or(&[]);
                let distance = match model.metric() {
                    DistanceMetric::Cosine => {
                        // Reuse pre-computed probe norm — saves one sqrt per candidate.
                        let dot: f32 = probe_vec.iter().zip(cand).map(|(a, b)| a * b).sum();
                        let cand_norm: f32 = cand.iter().map(|v| v * v).sum::<f32>().sqrt();
                        if probe_norm < 1e-8 || cand_norm < 1e-8 { 1.0 }
                        else { 1.0 - dot / (probe_norm * cand_norm) }
                    }
                    DistanceMetric::Euclidean => {
                        euclidean_distance_squared(&probe_vec, cand, probe_norm_sq).sqrt()
                    }
                    DistanceMetric::Manhattan => manhattan_distance(&probe_vec, cand),
                    DistanceMetric::Minkowski(p) => minkowski_distance(&probe_vec, cand, *p),
                };
                (distance, model.labels()[row_idx].clone())
            })
            .collect()
    });

    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut best: HashMap<ClassificationLabel, f32> = HashMap::new();
    let limit = model.k().min(ranked.len());
    for (distance, label) in ranked.into_iter().take(limit) {
        *best.entry(label).or_insert(0.0) += (-distance).exp();
    }

    softmax_scores(&best.into_iter().collect::<Vec<_>>())
}
