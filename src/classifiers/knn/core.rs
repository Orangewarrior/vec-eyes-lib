use std::collections::HashMap;

use ndarray::Axis;

use crate::classifier::softmax_scores;
use crate::classifiers::knn::{cosine_distance, euclidean_distance, manhattan_distance, minkowski_distance, DenseFeatureModel, DistanceMetric, KnnClassifier};
use crate::dataset::TrainingSample;
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::nlp::{dense_matrix_from_texts, FastTextConfigBuilder, NlpOption, WordEmbeddingModel};
use crate::parallel::install_pool;

pub(crate) fn train(
    samples: &[TrainingSample],
    nlp: NlpOption,
    metric: DistanceMetric,
    dims: usize,
    k: usize,
    threads: Option<usize>,
) -> Result<KnnClassifier, VecEyesError> {
    let texts: Vec<String> = samples.iter().map(|s| s.text.clone()).collect();
    let labels: Vec<ClassificationLabel> = samples.iter().map(|s| s.label.clone()).collect();

    let model = match nlp {
        NlpOption::Word2Vec => DenseFeatureModel::Word2Vec(WordEmbeddingModel::train_word2vec(&texts, dims)),
        NlpOption::FastText => {
            let config = FastTextConfigBuilder::new().build().expect("default FastTextConfigBuilder must be valid");
            DenseFeatureModel::FastText(WordEmbeddingModel::train_fasttext(&texts, dims, config))
        }
        _ => return Err(VecEyesError::invalid_config("classifier::KnnClassifier::train", "KNN requires Word2Vec or FastText")),
    };

    let matrix = match &model {
        DenseFeatureModel::Word2Vec(inner) => dense_matrix_from_texts(inner, &texts),
        DenseFeatureModel::FastText(inner) => dense_matrix_from_texts(inner, &texts),
    };

    Ok(KnnClassifier::from_parts(metric, threads, labels, matrix, model, k))
}

pub(crate) fn score_neighbors(model: &KnnClassifier, text: &str) -> Vec<(ClassificationLabel, f32)> {
    let probe = model.matrix_for_text(text);
    let probe_row = probe.index_axis(Axis(0), 0);
    let probe_vec = probe_row.to_vec();
    let mut ranked: Vec<(f32, ClassificationLabel)> = install_pool(model.threads(), || {
        use rayon::prelude::*;
        (0..model.matrix().shape()[0])
            .into_par_iter()
            .map(|row| {
                let candidate = model.matrix().index_axis(Axis(0), row);
                let candidate_slice = candidate.as_slice().unwrap_or(&[]);
                let distance = match model.metric() {
                    DistanceMetric::Cosine => cosine_distance(&probe_vec, candidate_slice),
                    DistanceMetric::Euclidean => euclidean_distance(&probe_vec, candidate_slice),
                    DistanceMetric::Manhattan => manhattan_distance(&probe_vec, candidate_slice),
                    DistanceMetric::Minkowski(p) => minkowski_distance(&probe_vec, candidate_slice, *p),
                };
                (distance, model.labels()[row].clone())
            })
            .collect::<Vec<_>>()
    });

    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut best: HashMap<ClassificationLabel, f32> = HashMap::new();
    let limit = model.k().min(ranked.len());
    for (distance, label) in ranked.into_iter().take(limit) {
        let score = (1.0 / (distance + 1e-6)).min(1000.0);
        *best.entry(label).or_insert(0.0) += score;
    }

    let raw: Vec<_> = best.into_iter().collect();
    softmax_scores(&raw)
}
