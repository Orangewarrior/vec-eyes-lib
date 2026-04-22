use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rayon::prelude::*;

use crate::advanced_models::LabelEncoder;
use crate::classifier::softmax_scores;
use crate::dataset::TrainingSample;
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::nlp::DenseMatrix;
use crate::parallel::install_pool;

#[derive(Debug, Clone)]
pub(crate) struct LogisticOVR {
    weights: Array2<f32>,
    bias: Array1<f32>,
    encoder: LabelEncoder,
}

impl LogisticOVR {
    pub(crate) fn fit(
        matrix: &DenseMatrix,
        samples: &[TrainingSample],
        epochs: usize,
        lr: f32,
        lambda: f32,
        threads: Option<usize>,
    ) -> Result<Self, VecEyesError> {
        let encoder = LabelEncoder::fit(samples);
        let classes = encoder.labels.len();
        let features = matrix.shape()[1];
        let x = matrix.clone();
        let y_idx: Vec<usize> = samples
            .iter()
            .map(|s| encoder.encode(&s.label))
            .collect::<Result<_, _>>()?;
        let n_samples = x.shape()[0];
        let batch_size = 64usize.min(n_samples.max(1));

        let models: Vec<(Vec<f32>, f32)> = install_pool(threads, || {
            (0..classes)
                .into_par_iter()
                .map(|class_id| {
                    let mut w = vec![0.0f32; features];
                    let mut b = 0.0f32;
                    // Each class binary gets its own RNG so parallel training is
                    // deterministic yet independent across classes.
                    let mut rng = StdRng::seed_from_u64(0xBAD_FEED ^ (class_id as u64 * 6364136223846793005));
                    let mut indices: Vec<usize> = (0..n_samples).collect();

                    for epoch in 0..epochs {
                        // Shuffle each epoch to break ordering bias
                        indices.shuffle(&mut rng);
                        let epoch_lr = lr / (1.0 + (epoch as f32) * 0.02);
                        let mut start = 0usize;
                        while start < indices.len() {
                            let end = (start + batch_size).min(indices.len());
                            let mut grad_w = vec![0.0f32; features];
                            let mut grad_b = 0.0f32;
                            for &row in &indices[start..end] {
                                let y = if y_idx[row] == class_id { 1.0 } else { 0.0 };
                                let mut z = b;
                                for col in 0..features { z += x[[row, col]] * w[col]; }
                                let diff = (1.0 / (1.0 + (-z).exp())) - y;
                                for col in 0..features { grad_w[col] += diff * x[[row, col]]; }
                                grad_b += diff;
                            }
                            let inv_n = 1.0 / ((end - start).max(1) as f32);
                            for col in 0..features {
                                grad_w[col] = grad_w[col] * inv_n + lambda * w[col];
                                w[col] -= epoch_lr * grad_w[col];
                            }
                            b -= epoch_lr * grad_b * inv_n;
                            start = end;
                        }
                    }
                    (w, b)
                })
                .collect()
        });

        let mut weights = Array2::<f32>::zeros((classes, features));
        let mut bias = Array1::<f32>::zeros(classes);
        for (class_id, (w, b)) in models.into_iter().enumerate() {
            for col in 0..features { weights[[class_id, col]] = w[col]; }
            bias[class_id] = b;
        }

        Ok(Self { weights, bias, encoder })
    }

    pub(crate) fn predict_scores(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
        let row = probe.index_axis(Axis(0), 0);
        let raw: Vec<(ClassificationLabel, f32)> = (0..self.encoder.labels.len())
            .map(|class_id| {
                let z = self.bias[class_id]
                    + row.iter().zip(self.weights.index_axis(Axis(0), class_id).iter()).map(|(a, b)| a * b).sum::<f32>();
                let p = 1.0 / (1.0 + (-z).exp());
                (self.encoder.decode(class_id), p.max(1e-9).ln())
            })
            .collect();
        softmax_scores(&raw)
    }
}
