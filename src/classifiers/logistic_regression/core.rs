use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;

use crate::advanced_models::LabelEncoder;
use crate::classifier::softmax_scores;
use crate::dataset::TrainingSample;
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
    pub(crate) fn fit(matrix: &DenseMatrix, samples: &[TrainingSample], epochs: usize, lr: f32, lambda: f32, threads: Option<usize>) -> Self {
        let encoder = LabelEncoder::fit(samples);
        let classes = encoder.labels.len();
        let features = matrix.shape()[1];
        let x = matrix.clone();
        let y_idx: Vec<usize> = samples.iter().map(|s| encoder.encode(&s.label)).collect();

        let models: Vec<(Vec<f32>, f32)> = install_pool(threads, || {
            (0..classes)
                .into_par_iter()
                .map(|class_id| {
                    let mut w = vec![0.0f32; features];
                    let mut b = 0.0f32;
                    for _ in 0..epochs {
                        let mut grad_w = vec![0.0f32; features];
                        let mut grad_b = 0.0f32;
                        for row in 0..x.shape()[0] {
                            let y = if y_idx[row] == class_id { 1.0 } else { 0.0 };
                            let mut z = b;
                            for col in 0..features {
                                z += x[[row, col]] * w[col];
                            }
                            let pred = 1.0 / (1.0 + (-z).exp());
                            let diff = pred - y;
                            for col in 0..features {
                                grad_w[col] += diff * x[[row, col]];
                            }
                            grad_b += diff;
                        }
                        let inv_n = 1.0 / (x.shape()[0].max(1) as f32);
                        for col in 0..features {
                            grad_w[col] = grad_w[col] * inv_n + lambda * w[col];
                            w[col] -= lr * grad_w[col];
                        }
                        b -= lr * grad_b * inv_n;
                    }
                    (w, b)
                })
                .collect()
        });

        let mut weights = Array2::<f32>::zeros((classes, features));
        let mut bias = Array1::<f32>::zeros(classes);
        for (class_id, (w, b)) in models.into_iter().enumerate() {
            for col in 0..features {
                weights[[class_id, col]] = w[col];
            }
            bias[class_id] = b;
        }

        Self { weights, bias, encoder }
    }

    pub(crate) fn predict_scores(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
        let row = probe.index_axis(Axis(0), 0);
        let mut raw = Vec::new();
        for class_id in 0..self.encoder.labels.len() {
            let mut z = self.bias[class_id];
            for col in 0..row.len() {
                z += self.weights[[class_id, col]] * row[col];
            }
            let p = 1.0 / (1.0 + (-z).exp());
            raw.push((self.encoder.decode(class_id), p.max(1e-6).ln()));
        }
        softmax_scores(&raw)
    }
}
