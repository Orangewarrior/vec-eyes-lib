
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;

use crate::advanced_models::{LabelEncoder, SvmConfig, SvmKernel};
use crate::classifier::softmax_scores;
use crate::dataset::TrainingSample;
use crate::labels::ClassificationLabel;
use crate::nlp::DenseMatrix;
use crate::parallel::install_pool;

fn pairwise_kernel_value(lhs: &[f32], rhs: &[f32], config: &SvmConfig) -> f32 {
    match config.kernel {
        SvmKernel::Linear => lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum(),
        SvmKernel::Rbf => {
            let squared_distance: f32 = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(a, b)| {
                    let delta = a - b;
                    delta * delta
                })
                .sum();
            (-config.gamma * squared_distance).exp()
        }
        SvmKernel::Polynomial => {
            let dot: f32 = lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum();
            (config.gamma * dot + config.coef0).powi(config.degree as i32)
        }
        SvmKernel::Sigmoid => {
            let dot: f32 = lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum();
            (config.gamma * dot + config.coef0).tanh()
        }
    }
}

fn kernelize(matrix: &DenseMatrix, reference: &DenseMatrix, config: &SvmConfig) -> DenseMatrix {
    if matches!(config.kernel, SvmKernel::Linear) && std::ptr::eq(matrix, reference) {
        return matrix.clone();
    }

    let rows = matrix.shape()[0];
    let cols = reference.shape()[0];
    let mut mapped = Array2::<f32>::zeros((rows, cols));

    for row in 0..rows {
        let lhs = matrix.index_axis(Axis(0), row).to_vec();
        for col in 0..cols {
            let rhs = reference.index_axis(Axis(0), col).to_vec();
            mapped[[row, col]] = pairwise_kernel_value(&lhs, &rhs, config);
        }
    }

    mapped
}

#[derive(Debug, Clone)]
pub(crate) struct LinearSvmOVR {
    weights: Array2<f32>,
    bias: Array1<f32>,
    encoder: LabelEncoder,
    config: SvmConfig,
    reference: DenseMatrix,
}

impl LinearSvmOVR {
    pub(crate) fn fit(matrix: &DenseMatrix, samples: &[TrainingSample], config: &SvmConfig, threads: Option<usize>) -> Self {
        let encoder = LabelEncoder::fit(samples);
        let reference = matrix.clone();
        let x = kernelize(matrix, &reference, config);
        let classes = encoder.labels.len();
        let features = x.shape()[1];
        let y_idx: Vec<usize> = samples.iter().map(|s| encoder.encode(&s.label)).collect();
        let lambda = 1.0 / config.c.max(1e-6);

        let models: Vec<(Vec<f32>, f32)> = install_pool(threads, || {
            (0..classes)
                .into_par_iter()
                .map(|class_id| {
                    let mut w = vec![0.0f32; features];
                    let mut b = 0.0f32;
                    for epoch in 0..config.epochs {
                        let local_lr = config.learning_rate / (1.0 + epoch as f32 * 0.05);
                        for row in 0..x.shape()[0] {
                            let y = if y_idx[row] == class_id { 1.0 } else { -1.0 };
                            let mut margin = b;
                            for col in 0..features {
                                margin += x[[row, col]] * w[col];
                            }
                            let signed = y * margin;
                            for col in 0..features {
                                w[col] *= 1.0 - local_lr * lambda;
                            }
                            if signed < 1.0 {
                                for col in 0..features {
                                    w[col] += local_lr * y * x[[row, col]];
                                }
                                b += local_lr * y;
                            }
                        }
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
        Self { weights, bias, encoder, config: config.clone(), reference }
    }

    pub(crate) fn predict_scores(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
        let mapped = kernelize(probe, &self.reference, &self.config);
        let row = mapped.index_axis(Axis(0), 0);
        let mut raw = Vec::new();
        for class_id in 0..self.encoder.labels.len() {
            let mut margin = self.bias[class_id];
            for col in 0..row.len() {
                margin += self.weights[[class_id, col]] * row[col];
            }
            raw.push((self.encoder.decode(class_id), margin));
        }
        softmax_scores(&raw)
    }
}
