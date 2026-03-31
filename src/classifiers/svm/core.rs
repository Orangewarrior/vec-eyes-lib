use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;

use crate::advanced_models::{LabelEncoder, SvmConfig, SvmKernel};
use crate::classifier::softmax_scores;
use crate::dataset::TrainingSample;
use crate::labels::ClassificationLabel;
use crate::nlp::DenseMatrix;
use crate::parallel::install_pool;

fn apply_svm_kernel_map(matrix: &DenseMatrix, config: &SvmConfig) -> DenseMatrix {
    let mut mapped = matrix.clone();
    match config.kernel {
        SvmKernel::Linear => mapped,
        SvmKernel::Rbf => {
            for row in 0..mapped.shape()[0] {
                for col in 0..mapped.shape()[1] {
                    let value = mapped[[row, col]];
                    mapped[[row, col]] = (-config.gamma * value * value).exp();
                }
            }
            mapped
        }
        SvmKernel::Polynomial => {
            for row in 0..mapped.shape()[0] {
                for col in 0..mapped.shape()[1] {
                    let value = mapped[[row, col]];
                    mapped[[row, col]] = (config.gamma * value + config.coef0).powi(config.degree as i32);
                }
            }
            mapped
        }
        SvmKernel::Sigmoid => {
            for row in 0..mapped.shape()[0] {
                for col in 0..mapped.shape()[1] {
                    let value = mapped[[row, col]];
                    mapped[[row, col]] = (config.gamma * value + config.coef0).tanh();
                }
            }
            mapped
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LinearSvmOVR {
    weights: Array2<f32>,
    bias: Array1<f32>,
    encoder: LabelEncoder,
    config: SvmConfig,
}

impl LinearSvmOVR {
    pub(crate) fn fit(matrix: &DenseMatrix, samples: &[TrainingSample], config: &SvmConfig, threads: Option<usize>) -> Self {
        let encoder = LabelEncoder::fit(samples);
        let x = apply_svm_kernel_map(matrix, config);
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
        Self { weights, bias, encoder, config: config.clone() }
    }

    pub(crate) fn predict_scores(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
        let mapped = apply_svm_kernel_map(probe, &self.config);
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
