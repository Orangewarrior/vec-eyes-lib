use ndarray::{Array1, Array2};
use rand::prelude::*;
use rayon::prelude::*;

use crate::advanced_models::LabelEncoder;
use crate::dataset::TrainingSample;
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::math::softmax_scores;
use crate::nlp::DenseMatrix;
use crate::parallel::install_pool;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
        let y_idx: Vec<usize> = samples
            .iter()
            .map(|s| encoder.encode(&s.label))
            .collect::<Result<_, _>>()?;
        let n_samples = matrix.shape()[0];
        let batch_size = 64usize.min(n_samples.max(1));

        let models: Vec<(Array1<f32>, f32)> = install_pool(threads, || {
            (0..classes)
                .into_par_iter()
                .map(|class_id| {
                    let mut w = Array1::<f32>::zeros(features);
                    let mut b = 0.0f32;
                    // Per-class deterministic RNG — independent across parallel classes.
                    let mut rng = StdRng::seed_from_u64(
                        0xBAD_FEED ^ (class_id as u64).wrapping_mul(6364136223846793005),
                    );
                    let mut indices: Vec<usize> = (0..n_samples).collect();
                    // Gradient accumulator reused across batches (zero-filled each batch).
                    let mut grad_w = Array1::<f32>::zeros(features);

                    for epoch in 0..epochs {
                        indices.shuffle(&mut rng);
                        let epoch_lr = lr / (1.0 + epoch as f32 * 0.02);
                        let mut start = 0usize;
                        while start < indices.len() {
                            let end = (start + batch_size).min(indices.len());
                            grad_w.fill(0.0);
                            let mut grad_b = 0.0f32;

                            for &row_idx in &indices[start..end] {
                                let y = if y_idx[row_idx] == class_id {
                                    1.0f32
                                } else {
                                    0.0f32
                                };
                                let x_row = matrix.row(row_idx);
                                // sdot: BLAS or LLVM auto-vec
                                let z = b + w.dot(&x_row);
                                let diff = 1.0 / (1.0 + (-z).exp()) - y;
                                // axpy: grad_w += diff * x_row (SIMD-friendly Zip)
                                ndarray::Zip::from(&mut grad_w)
                                    .and(&x_row)
                                    .for_each(|g, &xv| *g += diff * xv);
                                grad_b += diff;
                            }

                            let inv_n = 1.0 / (end - start).max(1) as f32;
                            // w *= (1 - lr·λ)  then  w -= lr/n · grad_w
                            // Equivalent to the original grad_w*inv_n + lambda*w update.
                            w.mapv_inplace(|v| v * (1.0 - epoch_lr * lambda));
                            // daxpy: w -= (lr/n) * grad_w
                            w.scaled_add(-epoch_lr * inv_n, &grad_w);
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
            weights.row_mut(class_id).assign(&w);
            bias[class_id] = b;
        }

        Ok(Self {
            weights,
            bias,
            encoder,
        })
    }

    pub(crate) fn predict_scores(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
        let row = probe.row(0);
        // (classes, features) @ (features,) + bias → (classes,): BLAS gemv when enabled
        let z_vec = self.weights.dot(&row) + &self.bias;
        let raw: Vec<(ClassificationLabel, f32)> = z_vec
            .iter()
            .enumerate()
            .map(|(i, &z)| {
                let p = 1.0 / (1.0 + (-z).exp());
                (self.encoder.decode(i), p.max(1e-9).ln())
            })
            .collect();
        softmax_scores(&raw)
    }
}
