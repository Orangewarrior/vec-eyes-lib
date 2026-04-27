use ndarray::{Array1, Array2, ArrayView1};
use rand::prelude::*;
use rayon::prelude::*;

use crate::advanced_models::{LabelEncoder, SvmConfig, SvmKernel};
use crate::math::softmax_scores;
use crate::dataset::TrainingSample;
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::nlp::DenseMatrix;
use crate::parallel::install_pool;

const N_COMPONENTS: usize = 256;
const MAX_LANDMARKS: usize = 128;

// ── Kernel map ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum KernelMap {
    Identity,
    /// RBF via Random Fourier Features (Rahimi & Recht 2007).
    /// ω ~ N(0, 2γ·I), b ~ U(0, 2π), φ(x) = √(2/D) cos(ωᵀx + b).
    Rff { weights: Array2<f32>, biases: Vec<f32>, scale: f32 },
    /// Polynomial / Sigmoid — explicit kernel against random landmarks.
    Landmark(Array2<f32>),
}

impl KernelMap {
    fn build(matrix: &DenseMatrix, config: &SvmConfig) -> Self {
        match config.kernel {
            SvmKernel::Linear => Self::Identity,
            SvmKernel::Rbf   => Self::build_rff(matrix.shape()[1], N_COMPONENTS, config.gamma),
            _                => Self::build_landmarks(matrix, config),
        }
    }

    fn build_rff(input_dim: usize, n_components: usize, gamma: f32) -> Self {
        let mut rng = StdRng::seed_from_u64(0xFEA7_C0DE);
        let std_dev = (2.0 * gamma).sqrt();
        let mut weights = Array2::<f32>::zeros((n_components, input_dim));
        let mut biases = Vec::with_capacity(n_components);
        for i in 0..n_components {
            for j in 0..input_dim {
                let u1 = rng.random_range(1e-7f32..1.0f32);
                let u2 = rng.random_range(0.0f32..1.0f32);
                let z = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
                weights[[i, j]] = z * std_dev;
            }
            biases.push(rng.random_range(0.0f32..std::f32::consts::TAU));
        }
        let scale = (2.0 / n_components as f32).sqrt();
        Self::Rff { weights, biases, scale }
    }

    fn build_landmarks(matrix: &DenseMatrix, config: &SvmConfig) -> Self {
        let rows = matrix.shape()[0];
        let cols = matrix.shape()[1];
        let keep = rows.min(MAX_LANDMARKS).max(1);
        let mut rng = StdRng::seed_from_u64(0xA9D_CAFE);
        let mut indices: Vec<usize> = (0..rows).collect();
        indices.shuffle(&mut rng);
        indices.truncate(keep);
        let mut lm = Array2::<f32>::zeros((keep, cols));
        for (out_row, &src_row) in indices.iter().enumerate() {
            lm.row_mut(out_row).assign(&matrix.row(src_row));
        }
        let _ = config;
        Self::Landmark(lm)
    }

    fn transform(&self, matrix: &DenseMatrix, config: &SvmConfig) -> DenseMatrix {
        match self {
            Self::Identity => matrix.clone(),

            Self::Rff { weights, biases, scale } => {
                // (N, D) @ (K, D)ᵀ → (N, K): one BLAS dgemm call when the
                // `blas` feature is enabled; LLVM-vectorised otherwise.
                let mut mapped = matrix.dot(&weights.t());

                // Fused bias-add + cos × scale row-by-row (SIMD element-wise).
                let biases_view = ArrayView1::from(biases.as_slice());
                for mut row in mapped.rows_mut() {
                    ndarray::Zip::from(&mut row)
                        .and(biases_view)
                        .for_each(|v, &b| *v = *scale * (*v + b).cos());
                }
                mapped
            }

            Self::Landmark(ref_matrix) => {
                let rows = matrix.shape()[0];
                let n_lm = ref_matrix.shape()[0];
                let mut mapped = Array2::<f32>::zeros((rows, n_lm));
                mapped
                    .axis_iter_mut(ndarray::Axis(0))
                    .into_par_iter()
                    .zip(matrix.axis_iter(ndarray::Axis(0)).into_par_iter())
                    .for_each(|(mut out_row, lhs)| {
                        let lhs_slice = lhs.as_slice().unwrap_or(&[]);
                        for col in 0..n_lm {
                            let rhs = ref_matrix.row(col);
                            out_row[col] = explicit_kernel(lhs_slice, rhs.as_slice().unwrap_or(&[]), config);
                        }
                    });
                mapped
            }
        }
    }
}

fn explicit_kernel(lhs: &[f32], rhs: &[f32], config: &SvmConfig) -> f32 {
    let dot: f32 = lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum();
    match config.kernel {
        SvmKernel::Polynomial => (config.gamma * dot + config.coef0).powi(config.degree as i32),
        SvmKernel::Sigmoid    => (config.gamma * dot + config.coef0).tanh(),
        _                     => dot,
    }
}

// ── LinearSvmOVR ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct LinearSvmOVR {
    weights: Array2<f32>,
    bias: Array1<f32>,
    encoder: LabelEncoder,
    config: SvmConfig,
    kernel_map: KernelMap,
}

impl LinearSvmOVR {
    pub(crate) fn fit(
        matrix: &DenseMatrix,
        samples: &[TrainingSample],
        config: &SvmConfig,
        threads: Option<usize>,
    ) -> Result<Self, VecEyesError> {
        let encoder = LabelEncoder::fit(samples);
        let kernel_map = KernelMap::build(matrix, config);
        let x = kernel_map.transform(matrix, config);
        let classes = encoder.labels.len();
        let features = x.shape()[1];
        let y_idx: Vec<usize> = samples
            .iter()
            .map(|s| encoder.encode(&s.label))
            .collect::<Result<_, _>>()?;
        let lambda = 1.0 / config.c.max(1e-6);

        let models: Vec<(Array1<f32>, f32)> = install_pool(threads, || {
            (0..classes)
                .into_par_iter()
                .map(|class_id| {
                    let mut w = Array1::<f32>::zeros(features);
                    let mut b = 0.0f32;
                    for epoch in 0..config.epochs {
                        let local_lr = config.learning_rate / (1.0 + epoch as f32 * 0.05);
                        let decay = 1.0 - local_lr * lambda;
                        for row in 0..x.shape()[0] {
                            let y = if y_idx[row] == class_id { 1.0f32 } else { -1.0f32 };
                            let x_row = x.row(row);
                            // sdot: BLAS or LLVM auto-vec
                            let margin = b + w.dot(&x_row);
                            // L2 regularisation shrink
                            w.mapv_inplace(|v| v * decay);
                            if y * margin < 1.0 {
                                // daxpy: w += lr·y·x_row
                                w.scaled_add(local_lr * y, &x_row);
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
            weights.row_mut(class_id).assign(&w);
            bias[class_id] = b;
        }
        Ok(Self { weights, bias, encoder, config: config.clone(), kernel_map })
    }

    pub(crate) fn predict_scores(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
        let mapped = self.kernel_map.transform(probe, &self.config);
        let row = mapped.row(0);
        // (classes, K) @ (K,) + bias → (classes,): BLAS gemv when enabled
        let margins = self.weights.dot(&row) + &self.bias;
        let raw: Vec<(ClassificationLabel, f32)> = margins
            .iter()
            .enumerate()
            .map(|(i, &m)| (self.encoder.decode(i), m))
            .collect();
        softmax_scores(&raw)
    }
}
