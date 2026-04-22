use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rayon::prelude::*;

use crate::advanced_models::{LabelEncoder, SvmConfig, SvmKernel};
use crate::classifier::softmax_scores;
use crate::dataset::TrainingSample;
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::nlp::DenseMatrix;
use crate::parallel::install_pool;

// Number of random Fourier features / landmark rows used for kernel approximation.
const N_COMPONENTS: usize = 256;
// Maximum landmark rows kept for Polynomial / Sigmoid kernels.
const MAX_LANDMARKS: usize = 128;

// ── Kernel map ────────────────────────────────────────────────────────────────

/// Kernel approximation strategy stored alongside the trained SVM.
#[derive(Debug, Clone)]
enum KernelMap {
    /// Linear SVM — no transformation needed.
    Identity,
    /// RBF kernel via Random Fourier Features (Rahimi & Recht 2007).
    /// Provably approximates exp(-γ‖x-y‖²) in expectation.
    Rff { weights: Array2<f32>, biases: Vec<f32>, scale: f32 },
    /// Polynomial / Sigmoid — explicit kernel matrix against random landmarks.
    Landmark(Array2<f32>),
}

impl KernelMap {
    fn build(matrix: &DenseMatrix, config: &SvmConfig) -> Self {
        match config.kernel {
            SvmKernel::Linear => Self::Identity,
            SvmKernel::Rbf => Self::build_rff(matrix.shape()[1], N_COMPONENTS, config.gamma),
            _ => Self::build_landmarks(matrix, config),
        }
    }

    /// Random Fourier Features for the RBF kernel.
    /// Samples ω ~ N(0, 2γ·I) and b ~ Uniform(0, 2π), then
    /// φ(x) = √(2/D) · cos(ωᵀx + b).
    fn build_rff(input_dim: usize, n_components: usize, gamma: f32) -> Self {
        let mut rng = StdRng::seed_from_u64(0xFEA7_C0DE);
        let std_dev = (2.0 * gamma).sqrt();
        let mut weights = Array2::<f32>::zeros((n_components, input_dim));
        let mut biases = Vec::with_capacity(n_components);
        for i in 0..n_components {
            for j in 0..input_dim {
                // Box-Muller transform → N(0, std_dev)
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

    /// Random landmark selection for Polynomial / Sigmoid kernels.
    fn build_landmarks(matrix: &DenseMatrix, config: &SvmConfig) -> Self {
        let rows = matrix.shape()[0];
        let cols = matrix.shape()[1];
        let keep = rows.min(MAX_LANDMARKS).max(1);
        // Shuffle row indices so landmarks are a random sample, not first-N rows.
        let mut rng = StdRng::seed_from_u64(0xA9D_CAFE);
        let mut indices: Vec<usize> = (0..rows).collect();
        indices.shuffle(&mut rng);
        indices.truncate(keep);
        let mut lm = Array2::<f32>::zeros((keep, cols));
        for (out_row, &src_row) in indices.iter().enumerate() {
            for col in 0..cols { lm[[out_row, col]] = matrix[[src_row, col]]; }
        }
        let _ = config; // stored in LinearSvmOVR for inference
        Self::Landmark(lm)
    }

    fn transform(&self, matrix: &DenseMatrix, config: &SvmConfig) -> DenseMatrix {
        match self {
            Self::Identity => matrix.clone(),
            Self::Rff { weights, biases, scale } => {
                let rows = matrix.shape()[0];
                let nc = biases.len();
                let mut mapped = Array2::<f32>::zeros((rows, nc));
                for row in 0..rows {
                    let x = matrix.index_axis(Axis(0), row);
                    for j in 0..nc {
                        let dot: f32 = x.iter()
                            .zip(weights.index_axis(Axis(0), j).iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        mapped[[row, j]] = scale * (dot + biases[j]).cos();
                    }
                }
                mapped
            }
            Self::Landmark(ref_matrix) => {
                let rows = matrix.shape()[0];
                let n_lm = ref_matrix.shape()[0];
                let mut mapped = Array2::<f32>::zeros((rows, n_lm));
                for row in 0..rows {
                    let lhs = matrix.index_axis(Axis(0), row).to_vec();
                    for col in 0..n_lm {
                        let rhs = ref_matrix.index_axis(Axis(0), col).to_vec();
                        mapped[[row, col]] = explicit_kernel(&lhs, &rhs, config);
                    }
                }
                mapped
            }
        }
    }
}

/// Explicit kernel evaluation for Polynomial / Sigmoid landmark maps.
fn explicit_kernel(lhs: &[f32], rhs: &[f32], config: &SvmConfig) -> f32 {
    let dot: f32 = lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum();
    match config.kernel {
        SvmKernel::Polynomial => (config.gamma * dot + config.coef0).powi(config.degree as i32),
        SvmKernel::Sigmoid    => (config.gamma * dot + config.coef0).tanh(),
        // Linear / RBF handled by KernelMap variants above.
        _                     => dot,
    }
}

// ── LinearSvmOVR ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
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
                            let margin = b + (0..features).map(|c| x[[row, c]] * w[c]).sum::<f32>();
                            for wt in &mut w { *wt *= 1.0 - local_lr * lambda; }
                            if y * margin < 1.0 {
                                for col in 0..features { w[col] += local_lr * y * x[[row, col]]; }
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
            for col in 0..features { weights[[class_id, col]] = w[col]; }
            bias[class_id] = b;
        }
        Ok(Self { weights, bias, encoder, config: config.clone(), kernel_map })
    }

    pub(crate) fn predict_scores(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
        let mapped = self.kernel_map.transform(probe, &self.config);
        let row = mapped.index_axis(Axis(0), 0);
        let raw: Vec<(ClassificationLabel, f32)> = (0..self.encoder.labels.len())
            .map(|class_id| {
                let margin = self.bias[class_id]
                    + row.iter().zip(self.weights.index_axis(Axis(0), class_id).iter()).map(|(a, b)| a * b).sum::<f32>();
                (self.encoder.decode(class_id), margin)
            })
            .collect();
        softmax_scores(&raw)
    }
}
