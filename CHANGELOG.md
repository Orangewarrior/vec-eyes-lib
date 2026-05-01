# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.2.2] - 2026-05-01

### Changed

- `GradientBoosting` now trains shallow regression trees and honors `max_depth` instead of behaving as a fixed-depth stump ensemble.
- README examples were updated to the current nested YAML format and current Rust builder API.
- README now documents production notes, model capability tradeoffs, and the safe regex-compatible VectorScan fallback.

### Performance

- KNN now caches training row norms for cosine/euclidean scoring, avoiding repeated candidate norm calculation at inference time.
- Applied small hot-path cleanups in SVM, RandomForest, and internal embedding training loops.

### Rust/API

- Added compatible `Default` implementations/derives for public builders and enums where `new()` or manual defaults already existed.
- Reduced Clippy noise for simple clamp/default patterns while preserving the public API.

---

## [3.2.1] - 2026-05-01

### Security

#### Hardened binary embedding loaders (`src/nlp/fasttext_bin.rs`, `src/nlp/word2vec_bin.rs`)

- Added strict bounds for dimensions, vocabulary size, bucket count, header length, token length, and total matrix bytes.
- Replaced trusted `rows * cols` allocation with checked multiplication before allocating embedding matrices.
- Rejects malformed fastText/word2vec headers early instead of allowing adversarial files to trigger huge allocations or integer overflow.

#### Safer model deserialization (`src/classifiers/*`, `src/advanced_models.rs`, `src/nlp/*`)

- Added bounded file reads for JSON and bincode model/embedding loaders.
- Added post-load invariants for KNN, Bayes, advanced classifiers, fastText embeddings, and word2vec embeddings.
- Split bincode loaders now validate both NLP and ML payload sizes before deserialization.

#### Stricter output path handling (`src/security.rs`, `src/config.rs`)

- Report output paths now reject absolute paths and `..` components.
- YAML-configured output paths are validated against the rules-file base directory so outputs cannot escape the expected tree.

#### Thread-count limits (`src/config.rs`, `src/parallel.rs`)

- `pipeline.threads` now rejects excessive values above the configured cap.
- Rayon pool creation clamps direct thread requests to the same cap, preventing unbounded thread-pool creation from hostile configuration.

### Performance

#### KNN neighbor selection (`src/classifiers/knn/core.rs`)

- Replaced full sorting of all candidate distances with partial selection of the nearest `k` neighbors.
- This reduces per-classification neighbor ranking work from full `O(n log n)` sorting to selection-oriented ranking for large training sets.

### Tests

- Added regression coverage for excessive YAML thread counts.
- Added regression coverage for absolute report output rejection.
- Added regression coverage for malicious word2vec headers that should fail before matrix allocation.

---

## [3.2.0] - 2026-04-28

### Changed

#### `RulesFile` refactored into sub-configs (`src/config.rs`)

The flat ~40-field `RulesFile` struct has been replaced with a nested design that groups related settings:

- **`DataConfig`** — training data paths, labels, recursive mode, score-sum mode
- **`PipelineConfig`** — NLP option, threads, embedding dimensions, security normalization flag
- **`ModelConfig`** — internally-tagged serde enum (`#[serde(tag = "method")]`); one variant per classifier, holding only the fields that variant needs

Old flat YAML format:
```yaml
method: RandomForest
nlp: TfIdf
hot_test_path: data/hot
cold_test_path: data/cold
random_forest_n_trees: 51
random_forest_max_depth: 8
```

New nested YAML format:
```yaml
data:
  hot_test_path: data/hot
  cold_test_path: data/cold
  recursive_way: On
  score_sum: Off

pipeline:
  nlp: TfIdf

model:
  method: RandomForest
  n_trees: 51
  max_depth: 8
```

All 35 YAML test fixtures updated. `ModelConfig`, `DataConfig`, and `PipelineConfig` are re-exported from `vec_eyes_lib`.

### Added

#### `EnsembleStrategy` enum (`src/classifier.rs`)

`EnsembleClassifier` now supports two combination strategies:

- **`WeightedAverage`** (default) — sums weighted probabilities across classifiers, normalises by total weight; produces intuitive linear mixtures
- **`ProductOfExperts`** — accumulates weighted log-probabilities then applies softmax; mathematically principled but sharper (more extreme) than linear averaging

Builder method: `EnsembleClassifier::with_strategy(EnsembleStrategy::ProductOfExperts)`.

#### Streaming dataset iterator (`src/dataset.rs`)

New `training_sample_iter(path, label, recursive)` returns a lazy `impl Iterator<Item = Result<TrainingSample, VecEyesError>>`. Reading a 10 GB directory no longer requires collecting all samples into memory before training begins. `load_training_samples` now delegates to this iterator.

#### Model versioning for save/load (`src/advanced_models.rs`)

- **JSON**: `VersionedJsonRef` / `VersionedJsonOwned` wrapper types inject a `"version": 1` field into serialised JSON. Files without a version field are treated as version 0 (legacy) and loaded without error.
- **Bincode**: `VersionedBincode { version: u32, classifier }` wrapper. Old unversioned bincode files now fail with a clear "please retrain" error instead of a panic.

#### Label validation in `ClassificationLabel::from_str` (`src/labels.rs`)

`from_str` now rejects empty strings, labels longer than 64 characters, and labels containing control characters, returning a `&'static str` error message instead of the previous `()`.

### Fixed

#### Path traversal guard for report output (`src/report.rs`)

`ClassificationReport::write_csv` and `write_json` now call `security::sanitize_output_path` before opening the file, rejecting any path that contains `..` components.

#### `IsolationForest` NLP restriction removed (`src/advanced_models.rs`)

`IsolationForestClassifier::train` previously rejected `NlpOption::TfIdf` and `NlpOption::Count`. That restriction has been lifted; all NLP options now work with IsolationForest.

### Performance

#### Rayon parallelism throughout classifier cores

- `IsolationForest::predict_scores` — per-tree path-length accumulation is now parallel via `rayon::par_iter`
- `RandomForest::predict_scores` — vote accumulation is parallel; per-tree feature-importance updates use `zip().for_each()` instead of indexed loops
- `GradientBoosting::fit_regression_stump` — best-split search across features is parallel
- `SVM Landmark.transform` — row-wise landmark projection now uses `axis_iter_mut().into_par_iter()`
- `AdvancedClassifier::transform_count` — row-wise embedding construction is parallel; L2 normalisation is inlined via `mapv_inplace`

#### Lower-contention thread-pool cache (`src/parallel.rs`)

Replaced `Mutex<HashMap<usize, Arc<ThreadPool>>>` with a `RwLock` double-check pattern: threads that request an already-cached pool size take the read path without ever acquiring an exclusive lock.

---

## [3.1.0] - 2026-04-27

### Added

#### word2vec binary loader (`src/nlp/word2vec_bin.rs`)

Pure-Rust parser for the Google word2vec binary format (`.bin`) produced by the original `word2vec` C tool and its Rust/Python reimplementations. Handles both variants of the trailing-newline encoding used by different encoders.

Key types:
- `Word2VecBin` — raw loader; holds word index + flat float32 matrix in memory
- `Word2VecEmbeddings` — extracted subset of vectors; serde-serializable for fast persistence; includes a pre-computed `centroid` vector for OOV tokens

Key methods:
- `Word2VecBin::load(path)` — parse a `.bin` file (header: `<vocab_size> <dims>\n`, then `<word> <float32 × dims> <\n>` per entry)
- `Word2VecBin::extract_all()` — extract every word vector
- `Word2VecBin::extract_for_vocab(vocab)` — extract only vectors needed by a given vocabulary; unknown words are represented by the centroid
- `Word2VecEmbeddings::vector_for(word)` — in-vocabulary lookup or centroid fallback for OOV
- `Word2VecEmbeddings::save_bincode / load_bincode` — persist/reload extracted embeddings

OOV strategy: vocabulary centroid (mean of all word vectors) pre-computed at extraction time. Unlike fastText, word2vec has no subword model, so the centroid is the most stable fallback that preserves dimensionality and roughly preserves the embedding space centre.

#### `ExternalEmbeddings` unified enum (`src/nlp/external_embeddings.rs`)

```rust
pub enum ExternalEmbeddings {
    FastText(FastTextEmbeddings),
    Word2Vec(Word2VecEmbeddings),
}
```

Single type accepted by every classifier's `train_with_external_embeddings` method. Dispatches to the appropriate embedding backend at inference time. Supports `save_bincode` / `load_bincode` for caching.

Methods:
- `ExternalEmbeddings::dims() -> usize`
- `ExternalEmbeddings::save_bincode(path) -> Result<()>`
- `ExternalEmbeddings::load_bincode(path) -> Result<Self>`

#### `train_with_external_embeddings` on all classifiers

Unified constructor accepting `ExternalEmbeddings` (fastText or word2vec) on every classifier type:

- `KnnClassifier::train_with_external_embeddings(samples, embeddings, metric, k, threads, normalize_features)`
- `LogisticClassifier::train_with_external_embeddings(samples, embeddings, config, threads)`
- `SvmClassifier::train_with_external_embeddings(samples, embeddings, config, threads)`
- `RandomForestClassifier::train_with_external_embeddings(samples, embeddings, config, threads)`
- `GradientBoostingClassifier::train_with_external_embeddings(samples, embeddings, config, threads)`
- `IsolationForestClassifier::train_with_external_embeddings(samples, embeddings, config, hot_label, cold_label, threads)`
- `AdvancedClassifier::train_with_external_embeddings(method, samples, embeddings, hot_label, cold_label, config)`

The older `train_with_external_fasttext` methods are kept for backwards compatibility.

#### `ClassifierSpec` / `KnnSpec` / `AdvancedSpec` — `external_embeddings` builder method

Both `KnnSpec` and `AdvancedSpec` gain `.external_embeddings(ExternalEmbeddings)` accepting the unified enum. The existing `.external_fasttext(FastTextEmbeddings)` is kept as a thin convenience wrapper.

#### Documentation updates

- `docs/save-load.md` — migrated all examples to the new `train_with_external_embeddings` API; added three end-to-end word2vec examples (News/KNN-Euclidean, SMS/RandomForest, Fraud/SVM) with full bash + Rust code
- `docs/real_examples.md` — four new external-embedding examples added (examples 4–7): SMS+fastText+KNN, News+word2vec+RandomForest, Splice+word2vec+SVM, multi-classifier comparison (KNN+LR fastText vs RF+KNN word2vec side-by-side)

### Changed

- `docs/save-load.md` — section renamed from "External fastText embeddings" to "External embeddings" (covers both fastText and word2vec)
- Table entry for external embedding persistence updated from `FastTextEmbeddings::save_bincode → train_with_external_fasttext` to `ExternalEmbeddings::save_bincode → train_with_external_embeddings`

---

## [3.0.0] - 2026-04-26

### Added

#### FastText `.bin` loader (`src/nlp/fasttext_bin.rs`)

Pure-Rust parser for binary files produced by the external fastText CLI (format v12). No C library or FFI required. Reads the magic number, args struct (dims, minn, maxn, bucket), dictionary, quantization flag, and flat float32 input matrix.

Key types:
- `FastTextBin` — raw loader; holds word index + full matrix in memory
- `FastTextEmbeddings` — extracted subset of vectors; serde-serializable for fast persistence

Key methods:
- `FastTextBin::load(path)` — parse a `.bin` file
- `FastTextBin::extract_all()` — extract every word and bucket vector
- `FastTextBin::extract_for_vocab(vocab)` — extract only vectors relevant to a given vocabulary (much smaller output)
- `FastTextEmbeddings::vector_for(word)` — in-vocabulary lookup or subword OOV composition
- `FastTextEmbeddings::save_bincode / load_bincode` — persist/reload extracted embeddings

#### Subword OOV composition (FNV-1a hash)

OOV words are represented by averaging the bucket vectors for all character n-grams in `[minn, maxn]`, matching fastText C++ exactly:
- Boundary markers `<` / `>` added around the word before n-gram extraction
- FNV-1a: `h ^= (b as i8 as i32 as u32)` (sign-extend byte) then `h.wrapping_mul(16777619)`
- Standalone `<` and `>` n-grams are excluded (matches fastText's `computeSubwords`)

#### `train_with_external_fasttext` on all classifiers

Every classifier type gains a constructor that accepts pre-loaded `FastTextEmbeddings` instead of training its own embedding model:

- `KnnClassifier::train_with_external_fasttext(samples, embeddings, metric, k, threads, normalize_features)`
- `LogisticClassifier::train_with_external_fasttext(samples, embeddings, config, threads, dims)`
- `SvmClassifier::train_with_external_fasttext(...)`
- `RandomForestClassifier::train_with_external_fasttext(...)`
- `GradientBoostingClassifier::train_with_external_fasttext(...)`
- `IsolationForestClassifier::train_with_external_fasttext(samples, embeddings, config, hot_label, cold_label, threads)`

#### Bincode persistence for all classifiers

All public classifier types now support fast binary save/load via `bincode v1`:

| Method pair | File count | Use case |
|---|---|---|
| `save_bincode / load_bincode` | 1 | Fast production reload |
| `save_split_bincode / load_split_bincode` | 2 (`.nlp.bin` + `.ml.bin`) | Share NLP pipeline across multiple ML heads |

Types with bincode support: `KnnClassifier`, `BayesClassifier`, `LogisticClassifier`, `SvmClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `IsolationForestClassifier`, `AdvancedClassifier`, `FastTextEmbeddings`.

#### Standalone typed classifier wrappers

Five new concrete `struct` types replacing the generic `AdvancedClassifier` wrapper. Each type wraps `AdvancedClassifier` internally but exposes a typed, method-specific API with the correct config type:

- `LogisticClassifier` — `LogisticRegressionConfig`
- `SvmClassifier` — `SvmConfig`
- `RandomForestClassifier` — `RandomForestConfig`
- `GradientBoostingClassifier` — `GradientBoostingConfig`
- `IsolationForestClassifier` — `IsolationForestConfig` + explicit `hot_label` / `cold_label`

All five implement the `Classifier` trait and are exported from the crate root.

#### `VecEyesError::Serialization` variant

New error variant for bincode encode/decode failures:
```rust
#[error("serialization error: {0}")]
Serialization(String),
```

#### `docs/save-load.md` — persistence documentation

Comprehensive guide covering:
- Format comparison table (JSON vs bincode single vs bincode split)
- Full working examples for every classifier type using the three UCI datasets
- External FastText workflow (parse once → save embeddings → reload → train)
- Error handling guide with `VecEyesError` match arms

#### `docs/real_examples.md` — real classification examples

Three complete standalone Rust projects (security, biology, finance) covering:
- UCI dataset download and preparation scripts
- Full `Cargo.toml` + `src/main.rs` for each example
- Build and run commands with expected output

#### Save/load test suite (`tests/save_load.rs`)

27 integration tests covering round-trip persistence for all classifier types:
- JSON save/load for `KnnClassifier` and `BayesClassifier`
- Bincode single-file round-trips for all classifier types
- Split bincode round-trips for all advanced classifier types
- `FastTextEmbeddings` bincode persistence
- Verify classification result is identical before and after round-trip

#### 3.1 `Classifier::classify_batch` — batch inference API
**Files:** `src/classifier.rs`, `src/advanced_models.rs`

Added `classify_batch(&self, texts: &[&str], ...) -> Vec<ClassificationResult>` to
the `Classifier` trait with a sequential default implementation. `AdvancedClassifier`
overrides it with a two-stage approach: transform the entire batch through the NLP
pipeline once (amortising embedding/TF-IDF cost), then score each row in parallel
via rayon, slicing the batch matrix with `ndarray::s!` to avoid copies.

#### 3.2 Model serialization — `save` / `load`
**Files:** `src/classifiers/bayes/mod.rs`, `src/classifiers/knn/mod.rs`,
`src/advanced_models.rs`, `src/nlp/feature_extractor.rs`, and all inner model structs

`serde::Serialize` / `serde::Deserialize` added to every internal type:
`TfIdfModel`, `WordEmbeddingModel`, `DistanceMetric`, `DenseFeatureModel`,
`FeaturePipeline`, `LabelEncoder`, `AdvancedInner`, and all six config structs
(`LogisticRegressionConfig`, `RandomForestConfig`, `SvmConfig`,
`GradientBoostingConfig`, `IsolationForestConfig`, `AdvancedModelConfig`).

All three public classifier types gain `save(path) -> Result<()>` and
`load(path) -> Result<Self>` backed by `serde_json`.

#### 3.3 `metrics` module
**File:** `src/metrics.rs` (new)

New public module exposed as `vec_eyes_lib::metrics` with:
- `accuracy(y_true, y_pred) -> f32`
- `precision(y_true, y_pred, label) -> f32`
- `recall(y_true, y_pred, label) -> f32`
- `f1(y_true, y_pred, label) -> f32`
- `macro_f1(y_true, y_pred, labels) -> f32`
- `weighted_f1(y_true, y_pred, labels) -> f32`
- `roc_auc(y_true, y_scores) -> f32` — trapezoidal rule
- `confusion_matrix(y_true, y_pred, labels) -> Vec<Vec<usize>>`
- `classification_report(y_true, y_pred, labels) -> ClassificationReport`
  (returns `ClassMetrics` per class: precision, recall, f1, support)

#### 3.4 `ClassifierBuilder::samples()` — preloaded training data
**File:** `src/classifier.rs`

Added `pub fn samples(mut self, samples: Vec<TrainingSample>) -> Self` to
`ClassifierBuilder` as an alternative to `hot_path` / `cold_path`. When
`samples()` is set, `build()` skips filesystem loading entirely. This enables
in-memory pipelines, unit tests without fixture directories, and programmatic
data augmentation.

#### 3.8 `ClassifierSpec` typed factory
**Files:** `src/factory/spec.rs` (new), `src/factory/mod.rs`

`ClassifierSpec` eliminates the `MethodKind` enum + `require_*_config` defensive
pattern. Each factory method returns a method-specific builder:

```rust
ClassifierSpec::bayes()                  -> BayesSpec
ClassifierSpec::knn_cosine(k)            -> KnnSpec
ClassifierSpec::knn_euclidean(k)         -> KnnSpec
ClassifierSpec::knn_manhattan(k)         -> KnnSpec
ClassifierSpec::knn_minkowski(k, p)      -> KnnSpec
ClassifierSpec::logistic_regression(cfg) -> AdvancedSpec
ClassifierSpec::random_forest(cfg)       -> AdvancedSpec
ClassifierSpec::svm(cfg)                 -> AdvancedSpec
ClassifierSpec::gradient_boosting(cfg)   -> AdvancedSpec
ClassifierSpec::isolation_forest(cfg)    -> AdvancedSpec
```

Each spec builder exposes only the relevant knobs (`.nlp()`, `.threads()`,
`.hot_label()`, `.cold_label()`, `.samples()`, `.training_data()`) and
provides `.build()` / `.build_boxed()` returning the concrete type or
`Box<dyn Classifier>`.

### Fixed

#### 2.8 Eliminate `Vec<String>` clones in NLP pipelines
**Files:** `src/nlp/feature_extractor.rs`, `src/advanced_models.rs`,
`src/classifiers/bayes/core.rs`, `src/classifiers/knn/core.rs`

All text-processing functions (`fit_tfidf`, `transform_tfidf`,
`dense_matrix_from_texts`, `train_word2vec`, `train_fasttext`,
`train_context_embeddings`, `transform_count`) are now generic over
`AsRef<str>`. Single-probe inference uses a zero-allocation `[text]` array
instead of `vec![text.to_string()]`.

#### 2.9 `LabelEncoder` sort/dedup invariant
**File:** `src/classifiers/*/core.rs`

Added explicit comment above `labels.dedup()` clarifying that the preceding
`sort()` is mandatory — `dedup` only collapses consecutive duplicates.

#### 2.10 Lazy rule-boost computation
**Files:** `src/advanced_models.rs`, `src/classifiers/bayes/mod.rs`,
`src/classifiers/knn/mod.rs`

`compute_rule_boost` (and the per-label score merge loop) is now skipped when
`ScoreSumMode` is `Off`. Matcher hits are still collected via the lighter
direct `find_matches` path, preserving telemetry without paying the full
scoring cost.

#### 2.11 Configurable per-file size limit
**Files:** `src/dataset.rs`, `src/config.rs`, `src/classifier.rs`

`MAX_TEXT_FILE_BYTES` renamed to `pub const DEFAULT_MAX_FILE_BYTES`. New
`read_text_file_limited(path, max_bytes)` accepts a caller-supplied cap.
`RulesFile` gains an optional `max_file_bytes: Option<u64>` YAML field honoured
by `run_rules_pipeline`. `collect_files_recursively_with_limit` now correctly
passes the `max_bytes` parameter through all checks instead of always using the
global constant.

#### 2.12 Bounded global thread-pool cache
**File:** `src/parallel.rs`

`cached_pool` enforces `MAX_POOL_CACHE = 32` entries with FIFO eviction so
the static `HashMap` cannot grow without bound across long-running processes.

### Changed

#### 3.5 `EnsembleClassifier` — correct product-of-experts math
**File:** `src/classifier.rs`

- `new()` normalises weights to sum 1.0 automatically.
- `classify_text` sums **weighted log-probabilities** (`weight * score.ln()`)
  then applies a single softmax — a proper product-of-experts ensemble.

#### 3.6 `softmax_scores` moved to private `math` module
**Files:** `src/math.rs` (new), all internal consumers updated

`softmax_scores` extracted into `pub(crate) mod math`. Public surface of
`classifier.rs` is unchanged.

#### 3.7 `compat.rs` types marked deprecated
**File:** `src/compat.rs`

All public types carry `#[deprecated(since = "2.8.0", note = "...")]`:
- `RepresentationKind` → use `NlpOption`
- `NlpPipeline` / `NlpPipelineBuilder` → use `ClassifierFactory::builder().nlp(...)`
- `OutputWriters` → no longer needed
- `alerts::AlertMatcher` → use `MatcherFactory::build_from_extra_match`
- `EngineBuilder` / `Engine` → use `ClassifierFactory::builder()`

---

## [2.9.0] - 2026-04-22

### Fixed — Critical correctness bugs (5 issues)

#### 2.1 remove_null_bytes: Remove null bytes instead of spaces
**File:** `src/nlp/normalizer.rs`

The `remove_null_bytes` function was incorrectly filtering space characters (`' '`)
instead of null bytes (`'\0'`). This corrupted tokenization inside
`decode_obfuscated_text`, silently removing all whitespace from processed text.

**Fix:** Changed filter condition from `*c != ' '` to `*c != '\0'`.

#### 2.2 decode_percent_encoding: Fix UTF-8 decoding
**File:** `src/nlp/normalizer.rs`

The percent-decoding function converted decoded bytes directly to `char` using
`value as char`, which only works for ASCII. Multi-byte UTF-8 sequences like
`%C3%A9` (é) were decoded as two invalid separate characters.

**Fix:**
- Changed output buffer from `String` to `Vec<u8>`
- Collect decoded bytes, then use `String::from_utf8_lossy()` at the end
- Now correctly handles all UTF-8 sequences

#### 2.3 flush_token: Preserve single-character tokens
**File:** `src/nlp/tokenizer.rs`

The tokenizer discarded all tokens with length < 2, silently removing critical
single-character symbols. For security-focused analysis (syscall/HTTP/malware
detection), tokens like `<`, `>`, `|`, and single digits are essential signals.

**Fix:**
- Removed `if current.len() >= 2` check
- Now preserves all non-empty tokens
- Added documentation explaining security rationale

#### 2.4 ClassifierBuilder: Remove duplicate new/build methods
**File:** `src/classifier.rs`

`ClassifierBuilder` had both `impl Builder<...> trait` and a separate
`impl ClassifierBuilder` block re-implementing `new()` and `build()`. This
duplication was confusing and prone to implementation drift.

**Fix:**
- Removed redundant inherent impl methods
- Kept only the trait implementation
- All builder methods now consistently defined in one place

#### 2.7 tokenizer: Remove duplicate lowercase conversion
**File:** `src/nlp/tokenizer.rs`

The tokenizer called `to_ascii_lowercase()` on each character even though
`normalize_text` already applies Unicode lowercase. This was redundant and
inconsistent (Unicode vs ASCII lowercase).

**Fix:** Removed duplicate `to_ascii_lowercase()` call in tokenizer loop.

### Performance Optimizations

#### softmax_scores: Optimize probability normalization
**File:** `src/classifier.rs`

Multiple micro-optimizations to the softmax function used in classification:

- Replaced manual max-finding loop with `fold(f32::NEG_INFINITY, f32::max)`
- Pre-allocate result vector with `Vec::with_capacity(input.len())`
- Replace division by sum with multiplication by inverse (`value * inv_sum`)
- Simplified zero-sum edge case handling

**Impact:** Reduced allocations, better SIMD utilization, faster computation
for large label sets.

### Changed

- All bug fixes maintain backward API compatibility
- Tokenization behavior change: single-char tokens now included (may increase
  token count slightly but improves detection accuracy)

---

## [2.8.0] - 2026-04-22

### Fixed — ML algorithmic correctness (10 issues)

#### 1.1 Sublinear TF in TF-IDF pipeline (`nlp/feature_extractor.rs`)
Raw term frequency divided by document length was replaced with the standard
sublinear form `tf = 1 + ln(count)`, eliminating length bias and aligning with
Salton & Buckley (1988).

#### 1.2 IDF re-fitted at inference time (`advanced_models.rs`)
`FeaturePipeline` variants `Word2Vec` and `FastText` now store the training
`TfIdfModel` as a named field (`idf`). Inference always uses the stationary
training IDF — the model no longer re-fits IDF on probe documents.

#### 1.3 IsolationForest trained on hot-only samples (`advanced_models.rs`)
The anomaly detector was previously fitted only on cold-label samples, preventing
it from learning the full data distribution. It now trains on the complete corpus;
contamination is derived automatically from the actual hot-label proportion
(`clamp(0.001, 0.49)`).

#### 1.4 Probe IDF used as NB weight (`classifiers/bayes/core.rs`)
The TF-IDF variant of Naïve Bayes was weighting token log-likelihoods by the
probe document's TF-IDF matrix rather than the corpus IDF. Fixed to use
`tfidf.idf[idx]` — a stationary weight from the training fit, consistent with
the IDF-weighted multinomial NB formulation.

#### 1.5 Singularity in k-NN distance weighting (`classifiers/knn/core.rs`)
Neighbor weight `1 / (d + 1e-6)`, clamped to 1000, was replaced with the
Gaussian kernel `exp(-d)`. The new weight is smooth, bounded in `(0, 1]`,
free of singularities, and valid for all supported distance metrics.

#### 1.6 Mini-batch SGD without epoch shuffle (`classifiers/logistic_regression/core.rs`)
Logistic regression batches always iterated samples in insertion order, biasing
gradients toward early samples. Each epoch now shuffles indices with a
per-class deterministic `StdRng` (seed `0xBAD_FEED ^ (class_id * large_prime)`),
ensuring unbiased mini-batches and reproducible parallel training.

#### 1.7 Incorrect Random Fourier Features for RBF SVM (`classifiers/svm/core.rs`)
The frequency matrix was not sampled from the correct distribution N(0, √(2γ)).
`KernelMap` enum introduced with three variants:
- `Identity` — linear kernel (no transform)
- `Rff` — RBF via Rahimi & Recht (2007): Box-Muller → N(0, √(2γ)), bias ~ U(0, 2π), scale √(2/D)
- `Landmark` — polynomial/sigmoid via random landmark rows (shuffled, seeded)

#### 1.8 Random threshold candidates in Gradient Boosting (`classifiers/gradient_boosting/core.rs`)
Split thresholds were generated as random jitter over `[min, max]`, missing the
optimal cut-points. Replaced with sorted quantile midpoints between adjacent
unique feature values (≤ 32 per feature — equivalent to a 32-bin histogram).
The implementation is now fully deterministic with no RNG dependency.

#### 1.9 Skip-gram with single embedding matrix (`nlp/feature_extractor.rs`)
Word2Vec Skip-gram used a single weight matrix for both center and context
words, deviating from Mikolov et al. (2013). Rewritten with two separate
matrices: `w_in` (deterministic hash initialization) and `w_out`
(zero-initialized). Negative sampling uses real RNG with the unigram noise
distribution P^0.75. Only `w_in` is returned as the final embedding.

#### 1.10 Inconsistent probability scale (`classifier.rs`, `matcher/mod.rs`)
`softmax_scores` was returning percentages `[0, 100]`; `merge_scores` received
`rule_boost` in `[0, 100]` and summed it directly to base probabilities,
producing scores > 1. Fixed:
- `softmax_scores` now returns true probabilities `[0, 1]`
- `score_percent` field computed as `top_score * 100.0` at report generation
- `merge_scores` normalises boost: `rule_boost / 100.0` before addition

### Added
- `.gitignore` — excludes `/target` build artefacts
- `Cargo.lock` — committed for reproducible builds

---

## [2.7.3] - (previous release)

See git history for earlier changes.
