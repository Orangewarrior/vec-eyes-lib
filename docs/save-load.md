# Save & Load — Persisting Trained Classifiers

Training an NLP classifier is expensive: it tokenises your corpus, builds TF-IDF or embedding tables, and fits the underlying ML model. Persistence lets you do that work **once**, write the result to disk, and reload in milliseconds on every subsequent run.

---

## Available formats

| Format | Method pair | Best for |
|---|---|---|
| **JSON** | `save` / `load` | Debugging, human inspection, git-diff-able models |
| **Bincode (single file)** | `save_bincode` / `load_bincode` | Fast production reload; smaller than JSON for large models |
| **Bincode split** | `save_split_bincode` / `load_split_bincode` | Sharing the NLP pipeline across multiple ML heads |

All three formats are lossless: classifying the same text before and after a round-trip returns the same top label.

---

## Classifiers that support persistence

Every public classifier type has all three format pairs:

| Type | `save/load` | `save/load_bincode` | `save/load_split_bincode` |
|---|---|---|---|
| `KnnClassifier` | ✓ | ✓ | ✓ |
| `BayesClassifier` | ✓ | ✓ | — |
| `LogisticClassifier` | ✓ | ✓ | ✓ |
| `SvmClassifier` | ✓ | ✓ | ✓ |
| `RandomForestClassifier` | ✓ | ✓ | ✓ |
| `GradientBoostingClassifier` | ✓ | ✓ | ✓ |
| `IsolationForestClassifier` | ✓ | ✓ | ✓ |
| `AdvancedClassifier` | ✓ | ✓ | ✓ |

`FastTextEmbeddings` (external embeddings loaded from a `.bin` file) can also be saved and loaded with `save_bincode` / `load_bincode`.

---

## UCI dataset examples

The examples below use the three small UCI-derived datasets included in `tests/data/`:

- **UCI SMS Spam** — `uci_sms/hot/` (spam), `uci_sms/cold/` (ham)
- **UCI Fraud Detection** — `uci_fraud/hot/` (fraud), `uci_fraud/cold/` (normal)
- **UCI Biology** — `uci_biology/hot/` (virus), `uci_biology/cold/` (human)

### Loading training data

```rust
use vec_eyes_lib::dataset::load_training_samples;
use vec_eyes_lib::ClassificationLabel;
use std::path::Path;

// Load SMS spam samples
let mut samples = load_training_samples(
    Path::new("tests/data/uci_sms/hot"),
    ClassificationLabel::Spam,
    false, // non-recursive
)?;
samples.extend(load_training_samples(
    Path::new("tests/data/uci_sms/cold"),
    ClassificationLabel::RawData,
    false,
)?);
```

---

## KnnClassifier

### Train → save → load (JSON)

```rust
use vec_eyes_lib::{KnnClassifier, NlpOption, DistanceMetric};

let classifier = KnnClassifier::train(
    &samples,
    NlpOption::Word2Vec,
    DistanceMetric::Cosine,
    /*embedding_dims=*/ 64,
    /*k=*/ 5,
    /*threads=*/ None,
    /*normalize_features=*/ false,
)?;

// Save (human-readable)
classifier.save("models/knn_sms.json")?;

// Fast reload
let classifier = KnnClassifier::load("models/knn_sms.json")?;
```

### Bincode (faster, more compact for large models)

```rust
classifier.save_bincode("models/knn_sms.bin")?;
let classifier = KnnClassifier::load_bincode("models/knn_sms.bin")?;
```

### Split bincode — share NLP pipeline between classifiers

```rust
// Save NLP embeddings and ML weights separately
classifier.save_split_bincode("models/knn.nlp.bin", "models/knn.ml.bin")?;

// Reload both parts
let classifier = KnnClassifier::load_split_bincode("models/knn.nlp.bin", "models/knn.ml.bin")?;
```

The split format is useful when you train the same Word2Vec/FastText model and then experiment with different `k` values or distance metrics: write the NLP file once, update only the ML file.

---

## BayesClassifier

```rust
use vec_eyes_lib::{BayesClassifier, NlpOption};

// Train on UCI SMS spam dataset
let classifier = BayesClassifier::train(&samples, NlpOption::TfIdf, None)?;

// JSON — human-readable, easy to inspect token scores
classifier.save("models/bayes_sms.json")?;
let classifier = BayesClassifier::load("models/bayes_sms.json")?;

// Bincode — faster I/O, better for production
classifier.save_bincode("models/bayes_sms.bin")?;
let classifier = BayesClassifier::load_bincode("models/bayes_sms.bin")?;
```

---

## LogisticClassifier

```rust
use vec_eyes_lib::{LogisticClassifier, LogisticRegressionConfig, NlpOption};

let config = LogisticRegressionConfig {
    learning_rate: 0.25,
    epochs: 200,
    lambda: 1e-3,
};

// Train on UCI fraud dataset using Word2Vec embeddings
let classifier = LogisticClassifier::train(
    &fraud_samples,
    NlpOption::Word2Vec,
    config,
    /*threads=*/ None,
    /*embedding_dims=*/ 64,
)?;

classifier.save_bincode("models/logistic_fraud.bin")?;
let classifier = LogisticClassifier::load_bincode("models/logistic_fraud.bin")?;

// Or split: NLP pipeline reused by another classifier
classifier.save_split_bincode("models/fraud.nlp.bin", "models/logistic.ml.bin")?;
```

---

## SvmClassifier

```rust
use vec_eyes_lib::{SvmClassifier, SvmConfig, SvmKernel, NlpOption};

let config = SvmConfig {
    kernel: SvmKernel::Linear,
    c: 1.0,
    learning_rate: 0.08,
    epochs: 40,
    ..Default::default()
};

// Train on UCI biology dataset
let classifier = SvmClassifier::train(
    &biology_samples,
    NlpOption::TfIdf,
    config,
    None,
    32,
)?;

classifier.save_bincode("models/svm_biology.bin")?;
let classifier = SvmClassifier::load_bincode("models/svm_biology.bin")?;
```

---

## RandomForestClassifier

```rust
use vec_eyes_lib::{RandomForestClassifier, RandomForestConfig, NlpOption};

let classifier = RandomForestClassifier::train(
    &samples,
    NlpOption::Word2Vec,
    RandomForestConfig { n_trees: 50, max_depth: 8, ..Default::default() },
    None,
    /*embedding_dims=*/ 64,
)?;

classifier.save_bincode("models/rf_sms.bin")?;
let classifier = RandomForestClassifier::load_bincode("models/rf_sms.bin")?;

// Split: reuse NLP embeddings across different classifiers
classifier.save_split_bincode("models/sms.nlp.bin", "models/rf.ml.bin")?;
```

---

## GradientBoostingClassifier

```rust
use vec_eyes_lib::{GradientBoostingClassifier, GradientBoostingConfig, NlpOption};

let classifier = GradientBoostingClassifier::train(
    &samples,
    NlpOption::TfIdf,
    GradientBoostingConfig { n_estimators: 30, learning_rate: 0.2, max_depth: 2 },
    None,
    32,
)?;

classifier.save_bincode("models/gb_fraud.bin")?;
let classifier = GradientBoostingClassifier::load_bincode("models/gb_fraud.bin")?;
```

---

## IsolationForestClassifier

IsolationForest is an anomaly detector. It requires dense embeddings (`Word2Vec` or `FastText`) and explicit `hot_label` / `cold_label` parameters that indicate which class is the anomaly.

```rust
use vec_eyes_lib::{
    IsolationForestClassifier, IsolationForestConfig, ClassificationLabel, NlpOption,
};

let classifier = IsolationForestClassifier::train(
    &fraud_samples,
    NlpOption::Word2Vec,
    IsolationForestConfig { n_trees: 100, contamination: 0.05, subsample_size: 64 },
    ClassificationLabel::Anomaly,   // hot: anomalous class
    ClassificationLabel::RawData,   // cold: normal class
    None,
    /*embedding_dims=*/ 64,
)?;

classifier.save_bincode("models/iforest_fraud.bin")?;
let classifier = IsolationForestClassifier::load_bincode("models/iforest_fraud.bin")?;

// Split save: share the Word2Vec NLP pipeline with a Logistic head
classifier.save_split_bincode("models/fraud.nlp.bin", "models/iforest.ml.bin")?;
```

---

## External FastText embeddings

When you load embeddings from an external fastText `.bin` file, you can persist the extracted `FastTextEmbeddings` separately so you never need to re-parse the large binary again.

```rust
use vec_eyes_lib::{FastTextBin, FastTextEmbeddings, KnnClassifier, ClassificationLabel, DistanceMetric};

// --- One-time setup: parse the .bin and save the embeddings ---

let bin = FastTextBin::load("fasttext_models/cc.en.300.bin")?;

// Extract only the vectors needed for your vocabulary (much smaller)
let vocab: Vec<&str> = samples.iter().map(|s| s.text.as_str()).collect();
let embeddings = bin.extract_for_vocab(&vocab);

// Persist the embeddings once — fast to reload later
embeddings.save_bincode("models/cc.en.embeddings.bin")?;

// --- Fast path: reload embeddings and train ---

let embeddings = FastTextEmbeddings::load_bincode("models/cc.en.embeddings.bin")?;

let classifier = KnnClassifier::train_with_external_fasttext(
    &samples,
    embeddings,
    DistanceMetric::Cosine,
    /*k=*/ 5,
    None,
    false,
)?;

// Save the full classifier (includes NLP pipeline + embeddings + KNN matrix)
classifier.save_bincode("models/knn_fasttext.bin")?;

// Or save split: NLP embeddings separately from the KNN training matrix
classifier.save_split_bincode("models/knn.nlp.bin", "models/knn.ml.bin")?;

// Reload in production
let classifier = KnnClassifier::load_bincode("models/knn_fasttext.bin")?;
```

The same `train_with_external_fasttext` method is available on every advanced classifier type (`LogisticClassifier`, `SvmClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `IsolationForestClassifier`).

---

## Choosing the right format

| Situation | Recommendation |
|---|---|
| Debugging a model, inspecting weights | JSON |
| Production service, reload on start | bincode single file |
| Large embedding model (Word2Vec / FastText) shared across pipelines | Split bincode — write NLP once, update ML head separately |
| External fastText with millions of buckets | `FastTextEmbeddings::save_bincode` → reload and pass to `train_with_external_fasttext` |

---

## Error handling

All persistence methods return `Result<_, VecEyesError>`. The two error variants you may encounter:

- `VecEyesError::Io` — file not found, permission denied, disk full.
- `VecEyesError::Serialization` — the file on disk does not match the expected format (e.g. a JSON file opened with `load_bincode`, or a truncated/corrupt file).

```rust
match KnnClassifier::load_bincode("models/knn.bin") {
    Ok(clf) => { /* use clf */ }
    Err(VecEyesError::Io(e)) => eprintln!("file error: {e}"),
    Err(VecEyesError::Serialization(msg)) => eprintln!("corrupt model: {msg}"),
    Err(e) => eprintln!("unexpected error: {e}"),
}
```
