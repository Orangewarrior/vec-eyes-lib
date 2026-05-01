# Vec-Eyes Core 🧠🔬

**Behavior intelligence engine for Rust**

Vec-Eyes Core is a modular behavior classification engine written in Rust, designed to power detection systems across security, data science, and biological domains.

It combines lightweight built-in ML models, NLP feature extraction, external embedding support, and rule-based matching into a unified engine for pattern detection and classification.

---

## 🚀 What is Vec-Eyes Core?

Vec-Eyes Core is the **engine behind Vec-Eyes CLI**.

It provides:

- 🧠 Machine Learning (KNN, Naive Bayes, Logistic Regression, SVM, Random Forest, Gradient Boosting, Isolation Forest)
- 🔡 NLP pipelines (Tokenization, TF-IDF, lightweight internal embeddings, external embeddings)
- ⚡ Vector similarity (Word2Vec, FastText)
- 🔎 Rule engine (Regex / optional VectorScan)
- 📊 Hybrid scoring system

---

## Production Notes

Vec-Eyes ships pure-Rust, dependency-light implementations that are useful for embedded classifiers, deterministic tests, security-oriented text features, and quick baselines. For high-stakes production ML, treat the built-in Word2Vec/FastText-style training and classical models as lightweight estimators rather than drop-in replacements for mature training stacks such as scikit-learn, XGBoost, or full fastText.

Recommended production path:

- Use built-in Count/TF-IDF, KNN, Bayes, and rules for small or medium text-classification workloads.
- Use external fastText or word2vec embeddings when semantic quality matters.
- Evaluate every pipeline with held-out metrics before deploying; see `src/metrics.rs` for accuracy, F1, ROC-AUC, confusion matrix, and reports.
- Load bincode models only from trusted sources. Prefer JSON or a controlled artifact pipeline for external model exchange.

### Model Capability Matrix

| Model | Best Fit | Notes |
|---|---|---|
| KNN | Similarity search on small/medium corpora | Brute-force exact search; cosine/euclidean paths cache training norms. |
| Bayes | Fast lexical baselines | Count and IDF-weighted modes; simple and explainable. |
| Logistic Regression | Linear text classification | One-vs-rest SGD-style trainer. |
| SVM | Linear and approximate kernels | RBF uses random Fourier features; polynomial/sigmoid use landmarks. |
| Random Forest | Robust tabular-like dense features | Supports balanced bootstrap, ExtraTrees-style splits, OOB score. |
| Gradient Boosting | Small dense feature sets | Uses shallow regression trees and honors `max_depth`. |
| Isolation Forest | Anomaly scoring | Best with enough representative cold/normal samples. |

---

## 📚 Documentation

| Guide | Description |
|---|---|
| [Real Classification Examples](docs/real_examples.md) | Three end-to-end projects (security, biology, finance) — download a UCI dataset, train a classifier, build and run |
| [Save & Load — Model Persistence](docs/save-load.md) | JSON, bincode, and split-bincode formats; external fastText workflow with UCI dataset examples |

### Quick links by topic

- **Getting started with a real dataset** → [real_examples.md — Prerequisites](docs/real_examples.md#prerequisites)
- **Security / spam detection** (KNN Euclidean + Word2Vec) → [Example 1](docs/real_examples.md#example-1--security-phishing--spam-detection)
- **Biology / sequence classification** (Naive Bayes + TF-IDF) → [Example 2](docs/real_examples.md#example-2--biology-splice-junction-sequence-classification)
- **Finance / sentiment** (Logistic Regression + Random Forest) → [Example 3](docs/real_examples.md#example-3--finance-sentiment-classification)
- **Comparing all four classifiers side by side** → [Comparison snippet](docs/real_examples.md#comparing-all-four-classifiers-on-the-same-dataset)
- **Persisting a trained model to disk** → [save-load.md](docs/save-load.md)
- **Loading an external fastText `.bin` from the CLI tool** → [External fastText workflow](docs/save-load.md#external-fasttext-embeddings--end-to-end-workflow)

---

## 🎯 Use Cases

### 🔐 Security & Threat Detection
- Spam classification
- Phishing detection
- Web attack identification (SQLi, XSS, fuzzing)
- Malware behavior analysis
- Log anomaly detection

### 💰 Fraud Detection
- Transaction anomaly detection
- Behavioral fraud patterns
- Suspicious activity classification

### 🧬 Biological & Scientific Analysis
Vec-Eyes Core can be adapted for:

- Virus pattern classification
- Human / biological signal classification
- Bacteria and fungus pattern detection
- Biomedical text/log classification

---

## ⚙️ Core Architecture

```
Input Text / Data
        ↓
Normalization / Tokenization
        ↓
Feature Extraction (TF-IDF / Embeddings)
        ↓
ML Engine (classifiers)
        ↓
Rule Engine (Regex / VectorScan)
        ↓
Hybrid Scoring
        ↓
Final Classification
```

---

## 🧠 Machine Learning

### ✔ Multiple ML Algorithms:
  - KNN (Cosine, Euclidean, Manhattan, Minkowski)
  - Naive Bayes (Count, TF-IDF)
  - Logistic Regression
  - SVM (Linear, RBF, Polynomial, Sigmoid)
  - Random Forest (Standard, Balanced, ExtraTrees + OOB)
  - Gradient Boosting
  - Isolation Forest
---


## 🔡 NLP Pipeline

- Tokenization
- Normalization
- TF-IDF
- Word2Vec (lightweight training)
- FastText-style embeddings (subword support)

---

## 🔎 Rule Engine

Vec-Eyes supports rule-based matching with scoring.

### ✔ Default (no dependencies)
- Regex-based matcher

### ✔ Optional (feature flag)
- VectorScan feature-gated backend. The current implementation keeps a safe regex-compatible fallback behind the feature until native scanning paths are wired.

---

## 🔥 YAML Pipeline

Vec-Eyes allows full pipeline definition via nested YAML. The current format groups data, NLP, model configuration, optional rules, and output settings explicitly.

### Minimal KNN + FastText

```yaml
report_name: Spam Security Classifier

data:
  hot_test_path: /data/email/spam/
  cold_test_path: /data/email/normal/
  hot_label: SPAM
  cold_label: RAW_DATA
  recursive_way: On
  score_sum: On

pipeline:
  nlp: FastText
  threads: 4

model:
  method: KnnCosine
  k: 5

extra_match:
  - engine: Regex
    path: /data/rules/spam_rules.txt
    score_add_points: 70
    title: Spam Keywords
    description: Detect common spam patterns
```

### Bayes + TF-IDF

```yaml
report_name: Fraud Narrative Classifier

data:
  hot_test_path: /data/fraud/hot/
  cold_test_path: /data/fraud/cold/
  hot_label: ANOMALY
  cold_label: RAW_DATA
  recursive_way: On
  score_sum: Off

pipeline:
  nlp: TfIdf
  threads: 2

model:
  method: Bayes
```

### Random Forest + OOB

```yaml
data:
  hot_test_path: /data/http/attacks/
  cold_test_path: /data/http/normal/
  hot_label: WEB_ATTACK
  cold_label: RAW_DATA

pipeline:
  nlp: TfIdf
  threads: 8

model:
  method: RandomForest
  n_trees: 200
  max_depth: 10
  mode: ExtraTrees
  max_features: Sqrt
  bootstrap: true
  oob_score: true
```

### Gradient Boosting With Shallow Trees

```yaml
data:
  hot_test_path: /data/fraud/high-risk/
  cold_test_path: /data/fraud/low-risk/
  hot_label: ANOMALY
  cold_label: RAW_DATA

pipeline:
  nlp: TfIdf
  threads: 4

model:
  method: GradientBoosting
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 3
```

### Isolation Forest

```yaml
data:
  hot_test_path: /data/anomaly/known_outliers/
  cold_test_path: /data/anomaly/normal/
  hot_label: ANOMALY
  cold_label: RAW_DATA

pipeline:
  nlp: FastText
  threads: 4

model:
  method: IsolationForest
  n_trees: 150
  contamination: 0.02
  subsample_size: 256
```

---

# Rust API Examples

## KNN + FastText

```rust
use vec_eyes_lib::{ClassificationLabel, ClassifierFactory, ClassifierMethod, NlpOption};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = ClassifierFactory::builder()
        .method(ClassifierMethod::KnnCosine)
        .nlp(NlpOption::FastText)
        .hot_path("data/spam")
        .cold_path("data/normal")
        .hot_label(ClassificationLabel::Spam)
        .cold_label(ClassificationLabel::RawData)
        .k(5)
        .threads(Some(4))
        .build()?;

    let result = classifier.classify_text(
        "claim your free casino bonus now",
        vec_eyes_lib::ScoreSumMode::Off,
        &[],
    );
    println!("{result:?}");

    Ok(())
}
```

---

## Bayes + TF-IDF

```rust
use vec_eyes_lib::{ClassificationLabel, ClassifierFactory, ClassifierMethod, NlpOption};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = ClassifierFactory::builder()
        .method(ClassifierMethod::Bayes)
        .nlp(NlpOption::TfIdf)
        .hot_path("data/fraud")
        .cold_path("data/normal")
        .hot_label(ClassificationLabel::Anomaly)
        .cold_label(ClassificationLabel::RawData)
        .threads(Some(2))
        .build()?;

    let result = classifier.classify_text(
        "urgent offshore transfer to anonymous account",
        vec_eyes_lib::ScoreSumMode::Off,
        &[],
    );
    println!("{result:?}");

    Ok(())
}
```

---

## Compatibility Matrix

The matrix below is designed to make Vec-Eyes easier to understand and safer to configure. It summarizes what each classifier is best at, which NLP representations fit best, which parameters are required, and which non-trivial parameters strongly affect results.

| Classifier | Best NLP / Feature Input | Required Parameters | Important Non-Trivial Parameters | Best Use Cases | Notes |
|---|---|---|---|---|---|
| **Bayes** | `Count`, `TfIdf` | None | `threads` | Spam detection, fast baseline text classification, simple fraud text screening | Very fast and stable. Best as a baseline. Not ideal for dense embeddings like Word2Vec/FastText. |
| **KnnCosine** | `FastText`, `Word2Vec` | `k` | `threads` | Similarity-based classification, noisy text, phishing, behavioral text matching | Strong default for embedding-based text classification. Cosine is usually the best first KNN metric for dense vectors. |
| **KnnEuclidean** | `FastText`, `Word2Vec` | `k` | `threads` | Distance-based embedding experiments, attack clustering | More sensitive to magnitude than cosine. Useful for controlled experiments. |
| **KnnManhattan** | `FastText`, `Word2Vec` | `k` | `threads` | Alternative distance profile for embeddings | Often used for experimentation rather than as the default production KNN choice. |
| **KnnMinkowski** | `FastText`, `Word2Vec` | `k`, `p` | `threads` | Research-style distance tuning, anomaly/similarity experiments | `p` changes the geometry of distance. Use only when you explicitly want to tune distance behavior. |
| **LogisticRegression** | `TfIdf`, `Count`, sometimes dense embeddings | `logistic_learning_rate`, `logistic_epochs` | `logistic_lambda`, `threads` | Fraud classification, text classification, strong production baseline | Great balance between interpretability, speed, and quality. Very practical model. |
| **SVM** | `TfIdf`, `Count`, sometimes dense embeddings | `svm_kernel`, `svm_c` | `svm_learning_rate`, `svm_epochs`, `svm_gamma`, `svm_degree`, `svm_coef0`, `threads` | Security text classification, spam, fraud, web attack text | `Linear` is usually the best first choice. `Rbf` is more expressive but more sensitive to tuning. |
| **RandomForest** | `TfIdf`, `FastText`, structured-ish feature sets | `random_forest_n_trees` | `random_forest_mode`, `random_forest_max_depth`, `random_forest_max_features`, `random_forest_min_samples_split`, `random_forest_min_samples_leaf`, `random_forest_bootstrap`, `random_forest_oob_score`, `threads` | Richer risk scoring, structured features, fraud, mixed-signal classification | Good when you want ensembles and model diversity. Supports `Standard`, `Balanced`, and `ExtraTrees`. |
| **GradientBoosting** | `TfIdf`, structured-ish feature sets | `n_estimators`, `learning_rate` | `max_depth`, `threads` | Fraud/risk scoring, more expressive tabular-like classification | Uses shallow regression trees and honors `max_depth`; more sensitive to hyperparameters than Bayes or Logistic Regression. |
| **IsolationForest** | `FastText`, `Word2Vec`, anomaly-oriented feature sets | `isolation_forest_n_trees`, `isolation_forest_contamination` | `isolation_forest_subsample_size`, `threads` | Anomaly detection, unusual behavior detection, outlier hunting | Best when the goal is finding what looks abnormal rather than choosing among many known labels. |

---


## 🧠 How Dataset Loading Works

- `hot` directories → labeled as target class
- `cold` directories → baseline / normal behavior
- All files are read **recursively**
- Multiple directories supported
- Each file contributes to training vectors

---


## 🔧 Optional VectorScan Support

### Fedora

```bash
sudo dnf install boost-devel cmake gcc gcc-c++
```

### Debian / Ubuntu

```bash
sudo apt install libboost-all-dev cmake build-essential
```

```bash
cargo build --features vectorscan
```

---

## ⚙️ Supported Methods

- Bayes
- KnnCosine
- KnnEuclidean
- KnnManhattan
- KnnMinkowski (requires `p`)
- LogisticRegression
- RandomForest
- Svm
- GradientBoosting
- IsolationForest

---

## ⚠️ Validation Rules

- KNN requires `k`
- Minkowski requires `p`
- Logistic Regression requires positive `learning_rate` and `epochs`
- Random Forest requires `n_trees >= 1`
- SVM requires `kernel` and positive `c`
- Gradient Boosting requires `n_estimators >= 1` and positive `learning_rate`
- Isolation Forest requires `n_trees >= 1` and `contamination` in `(0, 0.5)`
- YAML is validated before execution
---

## ⚡ Performance

- Rust-native
- Rayon parallelism
- ndarray + BLAS ready

---

## 🧩 Embedding in Your Project

Use `ClassifierFactory::builder()` for dynamic construction, or `ClassifierSpec` / `TypedClassifierBuilder` when you want stronger compile-time guidance.

---

## 🔗 Relationship with CLI

- `vec-eyes-lib` → core engine
- `vec-eyes-cli` → interface layer
- 
https://github.com/Orangewarrior/vec-eyes-cli
---
## 🧪 Testing Guide

https://github.com/Orangewarrior/vec-eyes-lib/blob/main/tests/tests.md
https://github.com/Orangewarrior/vec-eyes-lib/wiki/%F0%9F%A7%AA-Testing-Guide

## 🤝 Contributing

We welcome contributions in:

- ML improvements
- Performance optimization
- Rule engine enhancements
- Dataset integrations
- Biological classification extensions

---

## 👤 Author

Orangewarrior


If you like Vec-Eyes:

- ⭐ Star the repo
- 💡 Open issues
- 🔧 Contribute
