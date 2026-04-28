# Vec-Eyes Core 🧠🔬

**High-performance behavior intelligence engine for Rust**

Vec-Eyes Core is a modular, high-performance behavior classification engine written in Rust, designed to power advanced detection systems across security, data science, and biological domains.

It combines machine learning, NLP, vector embeddings, and rule-based matching into a unified engine for pattern detection and classification.

---

## 🚀 What is Vec-Eyes Core?

Vec-Eyes Core is the **engine behind Vec-Eyes CLI**.

It provides:

- 🧠 Machine Learning (KNN, Naive Bayes, Logistic Regression, SVM, Random Forest, Gradient Boosting, Isolation Forest)
- 🔡 NLP pipelines (Tokenization, TF-IDF, Embeddings)
- ⚡ Vector similarity (Word2Vec, FastText, external `.bin`)
- 🔎 Rule engine (Regex / optional VectorScan)
- 📊 Hybrid scoring system

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
- VectorScan (Hyperscan fork for high-speed matching)

---

## 🔥 YAML Pipeline (Advanced Examples)

Vec-Eyes allows full pipeline definition via YAML.

---

The examples below are designed to be:
- **realistic**
- **maintainable**
- **clear for contributors**
- **close to production usage**

---

# 1. YAML Examples

YAML pipelines follow a three-block structure: `data`, `pipeline`, and `model`.
An optional `extra_match` list attaches rule files to the scoring engine.

## 📄1.1 KNN + FastText for Spam / Security

A strong default for email classification and noisy text detection.

```yaml
report_name: Spam KNN FastText

data:
  hot_test_path: data/email/spam
  cold_test_path: data/email/normal
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
    path: rules/spam_keywords.txt
    score_add_points: 20
    title: Spam keywords
```

### When to use
- spam detection
- phishing-like text
- noisy or typo-heavy messages
- unstructured text with strong lexical patterns

---

## 📄1.2 KNN + Word2Vec for Web Attack Detection

Useful for request classification, payload similarity, and attack family grouping.

```yaml
report_name: Web Attack KNN Word2Vec

data:
  hot_test_path: data/http/attacks
  cold_test_path: data/http/normal
  hot_label: WEB_ATTACK
  cold_label: RAW_DATA
  recursive_way: On
  score_sum: On

pipeline:
  nlp: Word2Vec
  threads: 4

model:
  method: KnnEuclidean
  k: 3

extra_match:
  - engine: Regex
    path: rules/sqli_xss.txt
    score_add_points: 25
    title: SQLi / XSS patterns
```

### When to use
- HTTP request classification
- attack similarity analysis
- fuzzing / malicious payload detection

---

## 📄1.3 Bayes + TF-IDF for Financial Fraud Text Classification

A simple, fast baseline for suspicious transaction narratives and fraud-related documents.

```yaml
report_name: Fraud Bayes TfIdf

data:
  hot_test_path: data/fraud/transactions
  cold_test_path: data/legit/transactions
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

### When to use
- fraud screening
- suspicious transaction review
- baseline text classification for risk teams

---

## 📄1.4 Biological Classification with FastText

A lightweight example for biological text grouping and domain-specific keyword reinforcement.

```yaml
report_name: Biology KNN FastText

data:
  hot_test_path: data/bio/virus
  cold_test_path: data/bio/human
  hot_label: VIRUS
  cold_label: HUMAN
  recursive_way: On
  score_sum: On

pipeline:
  nlp: FastText
  threads: 4

model:
  method: KnnCosine
  k: 4

extra_match:
  - engine: Regex
    path: rules/virus_markers.txt
    score_add_points: 15
    title: Virus signature keywords
```

### When to use
- biological text classification
- biosignal labeling
- domain experiments in genomics / virology corpora

---

## 📄1.5 Random Forest + OOB + ExtraTrees

Example of a richer structured model configuration.

```yaml
report_name: Web Attack Random Forest

data:
  hot_test_path: data/http/attacks
  cold_test_path: data/http/normal
  hot_label: WEB_ATTACK
  cold_label: RAW_DATA
  recursive_way: On
  score_sum: Off

pipeline:
  nlp: FastText
  threads: 8

model:
  method: RandomForest
  mode: extra_trees
  n_trees: 200
  max_features: sqrt
  min_samples_split: 2
  min_samples_leaf: 1
  bootstrap: true
  oob_score: true
```

### When to use
- structured or semi-structured risk signals
- richer classification experiments
- Random Forest benchmarking
- OOB-based internal validation

---

## 📄1.6 SVM with Explicit Kernel Configuration

A clean example for more advanced text classification.

```yaml
report_name: Spam SVM Linear

data:
  hot_test_path: data/email/spam
  cold_test_path: data/email/normal
  hot_label: SPAM
  cold_label: RAW_DATA
  recursive_way: On
  score_sum: Off

pipeline:
  nlp: TfIdf
  threads: 4

model:
  method: Svm
  kernel: Linear
  c: 1.0
  learning_rate: 0.08
  epochs: 50
```

### Other valid kernels
- `Linear`
- `Rbf`
- `Polynomial`
- `Sigmoid`

---

## 📄1.7 Gradient Boosting

Good for more structured scoring scenarios.

```yaml
report_name: Fraud Gradient Boosting

data:
  hot_test_path: data/fraud/high-risk
  cold_test_path: data/fraud/low-risk
  hot_label: ANOMALY
  cold_label: RAW_DATA
  recursive_way: On
  score_sum: Off

pipeline:
  nlp: TfIdf
  threads: 4

model:
  method: GradientBoosting
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 3
```

---

## 📄1.8 Isolation Forest for Anomaly Detection

Best suited when your main signal is “normal vs strange”.

```yaml
report_name: Anomaly Isolation Forest

data:
  hot_test_path: data/anomaly/known_outliers
  cold_test_path: data/anomaly/normal
  hot_label: ANOMALY
  cold_label: RAW_DATA
  recursive_way: On
  score_sum: Off

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

# 2. Rust API Examples

## 📄2.1 KNN + FastText

```rust
use vec_eyes_lib::{
    ClassifierFactory, ClassifierMethod, ClassificationLabel, NlpOption, ScoreSumMode,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = ClassifierFactory::builder()
        .method(ClassifierMethod::KnnCosine)
        .nlp(NlpOption::FastText)
        .k(5)
        .threads(Some(4))
        .hot_path("data/email/spam")
        .cold_path("data/email/normal")
        .hot_label(ClassificationLabel::Spam)
        .cold_label(ClassificationLabel::RawData)
        .build()?;

    let result = classifier.classify_text("claim your free casino bonus now", ScoreSumMode::Off, &[]);
    println!("{:?}", result.labels);

    Ok(())
}
```

---

## 📄2.2 Bayes + TF-IDF

```rust
use vec_eyes_lib::{
    ClassifierFactory, ClassifierMethod, ClassificationLabel, NlpOption, ScoreSumMode,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = ClassifierFactory::builder()
        .method(ClassifierMethod::Bayes)
        .nlp(NlpOption::TfIdf)
        .threads(Some(2))
        .hot_path("data/fraud/hot")
        .cold_path("data/fraud/cold")
        .hot_label(ClassificationLabel::Anomaly)
        .cold_label(ClassificationLabel::RawData)
        .build()?;

    let result = classifier.classify_text("urgent offshore transfer to anonymous account", ScoreSumMode::Off, &[]);
    println!("{:?}", result.labels);

    Ok(())
}
```

---

## 📄2.3 Random Forest + Advanced Parameters

```rust
use vec_eyes_lib::{
    ClassifierFactory, ClassifierMethod, ClassificationLabel,
    NlpOption, RandomForestMaxFeatures, RandomForestMode, ScoreSumMode,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = ClassifierFactory::builder()
        .method(ClassifierMethod::RandomForest)
        .nlp(NlpOption::FastText)
        .threads(Some(8))
        .hot_path("data/http/attacks")
        .cold_path("data/http/normal")
        .hot_label(ClassificationLabel::WebAttack)
        .cold_label(ClassificationLabel::RawData)
        .random_forest_full_config(
            RandomForestMode::ExtraTrees,
            200,           // n_trees
            Some(12),      // max_depth
            Some(RandomForestMaxFeatures::Sqrt),
            Some(2),       // min_samples_split
            Some(1),       // min_samples_leaf
            Some(true),    // bootstrap
            Some(true),    // oob_score
        )
        .build()?;

    let result = classifier.classify_text("union select password from users where 1=1", ScoreSumMode::Off, &[]);
    println!("{:?}", result.labels);

    Ok(())
}
```

---

## 📄2.4 SVM + Explicit Kernel

```rust
use vec_eyes_lib::{
    ClassifierFactory, ClassifierMethod, ClassificationLabel,
    NlpOption, ScoreSumMode, SvmConfig, SvmKernel,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = ClassifierFactory::builder()
        .method(ClassifierMethod::Svm)
        .nlp(NlpOption::TfIdf)
        .threads(Some(4))
        .hot_path("data/email/spam")
        .cold_path("data/email/normal")
        .hot_label(ClassificationLabel::Spam)
        .cold_label(ClassificationLabel::RawData)
        .svm_config(SvmConfig {
            kernel: SvmKernel::Linear,
            c: 1.0,
            learning_rate: 0.08,
            epochs: 50,
            ..Default::default()
        })
        .build()?;

    let result = classifier.classify_text("win cash now bonus offer", ScoreSumMode::Off, &[]);
    println!("{:?}", result.labels);

    Ok(())
}
```

---

## 📄2.5 YAML pipeline (load and run)

```rust
use vec_eyes_lib::{classifier::run_rules_pipeline, config::RulesFile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rules = RulesFile::from_yaml_path("pipeline.yaml")?;
    let report = run_rules_pipeline(&rules, std::path::Path::new("data/classify"))?;
    report.write_json("output/report.json")?;
    println!("Classified {} files", report.records.len());
    Ok(())
}
```

---

## Compatibility Matrix

The matrix below summarises what each classifier is best at, which NLP representations fit, which `model:` parameters are required in YAML, and which non-trivial parameters strongly affect results.

| Classifier (`method:`) | Best NLP | Required params | Important optional params | Best Use Cases | Notes |
|---|---|---|---|---|---|
| **Bayes** | `Count`, `TfIdf` | — | `threads` | Spam, fast baseline, fraud text screening | Very fast and stable. Best as a first baseline. |
| **KnnCosine** | `FastText`, `Word2Vec` | `k` | `threads` | Phishing, noisy text, behavioral matching | Best default KNN metric for dense embeddings. |
| **KnnEuclidean** | `FastText`, `Word2Vec` | `k` | `threads` | Attack clustering, distance-based experiments | More magnitude-sensitive than cosine. |
| **KnnManhattan** | `FastText`, `Word2Vec` | `k` | `threads` | Alternative distance profile | Useful for experimentation. |
| **KnnMinkowski** | `FastText`, `Word2Vec` | `k`, `p` | `threads` | Distance geometry tuning | `p=2` ≡ Euclidean; `p=1` ≡ Manhattan. |
| **LogisticRegression** | `TfIdf`, `Count`, dense | `learning_rate`, `epochs` | `lambda`, `threads` | Fraud, text classification | Strong production baseline; interpretable. |
| **Svm** | `TfIdf`, `Count`, dense | `kernel`, `c` | `learning_rate`, `epochs`, `gamma`, `degree`, `coef0`, `threads` | Security, spam, web attacks | `Linear` first; `Rbf` for non-linear separation. |
| **RandomForest** | `TfIdf`, `FastText`, dense | `n_trees` | `mode`, `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf`, `bootstrap`, `oob_score`, `threads` | Mixed-signal classification, risk scoring | Supports `Standard`, `Balanced`, `ExtraTrees`. |
| **GradientBoosting** | `TfIdf`, dense | `n_estimators`, `learning_rate` | `max_depth`, `threads` | Fraud/risk scoring | More sensitive to tuning than RF or Bayes. |
| **IsolationForest** | Any (`TfIdf`, `FastText`, `Word2Vec`, …) | `n_trees`, `contamination` | `subsample_size`, `threads` | Anomaly detection, outlier hunting | Goal is "normal vs strange", not multi-class. |

---


## 🧠 How Dataset Loading Works

- `hot_test_path` → target class (e.g. spam, attacks, virus)
- `cold_test_path` → baseline / normal class
- `recursive_way: On` reads sub-directories recursively
- Each `.txt` file is one training sample
- Streaming iterator available via `training_sample_iter()` for large corpora

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

| Method | YAML `method:` value |
|---|---|
| K-Nearest Neighbours (Cosine) | `KnnCosine` |
| K-Nearest Neighbours (Euclidean) | `KnnEuclidean` |
| K-Nearest Neighbours (Manhattan) | `KnnManhattan` |
| K-Nearest Neighbours (Minkowski) | `KnnMinkowski` |
| Naive Bayes | `Bayes` |
| Logistic Regression | `LogisticRegression` |
| Support Vector Machine | `Svm` |
| Random Forest | `RandomForest` |
| Gradient Boosting | `GradientBoosting` |
| Isolation Forest | `IsolationForest` |

---

## ⚠️ Validation Rules

- KNN variants require `k ≥ 1`
- `KnnMinkowski` additionally requires `p > 0`
- `LogisticRegression` requires `learning_rate` and `epochs`
- `Svm` requires `kernel` and `c`
- `GradientBoosting` requires `n_estimators` and `learning_rate`
- `IsolationForest` requires `n_trees` and `contamination` (in `(0, 0.5)`)
- `RandomForest` with `oob_score: true` requires `bootstrap: true`
- `threads: 0` is rejected
- Output paths containing `..` are rejected (path-traversal guard)
- All validation runs before any training or file I/O

---

## ⚡ Performance

- Rust-native
- Rayon parallelism
- ndarray + BLAS ready

---

## 🧩 Embedding in Your Project

Add to `Cargo.toml`:

```toml
[dependencies]
vec-eyes-lib = { path = "../vec-eyes-lib" }
# or once published:
# vec-eyes-lib = "3.2.0"
```

Minimal classifier:

```rust
use vec_eyes_lib::{
    ClassifierFactory, ClassifierMethod, ClassificationLabel, NlpOption, ScoreSumMode,
};

let classifier = ClassifierFactory::builder()
    .method(ClassifierMethod::KnnCosine)
    .nlp(NlpOption::FastText)
    .k(5)
    .hot_path("data/hot")
    .cold_path("data/cold")
    .hot_label(ClassificationLabel::Spam)
    .cold_label(ClassificationLabel::RawData)
    .build()?;

let result = classifier.classify_text("free prize click here", ScoreSumMode::Off, &[]);
let top = result.labels.first();
println!("{top:?}");
```

YAML pipeline:

```rust
use vec_eyes_lib::{classifier::run_rules_pipeline, config::RulesFile};

let rules = RulesFile::from_yaml_path("pipeline.yaml")?;
let report = run_rules_pipeline(&rules, std::path::Path::new("data/classify"))?;
report.write_json("report.json")?;
```

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
