# Vec-Eyes Core 🧠🔬

**High-performance behavior intelligence engine for Rust**

Vec-Eyes Core is a modular, high-performance behavior classification engine written in Rust, designed to power advanced detection systems across security, data science, and biological domains.

It combines machine learning, NLP, vector embeddings, and rule-based matching into a unified engine for pattern detection and classification.

---

## 🚀 What is Vec-Eyes Core?

Vec-Eyes Core is the **engine behind Vec-Eyes CLI**.

It provides:

- 🧠 Machine Learning (KNN, Naive Bayes)
- 🔡 NLP pipelines (Tokenization, TF-IDF, Embeddings)
- ⚡ Vector similarity (Word2Vec, FastText)
- 🔎 Rule engine (Regex / optional VectorScan)
- 📊 Hybrid scoring system

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
ML Engine (KNN / Bayes)
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

## 📄1.1 KNN + FastText for Spam / Security

A strong default for email classification and noisy text detection.

```yaml
method: KnnCosine
nlp: FastText

k: 5
threads: 4

datasets:
  hot:
    - /data/email/spam/
  cold:
    - /data/email/normal/

rules:
  - title: Spam Keywords
    description: Detect common spam patterns
    match_rule: "free|bonus|win|casino|urgent"
    score: 70

  - title: Suspicious URL
    description: Detect promotional or deceptive links
    match_rule: "http://.*(promo|deal|bonus)"
    score: 80
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
method: KnnEuclidean
nlp: Word2Vec

k: 3
threads: 4

datasets:
  hot:
    - /data/http/attacks/
  cold:
    - /data/http/normal/

rules:
  - title: SQL Injection Pattern
    description: Common SQLi fragments
    match_rule: "union select|or 1=1|information_schema"
    score: 90

  - title: XSS Attempt
    description: Typical XSS payload markers
    match_rule: "<script>|alert\(|onerror="
    score: 85
```

### When to use
- HTTP request classification
- attack similarity analysis
- fuzzing / malicious payload detection

---

## 📄1.3 Bayes + TF-IDF for Financial Fraud Text Classification

A simple, fast baseline for suspicious transaction narratives and fraud-related documents.

```yaml
method: Bayes
nlp: TfIdf

threads: 2

datasets:
  hot:
    - /data/fraud/transactions/
  cold:
    - /data/legit/transactions/

rules:
  - title: Suspicious Transaction
    description: Transaction language associated with urgency or manipulation
    match_rule: "transfer|urgent|wire|immediate"
    score: 60

  - title: Known Fraud Pattern
    description: Indicators of laundering, anonymity, or offshore movement
    match_rule: "offshore|crypto|anonymous|shell company"
    score: 75
```

### When to use
- fraud screening
- suspicious transaction review
- baseline text classification for risk teams

---

## 📄1.4 Biological Classification with FastText

A lightweight example for biological text grouping and domain-specific keyword reinforcement.

```yaml
method: KnnCosine
nlp: FastText

k: 4
threads: 4

datasets:
  hot:
    - /data/bio/virus/
  cold:
    - /data/bio/human/

rules:
  - title: Virus Signature
    description: Vocabulary linked to viral sequences and mutations
    match_rule: "rna|mutation|viral|capsid"
    score: 80

  - title: Human Marker
    description: Terms associated with normal human biological context
    match_rule: "human tissue|somatic|host response"
    score: 20
```

### When to use
- biological text classification
- biosignal labeling
- domain experiments in genomics / virology corpora

---

## 📄1.5 Random Forest + OOB + ExtraTrees

Example of a richer structured model configuration.

```yaml
method: RandomForest
nlp: FastText

threads: 8

random_forest_mode: ExtraTrees
random_forest_n_trees: 200
random_forest_max_depth: null
random_forest_max_features: sqrt
random_forest_min_samples_split: 2
random_forest_min_samples_leaf: 1
random_forest_bootstrap: true
random_forest_oob_score: true

datasets:
  hot:
    - /data/http/attacks/
  cold:
    - /data/http/normal/

rules:
  - title: High Risk Attack Rule
    match_rule: "union select|<script>|../|xp_cmdshell"
    score: 90
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
method: SVM
nlp: TfIdf

threads: 4

svm_kernel: Linear
svm_c: 1.0
svm_learning_rate: 0.01
svm_epochs: 50

datasets:
  hot:
    - /data/email/spam/
  cold:
    - /data/email/normal/

rules:
  - title: Spam Promotion Rule
    match_rule: "bonus|prize|winner|cash"
    score: 50
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
method: GradientBoosting
nlp: TfIdf

threads: 4

gradient_boosting_n_estimators: 100
gradient_boosting_learning_rate: 0.1
gradient_boosting_max_depth: 3

datasets:
  hot:
    - /data/fraud/high-risk/
  cold:
    - /data/fraud/low-risk/

rules:
  - title: High Risk Pattern
    match_rule: "urgent transfer|offshore|anonymous wallet"
    score: 65
```

---

## 📄1.8 Isolation Forest for Anomaly Detection

Best suited when your main signal is “normal vs strange”.

```yaml
method: IsolationForest
nlp: FastText

threads: 4

isolation_forest_n_trees: 150
isolation_forest_contamination: 0.02
isolation_forest_subsample_size: 256

datasets:
  hot:
    - /data/anomaly/known_outliers/
  cold:
    - /data/anomaly/normal/

rules:
  - title: Rare Pattern
    match_rule: "unexpected syscall|rare endpoint|unusual payload"
    score: 40
```

---

# 2. Rust API Examples

## 📄2.1 KNN + FastText

```rust
use vec_eyes_lib::{ClassifierFactory, MethodKind, NlpOption};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = ClassifierFactory::new()
        .method(MethodKind::KnnCosine)
        .nlp(NlpOption::FastText)
        .k(Some(5))
        .threads(Some(4))
        .build()?;

    let result = classifier.classify_text("claim your free casino bonus now")?;
    println!("{result:?}");

    Ok(())
}
```

---

## 📄2.2 Bayes + TF-IDF

```rust
use vec_eyes_lib::{ClassifierFactory, MethodKind, NlpOption};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = ClassifierFactory::new()
        .method(MethodKind::Bayes)
        .nlp(NlpOption::TfIdf)
        .threads(Some(2))
        .build()?;

    let result = classifier.classify_text("urgent offshore transfer to anonymous account")?;
    println!("{result:?}");

    Ok(())
}
```

---

## 📄2.3 Random Forest + Advanced Parameters

```rust
use vec_eyes_lib::{
    ClassifierFactory,
    MethodKind,
    NlpOption,
    RandomForestMaxFeatures,
    RandomForestMode,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = ClassifierFactory::new()
        .method(MethodKind::RandomForest)
        .nlp(NlpOption::FastText)
        .threads(Some(8))
        .random_forest_mode(Some(RandomForestMode::ExtraTrees))
        .random_forest_n_trees(Some(200))
        .random_forest_max_features(Some(RandomForestMaxFeatures::Sqrt))
        .random_forest_bootstrap(Some(true))
        .random_forest_oob_score(Some(true))
        .build()?;

    let result = classifier.classify_text("union select password from users where 1=1")?;
    println!("{result:?}");

    Ok(())
}
```

---

## 📄 2.4 SVM + Explicit Kernel

```rust
use vec_eyes_lib::{ClassifierFactory, MethodKind, NlpOption, SvmKernel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = ClassifierFactory::new()
        .method(MethodKind::SVM)
        .nlp(NlpOption::TfIdf)
        .threads(Some(4))
        .svm_kernel(Some(SvmKernel::Linear))
        .svm_c(Some(1.0))
        .svm_learning_rate(Some(0.01))
        .svm_epochs(Some(50))
        .build()?;

    let result = classifier.classify_text("win cash now bonus offer")?;
    println!("{result:?}");

    Ok(())
}
```

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

- KnnCosine
- KnnEuclidean
- KnnManhattan
- KnnMinkowski (requires `p`)
- Bayes

---

## ⚠️ Validation Rules

- KNN requires `k`
- Minkowski requires `p`
- Bayes does not require extra parameters
- YAML is validated before execution
...
---

## ⚡ Performance

- Rust-native
- Rayon parallelism
- ndarray + BLAS ready

---

## 🧩 Embedding in Your Project

```rust
use vec_eyes_core::*;

let classifier = build_classifier(...)
    .with_method(MethodKind::KnnCosine { k: 5 })
    .with_nlp(NlpOption::FastText)
    .load_rules("rules.yaml")
    .train(datasets)?;

let result = classifier.classify("input data");
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
