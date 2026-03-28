# Vec-Eyes Core 🧠🔬

> **High-performance behavior intelligence engine for Rust**

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

### ✔ KNN
Supports:

- Cosine similarity
- Euclidean distance
- Manhattan distance
- Minkowski distance (requires `p`)

Required parameters:

- `k: usize`
- `p: Option<f32>` (only for Minkowski)

---

### ✔ Naive Bayes

- Count-based
- TF-IDF-based
- No mandatory hyperparameters

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

## 📄 YAML Rules Engine

Vec-Eyes Core can be fully configured via YAML.

### Example:

```yaml
method: KnnCosine
k: 5

rules:
  - title: Suspicious Keywords
    description: Detect spam patterns
    match_rule: "free|bonus|casino"
    score: 70

  - title: Known Malicious IP
    match_rule: "192\.168\.1\.100"
    score: 100
```

---

### 🧠 YAML Fields Explained

| Field        | Description |
|-------------|------------|
| `method`     | Classification method (`KnnCosine`, `KnnEuclidean`, `KnnManhattan`, `KnnMinkowski`, `Bayes`) |
| `k`          | Required for KNN methods |
| `p`          | Required only for Minkowski |
| `title`      | Rule name (used in reports/logs) |
| `description`| Optional explanation |
| `match_rule` | Regex or pattern |
| `score`      | Score (0–100) added to classification |

---

## 📄 YAML Example (Security)

```yaml
method: KnnCosine
k: 5

rules:
  - title: SQL Injection
    match_rule: "union select|or 1=1"
    score: 90

  - title: Suspicious Agent
    match_rule: "sqlmap|nikto"
    score: 80
```

---

## 📄 YAML Example (Biological)

```yaml
method: KnnEuclidean
k: 3

rules:
  - title: Virus Marker
    match_rule: "rna|mutation|virus"
    score: 85

  - title: Bacteria Pattern
    match_rule: "bacteria|e.coli"
    score: 60
```

---

## 🏷️ Supported Labels

SPAM, MALWARE, PHISHING, ANOMALY, WEB_ATTACK, FUZZING, FLOOD, FRAUD, BLOCK_LIST, RAW_DATA,  
VIRUS, HUMAN, ANIMAL, CANCER, FUNGUS, BACTERIA

---

## ⚡ Performance

- Rust-native 🦀
- ndarray + BLAS ready
- Rayon parallelism
- High-throughput design

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

---

## 🤝 Contributing

We welcome contributions in:

- ML improvements
- Performance optimization
- Rule engine enhancements
- Dataset integrations
- Biological classification extensions

---

## 🧠 Vision

Vec-Eyes aims to become:

> **A unified behavior intelligence engine for security, data science, and biological pattern analysis**

---

## 👤 Author

Orangewarrior

---

## ⭐ Support the Project

If you like Vec-Eyes:

- ⭐ Star the repo
- 💡 Open issues
- 🔧 Contribute
