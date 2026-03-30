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

## 🔥 YAML Pipeline (Advanced Examples)

Vec-Eyes allows full pipeline definition via YAML.

---

## 📄 Example 1 — KNN + FastText (Security / Spam)

```yaml
method: KnnCosine
nlp: FastText
k: 5

datasets:
  hot:
    - /data/email/spam/
  cold:
    - /data/email/normal/

rules:
  - title: Spam Keywords
    description: Detect common spam words
    match_rule: "free|bonus|win|casino"
    score: 70

  - title: Suspicious URL
    match_rule: "http://.*(promo|deal)"
    score: 80
```

---

## 📄 Example 2 — KNN + Word2Vec (Web Attack Detection)

```yaml
method: KnnEuclidean
nlp: Word2Vec
k: 3

datasets:
  hot:
    - /data/http/attacks/
  cold:
    - /data/http/normal/

rules:
  - title: SQL Injection Pattern
    match_rule: "union select|or 1=1"
    score: 90

  - title: XSS Attempt
    match_rule: "<script>|alert\("
    score: 85
```

---

## 📄 Example 3 — Bayes + TF-IDF (Fraud Detection)

```yaml
method: Bayes
nlp: TfIdf

datasets:
  hot:
    - /data/fraud/transactions/
  cold:
    - /data/legit/transactions/

rules:
  - title: Suspicious Transaction
    match_rule: "transfer|urgent|wire"
    score: 60

  - title: Known Fraud Pattern
    match_rule: "offshore|crypto|anonymous"
    score: 75
```

---

## 📄 Example 4 — Biological Classification (FastText)

```yaml
method: KnnCosine
nlp: FastText
k: 4

datasets:
  hot:
    - /data/bio/virus/
    - /data/bio/bacteria/
  cold:
    - /data/bio/human/

rules:
  - title: Virus Signature
    match_rule: "rna|mutation|viral"
    score: 80

  - title: Bacteria Pattern
    match_rule: "bacteria|e.coli"
    score: 70
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
---

## 🧠 How Dataset Loading Works

- `hot` directories → labeled as target class
- `cold` directories → baseline / normal behavior
- All files are read **recursively**
- Multiple directories supported
- Each file contributes to training vectors

---

## ⚙️ Supported NLP Options

- Count
- TfIdf
- Word2Vec
- FastText

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
