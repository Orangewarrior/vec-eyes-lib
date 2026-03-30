## 🧪 Testing Guide

Vec-Eyes includes a comprehensive test suite designed to validate real-world behavior across machine learning pipelines, NLP processing, and rule-based detection.

### 🚀 Running Tests

Run all tests:

```bash
cargo test
```

Run only YAML-based realistic tests:

```bash
cargo test --test yaml_realistic -- --nocapture
```

---

### 📂 Test Structure

```
tests/
├── basic.rs              # Core validation tests
├── yaml_realistic.rs     # Real-world pipeline tests
└── data/
    ├── rules/            # YAML rule files
    ├── fraud/            # Financial datasets (hot/cold)
    ├── biology/          # Biological datasets
    └── web/              # Web attack datasets
```

---

### 🧠 What Is Being Tested?

#### ✔ YAML Parsing & Validation
- Ensures required fields (`k`, `p`) are present
- Validates method correctness (KNN vs Bayes)

#### ✔ NLP + ML Pipelines
- KNN (Cosine, Euclidean, Manhattan, Minkowski)
- Naive Bayes (Count, TF-IDF)
- FastText / Word2Vec embeddings

#### ✔ Rule Engine
- Regex matching (default)
- Optional VectorScan support
- Score boosting behavior

#### ✔ Recursive Dataset Loading
- Reads multiple directories (`hot` / `cold`)
- Loads files recursively
- Builds training datasets dynamically

---

### ⚖️ Testing Philosophy

Vec-Eyes tests are designed to be:

- **Realistic** → Use real-like datasets instead of mocks
- **Robust** → Avoid fragile assertions (KNN is probabilistic)
- **Flexible** → Validate behavior, not exact scores

Example:

```rust
assert!(result.contains("ANOMALY:"));
```

Instead of:

```rust
assert_eq!(top_label, "ANOMALY");
```

---

### 📊 Supported Domains in Tests

- 🔐 Security (web attacks, spam, malware)
- 💰 Financial fraud detection
- 🧬 Biological classification (virus, bacteria, etc.)

---

### 🛠️ Debugging Tests

To print debug output:

```bash
cargo test -- --nocapture
```

Add debug logs inside tests:

```rust
println!("result = {}", result);
```

---

### 💡 Tips

- If a KNN test fails, check dataset separation
- If a rule test fails, verify `match_rule` patterns
- YAML errors usually indicate missing required fields

---

### 🤝 Contributing Tests

We welcome contributions:

- New datasets
- More realistic scenarios
- Performance/benchmark tests
- Edge-case validations

---

Vec-Eyes testing is built to simulate real-world pipelines — not just unit logic.
