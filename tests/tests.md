## 🧪 Testing Guide

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
| **GradientBoosting** | `TfIdf`, structured-ish feature sets | `gradient_boosting_n_estimators`, `gradient_boosting_learning_rate` | `gradient_boosting_max_depth`, `threads` | Fraud/risk scoring, more expressive tabular-like classification | More sensitive to hyperparameters than Bayes or Logistic Regression. |
| **IsolationForest** | `FastText`, `Word2Vec`, anomaly-oriented feature sets | `isolation_forest_n_trees`, `isolation_forest_contamination` | `isolation_forest_subsample_size`, `threads` | Anomaly detection, unusual behavior detection, outlier hunting | Best when the goal is finding what looks abnormal rather than choosing among many known labels. |

---

## Random Forest Modes

Vec-Eyes supports multiple Random Forest operating modes. These are not cosmetic flags — they change how the ensemble is built.

| Mode | Description | When to Use |
|---|---|---|
| `Standard` | Classic bagging-style Random Forest | General default use |
| `Balanced` | More balanced sampling behavior across classes | Imbalanced datasets, fraud/security |
| `ExtraTrees` | More randomized split behavior | Faster/high-variance ensemble experiments |

### Random Forest YAML fields

| Field | Required? | Description |
|---|---|---|
| `random_forest_mode` | Optional | `Standard`, `Balanced`, or `ExtraTrees` |
| `random_forest_n_trees` | **Required** | Number of trees in the forest |
| `random_forest_max_depth` | Optional | Maximum tree depth |
| `random_forest_max_features` | Optional | Feature subset strategy, such as `Sqrt`, `Log2`, `All`, `Half` |
| `random_forest_min_samples_split` | Optional | Minimum samples required to split a node |
| `random_forest_min_samples_leaf` | Optional | Minimum samples per leaf |
| `random_forest_bootstrap` | Optional | Enables bootstrap sampling |
| `random_forest_oob_score` | Optional | Enables Out-Of-Bag scoring; meaningful only when bootstrap is enabled |

---

## SVM Kernels

SVM is one of the most tunable models in Vec-Eyes. Kernel selection matters.

| Kernel | Required? | Typical Use |
|---|---|---|
| `Linear` | Yes, if SVM is selected | Strong first choice for TF-IDF / text classification |
| `Rbf` | Yes, if SVM is selected | More flexible non-linear separation |
| `Polynomial` | Optional advanced choice | Research / experimentation |
| `Sigmoid` | Optional advanced choice | Experimental / uncommon |

### SVM YAML fields

| Field | Required? | Description |
|---|---|---|
| `svm_kernel` | **Required** | `Linear`, `Rbf`, `Polynomial`, or `Sigmoid` |
| `svm_c` | **Required** | Regularization strength |
| `svm_learning_rate` | Optional | Optimization step size |
| `svm_epochs` | Optional | Number of training iterations |
| `svm_gamma` | Optional | Important for RBF and often for non-linear kernels |
| `svm_degree` | Optional | Used by Polynomial kernel |
| `svm_coef0` | Optional | Offset term for Polynomial / Sigmoid kernels |

---

## Logistic Regression Parameters

| Field | Required? | Description |
|---|---|---|
| `logistic_learning_rate` | **Required** | Optimization step size |
| `logistic_epochs` | **Required** | Number of training iterations |
| `logistic_lambda` | Optional | Regularization term |

### Why these matter
Logistic Regression is a strong production-grade baseline. The required parameters are intentionally explicit because convergence quality depends heavily on them.

---

## Gradient Boosting Parameters

| Field | Required? | Description |
|---|---|---|
| `gradient_boosting_n_estimators` | **Required** | Number of boosting rounds |
| `gradient_boosting_learning_rate` | **Required** | Shrinkage / learning rate |
| `gradient_boosting_max_depth` | Optional | Tree depth per weak learner |

### Why these matter
Gradient Boosting is powerful but sensitive. The required fields are the minimum meaningful configuration.

---

## Isolation Forest Parameters

| Field | Required? | Description |
|---|---|---|
| `isolation_forest_n_trees` | **Required** | Number of isolation trees |
| `isolation_forest_contamination` | **Required** | Expected anomaly ratio |
| `isolation_forest_subsample_size` | Optional | Subsample size for each tree |

### Why these matter
Isolation Forest is an anomaly-first model. `contamination` strongly affects sensitivity.

---

## KNN Parameters

| Field | Required? | Description |
|---|---|---|
| `k` | **Required** | Number of nearest neighbors |
| `p` | **Required only for `KnnMinkowski`** | Minkowski distance exponent |
| `threads` | Optional | Parallelism for distance computation |

### KNN Method Summary

| Method | Extra Required Params | Recommended NLP |
|---|---|---|
| `KnnCosine` | `k` | `FastText`, `Word2Vec` |
| `KnnEuclidean` | `k` | `FastText`, `Word2Vec` |
| `KnnManhattan` | `k` | `FastText`, `Word2Vec` |
| `KnnMinkowski` | `k`, `p` | `FastText`, `Word2Vec` |

---

## Bayes Parameters

| Field | Required? | Description |
|---|---|---|
| `threads` | Optional | Parallel scoring over labels |

### Recommended NLP
- `Count`
- `TfIdf`

Bayes is intentionally lightweight. It is the easiest classifier to configure and a very good baseline for open-source contributors and first-time users.

---

## Global Parameters

These fields affect multiple classifiers or the pipeline as a whole.

| Field | Required? | Description |
|---|---|---|
| `method` | **Required** | Selects the classifier family |
| `nlp` | Usually required in practice | Selects the text representation strategy |
| `threads` | Optional | Controls Rayon parallelism when supported |
| `datasets.hot` | Usually required | Positive / target data directories |
| `datasets.cold` | Usually required | Baseline / negative / normal data directories |
| `rules` | Optional | Regex / VectorScan-style score boosting rules |

---

## Model Selection Guide

### Use **Bayes**
When you want:
- a very fast baseline
- simple spam classification
- low operational complexity

### Use **KNN + FastText**
When you want:
- semantic similarity
- robust handling of noisy text
- phishing / security text classification

### Use **Logistic Regression**
When you want:
- a practical production baseline
- strong fraud or risk classification
- better control than Bayes with simpler tuning than SVM

### Use **SVM**
When you want:
- strong text classification
- a classic ML model with powerful decision boundaries
- explicit control over kernel behavior

### Use **Random Forest**
When you want:
- ensemble behavior
- richer structured-ish signals
- class balancing options
- OOB scoring

### Use **Gradient Boosting**
When you want:
- more expressive boosting-based classification
- richer fraud/risk scoring
- stronger tuning potential

### Use **Isolation Forest**
When you want:
- anomaly detection
- unknown / rare / suspicious behavior discovery
- outlier-oriented pipelines

---

## Example: Fully Configured YAML for Security

```yaml
method: RandomForest
nlp: FastText
threads: 8

random_forest_mode: ExtraTrees
random_forest_n_trees: 200
random_forest_max_depth: null
random_forest_max_features: Sqrt
random_forest_min_samples_split: 2
random_forest_min_samples_leaf: 1
random_forest_bootstrap: true
random_forest_oob_score: true

datasets:
  hot:
    - /data/http/attacks/
    - /data/email/phishing/
  cold:
    - /data/http/normal/
    - /data/email/normal/

rules:
  - title: SQL Injection Pattern
    description: Common SQL injection fragments
    match_rule: "union select|or 1=1|information_schema"
    score: 90

  - title: XSS Pattern
    description: Common XSS markers
    match_rule: "<script>|alert\(|onerror="
    score: 85
```

---

## Example: Fully Configured YAML for Fraud

```yaml
method: LogisticRegression
nlp: TfIdf
threads: 4

logistic_learning_rate: 0.01
logistic_epochs: 100
logistic_lambda: 0.001

datasets:
  hot:
    - /data/fraud/high-risk/
  cold:
    - /data/fraud/low-risk/

rules:
  - title: Urgent Transfer
    match_rule: "urgent transfer|wire now|immediate settlement"
    score: 60

  - title: Offshore Risk
    match_rule: "offshore|shell company|anonymous wallet|crypto mixer"
    score: 75
```

---

## Example: Fully Configured YAML for Anomaly Detection

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
  - title: Rare Endpoint
    match_rule: "/admin/debug|/internal/unsafe|unexpected syscall"
    score: 40
```

---

## Recommended README Additions

If you want to strengthen the main `README.md`, I would add:

1. **Quick Start**
   - a minimal working YAML
   - a minimal Rust example
   - one `cargo test` command

2. **Compatibility Matrix**
   - the matrix above, or a shortened version of it

3. **Required vs Optional Parameters**
   - especially useful for SVM, Random Forest, Gradient Boosting, Isolation Forest

4. **Threads / Parallelism**
   - document what `threads` affects
   - mention Rayon explicitly

5. **Model Selection Guide**
   - practical guidance for contributors and users

6. **Example Gallery**
   - security
   - fraud
   - biology
   - anomaly detection


# Vec-Eyes Examples & Testing Guide

> Elegant, production-minded examples for **Vec-Eyes Core** — designed for open-source contributors, Rust engineers, and practitioners working on security, fraud detection, anomaly analysis, and scientific classification.

Vec-Eyes has grown beyond a simple spam detector. It is now a **modular behavior intelligence engine** built in Rust, with support for multiple machine learning models, configurable NLP pipelines, YAML-driven workflows, and hybrid ML + rule-based scoring.

This guide gives you:
- modern **YAML examples**
- practical **Rust API examples**
- a clean overview of **how to run and understand tests**
- realistic examples for **security**, **fraud**, **biology**, and **anomaly detection**

---

## Why these examples matter

Vec-Eyes is highly configurable. That flexibility is powerful, but it also means examples need to reflect the **current architecture**:

- modular classifiers
- `Factory`-based selection
- `Builder`-driven configuration
- parallel execution via `threads`
- model-specific parameters
- YAML validation

The examples below are designed to be:
- **realistic**
- **maintainable**
- **clear for contributors**
- **close to production usage**

---

# 1. YAML Examples

## 1.1 KNN + FastText for Spam / Security

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

## 1.2 KNN + Word2Vec for Web Attack Detection

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

## 1.3 Bayes + TF-IDF for Financial Fraud Text Classification

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

## 1.4 Biological Classification with FastText

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

## 1.5 Random Forest + OOB + ExtraTrees

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

## 1.6 SVM with Explicit Kernel Configuration

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

## 1.7 Gradient Boosting

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

## 1.8 Isolation Forest for Anomaly Detection

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

## 2.1 KNN + FastText

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

## 2.2 Bayes + TF-IDF

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

## 2.3 Random Forest + Advanced Parameters

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

## 2.4 SVM + Explicit Kernel

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

# 3. Testing Guide

Vec-Eyes tests are designed to validate behavior across:
- machine learning pipelines
- YAML parsing and validation
- recursive dataset loading
- rule-based scoring
- domain-specific examples (security, fraud, biology)

## 3.1 Run all tests

```bash
cargo test
```

## 3.2 Run advanced model tests

```bash
cargo test --test advanced_models -- --nocapture
```

## 3.3 Run realistic YAML tests

```bash
cargo test --test yaml_realistic -- --nocapture
```

---

# 4. How the tests are structured

Typical test coverage includes:
- **YAML validation**
  - required fields like `k`, `p`, `svm_c`, `random_forest_n_trees`
- **pipeline digestion**
  - YAML → config → classifier → prediction
- **domain fixtures**
  - security
  - financial fraud
  - biology
- **recursive dataset loading**
  - `hot` and `cold` directories with multiple files
- **rule boosting**
  - regex / optional vectorscan-backed scoring

---

# 5. What contributors should know

Vec-Eyes follows a modular open-source design:
- **Factory** for selecting the classifier family
- **Builder** for classifier configuration
- dedicated modules per classifier under `src/classifiers/*`
- no intentionally monolithic algorithm files
- YAML-first workflow for reproducibility

This makes the project easier to:
- extend
- benchmark
- audit
- contribute to

---

# 6. Recommended additions for contributors

If you are opening a PR, good areas to improve are:
- new datasets
- benchmark coverage
- model calibration
- more realistic biological datasets
- fraud-specific scoring heuristics
- richer examples for scientific and industrial use cases

---

# 7. Final note

Vec-Eyes is no longer just a spam detector.

It is evolving into a **modular behavior intelligence engine for security, fraud, anomaly detection, and scientific classification** — built in Rust, designed for performance, and structured for long-term open-source growth.
