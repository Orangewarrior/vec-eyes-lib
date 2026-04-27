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

`FastTextEmbeddings` and `Word2VecEmbeddings` (external embeddings extracted from a `.bin` file) can both be saved and loaded via `save_bincode` / `load_bincode`. The unified `ExternalEmbeddings` enum wraps both and is accepted by every classifier's `train_with_external_embeddings` method.

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

## External embeddings — end-to-end workflow

This section walks through complete pipelines using two types of external word vectors:

- **fastText** (`.bin` from the `fasttext` CLI) — supports subword OOV composition
- **word2vec** (`.bin` from the `word2vec` or `word2vec-c` tool) — uses vocabulary centroid for OOV

Both are loaded into vec-eyes-lib's unified `ExternalEmbeddings` type and accepted by every classifier's `train_with_external_embeddings` method.

Steps covered:
1. Download a UCI text dataset
2. Build a corpus file and train word vectors with the **system CLI**
3. Load the resulting `.bin` into vec-eyes-lib
4. Train a classifier and persist it

### Step 0 — install fastText CLI

```bash
sudo apt install fasttext
fasttext --version   # 0.9.2 or similar
```

---

### Example 1 — UCI SMS Spam Collection

**Dataset**: [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
(4 837 ham + 747 spam SMS messages, single TSV file)

#### Download and prepare

```bash
# Download the dataset archive
wget -q "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip" \
     -O sms_spam.zip
unzip -o sms_spam.zip -d sms_spam_raw

# The archive contains SMSSpamCollection — a tab-separated file: label\tmessage
# Extract just the message text as a plain corpus for fastText
cut -f2 sms_spam_raw/SMSSpamCollection > corpus/sms_corpus.txt

# Separate spam / ham into the directory layout vec-eyes-lib expects
mkdir -p data/sms/hot data/sms/cold
awk -F'\t' '$1=="spam" {print $2 > "data/sms/hot/spam_" NR ".txt"}' \
    sms_spam_raw/SMSSpamCollection
awk -F'\t' '$1=="ham"  {print $2 > "data/sms/cold/ham_"  NR ".txt"}' \
    sms_spam_raw/SMSSpamCollection
```

#### Train fastText word vectors

```bash
mkdir -p corpus models

fasttext skipgram \
    -input  corpus/sms_corpus.txt \
    -output models/sms_ft \
    -dim    100 \
    -epoch  10  \
    -minCount 1 \
    -minn   3   \
    -maxn   6

# Produces:
#   models/sms_ft.bin   ← binary model (vectors + subword buckets)
#   models/sms_ft.vec   ← plain-text word vectors (not needed by vec-eyes-lib)
```

The flags that matter for vec-eyes-lib compatibility:

| Flag | Meaning |
|---|---|
| `-dim` | Vector dimensionality used by `train_with_external_embeddings` |
| `-minn` / `-maxn` | Character n-gram range used for OOV subword composition |
| `-minCount 1` | Include every word token; useful on small corpora |

#### Load in Rust and train a KNN classifier

```rust
use std::path::Path;
use vec_eyes_lib::{
    ClassificationLabel, DistanceMetric, ExternalEmbeddings, FastTextBin, KnnClassifier,
    dataset::load_training_samples,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load training samples from the directory layout we created above
    let mut samples = load_training_samples(
        Path::new("data/sms/hot"),
        ClassificationLabel::Spam,
        false,
    )?;
    samples.extend(load_training_samples(
        Path::new("data/sms/cold"),
        ClassificationLabel::RawData,
        false,
    )?);

    // ── One-time setup ────────────────────────────────────────────────────────
    // Parse the fastText .bin (reads word index + full matrix into RAM)
    let bin = FastTextBin::load("models/sms_ft.bin")?;

    // Extract only the vectors needed for this vocabulary.
    // Much smaller than extract_all() on large pre-trained models.
    let vocab: Vec<&str> = samples.iter().map(|s| s.text.as_str()).collect();
    let ft_embeddings = bin.extract_for_vocab(&vocab);

    // Wrap in the unified ExternalEmbeddings enum
    let embeddings = ExternalEmbeddings::FastText(ft_embeddings);

    // Cache the extracted embeddings — subsequent runs skip the .bin parse
    embeddings.save_bincode("models/sms_ft.embeddings.bin")?;
    // ─────────────────────────────────────────────────────────────────────────

    // Train KNN using the external embeddings
    let classifier = KnnClassifier::train_with_external_embeddings(
        &samples,
        embeddings,
        DistanceMetric::Cosine,
        /*k=*/ 5,
        /*threads=*/ None,
        /*normalize_features=*/ false,
    )?;

    // Persist — single file, fast reload on next startup
    classifier.save_bincode("models/sms_knn_ft.bin")?;

    // Classify new text
    let result = classifier.classify("Free entry win prize now click here")?;
    println!("label: {:?}  confidence: {:.2}", result.label, result.score);

    Ok(())
}
```

#### Fast path — reload without reparsing `.bin`

```rust
use vec_eyes_lib::{ExternalEmbeddings, KnnClassifier};

// Skip the heavy .bin parse — load pre-extracted embeddings directly
let embeddings = ExternalEmbeddings::load_bincode("models/sms_ft.embeddings.bin")?;

// If you already have a trained classifier, skip training too
let classifier = KnnClassifier::load_bincode("models/sms_knn_ft.bin")?;
```

---

### Example 2 — UCI Fraud Detection dataset

**Dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
or the lighter [PIMA Fraud proxy](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) — any dataset where each record can be serialized as a text line works.

For datasets stored as CSV with a `text` or `description` column, convert each row to a plain text file:

```bash
# Assuming fraud_raw.csv has columns: label,text
# label is "fraud" or "normal"
awk -F',' 'NR>1 && $1=="fraud"  {print $2 > "data/fraud/hot/fraud_"  NR ".txt"}' fraud_raw.csv
awk -F',' 'NR>1 && $1=="normal" {print $2 > "data/fraud/cold/normal_" NR ".txt"}' fraud_raw.csv

# Build corpus and train
cut -d',' -f2 fraud_raw.csv | tail -n +2 > corpus/fraud_corpus.txt

fasttext skipgram \
    -input  corpus/fraud_corpus.txt \
    -output models/fraud_ft \
    -dim    64 \
    -epoch  15 \
    -minCount 1
```

```rust
use std::path::Path;
use vec_eyes_lib::{
    ClassificationLabel, ExternalEmbeddings, FastTextBin, LogisticClassifier,
    LogisticRegressionConfig, dataset::load_training_samples,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut samples = load_training_samples(
        Path::new("data/fraud/hot"),
        ClassificationLabel::Anomaly,
        false,
    )?;
    samples.extend(load_training_samples(
        Path::new("data/fraud/cold"),
        ClassificationLabel::RawData,
        false,
    )?);

    let bin = FastTextBin::load("models/fraud_ft.bin")?;
    let vocab: Vec<&str> = samples.iter().map(|s| s.text.as_str()).collect();
    let embeddings = ExternalEmbeddings::FastText(bin.extract_for_vocab(&vocab));
    embeddings.save_bincode("models/fraud_ft.embeddings.bin")?;

    let config = LogisticRegressionConfig {
        learning_rate: 0.25,
        epochs: 200,
        lambda: 1e-3,
    };

    // Logistic regression with external fastText embeddings
    let classifier = LogisticClassifier::train_with_external_embeddings(
        &samples,
        embeddings,
        config,
        /*threads=*/ None,
    )?;

    // Split save: NLP pipeline reusable by a second classifier head
    classifier.save_split_bincode(
        "models/fraud_ft.nlp.bin",
        "models/fraud_logistic.ml.bin",
    )?;

    let result = classifier.classify("unauthorized transfer overseas account")?;
    println!("label: {:?}  score: {:.3}", result.label, result.score);

    Ok(())
}
```

---

### Example 3 — UCI Biology (sequence classification)

**Dataset**: [Molecular Biology — Splice-junction Gene Sequences](https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences)
Each sample is a DNA subsequence. Treating nucleotide sequences as "words" and running fastText on them captures n-gram patterns naturally.

```bash
# Convert the UCI splice dataset to one-sample-per-file layout
# The raw file has format: class, instance_name, sequence (comma-separated)
awk -F', ' '$1=="EI" {print $3 > "data/bio/hot/ei_" NR ".txt"}' splice.data
awk -F', ' '$1=="IE" || $1=="N" {print $3 > "data/bio/cold/neg_" NR ".txt"}' splice.data

# Build corpus (one sequence per line) and train character-level fastText
cut -d',' -f3 splice.data > corpus/bio_corpus.txt

fasttext skipgram \
    -input    corpus/bio_corpus.txt \
    -output   models/bio_ft \
    -dim      32 \
    -epoch    20 \
    -minCount 1  \
    -minn     2  \
    -maxn     4
# Short minn/maxn captures overlapping nucleotide k-mers (dinucleotides, trinucleotides)
```

```rust
use std::path::Path;
use vec_eyes_lib::{
    ClassificationLabel, ExternalEmbeddings, FastTextBin, IsolationForestClassifier,
    IsolationForestConfig, dataset::load_training_samples,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut samples = load_training_samples(
        Path::new("data/bio/hot"),
        ClassificationLabel::Anomaly,
        false,
    )?;
    samples.extend(load_training_samples(
        Path::new("data/bio/cold"),
        ClassificationLabel::RawData,
        false,
    )?);

    let bin = FastTextBin::load("models/bio_ft.bin")?;
    let vocab: Vec<&str> = samples.iter().map(|s| s.text.as_str()).collect();
    let embeddings = ExternalEmbeddings::FastText(bin.extract_for_vocab(&vocab));
    embeddings.save_bincode("models/bio_ft.embeddings.bin")?;

    let config = IsolationForestConfig {
        n_trees: 100,
        contamination: 0.1,
        subsample_size: 64,
    };

    let classifier = IsolationForestClassifier::train_with_external_embeddings(
        &samples,
        embeddings,
        config,
        ClassificationLabel::Anomaly, // hot: exon-intron junctions (signal class)
        ClassificationLabel::RawData, // cold: non-junctions (background)
        /*threads=*/ None,
    )?;

    classifier.save_bincode("models/bio_iforest.bin")?;

    let result = classifier.classify("AAGTTAAAGCAGGTGGGTATAAATGAATTTG")?;
    println!("label: {:?}  score: {:.3}", result.label, result.score);

    Ok(())
}
```

---

### Using a large pre-trained model (cc.en.300)

fastText publishes pre-trained Common Crawl models. The 300-dimensional English model is ~4 GB on disk but covers virtually any English vocabulary including subword OOV.

```bash
# Download (large — ~4.2 GB)
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz
```

```rust
use vec_eyes_lib::{ExternalEmbeddings, FastTextBin, KnnClassifier, DistanceMetric};

// ── One-time extraction (run this once, then use the cached embeddings) ────────

let bin = FastTextBin::load("cc.en.300.bin")?;

// extract_for_vocab reads only the vectors your data actually needs.
// For a 5 000-sample corpus this produces a few MB instead of 4 GB.
let vocab: Vec<&str> = samples.iter().map(|s| s.text.as_str()).collect();
let embeddings = ExternalEmbeddings::FastText(bin.extract_for_vocab(&vocab));
drop(bin); // release the 4 GB matrix

embeddings.save_bincode("models/cc300.embeddings.bin")?;

// ── Every subsequent run ───────────────────────────────────────────────────────

let embeddings = ExternalEmbeddings::load_bincode("models/cc300.embeddings.bin")?;

let classifier = KnnClassifier::train_with_external_embeddings(
    &samples,
    embeddings,
    DistanceMetric::Cosine,
    /*k=*/ 5,
    None,
    false,
)?;

classifier.save_bincode("models/cc300_knn.bin")?;
```

OOV words (tokens not in the Common Crawl vocabulary) are automatically handled via subword composition: vec-eyes-lib computes character n-gram hashes using the same FNV-1a algorithm as fastText C++, then averages the corresponding bucket vectors from the `.bin` file.

---

## External word2vec embeddings — end-to-end workflow

The Google word2vec binary format (`.bin`) produced by both the original `word2vec` C tool and its Rust/Python reimplementations is supported via `Word2VecBin`. Unlike fastText, word2vec has no subword model; out-of-vocabulary tokens fall back to the **vocabulary centroid** (mean of all word vectors).

### Step 0 — install word2vec CLI

```bash
sudo apt install word2vec   # Ubuntu/Debian
# or build from source: https://github.com/dav/word2vec
word2vec 2>&1 | head -1     # Starting training using file ...
```

---

### Example 4 — UCI News Category (word2vec + KNN Euclidean)

**Dataset**: [News Category Dataset](https://archive.ics.uci.edu/dataset/557/news+aggregator+dataset)  
Articles from Reuters with category labels: `b` (business), `t` (technology), `e` (entertainment), `m` (health).

```bash
# Download
wget -q "https://archive.ics.uci.edu/static/public/557/news+aggregator+dataset.zip" \
     -O news.zip
unzip -o news.zip -d news_raw

# newsCorpora.csv columns: ID, TITLE, URL, PUBLISHER, CATEGORY, ...
# Extract "business" vs all-others
mkdir -p data/news/hot data/news/cold corpus/news
awk -F'\t' 'NR>1 && $5=="b" {print $2 > "data/news/hot/b_" NR ".txt"}' \
    news_raw/newsCorpora.csv
awk -F'\t' 'NR>1 && $5!="b" {print $2 > "data/news/cold/o_" NR ".txt"}' \
    news_raw/newsCorpora.csv

# Build corpus and train word2vec
cut -f2 news_raw/newsCorpora.csv | tail -n +2 > corpus/news/titles.txt

word2vec \
    -train  corpus/news/titles.txt \
    -output models/news_w2v.bin     \
    -binary 1                       \
    -size   100                     \
    -window 5                       \
    -min-count 2                    \
    -iter   10
```

```rust
use std::path::Path;
use vec_eyes_lib::{
    ClassificationLabel, DistanceMetric, ExternalEmbeddings,
    Word2VecBin, KnnClassifier, dataset::load_training_samples,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut samples = load_training_samples(
        Path::new("data/news/hot"),
        ClassificationLabel::WebAttack, // "hot" label — business articles
        false,
    )?;
    samples.extend(load_training_samples(
        Path::new("data/news/cold"),
        ClassificationLabel::RawData,
        false,
    )?);

    // ── One-time: parse word2vec .bin and extract vocabulary vectors ──────────
    let bin = Word2VecBin::load("models/news_w2v.bin")?;
    let vocab: Vec<&str> = samples.iter().map(|s| s.text.as_str()).collect();
    let w2v_embeddings = bin.extract_for_vocab(&vocab);

    // Wrap in unified ExternalEmbeddings and cache
    let embeddings = ExternalEmbeddings::Word2Vec(w2v_embeddings);
    embeddings.save_bincode("models/news_w2v.embeddings.bin")?;
    // ─────────────────────────────────────────────────────────────────────────

    // KNN with Euclidean distance using word2vec features
    let classifier = KnnClassifier::train_with_external_embeddings(
        &samples,
        embeddings,
        DistanceMetric::Euclidean,
        /*k=*/ 7,
        /*threads=*/ None,
        /*normalize_features=*/ true,
    )?;

    classifier.save_bincode("models/news_knn_w2v.bin")?;

    let result = classifier.classify("Fed raises interest rates amid inflation concerns")?;
    println!("label: {:?}  confidence: {:.2}", result.label, result.score);

    Ok(())
}
```

#### Reload path (skip `.bin` reparsing)

```rust
use vec_eyes_lib::{ExternalEmbeddings, KnnClassifier};

let embeddings = ExternalEmbeddings::load_bincode("models/news_w2v.embeddings.bin")?;
let classifier = KnnClassifier::load_bincode("models/news_knn_w2v.bin")?;
```

---

### Example 5 — UCI SMS Spam (word2vec + Random Forest)

Same SMS Spam dataset as Example 1, but using word2vec vectors and a Random Forest classifier instead of KNN.

```bash
# Reuse corpus built in Example 1:  corpus/sms_corpus.txt
word2vec \
    -train  corpus/sms_corpus.txt \
    -output models/sms_w2v.bin    \
    -binary 1                     \
    -size   64                    \
    -window 3                     \
    -min-count 1                  \
    -iter   15
```

```rust
use std::path::Path;
use vec_eyes_lib::{
    ClassificationLabel, ExternalEmbeddings, RandomForestClassifier, RandomForestConfig,
    Word2VecBin, dataset::load_training_samples,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut samples = load_training_samples(
        Path::new("data/sms/hot"),
        ClassificationLabel::Spam,
        false,
    )?;
    samples.extend(load_training_samples(
        Path::new("data/sms/cold"),
        ClassificationLabel::RawData,
        false,
    )?);

    let bin = Word2VecBin::load("models/sms_w2v.bin")?;
    let vocab: Vec<&str> = samples.iter().map(|s| s.text.as_str()).collect();
    let embeddings = ExternalEmbeddings::Word2Vec(bin.extract_for_vocab(&vocab));
    embeddings.save_bincode("models/sms_w2v.embeddings.bin")?;

    let config = RandomForestConfig {
        n_trees: 150,
        max_depth: Some(12),
        ..Default::default()
    };

    // Random Forest with word2vec features
    let classifier = RandomForestClassifier::train_with_external_embeddings(
        &samples,
        embeddings,
        config,
        /*threads=*/ None,
    )?;

    classifier.save_bincode("models/sms_rf_w2v.bin")?;

    for text in &[
        "Congratulations! You have won a FREE iPhone",
        "Can we schedule a meeting for Tuesday?",
    ] {
        let r = classifier.classify(text)?;
        println!("{:<50}  {:?}  ({:.2})", text, r.label, r.score);
    }

    Ok(())
}
```

---

### Example 6 — UCI Fraud (word2vec + SVM)

```bash
# Reuse corpus from Example 2: corpus/fraud_corpus.txt
word2vec \
    -train  corpus/fraud_corpus.txt \
    -output models/fraud_w2v.bin    \
    -binary 1                       \
    -size   64                      \
    -window 5                       \
    -min-count 1                    \
    -iter   20
```

```rust
use std::path::Path;
use vec_eyes_lib::{
    ClassificationLabel, ExternalEmbeddings, SvmClassifier, SvmConfig, SvmKernel,
    Word2VecBin, dataset::load_training_samples,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut samples = load_training_samples(
        Path::new("data/fraud/hot"),
        ClassificationLabel::Anomaly,
        false,
    )?;
    samples.extend(load_training_samples(
        Path::new("data/fraud/cold"),
        ClassificationLabel::RawData,
        false,
    )?);

    let bin = Word2VecBin::load("models/fraud_w2v.bin")?;
    let vocab: Vec<&str> = samples.iter().map(|s| s.text.as_str()).collect();
    let embeddings = ExternalEmbeddings::Word2Vec(bin.extract_for_vocab(&vocab));
    embeddings.save_bincode("models/fraud_w2v.embeddings.bin")?;

    let config = SvmConfig {
        kernel: SvmKernel::Linear,
        c: 1.0,
        ..Default::default()
    };

    // SVM with word2vec features
    let classifier = SvmClassifier::train_with_external_embeddings(
        &samples,
        embeddings,
        config,
        /*threads=*/ None,
    )?;

    classifier.save_bincode("models/fraud_svm_w2v.bin")?;

    let result = classifier.classify("suspicious wire transfer to unknown offshore account")?;
    println!("label: {:?}  score: {:.3}", result.label, result.score);

    Ok(())
}
```

---

### When to use `extract_all` vs `extract_for_vocab`

| | `extract_all()` | `extract_for_vocab(&vocab)` |
|---|---|---|
| Output size | Full model (same as `.bin`) | Only vectors your data needs |
| Use case | Shipping the embeddings as a standalone artifact | Training directly from your dataset |
| OOV coverage | Complete | Subword buckets included for OOV tokens in `vocab` |
| Recommended for | Pre-trained models you share across many datasets | Single-dataset pipelines |

---

## Choosing the right format

| Situation | Recommendation |
|---|---|
| Debugging a model, inspecting weights | JSON |
| Production service, reload on start | bincode single file |
| Large embedding model (Word2Vec / FastText) shared across pipelines | Split bincode — write NLP once, update ML head separately |
| External fastText or word2vec embeddings | `ExternalEmbeddings::save_bincode` → reload and pass to `train_with_external_embeddings` |

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
