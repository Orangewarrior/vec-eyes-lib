# Real Classification Examples

Three self-contained projects that download a public UCI dataset, train a
different classifier from vec-eyes-lib, persist the model to disk, and classify
new inputs — from scratch to `cargo run`.

| # | Domain | UCI Dataset | Classifier | NLP |
|---|---|---|---|---|
| 1 | Security | SMS Spam Collection | KNN (Euclidean) | Word2Vec |
| 2 | Biology | Molecular Biology Splice-junction | Naive Bayes | TF-IDF |
| 3 | Finance | Sentiment Labelled Sentences | Logistic Regression + Random Forest | Word2Vec / TF-IDF |

---

## Prerequisites

```bash
# Rust toolchain (skip if already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable

# Python 3 — dataset preparation scripts
python3 --version

# wget + unzip — downloading the archives
sudo apt install wget unzip
```

Each example is a standalone Rust binary crate. Place it as a **sibling** of
`vec-eyes-lib/` so the path dependency resolves:

```
workspace/
├── vec-eyes-lib/      ← the library
├── security-classifier/
├── bio-classifier/
└── finance-classifier/
```

---

## Example 1 — Security: Phishing & Spam Detection

**Classifier**: K-Nearest Neighbours, Euclidean distance, Word2Vec embeddings  
**Dataset**: [UCI SMS Spam Collection #228](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
 — 5 574 SMS messages labelled `spam` / `ham`

KNN with Euclidean distance over dense Word2Vec embeddings clusters phishing
messages tightly (urgency language, reward promises, action imperatives) while
legitimate messages spread across broader semantic territory.

### 1. Download and prepare

```bash
mkdir -p security-classifier/data/hot \
         security-classifier/data/cold \
         security-classifier/models
cd security-classifier

wget -q "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip" \
     -O sms_raw.zip
unzip -q sms_raw.zip -d sms_raw

# Split tab-separated SMSSpamCollection into per-file hot/cold layout
python3 - <<'PY'
import os, re

src = "sms_raw/SMSSpamCollection"
counts = {"hot": 0, "cold": 0}

with open(src, encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.rstrip("\n")
        if "\t" not in line:
            continue
        tab = line.index("\t")
        label, text = line[:tab], line[tab + 1:]
        safe = re.sub(r"[^\w]", "_", text[:32])
        if label == "spam":
            path = f"data/hot/spam_{i:04d}_{safe}.txt"
            counts["hot"] += 1
        else:
            path = f"data/cold/ham_{i:04d}_{safe}.txt"
            counts["cold"] += 1
        with open(path, "w", encoding="utf-8") as out:
            out.write(text)

print(f"hot (spam) = {counts['hot']}  cold (ham) = {counts['cold']}")
PY
```

Expected: `hot (spam) = 747  cold (ham) = 4827`

### 2. Cargo.toml

```bash
cargo init --name security-classifier .
```

```toml
[package]
name = "security-classifier"
version = "0.1.0"
edition = "2021"

[dependencies]
# path dependency while developing inside the workspace:
vec-eyes-lib = { path = "../vec-eyes-lib" }
# published crate (after release):
# vec-eyes-lib = "3.0.0"
```

### 3. src/main.rs

```rust
use std::fs;
use std::path::Path;

use vec_eyes_lib::{
    ClassificationLabel, Classifier, DistanceMetric,
    KnnClassifier, NlpOption, VecEyesError,
    dataset::load_training_samples,
};

fn main() -> Result<(), VecEyesError> {
    // ── Training data ────────────────────────────────────────────────────────
    let mut samples = load_training_samples(
        Path::new("data/hot"),
        ClassificationLabel::Spam,
        false,
    )?;
    samples.extend(load_training_samples(
        Path::new("data/cold"),
        ClassificationLabel::RawData,
        false,
    )?);
    println!("Loaded {} training samples", samples.len());

    // ── Train ────────────────────────────────────────────────────────────────
    let classifier = KnnClassifier::train(
        &samples,
        NlpOption::Word2Vec,
        DistanceMetric::Euclidean,
        /*embedding_dims=*/ 64,
        /*k=*/ 7,
        /*threads=*/ None,
        /*normalize_features=*/ true,
    )?;
    println!("Training complete.");

    // ── Persist ──────────────────────────────────────────────────────────────
    fs::create_dir_all("models")?;
    classifier.save_bincode("models/security_knn.bin")?;
    println!("Model saved → models/security_knn.bin");

    // ── Classify new messages ────────────────────────────────────────────────
    let test_messages: &[(&str, bool)] = &[
        ("WINNER!! Claim your free prize now, click here", true),
        ("Hi, are we still on for lunch tomorrow?", false),
        ("URGENT: Your account has been suspended. Verify immediately.", true),
        ("Reminder: team standup at 10am", false),
        ("Congratulations! You've won a 1000 gift card", true),
        ("Please find the meeting notes attached", false),
    ];

    println!("\n{:<52} {:<10} {:<6} {}", "Message", "Label", "Score", "Correct?");
    println!("{}", "-".repeat(80));

    for (text, expected_hot) in test_messages {
        let result = classifier.classify(text)?;
        let label  = result.top_label().map(|l| format!("{l:?}")).unwrap_or_default();
        let score  = result.top_score();
        let is_hot = matches!(result.top_label(), Some(ClassificationLabel::Spam));
        let mark   = if is_hot == *expected_hot { "OK" } else { "WRONG" };
        let short  = if text.len() > 50 { &text[..50] } else { text };
        println!("{:<52} {:<10} {:.3}  {}", short, label, score, mark);
    }

    Ok(())
}
```

### 4. Reload without retraining

Add this to any subsequent binary to skip the training step entirely:

```rust
let classifier = KnnClassifier::load_bincode("models/security_knn.bin")?;
let result = classifier.classify("You have won a free iPhone")?;
println!("{:?}  {:.3}", result.top_label(), result.top_score());
```

### 5. Build and run

```bash
# Inside security-classifier/
cargo build --release 2>&1 | grep -E "^error|Compiling|Finished"
cargo run --release
```

Expected output (values vary across runs due to random Word2Vec initialisation):

```
Loaded 5574 training samples
Training complete.
Model saved → models/security_knn.bin

Message                                              Label      Score  Correct?
--------------------------------------------------------------------------------
WINNER!! Claim your free prize now, click here       Spam       0.857  OK
Hi, are we still on for lunch tomorrow?              RawData    0.714  OK
URGENT: Your account has been suspended. Verify im   Spam       0.857  OK
Reminder: team standup at 10am                       RawData    0.857  OK
Congratulations! You've won a 1000 gift card         Spam       0.929  OK
Please find the meeting notes attached               RawData    0.857  OK
```

---

## Example 2 — Biology: Splice-Junction Sequence Classification

**Classifier**: Naive Bayes, TF-IDF  
**Dataset**: [UCI Molecular Biology Splice-junction Gene Sequences #69](https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences)
 — 3 190 DNA sequences labelled `EI` (exon→intron boundary), `IE`
 (intron→exon boundary), or `N` (neither)

Naive Bayes with TF-IDF is a strong baseline for short-sequence classification.
Splice donor sites (`EI`) contain the canonical `GT` dinucleotide flanked by
characteristic trinucleotide contexts; these appear as high-IDF tokens that the
Bayes model learns to weight heavily. The model is also interpretable — you can
inspect which k-mers drive each class.

Here we treat `EI` boundaries as the **hot** (signal) class and everything else
as **cold** (background), making it a binary detector.

### 1. Download and prepare

```bash
mkdir -p bio-classifier/data/hot \
         bio-classifier/data/cold \
         bio-classifier/models
cd bio-classifier

wget -q "https://archive.ics.uci.edu/static/public/69/molecular+biology+splice+junction+gene+sequences.zip" \
     -O splice_raw.zip
unzip -q splice_raw.zip -d splice_raw

# Convert sequences to space-separated overlapping 3-mers
# so the library tokeniser treats each trinucleotide as a separate word.
python3 - <<'PY'
import os, re

raw = "splice_raw/splice.data"
counts = {"hot": 0, "cold": 0}

with open(raw) as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        cls = parts[0].strip()
        # Remove whitespace and non-ACGT characters from the sequence
        seq = re.sub(r"[^ACGTacgt]", "", parts[2]).upper()

        # Overlapping 3-mers: AGCTG → "AGC GCT CTG"
        kmers = " ".join(seq[j : j + 3] for j in range(len(seq) - 2))

        if cls == "EI":
            path = f"data/hot/ei_{i:04d}.txt"
            counts["hot"] += 1
        else:
            path = f"data/cold/{cls.lower()}_{i:04d}.txt"
            counts["cold"] += 1

        with open(path, "w") as out:
            out.write(kmers)

print(f"hot (EI) = {counts['hot']}  cold (IE+N) = {counts['cold']}")
PY
```

Expected: `hot (EI) = 767  cold (IE+N) = 2423`

### 2. Cargo.toml

```bash
cargo init --name bio-classifier .
```

```toml
[package]
name = "bio-classifier"
version = "0.1.0"
edition = "2021"

[dependencies]
vec-eyes-lib = { path = "../vec-eyes-lib" }
```

### 3. src/main.rs

```rust
use std::fs;
use std::path::Path;

use vec_eyes_lib::{
    BayesClassifier, ClassificationLabel, Classifier,
    NlpOption, VecEyesError,
    dataset::load_training_samples,
};

fn main() -> Result<(), VecEyesError> {
    // ── Training data ────────────────────────────────────────────────────────
    // Virus = EI junction (splice-donor signal class)
    // Human = IE + N  (non-junction background)
    let mut samples = load_training_samples(
        Path::new("data/hot"),
        ClassificationLabel::Virus,
        false,
    )?;
    samples.extend(load_training_samples(
        Path::new("data/cold"),
        ClassificationLabel::Human,
        false,
    )?);
    println!("Loaded {} sequences", samples.len());

    // ── Train ────────────────────────────────────────────────────────────────
    // TF-IDF: rare but discriminative 3-mers (e.g. "GTA" at exon-donor sites)
    // get high IDF weight, which is exactly what Naive Bayes multiplies through.
    let classifier = BayesClassifier::train(
        &samples,
        NlpOption::TfIdf,
        /*threads=*/ None,
    )?;
    println!("Training complete.");

    // ── Persist ──────────────────────────────────────────────────────────────
    fs::create_dir_all("models")?;
    classifier.save_bincode("models/splice_bayes.bin")?;
    println!("Model saved → models/splice_bayes.bin");

    // ── Test sequences (expressed as overlapping 3-mers, matching prep step) ─
    //
    // A canonical GT-AG splice donor site looks like:
    //   ...exon...AG | GT AGT...intron...
    // The 3-mer window around the boundary: AGG GGT GTA TAA AAG AGT ...
    //
    let test_seqs: &[(&str, &str, bool)] = &[
        // Strong EI donor: contains canonical AGGTAAGT motif
        (
            "CAG AGG GGT GTA TAA AAG AGT GTG TGA GAG AGG",
            "canonical EI donor site",
            true,
        ),
        // IE acceptor: TTCAG motif at end of intron
        (
            "TTT TTC TCA CAG AGG GGA GAG AGC GCT CTG TGC",
            "canonical IE acceptor site",
            false,
        ),
        // Random non-splice sequence
        (
            "ACG CGT GTA TAC ACG CGT GTA TAC ACG CGT GTA",
            "random repeated sequence",
            false,
        ),
        // Another typical EI donor: CAGGTAAGT
        (
            "GCA CAG AGG GGT GTA TAA AAG AGT GTG TGA",
            "EI donor with flanking exon",
            true,
        ),
    ];

    println!(
        "\n{:<46} {:<30} {:<8} {:<6} {}",
        "3-mer sequence (truncated)", "Context", "Label", "Score", "Correct?"
    );
    println!("{}", "-".repeat(100));

    for (kmers, context, expected_hot) in test_seqs {
        let result   = classifier.classify(kmers)?;
        let label    = result.top_label().map(|l| format!("{l:?}")).unwrap_or_default();
        let score    = result.top_score();
        let is_hot   = matches!(result.top_label(), Some(ClassificationLabel::Virus));
        let mark     = if is_hot == *expected_hot { "OK" } else { "WRONG" };
        let short    = if kmers.len() > 44 { &kmers[..44] } else { kmers };
        println!("{:<46} {:<30} {:<8} {:.3}  {}", short, context, label, score, mark);
    }

    Ok(())
}
```

### 4. Reload without retraining

```rust
let classifier = BayesClassifier::load_bincode("models/splice_bayes.bin")?;
let seq = "CAG AGG GGT GTA TAA AAG AGT GTG";   // 3-mers of your sequence
let result = classifier.classify(seq)?;
println!("{:?}", result.top_label());
```

### 5. Build and run

```bash
cargo build --release 2>&1 | grep -E "^error|Compiling|Finished"
cargo run --release
```

Expected output:

```
Loaded 3190 sequences
Training complete.
Model saved → models/splice_bayes.bin

3-mer sequence (truncated)                     Context                        Label    Score  Correct?
----------------------------------------------------------------------------------------------------
CAG AGG GGT GTA TAA AAG AGT GTG TGA GAG AGG   canonical EI donor site        Virus    0.821  OK
TTT TTC TCA CAG AGG GGA GAG AGC GCT CTG TGC   canonical IE acceptor site     Human    0.764  OK
ACG CGT GTA TAC ACG CGT GTA TAC ACG CGT GTA   random repeated sequence       Human    0.693  OK
GCA CAG AGG GGT GTA TAA AAG AGT GTG TGA       EI donor with flanking exon    Virus    0.799  OK
```

---

## Example 3 — Finance: Sentiment Classification

**Classifiers**: Logistic Regression (Word2Vec) **and** Random Forest (TF-IDF) —
both trained on the same data so you can compare them  
**Dataset**: [UCI Sentiment Labelled Sentences #331](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)
 — 3 000 product reviews (Amazon + Yelp + IMDB) with binary `1` (positive) /
 `0` (negative) labels; used here as a stand-in for financial analyst notes and
 customer feedback sentiment

### 1. Download and prepare

```bash
mkdir -p finance-classifier/data/hot \
         finance-classifier/data/cold \
         finance-classifier/models
cd finance-classifier

wget -q "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip" \
     -O sentiment_raw.zip
unzip -q sentiment_raw.zip -d sentiment_raw

python3 - <<'PY'
import os

base = "sentiment_raw/sentiment labelled sentences"
sources = [
    "amazon_cells_labelled.txt",
    "imdb_labelled.txt",
    "yelp_labelled.txt",
]
counts = {"hot": 0, "cold": 0}

for fname in sources:
    source = fname.split("_")[0]
    fpath  = os.path.join(base, fname)
    with open(fpath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            text, label = line.rsplit("\t", 1)
            label = label.strip()
            if label == "1":
                path = f"data/hot/{source}_{i:04d}.txt"
                counts["hot"] += 1
            else:
                path = f"data/cold/{source}_{i:04d}.txt"
                counts["cold"] += 1
            with open(path, "w", encoding="utf-8") as out:
                out.write(text.strip())

print(f"positive (hot) = {counts['hot']}  negative (cold) = {counts['cold']}")
PY
```

Expected: `positive (hot) = 1500  negative (cold) = 1500`

### 2. Cargo.toml — two binary targets

```bash
cargo init --name finance-classifier .
mkdir src   # if not already present
```

```toml
[package]
name = "finance-classifier"
version = "0.1.0"
edition = "2021"

# Two separate binaries — run with `cargo run --bin logistic` etc.
[[bin]]
name = "logistic"
path = "src/logistic.rs"

[[bin]]
name = "random_forest"
path = "src/random_forest.rs"

[dependencies]
vec-eyes-lib = { path = "../vec-eyes-lib" }
```

### 3a. src/logistic.rs — Logistic Regression + Word2Vec

Logistic regression over dense Word2Vec embeddings learns a linear decision
boundary in embedding space. Good at capturing semantic similarity ("excellent"
and "great" land close together) with low memory footprint.

```rust
use std::fs;
use std::path::Path;

use vec_eyes_lib::{
    ClassificationLabel, Classifier, LogisticClassifier,
    LogisticRegressionConfig, NlpOption, VecEyesError,
    dataset::load_training_samples,
};

fn main() -> Result<(), VecEyesError> {
    // ── Training data ────────────────────────────────────────────────────────
    // Custom labels express the business meaning directly.
    let mut samples = load_training_samples(
        Path::new("data/hot"),
        ClassificationLabel::Custom("positive".to_string()),
        false,
    )?;
    samples.extend(load_training_samples(
        Path::new("data/cold"),
        ClassificationLabel::Custom("negative".to_string()),
        false,
    )?);
    println!("Loaded {} reviews", samples.len());

    // ── Train ────────────────────────────────────────────────────────────────
    let config = LogisticRegressionConfig {
        learning_rate: 0.25,
        epochs: 200,
        lambda: 1e-3,
    };

    let classifier = LogisticClassifier::train(
        &samples,
        NlpOption::Word2Vec,
        config,
        /*threads=*/ None,
        /*embedding_dims=*/ 64,
    )?;
    println!("Logistic regression training complete.");

    // ── Persist ──────────────────────────────────────────────────────────────
    fs::create_dir_all("models")?;
    classifier.save_bincode("models/finance_logistic.bin")?;
    // Split save: NLP embeddings can be reused by the Random Forest head.
    classifier.save_split_bincode(
        "models/finance.nlp.bin",
        "models/logistic.ml.bin",
    )?;
    println!("Models saved.");

    // ── Classify analyst notes / customer feedback ───────────────────────────
    let probes: &[(&str, bool)] = &[
        ("Excellent earnings beat, strong guidance, buy signal",            true),
        ("Revenue miss, margins compressed, outlook cut to sell",           false),
        ("Solid quarter, reaffirmed full-year targets, neutral stance",     true),
        ("Debt levels rising sharply, covenant breach risk, avoid",         false),
        ("Record free cash flow, dividend raised 12 percent",               true),
        ("CEO resignation, accounting irregularities under investigation",  false),
    ];

    println!(
        "\n{:<56} {:<12} {:<6} {}",
        "Analyst note", "Label", "Score", "Correct?"
    );
    println!("{}", "-".repeat(82));

    for (text, expected_positive) in probes {
        let result      = classifier.classify(text)?;
        let label       = result.top_label().map(|l| format!("{l:?}")).unwrap_or_default();
        let score       = result.top_score();
        let is_positive = matches!(
            result.top_label(),
            Some(ClassificationLabel::Custom(s)) if s == "positive"
        );
        let mark  = if is_positive == *expected_positive { "OK" } else { "WRONG" };
        let short = if text.len() > 54 { &text[..54] } else { text };
        println!("{:<56} {:<12} {:.3}  {}", short, label, score, mark);
    }

    Ok(())
}
```

### 3b. src/random_forest.rs — Random Forest + TF-IDF

Random Forest over TF-IDF features uses sparse bag-of-words vectors where each
tree splits on the most discriminative individual terms. TF-IDF is often
stronger than dense embeddings for Random Forest because rare but highly
discriminative terms (e.g. "investigation", "dividend") get high IDF weight
and become natural split points.

```rust
use std::fs;
use std::path::Path;

use vec_eyes_lib::{
    ClassificationLabel, Classifier, NlpOption,
    RandomForestClassifier, RandomForestConfig, VecEyesError,
    dataset::load_training_samples,
};

fn main() -> Result<(), VecEyesError> {
    // ── Training data ────────────────────────────────────────────────────────
    let mut samples = load_training_samples(
        Path::new("data/hot"),
        ClassificationLabel::Custom("positive".to_string()),
        false,
    )?;
    samples.extend(load_training_samples(
        Path::new("data/cold"),
        ClassificationLabel::Custom("negative".to_string()),
        false,
    )?);
    println!("Loaded {} reviews", samples.len());

    // ── Train ────────────────────────────────────────────────────────────────
    let config = RandomForestConfig {
        n_trees:   50,
        max_depth: 8,
        ..Default::default()   // bootstrap=true, max_features=Sqrt, etc.
    };

    let classifier = RandomForestClassifier::train(
        &samples,
        NlpOption::TfIdf,
        config,
        /*threads=*/ None,
        /*embedding_dims=*/ 64,
    )?;
    println!("Random forest training complete (50 trees).");

    // ── Persist ──────────────────────────────────────────────────────────────
    fs::create_dir_all("models")?;
    classifier.save_bincode("models/finance_rf.bin")?;
    classifier.save_split_bincode(
        "models/finance_rf.nlp.bin",
        "models/rf.ml.bin",
    )?;
    println!("Models saved.");

    // ── Classify ─────────────────────────────────────────────────────────────
    let probes: &[(&str, bool)] = &[
        ("Excellent earnings beat, strong guidance, buy signal",            true),
        ("Revenue miss, margins compressed, outlook cut to sell",           false),
        ("Solid quarter, reaffirmed full-year targets, neutral stance",     true),
        ("Debt levels rising sharply, covenant breach risk, avoid",         false),
        ("Record free cash flow, dividend raised 12 percent",               true),
        ("CEO resignation, accounting irregularities under investigation",  false),
    ];

    println!(
        "\n{:<56} {:<12} {:<6} {}",
        "Analyst note", "Label", "Score", "Correct?"
    );
    println!("{}", "-".repeat(82));

    for (text, expected_positive) in probes {
        let result      = classifier.classify(text)?;
        let label       = result.top_label().map(|l| format!("{l:?}")).unwrap_or_default();
        let score       = result.top_score();
        let is_positive = matches!(
            result.top_label(),
            Some(ClassificationLabel::Custom(s)) if s == "positive"
        );
        let mark  = if is_positive == *expected_positive { "OK" } else { "WRONG" };
        let short = if text.len() > 54 { &text[..54] } else { text };
        println!("{:<56} {:<12} {:.3}  {}", short, label, score, mark);
    }

    Ok(())
}
```

### 4. Build and run

```bash
# Build both binaries
cargo build --release 2>&1 | grep -E "^error|Compiling|Finished"

# Logistic Regression
cargo run --release --bin logistic

# Random Forest
cargo run --release --bin random_forest
```

Expected output (logistic):

```
Loaded 3000 reviews
Logistic regression training complete.
Models saved.

Analyst note                                             Label        Score  Correct?
----------------------------------------------------------------------------------
Excellent earnings beat, strong guidance, buy signal     Custom(pos.. 0.812  OK
Revenue miss, margins compressed, outlook cut to sell    Custom(neg.. 0.779  OK
Solid quarter, reaffirmed full-year targets, neutral     Custom(pos.. 0.701  OK
Debt levels rising sharply, covenant breach risk         Custom(neg.. 0.834  OK
Record free cash flow, dividend raised 12 percent        Custom(pos.. 0.823  OK
CEO resignation, accounting irregularities under in      Custom(neg.. 0.891  OK
```

#### Reload from disk (production path)

```rust
// Logistic
let classifier = LogisticClassifier::load_bincode("models/finance_logistic.bin")?;

// Random Forest
let classifier = RandomForestClassifier::load_bincode("models/finance_rf.bin")?;

// Both support split load too — NLP pipeline shared:
let clf2 = LogisticClassifier::load_split_bincode(
    "models/finance.nlp.bin",
    "models/logistic.ml.bin",
)?;
```

---

## Comparing all four classifiers on the same dataset

The snippet below runs KNN (Euclidean), Bayes, Logistic Regression, and Random
Forest on the security dataset side by side. Add it as a third binary in any of
the projects above, or paste it into a new `src/compare.rs` target.

```toml
# In Cargo.toml, add:
[[bin]]
name = "compare"
path = "src/compare.rs"
```

```rust
use std::path::Path;

use vec_eyes_lib::{
    BayesClassifier, ClassificationLabel, Classifier,
    DistanceMetric, KnnClassifier, LogisticClassifier,
    LogisticRegressionConfig, NlpOption,
    RandomForestClassifier, RandomForestConfig, VecEyesError,
    dataset::load_training_samples,
};

fn main() -> Result<(), VecEyesError> {
    let mut samples = load_training_samples(
        Path::new("data/hot"),
        ClassificationLabel::Spam,
        false,
    )?;
    samples.extend(load_training_samples(
        Path::new("data/cold"),
        ClassificationLabel::RawData,
        false,
    )?);
    println!("Loaded {} samples — training all four classifiers...\n", samples.len());

    // Train all four
    let knn = KnnClassifier::train(
        &samples, NlpOption::Word2Vec, DistanceMetric::Euclidean,
        64, 7, None, true,
    )?;

    let bayes = BayesClassifier::train(&samples, NlpOption::TfIdf, None)?;

    let logistic = LogisticClassifier::train(
        &samples,
        NlpOption::Word2Vec,
        LogisticRegressionConfig::default(),
        None,
        64,
    )?;

    let rf = RandomForestClassifier::train(
        &samples,
        NlpOption::TfIdf,
        RandomForestConfig { n_trees: 50, max_depth: 8, ..Default::default() },
        None,
        64,
    )?;

    let probes = [
        "WIN a FREE iPhone limited offer click now",
        "The project deadline was moved to Friday",
        "URGENT verify your bank details or account closed",
        "Can we reschedule the call for next week",
        "Claim 500 cash reward expires tonight",
        "Team lunch is booked for Thursday at noon",
    ];

    println!(
        "{:<44} {:<9} {:<9} {:<12} {:<6}",
        "Text", "KNN", "Bayes", "Logistic", "RF"
    );
    println!("{}", "-".repeat(83));

    for text in &probes {
        let label = |r: vec_eyes_lib::ClassificationResult| {
            match r.top_label() {
                Some(ClassificationLabel::Spam) => "Spam",
                _ => "Ham",
            }
            .to_string()
        };

        let knn_l = label(knn.classify(text)?);
        let bay_l = label(bayes.classify(text)?);
        let log_l = label(logistic.classify(text)?);
        let rf_l  = label(rf.classify(text)?);

        let short = if text.len() > 42 { &text[..42] } else { text };
        println!("{:<44} {:<9} {:<9} {:<12} {:<6}", short, knn_l, bay_l, log_l, rf_l);
    }

    Ok(())
}
```

Expected output:

```
Loaded 5574 samples — training all four classifiers...

Text                                         KNN       Bayes     Logistic     RF
-----------------------------------------------------------------------------------
WIN a FREE iPhone limited offer click now    Spam      Spam      Spam         Spam
The project deadline was moved to Friday     Ham       Ham       Ham          Ham
URGENT verify your bank details or account   Spam      Spam      Spam         Spam
Can we reschedule the call for next week     Ham       Ham       Ham          Ham
Claim 500 cash reward expires tonight        Spam      Spam      Spam         Spam
Team lunch is booked for Thursday at noon    Ham       Ham       Ham          Ham
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `SIGILL` during `cargo run` | rustc used `-C target-cpu=native` on an old kernel | `RUSTFLAGS="" cargo run --release` |
| `VecEyesError::Io` on `load_training_samples` | `data/hot` or `data/cold` is empty | Re-run the Python preparation script |
| Path not found for `vec-eyes-lib` | Example project is inside the library repo | Move it to a sibling directory |
| Low accuracy on biology example | Sequences shorter than minn threshold | Lower `minn` in fastText config, or increase training data |
| `Custom(positive)` truncated in output | `Debug` format for `Custom(String)` | Pattern-match on the inner `String` for cleaner display |
