# Vec-Eyes Library

Vec-Eyes is a Rust library for behavior-oriented classification of raw text: mail buffers, HTTP requests, syscall traces, malware notes, anomaly descriptions, and rule-assisted detections.

## Why contributors care

Vec-Eyes is not limited to spam. It is a reusable classification core for:

- `SPAM`
- `MALWARE`
- `PHISHING`
- `ANOMALY`
- `FUZZING`
- `WEB_ATTACK`
- `FLOOD`
- `PORN`
- `RAW_DATA`
- `BLOCK_LIST`
- `VIRUS`
- `HUMAN`
- `ANIMAL`
- `CANCER`
- `FUNGUS`
- `BACTERIA`

## Main capabilities

- pure Rust default build
- optional native `vectorscan` backend behind a feature flag
- regex fallback matcher for portable builds
- Bayes support for `Count` and `TfIdf`
- KNN support for `Word2Vec` and `FastText`
- YAML validation for complete pipelines
- mandatory `k` for all KNN runs
- mandatory `p` for `KnnMinkowski`
- report export in CSV and JSON

## Build modes

### Default build

```bash
cargo test
```

### Native Vectorscan build

```bash
cargo test --features vectorscan
```

## System packages for native Vectorscan

### Fedora

```bash
sudo dnf install cmake gcc gcc-c++ boost-devel
```

### Debian / Ubuntu

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libboost-all-dev
```

## Example YAML pipeline

```yaml
report_name: Classification web attack
method: KnnCosine
nlp: FastText
threads: 8
csv_output: log.csv
json_output: log.json
recursive_way: On
hot_test_path: test/http/attack/requests/
cold_test_path: test/http/regular/requests/
hot_label: WEB_ATTACK
cold_label: RAW_DATA
score_sum: On
k: 5
extra_match:
  - recursive_way: On
    engine: Regex
    path: test/sql_injection/querys/
    score_add_points: 15
    title: sql-injection-indicators
    description: Signature boost for SQL injection markers
```

## Validation rules

- Bayes does not require `k` or `p`
- every KNN method requires `k`
- `KnnMinkowski` requires both `k` and `p`

## Design

- `ClassifierFactory` builds Bayes or KNN classifiers
- `MatcherFactory` builds regex or optional Vectorscan matchers
- `RulesFile` validates the pipeline before execution
- `ClassificationReport` exports results for automation

## More documentation

See `../helper.md` for a full explanation of the YAML rule file and score-merging behavior.
