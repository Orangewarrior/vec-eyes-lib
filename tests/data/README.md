# Test fixture notes

These fixtures are intentionally tiny and self-contained so the test suite remains deterministic and fast.

The text patterns were inspired by public datasets and benchmark families commonly used in:

- SMS spam and ham classification
- HTTP attack and benign request classification
- fraud / risk anomaly classification
- biology-oriented text classification

The fixtures are not full redistributed corpora. They are reduced test samples designed to validate:

- YAML ingestion
- recursive dataset loading
- NLP pipeline selection
- Bayes and KNN builder wiring
- rule matcher score boosting
