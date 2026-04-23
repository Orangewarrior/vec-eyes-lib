# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.9.0] - 2026-04-22

### Fixed — Critical correctness bugs (5 issues)

#### 2.1 remove_null_bytes: Remove null bytes instead of spaces
**File:** `src/nlp/normalizer.rs`

The `remove_null_bytes` function was incorrectly filtering space characters (`' '`)
instead of null bytes (`'\0'`). This corrupted tokenization inside
`decode_obfuscated_text`, silently removing all whitespace from processed text.

**Fix:** Changed filter condition from `*c != ' '` to `*c != '\0'`.

#### 2.2 decode_percent_encoding: Fix UTF-8 decoding
**File:** `src/nlp/normalizer.rs`

The percent-decoding function converted decoded bytes directly to `char` using
`value as char`, which only works for ASCII. Multi-byte UTF-8 sequences like
`%C3%A9` (é) were decoded as two invalid separate characters.

**Fix:** 
- Changed output buffer from `String` to `Vec<u8>`
- Collect decoded bytes, then use `String::from_utf8_lossy()` at the end
- Now correctly handles all UTF-8 sequences

#### 2.3 flush_token: Preserve single-character tokens
**File:** `src/nlp/tokenizer.rs`

The tokenizer discarded all tokens with length < 2, silently removing critical
single-character symbols. For security-focused analysis (syscall/HTTP/malware
detection), tokens like `<`, `>`, `|`, and single digits are essential signals.

**Fix:**
- Removed `if current.len() >= 2` check
- Now preserves all non-empty tokens
- Added documentation explaining security rationale

#### 2.4 ClassifierBuilder: Remove duplicate new/build methods
**File:** `src/classifier.rs`

`ClassifierBuilder` had both `impl Builder<...> trait` and a separate
`impl ClassifierBuilder` block re-implementing `new()` and `build()`. This
duplication was confusing and prone to implementation drift.

**Fix:**
- Removed redundant inherent impl methods
- Kept only the trait implementation
- All builder methods now consistently defined in one place

#### 2.7 tokenizer: Remove duplicate lowercase conversion
**File:** `src/nlp/tokenizer.rs`

The tokenizer called `to_ascii_lowercase()` on each character even though
`normalize_text` already applies Unicode lowercase. This was redundant and
inconsistent (Unicode vs ASCII lowercase).

**Fix:** Removed duplicate `to_ascii_lowercase()` call in tokenizer loop.

### Performance Optimizations

#### softmax_scores: Optimize probability normalization
**File:** `src/classifier.rs`

Multiple micro-optimizations to the softmax function used in classification:

- Replaced manual max-finding loop with `fold(f32::NEG_INFINITY, f32::max)`
- Pre-allocate result vector with `Vec::with_capacity(input.len())`
- Replace division by sum with multiplication by inverse (`value * inv_sum`)
- Simplified zero-sum edge case handling

**Impact:** Reduced allocations, better SIMD utilization, faster computation
for large label sets.

### Changed

- All bug fixes maintain backward API compatibility
- Tokenization behavior change: single-char tokens now included (may increase
  token count slightly but improves detection accuracy)

---

## [2.8.0] - 2026-04-22

### Fixed — ML algorithmic correctness (10 issues)

#### 1.1 Sublinear TF in TF-IDF pipeline (`nlp/feature_extractor.rs`)
Raw term frequency divided by document length was replaced with the standard
sublinear form `tf = 1 + ln(count)`, eliminating length bias and aligning with
Salton & Buckley (1988).

#### 1.2 IDF re-fitted at inference time (`advanced_models.rs`)
`FeaturePipeline` variants `Word2Vec` and `FastText` now store the training
`TfIdfModel` as a named field (`idf`). Inference always uses the stationary
training IDF — the model no longer re-fits IDF on probe documents.

#### 1.3 IsolationForest trained on hot-only samples (`advanced_models.rs`)
The anomaly detector was previously fitted only on cold-label samples, preventing
it from learning the full data distribution. It now trains on the complete corpus;
contamination is derived automatically from the actual hot-label proportion
(`clamp(0.001, 0.49)`).

#### 1.4 Probe IDF used as NB weight (`classifiers/bayes/core.rs`)
The TF-IDF variant of Naïve Bayes was weighting token log-likelihoods by the
probe document's TF-IDF matrix rather than the corpus IDF. Fixed to use
`tfidf.idf[idx]` — a stationary weight from the training fit, consistent with
the IDF-weighted multinomial NB formulation.

#### 1.5 Singularity in k-NN distance weighting (`classifiers/knn/core.rs`)
Neighbor weight `1 / (d + 1e-6)`, clamped to 1000, was replaced with the
Gaussian kernel `exp(-d)`. The new weight is smooth, bounded in `(0, 1]`,
free of singularities, and valid for all supported distance metrics.

#### 1.6 Mini-batch SGD without epoch shuffle (`classifiers/logistic_regression/core.rs`)
Logistic regression batches always iterated samples in insertion order, biasing
gradients toward early samples. Each epoch now shuffles indices with a
per-class deterministic `StdRng` (seed `0xBAD_FEED ^ (class_id * large_prime)`),
ensuring unbiased mini-batches and reproducible parallel training.

#### 1.7 Incorrect Random Fourier Features for RBF SVM (`classifiers/svm/core.rs`)
The frequency matrix was not sampled from the correct distribution N(0, √(2γ)).
`KernelMap` enum introduced with three variants:
- `Identity` — linear kernel (no transform)
- `Rff` — RBF via Rahimi & Recht (2007): Box-Muller → N(0, √(2γ)), bias ~ U(0, 2π), scale √(2/D)
- `Landmark` — polynomial/sigmoid via random landmark rows (shuffled, seeded)

#### 1.8 Random threshold candidates in Gradient Boosting (`classifiers/gradient_boosting/core.rs`)
Split thresholds were generated as random jitter over `[min, max]`, missing the
optimal cut-points. Replaced with sorted quantile midpoints between adjacent
unique feature values (≤ 32 per feature — equivalent to a 32-bin histogram).
The implementation is now fully deterministic with no RNG dependency.

#### 1.9 Skip-gram with single embedding matrix (`nlp/feature_extractor.rs`)
Word2Vec Skip-gram used a single weight matrix for both center and context
words, deviating from Mikolov et al. (2013). Rewritten with two separate
matrices: `w_in` (deterministic hash initialization) and `w_out`
(zero-initialized). Negative sampling uses real RNG with the unigram noise
distribution P^0.75. Only `w_in` is returned as the final embedding.

#### 1.10 Inconsistent probability scale (`classifier.rs`, `matcher/mod.rs`)
`softmax_scores` was returning percentages `[0, 100]`; `merge_scores` received
`rule_boost` in `[0, 100]` and summed it directly to base probabilities,
producing scores > 1. Fixed:
- `softmax_scores` now returns true probabilities `[0, 1]`
- `score_percent` field computed as `top_score * 100.0` at report generation
- `merge_scores` normalises boost: `rule_boost / 100.0` before addition

### Added
- `.gitignore` — excludes `/target` build artefacts
- `Cargo.lock` — committed for reproducible builds

---

## [2.7.3] - (previous release)

See git history for earlier changes.
