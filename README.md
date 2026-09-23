# StegoShield

**Explainable ML-Based Image Steganalysis Platform**

StegoShield is a cybersecurity application for detecting possible hidden information in digital images. The MVP focuses on LSB-based image steganography and uses handcrafted statistical/bit-plane features with machine-learning classifiers.

## Core goals

- Detect cover vs. LSB-stego images.
- Return a steganography likelihood score with calibrated wording (not a claim of maliciousness).
- Explain which feature groups influenced the prediction.
- Compare Logistic Regression, SVM and Random Forest.
- Study payload sensitivity and robustness to common image transformations.
- Provide a polished React/Vite security-analysis dashboard backed by FastAPI.
- Keep the architecture modular so additional steganographic algorithms or CNN models can be added later.

## Current scope

The first implementation is intentionally **LSB-focused**. It is not a universal steganography detector.

## Project status

The ML core and its evaluation (Phases 1A and 1B) are complete. The FastAPI service and the React dashboard are still skeletons (Phases 2 and 3 are not started). All numbers below come from generated experiment reports (`data/reports/`, git-ignored); the complete write-up is produced locally by `scripts/finalize_phase1b_report.py` as `data/reports/phase1b_final_report.md`.

### Phase 1A — in-domain ML core (frozen)

- **Data:** BOSSBase 1.01, 10,000 grayscale 512×512 PGM covers. Exact pixel-content duplicates (4 pairs) are excluded, leaving 9,996 unique sources.
- **Leakage control:** train/validation/test (6,997 / 1,499 / 1,500 sources) are split by cover identity; a cover and all its stego derivatives always share a split, and stego images are generated in memory from a deterministic seed.
- **Embedding:** controlled LSB replacement; payload = embedded bits per pixel (0.01–0.40).
- **Model:** Random Forest (300 trees, depth 6, min leaf 20) on 22 pixel/bit-plane statistics, chosen over Logistic Regression and SVM on validation F1; test evaluated once. The artifact is frozen (SHA-256 `34707f83…2523d`, git-ignored).
- **In-domain result (1,500 test cover/stego pairs, 0.10 bits per pixel):** accuracy 0.541, F1 0.635, ROC-AUC 0.553; at 0.40 bits per pixel ROC-AUC 0.695. This is a weak detector, and its scores are likelihood scores, not calibrated probabilities.

### Phase 1B — external generalization and robustness (Phase 1A model unchanged)

External corpus: ALASKA2 (JPEG, RGB, JPEG quality factors 75/90/95), 151 complete source groups / 604 images (Cover, JMiPOD, J-UNIWARD, UERD). All statistics treat the source as the unit; the sample is small (about 50 sources per quality factor), so intervals are wide.

| evaluation | ROC-AUC (95% CI over sources) |
|---|---|
| BOSSBase in-domain, 0.10 / 0.40 bits per pixel | 0.553 / 0.695 |
| ALASKA2, our controlled LSB, 0.10 bits per pixel per channel | 0.535 [0.518, 0.555] |
| ALASKA2, our controlled LSB, 0.40 bits per pixel per channel | 0.651 [0.617, 0.687] |
| ALASKA2 native JMiPOD | 0.500 [0.483, 0.517] |
| ALASKA2 native J-UNIWARD | 0.514 [0.498, 0.532] |
| ALASKA2 native UERD | 0.507 [0.490, 0.525] |

- Clean ALASKA2 cover scores are not distinguishable from in-domain BOSSBase cover scores at this sample size (75.5% vs 71.9% of covers score at or above 0.5), so that statistic is a property of the model's score distribution, not evidence of domain shift.
- A small signal from our own controlled LSB embedding transfers to ALASKA2; the detector shows no measurable ability to separate ALASKA2's native DCT-domain steganography (JMiPOD, J-UNIWARD, UERD) from cover. That is a negative result about this pixel-domain detector only.
- **Robustness** (1,500 BOSSBase test sources, same transformation applied to cover and stego; change in ROC-AUC at 0.10 / 0.40 bits per pixel): JPEG QF90 −0.044 / −0.156, JPEG QF75 −0.051 / −0.188, Gaussian noise σ=2 −0.056 / −0.194, small crop +0.000 / −0.009, 0.9× bicubic resize +0.021 / +0.001. JPEG re-compression and mild noise remove essentially all signal; small crop and mild downscaling do not. The reason resize does not hurt is not established.

### What these results do not show

They do not show the detector is production-ready; ALASKA2 is one corpus and does not represent steganography in general; BOSSBase→ALASKA2 and LSB→DCT-domain results are not equivalent to the in-domain task (different sensors, colour handling, compression and embedding paradigms); and camera-level source leakage cannot be assessed because camera metadata is unavailable. See the final report for full limitations.

### Data handling

Datasets, generated features, model binaries and reports are git-ignored. ALASKA2 is licensed for non-commercial use without redistribution of derived material and must stay local.

## Quick architecture

```text
React/Vite UI
     |
     v
FastAPI API
     |
     +--> Image validation / preprocessing
     |
     +--> Feature extraction
     |
     +--> ML inference
     |
     +--> Risk/explanation service
     |
     +--> SQLite metadata (optional MVP)
     |
     +--> model artifact (.joblib)
```

See `System-Design/` for the full design package.
