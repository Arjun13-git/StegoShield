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
