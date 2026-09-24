# StegoShield

**Explainable ML-Based Image Steganalysis Platform**

StegoShield is a cybersecurity application for detecting possible hidden information in digital images. The MVP focuses on LSB-based image steganography and uses handcrafted statistical/bit-plane features with machine-learning classifiers.

## Core goals

- Detect cover vs. LSB-stego images.
- Return a steganography likelihood score with calibrated wording (not a claim of maliciousness).
- Explain which feature groups influenced the prediction.
- Compare Logistic Regression, SVM and Random Forest.
- Study payload sensitivity and robustness to common image transformations.
- Provide a Next.js security-analysis dashboard backed by FastAPI.
- Keep the architecture modular so additional steganographic algorithms or CNN models can be added later.

## Current scope

The first implementation is intentionally **LSB-focused**. It is not a universal steganography detector.

## Project status

The ML core and its evaluation (Phases 1A and 1B) and the FastAPI inference backend (Phase 2, see [Backend API](#backend-api-phase-2)) are complete. The Next.js dashboard (Phase 3, see [Frontend](#frontend-phase-3)) provides Analyze, Encode and Research pages on top of that API. All numbers below come from generated experiment reports (`data/reports/`, git-ignored); the complete write-up is produced locally by `scripts/finalize_phase1b_report.py` as `data/reports/phase1b_final_report.md`.

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

## Backend API (Phase 2)

A FastAPI service exposes the frozen Phase 1 detector. It reuses the Phase 1 feature extractor and `ModelBundle` unchanged; it never trains, refits or replaces a model.

### Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate   # or any Python >= 3.11 environment
uv pip install -e ".[dev]"
```

The model is **not** in git. Produce it with the Phase 1 scripts (`generate_dataset.py` then `train_models.py`, see `scripts/README.md`) or place your copy at `data/models/stegoshield_rf.joblib` (or set `MODEL_PATH`). At start-up the API hashes the file and refuses any artifact that does not match the pinned SHA-256 of the frozen model (`34707f83…2523d`) before deserialising it (the artifact is a pickle, so an unverified file is a code-execution risk). Set `MODEL_SHA256` to another value to use a different artifact, or to an empty value to disable the check. `backend/app/resources/reference_scores_v1.json` is tracked; it is bound to the same hash and is rebuilt with `scripts/build_reference_scores.py`. All configuration variables are listed in `.env.example`.

### Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000        # interactive docs at /docs
curl http://127.0.0.1:8000/ready
```

### Endpoints

| method and path | purpose |
|---|---|
| `GET /health` | liveness only: the process is up |
| `GET /ready` | 200 only if the frozen model is verified and loaded; otherwise 503 with a short `reason` tag (for example `artifact_missing`) |
| `GET /api/v1/health` | API-level health (per the design) |
| `GET /api/v1/model-info` | model identity (artifact SHA-256, schema), stored in-domain test metrics, the risk-level definitions and limitations |
| `POST /api/v1/analyze` | multipart field `file` (PNG or JPEG); returns the analysis below |
| `POST /api/v1/encode` | multipart `file` (cover image) and `message` (text); returns a PNG with the message embedded in the pixel LSBs (educational demonstration, sequential embedding, alpha untouched) |

`/ready` is required by the Phase 2 task but is not in `System-Design/05-api-design.md`. Errors always have the form `{"detail", "code", "request_id"}` with fixed, non-leaky messages (400 invalid image/format/mode/dimensions or missing message, 413 too large, 422 malformed request, 500 internal, 503 model unavailable).

### Example

```bash
python -c "import numpy as np; from PIL import Image; Image.fromarray(np.random.default_rng(1).integers(0,256,(256,256),dtype=np.uint8)).save('example.png')"
curl -F "file=@example.png" http://127.0.0.1:8000/api/v1/analyze
```

Response (real output of the frozen model on that image, `indicators` truncated to two entries and `request_id` replaced):

```json
{
  "request_id": "…",
  "verdict": "likely_cover",
  "risk_level": "low",
  "stego_score": 51.99,
  "score_kind": "uncalibrated_model_score",
  "cover_reference_percentile": 83.59,
  "risk_context": "Below the 'elevated' level (top 10% of reference clean images). In Phase 1 held-out testing that level flagged 11.3% of clean images and 13.9% of 0.10 bits-per-pixel LSB stego images.",
  "model": "RandomForest", "model_version": "v1", "feature_schema_version": "v1", "feature_count": 22,
  "indicators": [
    {"name": "lsb_transition_b", "label": "LSB transition rate (blue channel)", "value": 0.5011565685272217,
     "direction": "increases_stego_signal", "deviation_from_reference_cover": 0.719, "model_importance": 0.10854}
  ],
  "explanation_method": "Features ranked by global model importance times deviation from the reference clean-image distribution. Heuristic model-based evidence, not a causal or per-prediction attribution.",
  "image": {"format": "PNG", "width": 256, "height": 256, "original_mode": "L", "size_bytes": 65917, "alpha_discarded": false, "analyzed_as": "RGB (grey replicated)"},
  "warnings": ["The image is not 512x512, the size used in the research evaluations."],
  "limitations": ["The score is an uncalibrated model output, not a probability that the image contains hidden data.", "…"]
}
```

### How to read a result

- `stego_score` is the model output x 100. It is **uncalibrated and is not a probability** that the image contains hidden data.
- The raw score clusters near 0.5 (Phase 1: 71.9% of clean held-out images score at or above 0.5), so a fixed 0.5 cut-off would flag most clean images. `risk_level` is therefore defined against the score distribution of clean reference images: `elevated` is the top 10% and `high` the top 1%. On the held-out Phase 1 test set `elevated` flagged 11.3% of clean and 13.9% of 0.10 bits-per-pixel LSB stego images, and `high` flagged 1.1% and 2.1%, so even a flagged image is weak evidence. If no matching reference file is loaded, `risk_level` is `not_assessed` and `verdict` is `inconclusive`.
- `indicators` are model-based evidence (global feature importance x deviation from clean reference images), not a causal explanation; a deviation does not prove hidden data.
- Grayscale images are analysed exactly as in Phase 1 (grey replicated to RGB). The alpha channel of RGBA images is discarded and reported. JPEG and colour inputs are outside the detector's tested setting and are flagged in `warnings`.

### Security controls and remaining limitations

Implemented: a body-size cap enforced before multipart parsing (declared and streamed); format decided by magic bytes with Pillow restricted to the PNG/JPEG decoders; dimensions, pixel count (default 4 MP), colour mode and frame count checked from the header before any decode; no file is written and no user filename or Content-Type is used; a concurrency limit; server-generated request IDs; fixed non-leaky error messages; explicit CORS origins (defaults: the Next.js dev server on port 3000; wildcard refused; `CORS_ORIGINS` overrides); model hash verified before deserialisation.

Not provided: authentication, rate limiting and TLS (put the service behind a reverse proxy before exposing it; the design also asks for proxy-level size limits and timeouts), slow-client (slow-loris) protection, a per-request timeout, model reloading without a restart, or any guarantee about images outside the tested setting. Measured on a development machine: about 47 ms per 512x512 request and about 375 ms for the largest allowed image (2048x2048), with peak server memory of roughly 0.46 GB for one largest-size request and 0.83 GB for two concurrent ones (the default concurrency limit). These are single-machine measurements, not guarantees. The API is a research prototype interface and is not production-hardened.

## Frontend (Phase 3)

`frontend/` is a Next.js (App Router, TypeScript, Tailwind) application with **Analyze**, **Encode** and **Research** pages. It only calls the FastAPI endpoints above and contains no ML logic. The Research page shows aggregate results exported from the generated experiment reports (`python scripts/export_research_summary.py` writes `frontend/src/data/research-summary.json`), plus live `GET /api/v1/model-info`. The backend's default CORS origins are `http://localhost:3000` and `http://127.0.0.1:3000`; see `frontend/README.md`.

## Quick architecture

```text
Next.js UI
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
