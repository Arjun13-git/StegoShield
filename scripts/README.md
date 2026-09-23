# Offline scripts

Command-line orchestration for the ML pipeline. Reusable logic lives in
`backend/app/ml/`; these scripts only wire it together. Run from the
repository root inside the project virtual environment. Generated outputs go
under `data/` (git-ignored) and contain only measured values.

## Phase 1A — BOSSBase, controlled LSB, model comparison

- `generate_dataset.py`: index BOSSBase, exclude exact pixel-content duplicates, assign a source-level train/val/test split, and write per-payload feature CSVs (stego images are generated in memory, never stored).
- `train_models.py`: train Logistic Regression / SVM / Random Forest, select on validation F1, evaluate the frozen choice once on test, save the model bundle.
- `evaluate.py`: re-evaluate a saved artifact on any split and emit ROC-curve points.
- `generate_payload_report.py`: evaluate one frozen artifact across payload levels.

## Phase 1B — external (ALASKA2) evaluation and robustness (frozen Phase 1A model)

- `download_alaska2_pilot.py`: deterministic, resumable, rate-limit-aware selective download (requires Kaggle credentials and acceptance of the competition rules; never downloads the full corpus).
- `inspect_alaska2.py`: read-only inspection; `--full-validate` decodes every file and tabulates format, size, JPEG structure and QF.
- `prepare_external_dataset.py`: deterministic source selection, source-level split, duplicate checks and manifest; `--require-complete` refuses partial source groups.
- `evaluate_external_pilot.py`: clean-cover score control, controlled LSB test, and separate Cover-vs-JMiPOD / JUNIWARD / UERD comparisons with source-level statistics.
- `robustness_eval.py`: JPEG QF90/QF75, resize, Gaussian noise and crop applied identically to BOSSBase test covers and their stego.
- `analyze_robustness.py`: source-level bootstrap intervals for the robustness results.
- `finalize_phase1b_report.py`: independently re-derives and cross-checks every Phase 1B result and writes `data/reports/phase1b_final_report.md`.

Notes:

- Read per-source score CSVs with `pandas.read_csv(..., float_precision="round_trip")`; the default parser perturbs values by ~1e-16, which changes tie handling in AUC.
- ALASKA2 is licensed for non-commercial use without redistribution of derived material; keep it and everything derived from it local.
