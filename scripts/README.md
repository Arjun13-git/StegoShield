# Offline scripts

These scripts are intentionally left as implementation targets for the first Claude Code pass.

- `generate_dataset.py`: create source-aware cover/stego pairs and manifests.
- `train_models.py`: train LR/SVM/RF using the shared feature extractor.
- `evaluate.py`: produce metrics, confusion matrix and ROC data.
- `generate_payload_report.py`: evaluate multiple payload densities.
- `robustness_eval.py`: evaluate transformed images.

All outputs should go under `data/` or `docs/` and should never contain fabricated metrics.
