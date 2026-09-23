import json

import numpy as np
import pytest

from app.core.config import DEFAULT_REFERENCE_PATH, FROZEN_MODEL_SHA256
from app.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from app.services.explanation import build_indicators, feature_label
from app.services.reference import (
    FeatureReference, ReferenceData, build_reference, load_reference, risk_from_percentile,
)

NAMES = list(FEATURE_NAMES)


def synthetic_frames(seed: int = 0):
    rng = np.random.default_rng(seed)
    cover = lambda n: rng.normal(0.0, 1.0, size=(n, len(NAMES)))
    stego = lambda n: rng.normal(0.0, 1.0, size=(n, len(NAMES))) + np.eye(len(NAMES))[0] * 1.0  # only feature 0 separates
    return cover(500), stego(500), cover(300), stego(300), cover(300), stego(300)


def synthetic_reference(tmp_path, sha="a" * 64):
    tc, ts, vc, vs, ec, es = synthetic_frames()
    ref = build_reference(
        model_sha256=sha, feature_schema_version=FEATURE_SCHEMA_VERSION, feature_names=NAMES,
        predict_proba=lambda x: 1 / (1 + np.exp(-x[:, 0])), train_x_cover=tc, train_x_stego=ts, val_x_cover=vc,
        val_x_stego=vs, test_x_cover=ec, test_x_stego=es, payload_note="synthetic", description="synthetic",
    )
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(ref))
    return path, ref


def test_risk_bands() -> None:
    assert [risk_from_percentile(p) for p in (0.0, 0.8999, 0.90, 0.9899, 0.99, 1.0)] == ["low", "low", "elevated", "elevated", "high", "high"]


def test_builder_flag_rates_match_the_definition(tmp_path) -> None:
    _, ref = synthetic_reference(tmp_path)
    e = ref["operating_points"]["elevated"]["flag_rate"]
    assert e["validation_clean"] == pytest.approx(0.10, abs=0.01)  # by construction ~10% of the reference covers
    assert e["validation_lsb_stego"] > e["validation_clean"]        # the synthetic stego really is shifted
    h = ref["operating_points"]["high"]["flag_rate"]
    assert h["validation_clean"] == pytest.approx(0.01, abs=0.005)
    assert len(ref["val_cover_scores"]) == 300 and ref["format_version"] == 1


def test_percentile_is_the_empirical_cdf(tmp_path) -> None:
    path, ref = synthetic_reference(tmp_path)
    data = load_reference(path, model_sha256="a" * 64, feature_names=NAMES, feature_schema_version=FEATURE_SCHEMA_VERSION)
    scores = np.array(ref["val_cover_scores"])
    assert data.cover_percentile(scores.min() - 1) == 0.0 and data.cover_percentile(scores.max() + 1) == 1.0
    assert data.cover_percentile(float(np.median(scores))) == pytest.approx(0.5, abs=0.01)


def test_reference_bound_to_model_and_schema(tmp_path) -> None:
    path, _ = synthetic_reference(tmp_path)
    ok = dict(model_sha256="a" * 64, feature_names=NAMES, feature_schema_version=FEATURE_SCHEMA_VERSION)
    assert load_reference(path, **ok) is not None
    assert load_reference(path, **{**ok, "model_sha256": "b" * 64}) is None            # different model
    assert load_reference(path, **{**ok, "feature_schema_version": "v2"}) is None      # different schema
    assert load_reference(path, **{**ok, "feature_names": NAMES[::-1]}) is None        # different order
    assert load_reference(tmp_path / "missing.json", **ok) is None
    (tmp_path / "bad.json").write_text("{not json")
    assert load_reference(tmp_path / "bad.json", **ok) is None


def test_shipped_reference_is_bound_to_the_frozen_model() -> None:
    raw = json.loads(DEFAULT_REFERENCE_PATH.read_text())
    assert raw["model_sha256"] == FROZEN_MODEL_SHA256
    assert raw["feature_names"] == NAMES and raw["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert raw["n"]["val_cover"] == len(raw["val_cover_scores"]) == 1499


# ------------------------------------------------------------------ explanation
def make_reference() -> ReferenceData:
    feats = {n: FeatureReference(cover_mean=0.0, cover_std=1.0, stego_mean=0.0) for n in NAMES}
    feats["lsb_transition_r"] = FeatureReference(0.0, 1.0, 1.0)      # stego higher
    feats["lsb_local_variance_r"] = FeatureReference(0.0, 1.0, -1.0)  # stego lower
    return ReferenceData("a" * 64, "v1", "d", "p", tuple(np.linspace(0, 1, 200)), feats, {})


def test_labels_are_static_and_specific() -> None:
    assert feature_label("lsb_transition_g") == "LSB transition rate (green channel)"
    assert feature_label("highpass_std") == "High-pass residual standard deviation"
    assert all(feature_label(n) != n for n in NAMES)


def test_direction_reflects_reference_class_shift_and_discriminative_features_come_first() -> None:
    vec = np.zeros(len(NAMES))
    vec[NAMES.index("lsb_transition_r")] = 2.0       # above clean mean, and stego is higher -> toward stego
    vec[NAMES.index("lsb_local_variance_r")] = 2.0   # above clean mean, but stego is lower -> away from stego
    vec[NAMES.index("highpass_std")] = 50.0          # huge deviation but does not separate classes
    imp = np.full(len(NAMES), 0.01); imp[NAMES.index("highpass_std")] = 1.0
    out = {i.name: i for i in build_indicators(vec, NAMES, imp, make_reference())}
    assert out["lsb_transition_r"].direction == "increases_stego_signal"
    assert out["lsb_local_variance_r"].direction == "decreases_stego_signal"
    ordered = [i.name for i in build_indicators(vec, NAMES, imp, make_reference())]
    assert "highpass_std" in ordered
    assert ordered.index("lsb_transition_r") < ordered.index("highpass_std")
    assert ordered.index("lsb_local_variance_r") < ordered.index("highpass_std")
    assert ordered[0] in ("lsb_transition_r", "lsb_local_variance_r")


def test_without_reference_indicators_are_informational_only() -> None:
    imp = np.arange(len(NAMES), dtype=float)
    out = build_indicators(np.zeros(len(NAMES)), NAMES, imp, None)
    assert len(out) == 5 and all(i.direction == "informational" and i.deviation_from_reference_cover is None for i in out)
    assert out[0].name == NAMES[-1]  # highest importance first
