"""Benchmark framework smoke tests."""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.benchmark_core import (
    aggregate_evaluations,
    boundary_errors_ms,
    build_config_for_run,
    evaluate_sample,
    load_annotations,
    write_annotation_template,
)
from benchmark.generate_corpus import SYNTHETIC_SPECS, generate_corpus
import audio_segment_core as core


def test_generate_corpus_count():
    with tempfile.TemporaryDirectory() as tmp:
        ann = generate_corpus(Path(tmp))
        data = json.loads(ann.read_text())
        assert len(data["samples"]) == len(SYNTHETIC_SPECS) == 40
        assert all((Path(tmp) / s["file"]).exists() for s in data["samples"])


def test_benchmark_evaluate_sample():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        generate_corpus(root)
        samples = load_annotations(root / "annotations.json")
        cfg = build_config_for_run("Medium (1.5-3.0s)", "smart", "expand")
        ev = evaluate_sample(samples[16], cfg)
        assert ev.errors_ms.mean < 200.0


def test_boundary_errors_zero_on_perfect_match():
    truth = type("T", (), {"t_att": 1.0, "t_dec": 2.0, "t_end": 2.5})()
    pred = core.SegmentResult(1.0, 2.0, 2.5, core.TrimInfo(0, 100, 0, 2.5, 2.5))
    err = boundary_errors_ms(pred, truth)
    assert err.mean == 0.0


def test_aggregate_metrics():
    from benchmark.benchmark_core import BoundaryErrorsMs, SampleEvaluation

    ev = SampleEvaluation(
        sample_id="a",
        file="a.wav",
        preset="Medium (1.5-3.0s)",
        mode="smart",
        pitch_refine_mode="expand",
        errors_ms=BoundaryErrorsMs(10.0, 20.0, 30.0),
        predicted={},
        ground_truth={},
    )
    agg = aggregate_evaluations([ev], "smart", "Medium (1.5-3.0s)", "expand", 50.0)
    assert agg.mae_mean_ms == 20.0
    assert agg.n_samples == 1


def test_annotation_template_csv():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "template.csv"
        write_annotation_template(path)
        assert path.exists()
        text = path.read_text()
        assert "t_att" in text and "t_dec" in text
