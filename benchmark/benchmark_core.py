"""
Boundary-error evaluation for ADSR segmentation benchmarks.

Ground-truth times are absolute file seconds: t_att, t_dec, t_end.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np

import audio_segment_core as core

BOUNDARY_NAMES = ("t_att", "t_dec", "t_end")
DEFAULT_TOLERANCE_MS = 50.0


@dataclass
class GroundTruthSample:
    sample_id: str
    file: str
    sr: int
    t_att: float
    t_dec: float
    t_end: float
    preset: str = "Medium (1.5-3.0s)"
    category: str = "unknown"
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Optional[Path] = None) -> "GroundTruthSample":
        path = data["file"]
        if base_dir and not Path(path).is_absolute():
            path = str(base_dir / path)
        return cls(
            sample_id=str(data.get("id") or data.get("sample_id") or Path(path).stem),
            file=path,
            sr=int(data.get("sr") or data.get("sample_rate") or 22050),
            t_att=float(data["t_att"]),
            t_dec=float(data["t_dec"]),
            t_end=float(data["t_end"]),
            preset=str(data.get("preset") or "Medium (1.5-3.0s)"),
            category=str(data.get("category") or "unknown"),
            notes=str(data.get("notes") or ""),
        )


@dataclass
class BoundaryErrorsMs:
    t_att: float
    t_dec: float
    t_end: float

    @property
    def mean(self) -> float:
        return (self.t_att + self.t_dec + self.t_end) / 3.0

    @property
    def max(self) -> float:
        return max(self.t_att, self.t_dec, self.t_end)

    def within_tolerance(self, tolerance_ms: float) -> bool:
        return self.max <= tolerance_ms


@dataclass
class SampleEvaluation:
    sample_id: str
    file: str
    preset: str
    mode: str
    pitch_refine_mode: str
    errors_ms: BoundaryErrorsMs
    predicted: Dict[str, float]
    ground_truth: Dict[str, float]


@dataclass
class AggregateMetrics:
    mode: str
    preset: str
    pitch_refine_mode: str
    n_samples: int
    mae_att_ms: float
    mae_dec_ms: float
    mae_end_ms: float
    mae_mean_ms: float
    within_tolerance_pct: float
    tolerance_ms: float = DEFAULT_TOLERANCE_MS


@dataclass
class BenchmarkReport:
    created_at: str
    corpus_dir: str
    tolerance_ms: float
    aggregates: List[AggregateMetrics] = field(default_factory=list)
    samples: List[SampleEvaluation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "corpus_dir": self.corpus_dir,
            "tolerance_ms": self.tolerance_ms,
            "aggregates": [asdict(a) for a in self.aggregates],
            "samples": [
                {
                    **asdict(s),
                    "errors_ms": asdict(s.errors_ms),
                }
                for s in self.samples
            ],
        }


def boundary_errors_ms(
    predicted: core.SegmentResult, truth: GroundTruthSample
) -> BoundaryErrorsMs:
    return BoundaryErrorsMs(
        t_att=abs(predicted.t_att - truth.t_att) * 1000.0,
        t_dec=abs(predicted.t_dec - truth.t_dec) * 1000.0,
        t_end=abs(predicted.t_end - truth.t_end) * 1000.0,
    )


def load_annotations_json(path: Path) -> List[GroundTruthSample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    samples = data.get("samples") or data
    if isinstance(samples, dict):
        samples = samples.get("samples", [])
    return [GroundTruthSample.from_dict(s, base) for s in samples]


def load_annotations_csv(path: Path) -> List[GroundTruthSample]:
    base = path.parent
    out: List[GroundTruthSample] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                GroundTruthSample.from_dict(
                    {
                        "id": row.get("id") or row.get("sample_id"),
                        "file": row["file"],
                        "sr": row.get("sr") or row.get("sample_rate") or 22050,
                        "t_att": row["t_att"],
                        "t_dec": row["t_dec"],
                        "t_end": row["t_end"],
                        "preset": row.get("preset") or "Medium (1.5-3.0s)",
                        "category": row.get("category") or "manual",
                        "notes": row.get("notes") or "",
                    },
                    base,
                )
            )
    return out


def load_annotations(path: Path) -> List[GroundTruthSample]:
    if path.suffix.lower() == ".csv":
        return load_annotations_csv(path)
    return load_annotations_json(path)


def detection_mode_label(cfg: core.SegmentConfig) -> str:
    if cfg.use_advanced:
        return "advanced"
    if cfg.use_smart:
        return "smart"
    return "proportional"


def build_config_for_run(
    preset: str,
    mode: str,
    pitch_refine_mode: str = "expand",
) -> core.SegmentConfig:
    cfg = core.SegmentConfig.from_preset(preset)
    mode = mode.lower()
    if mode == "advanced":
        cfg.use_advanced = True
        cfg.use_smart = False
    elif mode == "proportional":
        cfg.use_advanced = False
        cfg.use_smart = False
    else:
        cfg.use_advanced = False
        cfg.use_smart = True
    cfg.pitch_refine_mode = pitch_refine_mode
    cfg.use_pitch_refine = pitch_refine_mode != "off"
    return cfg


def evaluate_sample(
    sample: GroundTruthSample,
    cfg: core.SegmentConfig,
) -> SampleEvaluation:
    y, sr = librosa.load(sample.file, sr=None)
    if sr != sample.sr:
        sample = GroundTruthSample(
            sample_id=sample.sample_id,
            file=sample.file,
            sr=sr,
            t_att=sample.t_att,
            t_dec=sample.t_dec,
            t_end=sample.t_end,
            preset=sample.preset,
            category=sample.category,
            notes=sample.notes,
        )
    result = core.detect_segments(y, sr, cfg, file_path=Path(sample.file))
    errors = boundary_errors_ms(result, sample)
    return SampleEvaluation(
        sample_id=sample.sample_id,
        file=sample.file,
        preset=sample.preset,
        mode=detection_mode_label(cfg),
        pitch_refine_mode=cfg.pitch_refine_mode if cfg.use_pitch_refine else "off",
        errors_ms=errors,
        predicted={"t_att": result.t_att, "t_dec": result.t_dec, "t_end": result.t_end},
        ground_truth={"t_att": sample.t_att, "t_dec": sample.t_dec, "t_end": sample.t_end},
    )


def aggregate_evaluations(
    evaluations: Sequence[SampleEvaluation],
    mode: str,
    preset: str,
    pitch_refine_mode: str,
    tolerance_ms: float = DEFAULT_TOLERANCE_MS,
) -> AggregateMetrics:
    if not evaluations:
        return AggregateMetrics(
            mode=mode,
            preset=preset,
            pitch_refine_mode=pitch_refine_mode,
            n_samples=0,
            mae_att_ms=0.0,
            mae_dec_ms=0.0,
            mae_end_ms=0.0,
            mae_mean_ms=0.0,
            within_tolerance_pct=0.0,
            tolerance_ms=tolerance_ms,
        )
    n = len(evaluations)
    mae_att = sum(e.errors_ms.t_att for e in evaluations) / n
    mae_dec = sum(e.errors_ms.t_dec for e in evaluations) / n
    mae_end = sum(e.errors_ms.t_end for e in evaluations) / n
    mae_mean = sum(e.errors_ms.mean for e in evaluations) / n
    within = sum(1 for e in evaluations if e.errors_ms.within_tolerance(tolerance_ms))
    return AggregateMetrics(
        mode=mode,
        preset=preset,
        pitch_refine_mode=pitch_refine_mode,
        n_samples=n,
        mae_att_ms=mae_att,
        mae_dec_ms=mae_dec,
        mae_end_ms=mae_end,
        mae_mean_ms=mae_mean,
        within_tolerance_pct=100.0 * within / n,
        tolerance_ms=tolerance_ms,
    )


def format_report_table(report: BenchmarkReport) -> str:
    lines = [
        "",
        f"ADSR Boundary Benchmark  |  tolerance +/-{report.tolerance_ms:.0f} ms  |  {report.created_at}",
        f"Corpus: {report.corpus_dir}",
        "",
        f"{'Mode':<14} {'Preset':<28} {'Pitch':<10} {'N':>3} "
        f"{'MAE att':>8} {'MAE dec':>8} {'MAE end':>8} {'MAE mean':>9} {'<=tol%':>6}",
        "-" * 102,
    ]
    for a in sorted(report.aggregates, key=lambda x: (x.mode, x.preset, x.pitch_refine_mode)):
        lines.append(
            f"{a.mode:<14} {a.preset:<28} {a.pitch_refine_mode:<10} {a.n_samples:>3} "
            f"{a.mae_att_ms:>7.1f}ms {a.mae_dec_ms:>7.1f}ms {a.mae_end_ms:>7.1f}ms "
            f"{a.mae_mean_ms:>8.1f}ms {a.within_tolerance_pct:>5.1f}%"
        )
    lines.append("")
    return "\n".join(lines)


def write_annotation_template(path: Path) -> None:
    """CSV template for manual labeling of real one-shots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "file", "sr", "t_att", "t_dec", "t_end", "preset", "category", "notes",
        ])
        writer.writerow([
            "violin_A4_01",
            "violin_A4_01.wav",
            "44100",
            "0.180",
            "2.450",
            "2.920",
            "Medium (1.5-3.0s)",
            "manual",
            "attack end / decay start / active end (seconds from file start)",
        ])
