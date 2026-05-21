"""
Run ADSR boundary benchmark and report mean error (ms) per detection mode and preset.

Examples:
  python run_benchmark.py --generate-corpus          # build 40 synthetic labeled one-shots
  python run_benchmark.py                            # run on default synthetic corpus
  python run_benchmark.py --annotations my.csv --audio-dir D:/Samples/labeled
  python run_benchmark.py --template annotations_template.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmark.benchmark_core import (  # noqa: E402
    DEFAULT_TOLERANCE_MS,
    AggregateMetrics,
    BenchmarkReport,
    GroundTruthSample,
    aggregate_evaluations,
    build_config_for_run,
    evaluate_sample,
    format_report_table,
    load_annotations,
    write_annotation_template,
)
from benchmark.generate_corpus import generate_corpus  # noqa: E402

DEFAULT_CORPUS = ROOT / "benchmark" / "corpus"
DEFAULT_RESULTS = ROOT / "benchmark" / "results"

# (detection_mode, pitch_refine_mode)
BENCHMARK_CONDITIONS: List[Tuple[str, str]] = [
    ("smart", "expand"),
    ("smart", "annotate"),
    ("smart", "off"),
    ("advanced", "expand"),
    ("advanced", "off"),
    ("proportional", "off"),
]

PRESETS_FOR_BENCHMARK = [
    "Very Short (< 0.5s)",
    "Short (0.5-1.5s)",
    "Medium (1.5-3.0s)",
    "Long (3.0-6.0s)",
    "Very Long (> 6.0s)",
    "Legato / Bow",
]


def _samples_for_preset(all_samples: Sequence[GroundTruthSample], preset: str) -> List[GroundTruthSample]:
    matched = [s for s in all_samples if s.preset == preset]
    if matched:
        return matched
    if preset == "Legato / Bow":
        return [s for s in all_samples if s.category == "bow"]
    if preset == "Staccato / Pluck":
        return [s for s in all_samples if s.category in ("pluck", "marcato")]
    return []


def run_benchmark(
    samples: Sequence[GroundTruthSample],
    tolerance_ms: float = DEFAULT_TOLERANCE_MS,
    presets: Sequence[str] = PRESETS_FOR_BENCHMARK,
    conditions: Sequence[Tuple[str, str]] = BENCHMARK_CONDITIONS,
    corpus_dir: Path = DEFAULT_CORPUS,
) -> BenchmarkReport:
    report = BenchmarkReport(
        created_at=datetime.now().isoformat(timespec="seconds"),
        corpus_dir=str(corpus_dir),
        tolerance_ms=tolerance_ms,
    )

    for preset in presets:
        subset = _samples_for_preset(samples, preset)
        if not subset:
            continue
        for mode, pitch_mode in conditions:
            cfg = build_config_for_run(preset, mode, pitch_mode)
            evals = [evaluate_sample(s, cfg) for s in subset]
            report.samples.extend(evals)
            report.aggregates.append(
                aggregate_evaluations(evals, mode, preset, pitch_mode, tolerance_ms)
            )

    return report


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ADSR boundary error benchmark")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Folder with WAV files and annotations.json",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="annotations.json or .csv (default: <corpus-dir>/annotations.json)",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Base folder for audio paths in CSV (default: corpus-dir)",
    )
    parser.add_argument("--generate-corpus", action="store_true", help="Generate 40 synthetic labeled one-shots")
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Write a CSV template for manual labeling of real recordings",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=DEFAULT_TOLERANCE_MS,
        help="Boundary tolerance for within-tol %% column (default 50 ms)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Directory for benchmark_results.json and .txt report",
    )
    args = parser.parse_args(argv)

    if args.template:
        write_annotation_template(args.template)
        print(f"Wrote labeling template: {args.template}")
        return 0

    corpus_dir = args.corpus_dir.resolve()
    if args.generate_corpus:
        ann = generate_corpus(corpus_dir)
        print(f"Generated synthetic corpus: {ann} ({len(json.loads(ann.read_text())['samples'])} samples)")

    ann_path = args.annotations or (corpus_dir / "annotations.json")
    if not ann_path.exists():
        print(f"Annotations not found: {ann_path}", file=sys.stderr)
        print("Run: python run_benchmark.py --generate-corpus", file=sys.stderr)
        return 1

    samples = load_annotations(ann_path.resolve())
    if args.audio_dir:
        base = args.audio_dir.resolve()
        samples = [
            GroundTruthSample(
                sample_id=s.sample_id,
                file=str(base / Path(s.file).name),
                sr=s.sr,
                t_att=s.t_att,
                t_dec=s.t_dec,
                t_end=s.t_end,
                preset=s.preset,
                category=s.category,
                notes=s.notes,
            )
            for s in samples
        ]

    missing = [s for s in samples if not Path(s.file).exists()]
    if missing:
        print(f"Missing {len(missing)} audio file(s), e.g. {missing[0].file}", file=sys.stderr)
        return 1

    print(f"Benchmarking {len(samples)} labeled one-shots...")
    report = run_benchmark(samples, tolerance_ms=args.tolerance_ms, corpus_dir=corpus_dir)

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "benchmark_results.json"
    txt_path = args.output / "benchmark_report.txt"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    table = format_report_table(report)
    txt_path.write_text(table, encoding="utf-8")
    print(table)
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")

    best = min(report.aggregates, key=lambda a: a.mae_mean_ms if a.n_samples else 1e9)
    print(
        f"Best mean error: {best.mode} + {best.preset} + pitch={best.pitch_refine_mode} "
        f"-> {best.mae_mean_ms:.1f} ms (n={best.n_samples})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
