"""
Headless batch ADSR splitter — same core as the GUI, no Tkinter required.

Usage:
  python split_audio_cli.py --folder "D:/Samples/Violin"
  python split_audio_cli.py --folder . --preset "Legato / Bow" --export-metadata
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import audio_segment_core as core

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _preset_fade_ms(name: str) -> float:
    preset = core.ALL_PRESETS.get(name, {})
    return float(preset.get("fade_ms", 50.0))


def build_config(args: argparse.Namespace) -> core.SegmentConfig:
    cfg = core.SegmentConfig.from_preset(args.preset)
    if args.attack_threshold is not None:
        cfg.attack_threshold = args.attack_threshold
    if args.decay_threshold is not None:
        cfg.decay_threshold = args.decay_threshold
    if args.min_sustain is not None:
        cfg.min_sustain_duration = args.min_sustain
    if args.advanced:
        cfg.use_advanced = True
        cfg.use_smart = False
    elif args.proportional:
        cfg.use_advanced = False
        cfg.use_smart = False
    else:
        cfg.use_advanced = False
        cfg.use_smart = True
    if args.no_pitch_refine:
        cfg.use_pitch_refine = False
    if args.pitch_refine_mode:
        cfg.pitch_refine_mode = args.pitch_refine_mode
    if args.no_vibrato_robust:
        cfg.vibrato_robust = False
    return cfg


def export_metadata(folder: Path, results: List[Dict[str, Any]], cfg: core.SegmentConfig, fade_ms: float, fade_type: str) -> None:
    json_path = folder / "segmentation_metadata.json"
    payload = {
        "export_date": datetime.now().isoformat(),
        "parameters": {
            "attack_threshold": cfg.attack_threshold,
            "decay_threshold": cfg.decay_threshold,
            "attack_pct": cfg.attack_pct,
            "sustain_pct": cfg.sustain_pct,
            "decay_pct": cfg.decay_pct,
            "min_sustain_duration": cfg.min_sustain_duration,
            "use_smart": cfg.use_smart,
            "use_advanced": cfg.use_advanced,
            "vibrato_robust": cfg.vibrato_robust,
            "fade_ms": fade_ms,
            "fade_type": fade_type,
        },
        "files": [],
    }
    for info in results:
        if "error" in info:
            payload["files"].append({"file_path": info["file_path"], "error": info["error"]})
            continue
        payload["files"].append(
            {
                "file_path": info["file_path"],
                "sample_rate": info["sr"],
                "detection_mode": info.get("detection_mode"),
                "segments": {
                    "attack_end": info["t_att"],
                    "decay_start": info["t_dec"],
                    "end": info["t_end"],
                    "durations": {
                        "attack": info["dur_att"],
                        "sustain": info["dur_sus"],
                        "decay": info["dur_dec"],
                        "release": info["dur_rel"],
                    },
                    "pitch_stability": info.get("pitch_refine", {}),
                },
            }
        )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    csv_path = folder / "segmentation_metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "File", "Sample Rate", "Mode", "Attack End (s)", "Decay Start (s)", "End (s)",
            "Attack Dur (s)", "Sustain Dur (s)", "Decay Dur (s)", "Release Dur (s)",
            "Pitch Stable Used", "Pitch Std (cents)",
        ])
        for info in results:
            if "error" in info:
                writer.writerow([Path(info["file_path"]).name, "", "", "", "", "", "", "", "", "", "", info["error"]])
                continue
            pitch = info.get("pitch_refine") or {}
            writer.writerow([
                Path(info["file_path"]).name,
                info["sr"],
                info.get("detection_mode", ""),
                f"{info['t_att']:.4f}",
                f"{info['t_dec']:.4f}",
                f"{info['t_end']:.4f}",
                f"{info['dur_att']:.4f}",
                f"{info['dur_sus']:.4f}",
                f"{info['dur_dec']:.4f}",
                f"{info['dur_rel']:.4f}",
                pitch.get("used", False),
                "" if pitch.get("std_cents") is None else f"{pitch['std_cents']:.4f}",
            ])
    logger.info("Metadata: %s, %s", json_path.name, csv_path.name)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch ADSR audio segment splitter")
    parser.add_argument("--folder", "-f", type=Path, required=True, help="Folder with audio files")
    parser.add_argument(
        "--preset", "-p", default="Medium (1.5-3.0s)",
        choices=sorted(core.ALL_PRESETS.keys()),
        help="Duration or articulation preset",
    )
    parser.add_argument("--fade-ms", type=float, default=None, help="Fade length in ms (default: from preset)")
    parser.add_argument("--fade-type", choices=["cosine", "hann", "linear"], default="cosine")
    parser.add_argument("--attack-threshold", type=float, default=None)
    parser.add_argument("--decay-threshold", type=float, default=None)
    parser.add_argument("--min-sustain", type=float, default=None, help="Minimum sustain duration (s)")
    parser.add_argument("--advanced", action="store_true", help="Derivative + spectral flux mode")
    parser.add_argument("--proportional", action="store_true", help="Proportional-only mode")
    parser.add_argument("--no-vibrato-robust", action="store_true", help="Disable vibrato-aware pitch scoring")
    parser.add_argument("--no-pitch-refine", action="store_true", help="Disable pitch-based sustain refinement")
    parser.add_argument(
        "--pitch-refine-mode",
        choices=["expand", "annotate", "crop"],
        default=None,
        help="expand= grow stable region (default); annotate= full energy sustain + metadata; crop= tight window",
    )
    parser.add_argument("--export-metadata", action="store_true", help="Write JSON/CSV metadata")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output directory (default: same as folder)")
    args = parser.parse_args(argv)

    folder = args.folder.resolve()
    if not folder.is_dir():
        logger.error("Folder not found: %s", folder)
        return 1

    files = core.list_audio_files(folder)
    if not files:
        logger.error("No audio files in %s", folder)
        return 1

    cfg = build_config(args)
    fade_ms = args.fade_ms if args.fade_ms is not None else _preset_fade_ms(args.preset)
    out_dir = (args.output or folder).resolve()

    logger.info("Processing %d file(s) | preset=%s | fade=%.0fms %s", len(files), args.preset, fade_ms, args.fade_type)
    results = core.batch_process_folder(folder, cfg, fade_ms, args.fade_type, out_dir)

    ok = sum(1 for r in results if "error" not in r)
    logger.info("Done: %d/%d succeeded", ok, len(results))
    for r in results:
        if "error" in r:
            logger.warning("  FAIL %s: %s", Path(r["file_path"]).name, r["error"])
        else:
            logger.info(
                "  OK   %s | Att %.2fs Sus %.2fs Dec %.2fs",
                Path(r["file_path"]).name, r["dur_att"], r["dur_sus"], r["dur_dec"],
            )

    if args.export_metadata:
        export_metadata(out_dir, results, cfg, fade_ms, args.fade_type)

    return 0 if ok == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
