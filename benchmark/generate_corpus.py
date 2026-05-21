"""
Generate a reproducible synthetic benchmark corpus (~40 one-shots) with known ADSR boundaries.

WAV files + annotations.json are written to benchmark/corpus/ by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf

import audio_segment_core as core

SR = 22050

# (attack_s, sustain_s, decay_s, gap_s, freq_hz, preset, category, vibrato_rate, vibrato_depth_cents)
SYNTHETIC_SPECS: List[Tuple[float, float, float, float, float, str, str, float, float]] = [
    # Very Short (< 0.5 s active) — 8 samples
    (0.02, 0.08, 0.04, 0.05, 880.0, "Very Short (< 0.5s)", "pluck", 0.0, 0.0),
    (0.03, 0.10, 0.05, 0.06, 660.0, "Very Short (< 0.5s)", "pluck", 0.0, 0.0),
    (0.025, 0.09, 0.045, 0.04, 988.0, "Very Short (< 0.5s)", "pluck", 0.0, 0.0),
    (0.02, 0.07, 0.05, 0.05, 523.0, "Very Short (< 0.5s)", "pluck", 0.0, 0.0),
    (0.035, 0.12, 0.06, 0.05, 740.0, "Very Short (< 0.5s)", "marcato", 0.0, 0.0),
    (0.03, 0.11, 0.05, 0.06, 440.0, "Very Short (< 0.5s)", "pluck", 5.5, 12.0),
    (0.025, 0.09, 0.04, 0.05, 1175.0, "Very Short (< 0.5s)", "pluck", 0.0, 0.0),
    (0.04, 0.10, 0.06, 0.04, 554.0, "Very Short (< 0.5s)", "marcato", 0.0, 0.0),
    # Short (0.5–1.5 s) — 8 samples
    (0.05, 0.45, 0.18, 0.10, 440.0, "Short (0.5-1.5s)", "bow", 0.0, 0.0),
    (0.06, 0.55, 0.20, 0.12, 392.0, "Short (0.5-1.5s)", "bow", 5.0, 18.0),
    (0.07, 0.50, 0.22, 0.10, 494.0, "Short (0.5-1.5s)", "bow", 0.0, 0.0),
    (0.05, 0.60, 0.15, 0.08, 330.0, "Short (0.5-1.5s)", "bow", 6.0, 22.0),
    (0.08, 0.48, 0.20, 0.12, 587.0, "Short (0.5-1.5s)", "bow", 0.0, 0.0),
    (0.06, 0.52, 0.18, 0.10, 698.0, "Short (0.5-1.5s)", "bow", 5.5, 15.0),
    (0.09, 0.42, 0.24, 0.11, 262.0, "Short (0.5-1.5s)", "bow", 0.0, 0.0),
    (0.07, 0.58, 0.16, 0.09, 440.0, "Short (0.5-1.5s)", "bow", 0.0, 0.0),
    # Medium (1.5–3.0 s) — 8 samples
    (0.08, 1.20, 0.35, 0.15, 440.0, "Medium (1.5-3.0s)", "bow", 5.5, 20.0),
    (0.10, 1.35, 0.30, 0.18, 392.0, "Medium (1.5-3.0s)", "bow", 0.0, 0.0),
    (0.09, 1.10, 0.40, 0.16, 523.0, "Medium (1.5-3.0s)", "bow", 6.0, 25.0),
    (0.12, 1.25, 0.32, 0.14, 349.0, "Medium (1.5-3.0s)", "bow", 0.0, 0.0),
    (0.08, 1.40, 0.28, 0.15, 587.0, "Medium (1.5-3.0s)", "bow", 5.0, 18.0),
    (0.11, 1.15, 0.38, 0.17, 440.0, "Medium (1.5-3.0s)", "bow", 0.0, 0.0),
    (0.09, 1.30, 0.33, 0.15, 466.0, "Medium (1.5-3.0s)", "bow", 5.8, 22.0),
    (0.10, 1.22, 0.36, 0.16, 415.0, "Medium (1.5-3.0s)", "bow", 0.0, 0.0),
    # Long (3.0–6.0 s) — 8 samples
    (0.15, 2.80, 0.55, 0.20, 440.0, "Long (3.0-6.0s)", "bow", 5.5, 20.0),
    (0.18, 2.60, 0.60, 0.22, 392.0, "Long (3.0-6.0s)", "bow", 0.0, 0.0),
    (0.12, 3.00, 0.50, 0.18, 523.0, "Long (3.0-6.0s)", "bow", 6.0, 18.0),
    (0.20, 2.50, 0.65, 0.25, 330.0, "Long (3.0-6.0s)", "bow", 0.0, 0.0),
    (0.16, 2.70, 0.58, 0.20, 587.0, "Long (3.0-6.0s)", "bow", 5.2, 24.0),
    (0.14, 2.90, 0.52, 0.22, 440.0, "Long (3.0-6.0s)", "bow", 0.0, 0.0),
    (0.17, 2.65, 0.62, 0.21, 349.0, "Long (3.0-6.0s)", "bow", 5.5, 16.0),
    (0.13, 2.85, 0.54, 0.19, 494.0, "Long (3.0-6.0s)", "bow", 0.0, 0.0),
    # Very Long (> 6.0 s) — 8 samples
    (0.25, 4.50, 0.70, 0.25, 440.0, "Very Long (> 6.0s)", "bow", 5.5, 22.0),
    (0.30, 4.20, 0.80, 0.28, 392.0, "Very Long (> 6.0s)", "bow", 0.0, 0.0),
    (0.22, 4.80, 0.65, 0.24, 523.0, "Very Long (> 6.0s)", "bow", 6.0, 20.0),
    (0.28, 4.40, 0.75, 0.26, 330.0, "Very Long (> 6.0s)", "bow", 0.0, 0.0),
    (0.26, 4.60, 0.68, 0.25, 587.0, "Very Long (> 6.0s)", "bow", 5.0, 18.0),
    (0.24, 4.70, 0.72, 0.27, 440.0, "Very Long (> 6.0s)", "bow", 0.0, 0.0),
    (0.32, 4.10, 0.82, 0.30, 262.0, "Very Long (> 6.0s)", "bow", 5.8, 25.0),
    (0.27, 4.55, 0.69, 0.26, 466.0, "Very Long (> 6.0s)", "bow", 0.0, 0.0),
]


def _synthesize(
    sr: int,
    attack_s: float,
    sustain_s: float,
    decay_s: float,
    gap_s: float,
    freq: float,
    vib_rate: float = 0.0,
    vib_depth_cents: float = 0.0,
    attack_noise: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    n_gap = int(gap_s * sr)
    n_att = int(attack_s * sr)
    n_sus = int(sustain_s * sr)
    n_dec = int(decay_s * sr)

    att = np.linspace(0, 1, n_att, endpoint=False) if n_att else np.array([])
    sus = np.ones(n_sus)
    dec = np.linspace(1, 0, n_dec, endpoint=True) if n_dec else np.array([])
    env = np.concatenate([x for x in (att, sus, dec) if len(x)])
    t_body = np.arange(len(env)) / sr

    if vib_rate > 0 and vib_depth_cents > 0:
        freq_t = freq * (2 ** ((vib_depth_cents / 1200.0) * np.sin(2 * np.pi * vib_rate * t_body)))
        phase = np.cumsum(2 * np.pi * freq_t / sr)
        tone = 0.45 * np.sin(phase) * env
    else:
        tone = 0.45 * np.sin(2 * np.pi * freq * t_body) * env

    if attack_noise > 0 and n_att > 0:
        noise = attack_noise * np.random.randn(n_att)
        noise *= np.linspace(1, 0, n_att)
        tone[:n_att] += noise

    y = np.concatenate([np.zeros(n_gap), tone, np.zeros(n_gap)])
    t_start = gap_s
    t_att = t_start + attack_s
    t_dec = t_att + sustain_s
    t_end = t_dec + decay_s
    return y, {"t_start": t_start, "t_att": t_att, "t_dec": t_dec, "t_end": t_end}


def generate_corpus(output_dir: Path, seed: int = 42) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples: List[Dict[str, Any]] = []

    for idx, spec in enumerate(SYNTHETIC_SPECS, start=1):
        att, sus, dec, gap, freq, preset, category, vib_r, vib_d = spec
        noise = 0.04 if category in ("marcato", "pluck") and idx % 2 == 0 else 0.0
        y, times = _synthesize(
            SR, att, sus, dec, gap, freq, vib_r, vib_d,
            attack_noise=noise if noise else 0.02 * rng.random(),
        )
        y = core.preprocess_signal(y, remove_dc=True)
        _, trim = core.trim_active_region(y, SR, core.DEFAULT_TRIM_DB)
        # Align t_end with trim (same reference frame as detect_segments)
        t_att = times["t_att"]
        t_dec = times["t_dec"]
        t_end = trim.t_end
        name = f"syn_{idx:03d}.wav"
        sf.write(output_dir / name, y, SR)
        samples.append(
            {
                "id": f"syn_{idx:03d}",
                "file": name,
                "sr": SR,
                "t_att": round(t_att, 6),
                "t_dec": round(t_dec, 6),
                "t_end": round(t_end, 6),
                "preset": preset,
                "category": category,
                "notes": f"freq={freq}Hz att={att}s sus={sus}s dec={dec}s",
            }
        )

    ann_path = output_dir / "annotations.json"
    payload = {
        "version": 1,
        "description": "Synthetic ADSR benchmark corpus (40 one-shots, reproducible)",
        "sr_default": SR,
        "samples": samples,
    }
    ann_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ann_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "corpus"
    path = generate_corpus(root)
    print(f"Generated {len(SYNTHETIC_SPECS)} samples -> {path}")
