"""
Pure audio segmentation logic (no GUI).
Times are trim-relative inside detection, converted to absolute file times at the end.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TRIM_DB = 60.0
DEFAULT_FRAME_LENGTH = 1024
DEFAULT_HOP_LENGTH = 512
DEFAULT_MIN_SUSTAIN_FRAMES = 40
DEFAULT_SUSTAIN_VARIANCE_THRESHOLD = 0.2
DEFAULT_ZERO_CROSSING_SEARCH_MS = 100.0
SMART_ENERGY_BLEND = 0.7
SMART_PROP_BLEND = 0.3
DEFAULT_VIBRATO_MEDIAN_WINDOW_S = 0.12
DEFAULT_PITCH_REFINE_MIN_FRACTION = 0.70
DEFAULT_SUSTAIN_FRACTION_BEFORE_DECAY = 0.75  # min % through proportional sustain before decay

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg", ".m4a", ".wma", ".mp4", ".mka",
}

OUTPUT_FOLDERS = (
    "_Attacks",
    "_Sustains",
    "_Decays",
    "_Release_Silence",
    "_Full_Active_Sound",
)

PRESETS = {
    "Very Short (< 0.5s)": {
        "attack_pct": 0.20,
        "sustain_pct": 0.50,
        "decay_pct": 0.30,
        "fade_ms": 30.0,
        "min_sustain_duration": 0.06,
        "attack_threshold": 0.85,
        "decay_threshold": 0.45,
    },
    "Short (0.5-1.5s)": {
        "attack_pct": 0.15,
        "sustain_pct": 0.60,
        "decay_pct": 0.25,
        "fade_ms": 40.0,
        "min_sustain_duration": 0.15,
        "attack_threshold": 0.90,
        "decay_threshold": 0.50,
    },
    "Medium (1.5-3.0s)": {
        "attack_pct": 0.12,
        "sustain_pct": 0.65,
        "decay_pct": 0.23,
        "fade_ms": 50.0,
        "min_sustain_duration": 0.35,
        "attack_threshold": 0.90,
        "decay_threshold": 0.50,
    },
    "Long (3.0-6.0s)": {
        "attack_pct": 0.10,
        "sustain_pct": 0.70,
        "decay_pct": 0.20,
        "fade_ms": 60.0,
        "min_sustain_duration": 0.60,
        "attack_threshold": 0.90,
        "decay_threshold": 0.45,
        "pitch_refine_mode": "expand",
        "pitch_refine_min_fraction": 0.72,
    },
    "Very Long (> 6.0s)": {
        "attack_pct": 0.08,
        "sustain_pct": 0.75,
        "decay_pct": 0.17,
        "fade_ms": 70.0,
        "min_sustain_duration": 1.00,
        "attack_threshold": 0.90,
        "decay_threshold": 0.40,
        "pitch_refine_mode": "expand",
        "pitch_refine_min_fraction": 0.75,
    },
    "Custom": {
        "attack_pct": 0.15,
        "sustain_pct": 0.60,
        "decay_pct": 0.25,
        "fade_ms": 50.0,
        "min_sustain_duration": 0.35,
        "attack_threshold": 0.90,
        "decay_threshold": 0.50,
    },
}

# Orchestral articulation profiles (same ADSR objective, tuned thresholds)
ARTICULATION_PRESETS = {
    "Staccato / Pluck": {
        "attack_pct": 0.22,
        "sustain_pct": 0.45,
        "decay_pct": 0.33,
        "fade_ms": 25.0,
        "min_sustain_duration": 0.04,
        "attack_threshold": 0.82,
        "decay_threshold": 0.55,
        "use_advanced": True,
        "use_smart": False,
    },
    "Legato / Bow": {
        "attack_pct": 0.10,
        "sustain_pct": 0.72,
        "decay_pct": 0.18,
        "fade_ms": 55.0,
        "min_sustain_duration": 0.45,
        "attack_threshold": 0.88,
        "decay_threshold": 0.42,
        "use_advanced": False,
        "use_smart": True,
        "pitch_stability_cents": 8.0,
    },
    "Marcato / Accent": {
        "attack_pct": 0.18,
        "sustain_pct": 0.52,
        "decay_pct": 0.30,
        "fade_ms": 35.0,
        "min_sustain_duration": 0.12,
        "attack_threshold": 0.80,
        "decay_threshold": 0.48,
        "use_advanced": True,
        "use_smart": False,
    },
}

ALL_PRESETS = {**PRESETS, **ARTICULATION_PRESETS}


@dataclass
class SegmentConfig:
    trim_db: float = DEFAULT_TRIM_DB
    attack_threshold: float = 0.9
    decay_threshold: float = 0.5
    attack_pct: float = 0.15
    sustain_pct: float = 0.60
    decay_pct: float = 0.25
    min_sustain_duration: float = 0.35
    pitch_window_duration: float = 0.5
    pitch_stability_cents: float = 5.0
    use_advanced: bool = False
    use_smart: bool = True
    sustain_variance_threshold: float = DEFAULT_SUSTAIN_VARIANCE_THRESHOLD
    frame_length: int = DEFAULT_FRAME_LENGTH
    hop_length: int = DEFAULT_HOP_LENGTH
    min_sustain_frames: int = DEFAULT_MIN_SUSTAIN_FRAMES
    vibrato_robust: bool = True
    vibrato_median_window_s: float = DEFAULT_VIBRATO_MEDIAN_WINDOW_S
    remove_dc: bool = True
    use_pitch_refine: bool = True
    # annotate = keep energy sustain, record stable window in metadata only (best for STFT)
    # expand   = grow stable seed outward (default; keeps long sustains for spectral work)
    # crop     = tightest stable window only (legacy sampler-style)
    pitch_refine_mode: str = "expand"
    pitch_refine_min_fraction: float = DEFAULT_PITCH_REFINE_MIN_FRACTION
    sustain_fraction_before_decay: float = DEFAULT_SUSTAIN_FRACTION_BEFORE_DECAY

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "SegmentConfig":
        """Build config from a named preset with optional field overrides."""
        preset = ALL_PRESETS.get(name)
        if preset is None:
            raise ValueError(f"Unknown preset: {name!r}")
        fields = {
            "attack_pct": preset.get("attack_pct", 0.15),
            "sustain_pct": preset.get("sustain_pct", 0.60),
            "decay_pct": preset.get("decay_pct", 0.25),
            "min_sustain_duration": preset.get("min_sustain_duration", 0.35),
            "attack_threshold": preset.get("attack_threshold", 0.9),
            "decay_threshold": preset.get("decay_threshold", 0.5),
            "use_advanced": preset.get("use_advanced", False),
            "use_smart": preset.get("use_smart", True),
            "pitch_stability_cents": preset.get("pitch_stability_cents", 5.0),
            "pitch_refine_mode": preset.get("pitch_refine_mode", "expand"),
            "pitch_refine_min_fraction": preset.get(
                "pitch_refine_min_fraction", DEFAULT_PITCH_REFINE_MIN_FRACTION
            ),
        }
        fields.update(overrides)
        return cls(**fields)


@dataclass
class TrimInfo:
    idx_start: int
    idx_end: int
    t_start: float
    t_end: float
    active_len: float


@dataclass
class SegmentResult:
    t_att: float
    t_dec: float
    t_end: float
    trim: TrimInfo
    pitch_refine: Dict = field(default_factory=dict)


def preprocess_signal(y: np.ndarray, remove_dc: bool = True) -> np.ndarray:
    """Optional DC removal before envelope analysis."""
    if not remove_dc or len(y) == 0:
        return y
    return y - float(np.mean(y))


def trim_active_region(y: np.ndarray, sr: int, trim_db: float = DEFAULT_TRIM_DB) -> Tuple[np.ndarray, TrimInfo]:
    y_trimmed, index = librosa.effects.trim(y, top_db=trim_db)
    idx_start, idx_end = int(index[0]), int(index[1])
    t_start = idx_start / sr
    t_end_trimmed = idx_end / sr
    t_end_signal = len(y) / sr
    t_end = min(t_end_trimmed, t_end_signal - 1e-3)
    active_len = max(0.0, t_end - t_start)
    return y_trimmed, TrimInfo(idx_start, idx_end, t_start, t_end, active_len)


def compute_rms_envelope(
    y: np.ndarray, sr: int, frame_length: int = DEFAULT_FRAME_LENGTH, hop_length: int = DEFAULT_HOP_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    return rms, times


def compute_spectral_flux(
    y: np.ndarray, sr: int, frame_length: int = DEFAULT_FRAME_LENGTH, hop_length: int = DEFAULT_HOP_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(stft)
    diff = np.diff(magnitude, axis=1)
    flux = np.sum(np.maximum(diff, 0.0), axis=0)
    times = librosa.times_like(flux, sr=sr, hop_length=hop_length)
    return flux, times


def _moving_median(arr: np.ndarray, win: int) -> np.ndarray:
    """Odd-length moving median (used to suppress vibrato in pitch stability scoring)."""
    win = max(3, win | 1)
    half = win // 2
    out = np.empty_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        out[i] = float(np.median(arr[lo:hi]))
    return out


def pitch_stability_std_cents(
    cents: np.ndarray,
    times: np.ndarray,
    cfg: SegmentConfig,
) -> Optional[float]:
    """
    Pitch stability in cents after linear detrend and optional vibrato suppression.
    Vibrato (~4–7 Hz on strings) is attenuated via a short moving-median residual.
    """
    valid = np.isfinite(cents)
    if valid.sum() < 3:
        return None
    c = cents[valid].astype(np.float64)
    t = times[valid].astype(np.float64)
    if len(c) >= 3:
        slope, intercept = np.polyfit(t - t[0], c, 1)
        c = c - np.polyval([slope, intercept], t - t[0])
    if cfg.vibrato_robust and len(c) >= 5:
        dt = float(np.median(np.diff(t))) if len(t) > 1 else cfg.vibrato_median_window_s
        if dt > 0:
            win = max(3, int(round(cfg.vibrato_median_window_s / dt)) | 1)
            if len(c) >= win:
                c = c - _moving_median(c, win)
    return float(np.std(c))


def detect_attack_energy(
    rms: np.ndarray, times: np.ndarray, peak_idx: int, threshold: float
) -> float:
    """Attack end: first frame at/above threshold * peak while ascending toward peak."""
    if len(rms) < 2:
        return float(times[0])
    peak_val = float(rms[peak_idx])
    if peak_val < 1e-12:
        return float(times[0])
    level = threshold * peak_val
    search_end = max(1, peak_idx + 1)
    for i in range(search_end):
        if rms[i] >= level:
            return float(times[i])
    return float(times[min(peak_idx, len(times) - 1)])


def detect_attack_derivative(
    rms: np.ndarray,
    times: np.ndarray,
    peak_idx: int,
    spectral_flux: Optional[np.ndarray] = None,
    flux_times: Optional[np.ndarray] = None,
) -> float:
    if len(rms) < 2:
        return float(times[0])
    drms = np.diff(rms)
    search_end = max(1, min(peak_idx, len(drms)))
    max_derivative_idx = int(np.argmax(drms[:search_end]))
    attack_time = float(times[max_derivative_idx])
    if spectral_flux is not None and flux_times is not None and len(spectral_flux) > 0:
        flux_peak_time = float(flux_times[int(np.argmax(spectral_flux))])
        attack_time = min(attack_time, flux_peak_time)
    min_attack_time = float(times[int(len(times) * 0.05)])
    attack_time = max(min_attack_time, attack_time)
    peak_time = float(times[peak_idx])
    attack_time = min(attack_time, peak_time * 0.85)
    return attack_time


def detect_attack_combined(
    rms: np.ndarray,
    times: np.ndarray,
    peak_idx: int,
    threshold: float,
    spectral_flux: Optional[np.ndarray] = None,
    flux_times: Optional[np.ndarray] = None,
    use_derivative: bool = False,
) -> float:
    energy_t = detect_attack_energy(rms, times, peak_idx, threshold)
    if not use_derivative:
        return energy_t
    deriv_t = detect_attack_derivative(rms, times, peak_idx, spectral_flux, flux_times)
    return min(energy_t, deriv_t)


def _normalized_adsr_fractions(
    attack_pct: float, sustain_pct: float, decay_pct: float
) -> Tuple[float, float, float]:
    total = attack_pct + sustain_pct + decay_pct
    if total <= 0:
        return 0.15, 0.60, 0.25
    return attack_pct / total, sustain_pct / total, decay_pct / total


def min_decay_time_proportional(
    active_len: float, attack_pct: float, sustain_pct: float, decay_pct: float, sustain_fraction: float
) -> float:
    """Earliest allowed decay start: after sustain_fraction of the proportional sustain zone."""
    att_f, sus_f, _ = _normalized_adsr_fractions(attack_pct, sustain_pct, decay_pct)
    return active_len * (att_f + sus_f * sustain_fraction)


def detect_decay_energy(
    rms: np.ndarray,
    times: np.ndarray,
    attack_idx: int,
    peak_idx: int,
    threshold: float,
    min_decay_time: Optional[float] = None,
) -> float:
    peak_val = float(np.max(rms))
    if peak_val < 1e-12:
        return float(times[-1])
    level = threshold * peak_val
    search_start = max(attack_idx, peak_idx)
    if min_decay_time is not None:
        search_start = max(
            search_start,
            int(np.searchsorted(times, min_decay_time, side="left")),
        )
    for i in range(search_start, len(rms)):
        if rms[i] <= level:
            return float(times[i])
    return float(times[int(len(times) * 0.85)])


def detect_decay_derivative(
    rms: np.ndarray, times: np.ndarray, attack_idx: int, peak_idx: int, threshold: float
) -> float:
    if len(rms) < 2:
        return float(times[-1])
    drms = np.diff(rms)
    search_start = max(attack_idx, peak_idx)
    if peak_idx < len(times):
        peak_time = float(times[peak_idx])
        min_decay_delay = max(0.05, (times[-1] - peak_time) * 0.15)
        min_decay_time = peak_time + min_decay_delay
        search_start = max(search_start, int(np.searchsorted(times, min_decay_time, side="right")))
    search_start = min(search_start, len(drms) - 1)
    negative_count = 0
    for i in range(search_start, len(drms)):
        if drms[i] < 0:
            negative_count += 1
            if negative_count >= 3:
                return float(times[i - 2])
        else:
            negative_count = 0
    return detect_decay_energy(rms, times, attack_idx, peak_idx, threshold)


def detect_segments_proportional(
    active_len: float,
    attack_pct: float,
    sustain_pct: float,
    decay_pct: float,
    min_sustain_duration: float,
) -> Tuple[float, float]:
    total = attack_pct + sustain_pct + decay_pct
    if total > 0:
        attack_pct /= total
        sustain_pct /= total
        decay_pct /= total
    t_attack_end = active_len * attack_pct
    t_decay_start = active_len * (attack_pct + sustain_pct)
    min_sustain_actual = max(min_sustain_duration, active_len * sustain_pct * 0.4)
    t_decay_start = max(t_attack_end + min_sustain_actual, t_decay_start)
    margin = active_len * 0.05
    t_decay_start = min(t_decay_start, active_len - margin)
    if t_decay_start <= t_attack_end:
        t_decay_start = min(t_attack_end + active_len * sustain_pct, active_len - 0.02)
    return t_attack_end, t_decay_start


def detect_sustain_plateau(
    rms: np.ndarray,
    times: np.ndarray,
    attack_idx: int,
    decay_idx: int,
    min_duration: float,
    variance_threshold: float = DEFAULT_SUSTAIN_VARIANCE_THRESHOLD,
) -> Tuple[Optional[int], Optional[int]]:
    if decay_idx <= attack_idx + 1:
        return None, None
    sustain_rms = rms[attack_idx:decay_idx]
    if len(sustain_rms) < 2:
        return None, None
    mean_rms = np.mean(sustain_rms)
    if mean_rms < 1e-10:
        return None, None
    variance = np.var(sustain_rms) / (mean_rms ** 2)
    duration = times[decay_idx] - times[attack_idx]
    if variance < variance_threshold and duration >= min_duration:
        return attack_idx, decay_idx
    return None, None


def effective_min_sustain_duration(
    cfg: SegmentConfig, sr: int, active_len: Optional[float] = None
) -> float:
    min_by_frames = (cfg.min_sustain_frames * cfg.hop_length) / max(float(sr), 1.0)
    min_required = max(cfg.min_sustain_duration, cfg.pitch_window_duration, min_by_frames)
    if active_len is not None and 0 < active_len < min_required:
        min_required = max(active_len * 0.25, min_by_frames, 0.02)
    return min_required


def parse_note_hz_from_filename(path: Optional[Path]) -> Optional[float]:
    if path is None:
        return None
    match = re.search(r"([A-Ga-g])(#|b)?(\d+)", path.stem)
    if not match:
        return None
    letter, accidental, octave = match.group(1).upper(), match.group(2), match.group(3)
    note_str = letter + (accidental or "") + octave
    try:
        return float(librosa.note_to_hz(note_str))
    except Exception:
        return None


def _expand_stable_pitch_window(
    cents: np.ndarray,
    times: np.ndarray,
    seed_lo: int,
    seed_hi: int,
    cfg: SegmentConfig,
) -> Tuple[int, int]:
    """Grow a pitch-stable seed window outward while stability stays within tolerance."""
    lo, hi = seed_lo, seed_hi
    threshold = cfg.pitch_stability_cents * 1.25

    while lo > 0:
        std = pitch_stability_std_cents(cents[lo - 1 : hi], times[lo - 1 : hi], cfg)
        if std is None or std > threshold:
            break
        lo -= 1

    while hi < len(cents):
        std = pitch_stability_std_cents(cents[lo : hi + 1], times[lo : hi + 1], cfg)
        if std is None or std > threshold:
            break
        hi += 1

    return lo, hi


def refine_sustain_by_pitch(
    y_trimmed: np.ndarray,
    sr: int,
    t_att_rel: float,
    t_dec_rel: float,
    cfg: SegmentConfig,
    expected_note_hz: Optional[float] = None,
) -> Tuple[float, float, Dict]:
    info: Dict = {
        "used": False,
        "std_cents": None,
        "window_start": None,
        "window_end": None,
        "window_duration": None,
        "expected_note_hz": expected_note_hz,
        "mean_abs_cents_from_note": None,
        "mode": cfg.pitch_refine_mode,
        "energy_sustain_duration": None,
        "kept_energy_boundaries": False,
    }
    if not cfg.use_pitch_refine:
        info["kept_energy_boundaries"] = True
        return t_att_rel, t_dec_rel, info

    energy_att, energy_dec = t_att_rel, t_dec_rel
    energy_sustain_dur = max(0.0, energy_dec - energy_att)
    info["energy_sustain_duration"] = energy_sustain_dur

    min_duration = effective_min_sustain_duration(cfg, sr, len(y_trimmed) / sr)
    # Analysis grain for seed search — not the export length in expand mode
    window_duration = max(cfg.pitch_window_duration, min(0.5, min_duration))
    total_len = len(y_trimmed) / sr
    sustain_start = max(0.0, min(t_att_rel, total_len))
    sustain_end = max(sustain_start, min(t_dec_rel, total_len))
    if sustain_end - sustain_start < min_duration:
        info["kept_energy_boundaries"] = True
        return t_att_rel, t_dec_rel, info

    start_idx = int(sustain_start * sr)
    end_idx = int(sustain_end * sr)
    y_sustain = y_trimmed[start_idx:end_idx]
    if len(y_sustain) < cfg.frame_length:
        info["kept_energy_boundaries"] = True
        return t_att_rel, t_dec_rel, info

    try:
        f0 = librosa.yin(
            y_sustain,
            fmin=librosa.note_to_hz("A0"),
            fmax=librosa.note_to_hz("C8"),
            sr=sr,
            frame_length=cfg.frame_length,
            hop_length=cfg.hop_length,
        )
    except Exception:
        info["kept_energy_boundaries"] = True
        return t_att_rel, t_dec_rel, info

    times = librosa.times_like(f0, sr=sr, hop_length=cfg.hop_length)
    valid = np.isfinite(f0) & (f0 > 0)
    if valid.sum() < 3:
        info["kept_energy_boundaries"] = True
        return t_att_rel, t_dec_rel, info

    f0_med = float(np.median(f0[valid]))
    if f0_med <= 0:
        info["kept_energy_boundaries"] = True
        return t_att_rel, t_dec_rel, info

    cents = np.full_like(f0, np.nan, dtype=np.float64)
    cents[valid] = 1200.0 * np.log2(f0[valid] / f0_med)
    cents_from_note = np.full_like(f0, np.nan, dtype=np.float64)
    if expected_note_hz is not None and expected_note_hz > 0:
        cents_from_note[valid] = 1200.0 * np.log2(f0[valid] / expected_note_hz)

    window_frames = max(1, int(np.ceil(window_duration * sr / cfg.hop_length)))
    best_std = None
    best_start = None
    best_mean_abs_from_note = None
    best_score = None

    for i in range(0, len(cents) - window_frames + 1):
        window = cents[i : i + window_frames]
        w_times = times[i : i + window_frames]
        w_valid = np.isfinite(window)
        if w_valid.sum() < max(3, int(0.6 * window_frames)):
            continue
        std = pitch_stability_std_cents(window, w_times, cfg)
        if std is None:
            continue
        mean_abs_from_note = 0.0
        if expected_note_hz is not None:
            w_note = np.isfinite(cents_from_note[i : i + window_frames])
            if w_note.sum() >= max(3, int(0.6 * window_frames)):
                mean_abs_from_note = float(
                    np.mean(np.abs(cents_from_note[i : i + window_frames][w_note]))
                )
        score = std + mean_abs_from_note
        if best_score is None or score < best_score:
            best_score = score
            best_std = std
            best_start = i
            best_mean_abs_from_note = mean_abs_from_note if expected_note_hz else None

    if best_std is None or best_start is None:
        info["kept_energy_boundaries"] = True
        return t_att_rel, t_dec_rel, info

    seed_lo = best_start
    seed_hi = best_start + window_frames

    if best_std <= cfg.pitch_stability_cents:
        if cfg.pitch_refine_mode == "expand":
            seed_lo, seed_hi = _expand_stable_pitch_window(cents, times, seed_lo, seed_hi, cfg)
        elif cfg.pitch_refine_mode == "crop":
            pass  # keep seed window only

        win_start_t = sustain_start + float(times[seed_lo])
        win_end_t = sustain_start + float(times[min(seed_hi - 1, len(times) - 1)])
        win_end_t = min(max(win_end_t, win_start_t + min_duration), sustain_end)

        refined_dur = win_end_t - win_start_t
        min_allowed = energy_sustain_dur * cfg.pitch_refine_min_fraction

        if refined_dur < min_allowed:
            info.update(
                {
                    "std_cents": best_std,
                    "window_start": win_start_t,
                    "window_end": win_end_t,
                    "window_duration": refined_dur,
                    "mean_abs_cents_from_note": best_mean_abs_from_note,
                    "kept_energy_boundaries": True,
                }
            )
            return energy_att, energy_dec, info

        if cfg.pitch_refine_mode == "annotate":
            info.update(
                {
                    "used": True,
                    "std_cents": pitch_stability_std_cents(
                        cents[seed_lo:seed_hi], times[seed_lo:seed_hi], cfg
                    ),
                    "window_start": win_start_t,
                    "window_end": win_end_t,
                    "window_duration": refined_dur,
                    "mean_abs_cents_from_note": best_mean_abs_from_note,
                    "kept_energy_boundaries": True,
                }
            )
            return energy_att, energy_dec, info

        info.update(
            {
                "used": True,
                "std_cents": pitch_stability_std_cents(
                    cents[seed_lo:seed_hi], times[seed_lo:seed_hi], cfg
                ),
                "window_start": win_start_t,
                "window_end": win_end_t,
                "window_duration": win_end_t - win_start_t,
                "mean_abs_cents_from_note": best_mean_abs_from_note,
                "kept_energy_boundaries": False,
            }
        )
        return win_start_t, win_end_t, info

    info["std_cents"] = best_std
    info["mean_abs_cents_from_note"] = best_mean_abs_from_note
    info["kept_energy_boundaries"] = True
    return t_att_rel, t_dec_rel, info


def _clamp_segment_rel(
    t_att: float, t_dec: float, active_len: float, min_sustain: float, min_decay: float = 0.02
) -> Tuple[float, float]:
    min_tail = max(min_decay, 0.01)
    t_att = max(0.0, min(t_att, active_len - min_sustain - min_tail))
    t_dec = max(t_att + min_sustain, min(t_dec, active_len - min_tail))
    if t_dec <= t_att:
        t_dec = min(t_att + min_sustain, active_len - min_tail)
    if t_dec >= active_len - 1e-4:
        t_dec = max(t_att + min_sustain, active_len - min_tail)
    return t_att, t_dec


def detect_segments_advanced_rel(
    y_trimmed: np.ndarray, sr: int, cfg: SegmentConfig, min_sustain: float
) -> Tuple[float, float]:
    rms, times = compute_rms_envelope(y_trimmed, sr, cfg.frame_length, cfg.hop_length)
    peak_idx = int(np.argmax(rms))
    flux, flux_times = compute_spectral_flux(y_trimmed, sr, cfg.frame_length, cfg.hop_length)

    attack_time = detect_attack_combined(
        rms, times, peak_idx, cfg.attack_threshold, flux, flux_times, use_derivative=True
    )
    attack_idx = min(int(np.searchsorted(times, attack_time, side="right")), len(rms) - 1)

    active_len = len(y_trimmed) / sr
    min_decay_t = min_decay_time_proportional(
        active_len, cfg.attack_pct, cfg.sustain_pct, cfg.decay_pct, cfg.sustain_fraction_before_decay
    )
    decay_time = detect_decay_derivative(rms, times, attack_idx, peak_idx, cfg.decay_threshold)
    decay_time = max(decay_time, min_decay_t)
    decay_idx = min(int(np.searchsorted(times, decay_time, side="right")), len(rms) - 1)

    plateau = detect_sustain_plateau(
        rms, times, attack_idx, decay_idx, min_sustain, cfg.sustain_variance_threshold
    )
    if plateau[0] is not None:
        attack_time = float(times[plateau[0]])
        decay_time = float(times[plateau[1]])

    active_len = len(y_trimmed) / sr
    return _clamp_segment_rel(attack_time, decay_time, active_len, min_sustain)


def detect_segments_smart_rel(
    y_trimmed: np.ndarray, sr: int, cfg: SegmentConfig, min_sustain: float
) -> Tuple[float, float]:
    """Energy-guided boundaries blended with proportional anchors."""
    active_len = len(y_trimmed) / sr
    prop_att, prop_dec = detect_segments_proportional(
        active_len, cfg.attack_pct, cfg.sustain_pct, cfg.decay_pct, min_sustain
    )
    rms, times = compute_rms_envelope(y_trimmed, sr, cfg.frame_length, cfg.hop_length)
    peak_idx = int(np.argmax(rms))
    energy_att = detect_attack_energy(rms, times, peak_idx, cfg.attack_threshold)
    attack_idx = min(int(np.searchsorted(times, energy_att, side="right")), len(rms) - 1)
    min_decay_t = min_decay_time_proportional(
        active_len, cfg.attack_pct, cfg.sustain_pct, cfg.decay_pct, cfg.sustain_fraction_before_decay
    )
    energy_dec = detect_decay_energy(
        rms, times, attack_idx, peak_idx, cfg.decay_threshold, min_decay_time=min_decay_t
    )

    t_att = SMART_ENERGY_BLEND * energy_att + SMART_PROP_BLEND * prop_att
    t_dec = SMART_ENERGY_BLEND * energy_dec + SMART_PROP_BLEND * prop_dec
    t_dec = max(t_dec, min_decay_t)
    return _clamp_segment_rel(t_att, t_dec, active_len, min_sustain)


def detect_segments(
    y: np.ndarray, sr: int, cfg: SegmentConfig, file_path: Optional[Path] = None
) -> SegmentResult:
    empty_pitch = {
        "used": False,
        "std_cents": None,
        "window_start": None,
        "window_end": None,
        "window_duration": None,
        "expected_note_hz": None,
        "mean_abs_cents_from_note": None,
        "vibrato_robust": cfg.vibrato_robust,
    }
    y = preprocess_signal(y, remove_dc=cfg.remove_dc)
    try:
        y_trimmed, trim = trim_active_region(y, sr, cfg.trim_db)
        active_len = trim.active_len

        if active_len <= 0 or len(y_trimmed) < cfg.frame_length * 2:
            t_att = trim.t_start + cfg.attack_pct * active_len
            t_dec = trim.t_start + (cfg.attack_pct + cfg.sustain_pct) * active_len
            return SegmentResult(t_att, t_dec, trim.t_end, trim, empty_pitch)

        min_sustain = effective_min_sustain_duration(cfg, sr, active_len)

        if cfg.use_advanced:
            t_att_rel, t_dec_rel = detect_segments_advanced_rel(y_trimmed, sr, cfg, min_sustain)
        elif cfg.use_smart:
            t_att_rel, t_dec_rel = detect_segments_smart_rel(y_trimmed, sr, cfg, min_sustain)
        else:
            t_att_rel, t_dec_rel = detect_segments_proportional(
                active_len, cfg.attack_pct, cfg.sustain_pct, cfg.decay_pct, min_sustain
            )

        expected_hz = parse_note_hz_from_filename(file_path)
        t_att_rel, t_dec_rel, pitch_info = refine_sustain_by_pitch(
            y_trimmed, sr, t_att_rel, t_dec_rel, cfg, expected_hz
        )
        t_att_rel, t_dec_rel = _clamp_segment_rel(t_att_rel, t_dec_rel, active_len, min_sustain)
        if pitch_info.get("used"):
            pitch_info["window_start"] = trim.t_start + pitch_info["window_start"]
            pitch_info["window_end"] = trim.t_start + pitch_info["window_end"]

        min_decay = max(0.02, active_len * 0.05)
        t_att = trim.t_start + t_att_rel
        t_dec = min(trim.t_start + t_dec_rel, trim.t_end - min_decay)
        t_att = min(t_att, t_dec - min(min_sustain, active_len * 0.5))
        return SegmentResult(t_att, t_dec, trim.t_end, trim, pitch_info)

    except Exception as exc:
        logger.warning("detect_segments fallback to proportional: %s", exc, exc_info=True)
        active_len = len(y) / sr
        t_att, t_dec = detect_segments_proportional(
            active_len, cfg.attack_pct, cfg.sustain_pct, cfg.decay_pct, cfg.min_sustain_duration
        )
        trim = TrimInfo(0, len(y), 0.0, active_len - 1e-3, active_len)
        return SegmentResult(t_att, t_dec, trim.t_end, trim, empty_pitch)


def validate_segments(
    t_att: float, t_dec: float, t_end: float, min_duration: float = 0.01
) -> bool:
    if t_att >= t_dec or t_dec >= t_end:
        return False
    if t_dec - t_att < min_duration or t_end - t_dec < min_duration:
        return False
    return True


def find_zero_crossing(
    y: np.ndarray, idx: int, sr: int, search_ms: float = DEFAULT_ZERO_CROSSING_SEARCH_MS
) -> int:
    idx = max(0, min(idx, len(y) - 1))
    search_samples = int(sr * (search_ms / 1000.0))
    start = max(0, idx - search_samples)
    end = min(len(y), idx + search_samples + 1)
    chunk = y[start:end]
    if len(chunk) < 2:
        return idx
    sign_changes = np.where(np.diff(np.signbit(chunk)))[0]
    if len(sign_changes) == 0:
        expanded = min(search_samples * 2, len(y) // 4)
        start = max(0, idx - expanded)
        end = min(len(y), idx + expanded)
        chunk = y[start:end]
        if len(chunk) < 2:
            return idx
        sign_changes = np.where(np.diff(np.signbit(chunk)))[0]
        if len(sign_changes) == 0:
            return idx
    target_offset = max(0, min(idx - start, len(chunk) - 1))
    crossing_idx = sign_changes[int(np.argmin(np.abs(sign_changes - target_offset)))]
    if crossing_idx < len(chunk) - 1:
        y1, y2 = chunk[crossing_idx], chunk[crossing_idx + 1]
        t = -y1 / (y2 - y1) if abs(y2 - y1) > 1e-10 else 0.0
        exact = crossing_idx + t
    else:
        exact = float(crossing_idx)
    return max(0, min(start + int(round(exact)), len(y) - 1))


def _fade_curve(n: int, fade_type: str, rising: bool) -> np.ndarray:
    if n < 1:
        return np.array([], dtype=np.float64)
    t = np.linspace(0.0, 1.0, n)
    if fade_type == "linear":
        curve = t if rising else 1.0 - t
    elif fade_type == "hann":
        curve = np.hanning(n * 2)[:n] if rising else np.hanning(n * 2)[n:]
    else:
        # cosine (raised cosine)
        curve = 0.5 * (1.0 - np.cos(np.pi * t)) if rising else 0.5 * (1.0 + np.cos(np.pi * t))
    return curve.astype(np.float64)


def apply_fades(audio: np.ndarray, sr: int, fade_ms: float, fade_type: str = "cosine") -> np.ndarray:
    if len(audio) == 0:
        return audio
    fade_samples = int(sr * (fade_ms / 1000.0))
    min_fade = int(sr / 20.0)
    fade_samples = max(fade_samples, min(min_fade, len(audio) // 4))
    fade_samples = min(fade_samples, len(audio) // 2)
    if fade_samples < 1:
        return audio
    fade_in = _fade_curve(fade_samples, fade_type, rising=True)
    fade_out = _fade_curve(fade_samples, fade_type, rising=False)
    out = audio.copy()
    out[:fade_samples] *= fade_in
    out[-fade_samples:] *= fade_out
    if abs(out[0]) < 0.001:
        out[0] = 0.0
    if abs(out[-1]) < 0.001:
        out[-1] = 0.0
    return out


def edge_click_severity(audio: np.ndarray, edge_samples: int = 32) -> float:
    """Boundary discontinuity vs typical interior sample-to-sample change (0 = clean)."""
    if len(audio) < 8:
        return 0.0
    edge = min(edge_samples, len(audio) // 4, len(audio) - 1)
    mid_lo = edge
    mid_hi = max(mid_lo + 4, len(audio) - edge)
    if mid_hi <= mid_lo + 4:
        ref_diff = float(np.median(np.abs(np.diff(audio)))) + 1e-10
    else:
        ref_diff = float(np.median(np.abs(np.diff(audio[mid_lo:mid_hi])))) + 1e-10

    start_amp = abs(float(audio[0]))
    end_amp = abs(float(audio[-1]))
    start_spike = float(np.max(np.abs(np.diff(audio[: max(2, edge)])))) if edge > 1 else start_amp
    end_spike = float(np.max(np.abs(np.diff(audio[-max(2, edge) :])))) if edge > 1 else end_amp
    return max(start_amp / ref_diff, end_amp / ref_diff, start_spike / (ref_diff * 2), end_spike / (ref_diff * 2))


def verify_no_clicks(audio: np.ndarray, tolerance: float = 0.01, max_severity: float = 4.0) -> bool:
    if len(audio) < 2:
        return True
    if abs(audio[0]) > tolerance or abs(audio[-1]) > tolerance:
        return False
    return edge_click_severity(audio) <= max_severity


def extract_and_fade_segments(
    y: np.ndarray,
    sr: int,
    t_att: float,
    t_dec: float,
    t_end: float,
    trim: TrimInfo,
    fade_ms: float,
    fade_type: str,
) -> Tuple[Dict[str, np.ndarray], int, int, int]:
    max_idx = len(y) - 1
    idx_start = max(0, min(trim.idx_start, max_idx))
    idx_att_target = max(idx_start, min(int(t_att * sr), max_idx))
    idx_dec_target = max(idx_att_target + 1, min(int(t_dec * sr), max_idx))
    idx_end_target = max(idx_dec_target + 1, min(int(t_end * sr), max_idx))

    idx_att = find_zero_crossing(y, idx_att_target, sr)
    idx_dec = find_zero_crossing(y, max(idx_att + int(0.02 * sr), idx_dec_target), sr)
    idx_end = find_zero_crossing(y, idx_end_target, sr)

    idx_att = max(idx_start, min(idx_att, max_idx))
    idx_dec = max(idx_att + 1, min(idx_dec, max_idx))
    idx_end = max(idx_dec + 1, min(idx_end, len(y)))

    attack_seg = y[idx_start:idx_att].copy()
    sustain_seg = y[idx_att:idx_dec].copy()
    decay_seg = y[idx_dec:idx_end].copy()
    release_seg = y[idx_end:].copy()
    active_sound = y[idx_start:idx_end].copy()

    parts = {
        "_Attacks": apply_fades(attack_seg, sr, fade_ms, fade_type),
        "_Sustains": apply_fades(sustain_seg, sr, fade_ms, fade_type),
        "_Decays": apply_fades(decay_seg, sr, fade_ms, fade_type),
        "_Release_Silence": release_seg,
        "_Full_Active_Sound": apply_fades(active_sound, sr, fade_ms, fade_type),
    }
    for name, seg in parts.items():
        if len(seg) > 0 and name != "_Release_Silence" and not verify_no_clicks(seg):
            parts[name] = apply_fades(seg, sr, fade_ms * 1.5, fade_type)
    return parts, idx_att, idx_dec, idx_end


def _soundfile_format_for_extension(ext: str) -> Optional[str]:
    return {
        ".wav": "WAV",
        ".aif": "AIFF",
        ".aiff": "AIFF",
        ".flac": "FLAC",
        ".ogg": "OGG",
    }.get(ext.lower())


def write_audio(output_path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf_format = _soundfile_format_for_extension(output_path.suffix)
    if sf_format:
        sf.write(output_path, audio, sr, format=sf_format)
    else:
        sf.write(output_path, audio, sr)


def list_audio_files(folder: Path) -> List[Path]:
    files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        and not f.stem.lower().endswith("_backup")
    ]
    return sorted(files, key=lambda p: p.name.lower())


def process_audio_file(
    input_path: Path,
    output_dir: Path,
    cfg: SegmentConfig,
    fade_ms: float = 50.0,
    fade_type: str = "cosine",
) -> Dict:
    """
    Headless ADSR split for one file. Writes segment folders under output_dir.
    Returns metadata dict for the processed file.
    """
    import librosa

    y, sr = librosa.load(str(input_path), sr=None)
    result = detect_segments(y, sr, cfg, file_path=input_path)
    if not validate_segments(result.t_att, result.t_dec, result.t_end):
        raise ValueError(f"Invalid segment boundaries for {input_path.name}")

    parts, idx_att, idx_dec, idx_end = extract_and_fade_segments(
        y, sr, result.t_att, result.t_dec, result.t_end, result.trim, fade_ms, fade_type
    )

    for folder, audio in parts.items():
        target_dir = output_dir / folder
        target_dir.mkdir(exist_ok=True, parents=True)
        if len(audio) == 0:
            continue
        if folder == "_Full_Active_Sound":
            tag = "FullActive"
        elif folder == "_Release_Silence":
            tag = "Release"
        else:
            tag = folder.strip("_")
        write_audio(target_dir / f"{input_path.stem}_{tag}{input_path.suffix}", audio, sr)

    trim = result.trim
    idx_start = trim.idx_start
    return {
        "file_path": str(input_path),
        "sr": sr,
        "t_start": idx_start / sr,
        "t_att": idx_att / sr,
        "t_dec": idx_dec / sr,
        "t_end": idx_end / sr,
        "dur_att": (idx_att - idx_start) / sr,
        "dur_sus": (idx_dec - idx_att) / sr,
        "dur_dec": (idx_end - idx_dec) / sr,
        "dur_rel": (len(y) - idx_end) / sr,
        "pitch_refine": result.pitch_refine,
        "detection_mode": (
            "advanced" if cfg.use_advanced else ("smart" if cfg.use_smart else "proportional")
        ),
    }


def batch_process_folder(
    folder: Path,
    cfg: SegmentConfig,
    fade_ms: float = 50.0,
    fade_type: str = "cosine",
    output_dir: Optional[Path] = None,
) -> List[Dict]:
    """Process all audio files in folder; returns list of per-file metadata."""
    out = output_dir or folder
    results: List[Dict] = []
    for f_path in list_audio_files(folder):
        try:
            results.append(process_audio_file(f_path, out, cfg, fade_ms, fade_type))
        except Exception as exc:
            logger.error("Failed %s: %s", f_path.name, exc)
            results.append({"file_path": str(f_path), "error": str(exc)})
    return results
