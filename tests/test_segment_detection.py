"""Unit tests for audio_segment_core (no GUI)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audio_segment_core as core


def _sine_burst(sr: int, freq: float, attack_s: float, sustain_s: float, decay_s: float, gap_s: float = 0.2):
    """Synthetic envelope: silence, attack ramp, sustain, decay, silence."""
    n_att = int(attack_s * sr)
    n_sus = int(sustain_s * sr)
    n_dec = int(decay_s * sr)
    n_gap = int(gap_s * sr)
    att = np.linspace(0, 1, n_att, endpoint=False) if n_att else np.array([])
    sus = np.ones(n_sus)
    dec = np.linspace(1, 0, n_dec, endpoint=True) if n_dec else np.array([])
    body = np.concatenate([x for x in (att, sus, dec) if len(x)])
    t = np.arange(len(body)) / sr
    tone = 0.5 * np.sin(2 * np.pi * freq * t) * body
    return np.concatenate([np.zeros(n_gap), tone, np.zeros(n_gap)])


@pytest.fixture
def sr():
    return 22050


def test_trim_active_region(sr):
    y = _sine_burst(sr, 440.0, 0.05, 0.4, 0.15, gap_s=0.1)
    y_trim, trim = core.trim_active_region(y, sr)
    assert trim.active_len > 0.4
    assert len(y_trim) < len(y)


def test_energy_attack_before_peak(sr):
    y = _sine_burst(sr, 440.0, 0.08, 0.5, 0.2)
    y_trim, _ = core.trim_active_region(y, sr)
    rms, times = core.compute_rms_envelope(y_trim, sr)
    peak_idx = int(np.argmax(rms))
    t = core.detect_attack_energy(rms, times, peak_idx, 0.9)
    assert t < times[peak_idx]


def test_detect_segments_smart_ordering(sr):
    y = _sine_burst(sr, 440.0, 0.06, 0.5, 0.25)
    cfg = core.SegmentConfig(use_smart=True, use_advanced=False, min_sustain_duration=0.08)
    result = core.detect_segments(y, sr, cfg)
    assert result.t_att < result.t_dec < result.t_end
    assert core.validate_segments(result.t_att, result.t_dec, result.t_end, 0.02)


def test_detect_segments_short_sound(sr):
    y = _sine_burst(sr, 880.0, 0.02, 0.08, 0.04, gap_s=0.05)
    cfg = core.SegmentConfig(
        use_smart=True,
        attack_pct=0.2,
        sustain_pct=0.5,
        decay_pct=0.3,
        min_sustain_duration=0.03,
    )
    result = core.detect_segments(y, sr, cfg)
    assert result.t_dec - result.t_att >= 0.02


def test_extract_starts_at_trim_not_file_start(sr):
    y = _sine_burst(sr, 440.0, 0.05, 0.35, 0.15, gap_s=0.15)
    cfg = core.SegmentConfig(use_smart=True, min_sustain_duration=0.08)
    seg = core.detect_segments(y, sr, cfg)
    parts, idx_att, idx_dec, idx_end = core.extract_and_fade_segments(
        y, sr, seg.t_att, seg.t_dec, seg.t_end, seg.trim, 30.0, "cosine"
    )
    assert len(parts["_Attacks"]) < len(y)
    assert idx_att >= seg.trim.idx_start
    assert len(parts["_Sustains"]) > 0


def test_parse_note_from_filename():
    assert core.parse_note_hz_from_filename(Path("Violin_A4_test.wav")) == pytest.approx(440.0, rel=1e-3)
    assert core.parse_note_hz_from_filename(Path("noise.wav")) is None


def test_proportional_percentages_sum(sr):
    y = np.random.randn(sr) * 0.01
    y[sr // 4 : 3 * sr // 4] += np.sin(2 * np.pi * 440 * np.arange(sr // 2) / sr)
    cfg = core.SegmentConfig(use_smart=False, use_advanced=False)
    result = core.detect_segments(y, sr, cfg)
    assert result.t_att <= result.t_dec
