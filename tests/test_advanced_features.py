"""Extended tests: vibrato, accuracy, fades, batch, articulation presets."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audio_segment_core as core


def _sine_burst(sr, freq, attack_s, sustain_s, decay_s, gap_s=0.2):
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


def _vibrato_burst(sr, freq, attack_s, sustain_s, decay_s, vib_rate=5.5, vib_depth_cents=25.0, gap_s=0.2):
    """Sustain with sinusoidal pitch modulation (string-like vibrato)."""
    y = _sine_burst(sr, freq, attack_s, sustain_s, decay_s, gap_s)
    y_trim, trim = core.trim_active_region(y, sr)
    t = np.arange(len(y_trim)) / sr
    att_end = attack_s
    dec_start = attack_s + sustain_s
    env = np.ones(len(y_trim))
    if att_end > 0:
        env[: int(att_end * sr)] = np.linspace(0, 1, int(att_end * sr), endpoint=False)
    if decay_s > 0:
        env[int(dec_start * sr) :] = np.linspace(1, 0, len(y_trim) - int(dec_start * sr))
    phase = 2 * np.pi * vib_rate * t
    freq_mod = freq * (2 ** ((vib_depth_cents / 1200.0) * np.sin(phase)))
    phase_acc = np.cumsum(2 * np.pi * freq_mod / sr)
    return np.concatenate([
        np.zeros(trim.idx_start),
        0.45 * np.sin(phase_acc) * env,
        np.zeros(len(y) - trim.idx_end),
    ])


@pytest.fixture
def sr():
    return 22050


def test_spectral_flux_vectorized(sr):
    y = _sine_burst(sr, 440.0, 0.05, 0.4, 0.15)
    flux, times = core.compute_spectral_flux(y, sr)
    assert len(flux) == len(times)
    assert flux.max() > 0


def test_hann_differs_from_cosine(sr):
    n = sr // 2
    tone = np.sin(2 * np.pi * 440 * np.arange(n) / sr) * 0.5
    hann = core.apply_fades(tone.copy(), sr, 40.0, "hann")
    cos = core.apply_fades(tone.copy(), sr, 40.0, "cosine")
    assert not np.allclose(hann[:50], cos[:50])


def test_vibrato_robust_stability_lower_than_raw(sr):
    t = np.linspace(0, 0.5, 500)
    vibrato_cents = 30.0 * np.sin(2 * np.pi * 5.5 * t)
    cfg = core.SegmentConfig(vibrato_robust=True)
    std_robust = core.pitch_stability_std_cents(vibrato_cents, t, cfg)
    cfg.vibrato_robust = False
    std_raw = core.pitch_stability_std_cents(vibrato_cents, t, cfg)
    assert std_robust is not None and std_raw is not None
    assert std_robust < std_raw


def test_vibrato_sustain_segments_valid(sr):
    y = _vibrato_burst(sr, 440.0, 0.06, 0.55, 0.2, vib_rate=5.5, vib_depth_cents=30.0)
    cfg = core.SegmentConfig.from_preset("Legato / Bow", vibrato_robust=True)
    result = core.detect_segments(y, sr, cfg, file_path=Path("Violin_A4.wav"))
    assert result.t_att < result.t_dec < result.t_end
    assert core.validate_segments(result.t_att, result.t_dec, result.t_end, 0.02)


def test_boundary_accuracy_smart_mode(sr):
    attack_s, sustain_s, decay_s = 0.08, 0.50, 0.20
    y = _sine_burst(sr, 440.0, attack_s, sustain_s, decay_s, gap_s=0.15)
    y_trim, trim = core.trim_active_region(y, sr)
    gap = trim.t_start
    expected_att = gap + attack_s
    expected_dec = gap + attack_s + sustain_s

    cfg = core.SegmentConfig(use_smart=True, use_advanced=False, min_sustain_duration=0.08)
    result = core.detect_segments(y, sr, cfg)
    assert abs(result.t_att - expected_att) < 0.12
    assert abs(result.t_dec - expected_dec) < 0.28


def test_advanced_mode_ordering(sr):
    y = _sine_burst(sr, 660.0, 0.05, 0.45, 0.18)
    cfg = core.SegmentConfig(use_advanced=True, use_smart=False, min_sustain_duration=0.06)
    result = core.detect_segments(y, sr, cfg)
    assert result.t_att < result.t_dec < result.t_end


def test_articulation_preset_staccato(sr):
    cfg = core.SegmentConfig.from_preset("Staccato / Pluck")
    assert cfg.use_advanced is True
    y = _sine_burst(sr, 880.0, 0.02, 0.06, 0.05, gap_s=0.08)
    result = core.detect_segments(y, sr, cfg)
    assert result.t_dec - result.t_att >= 0.02


def test_preprocess_removes_dc(sr):
    y = np.sin(2 * np.pi * 440 * np.arange(sr) / sr) + 0.25
    out = core.preprocess_signal(y, remove_dc=True)
    assert abs(float(np.mean(out))) < 1e-10


def test_edge_click_severity_clean_faded(sr):
    tone = np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 0.3
    faded = core.apply_fades(tone, sr, 50.0, "cosine")
    assert core.edge_click_severity(faded) < 4.0
    assert core.verify_no_clicks(faded)


def test_process_audio_file_writes_outputs(sr):
    y = _sine_burst(sr, 440.0, 0.05, 0.35, 0.12, gap_s=0.1)
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "test_A4.wav"
        sf.write(inp, y, sr)
        cfg = core.SegmentConfig(use_smart=True, min_sustain_duration=0.05)
        meta = core.process_audio_file(inp, Path(tmp), cfg, fade_ms=30.0, fade_type="cosine")
        assert meta["dur_sus"] > 0
        for folder in ("_Attacks", "_Sustains", "_Decays"):
            assert (Path(tmp) / folder).exists()
            wavs = list((Path(tmp) / folder).glob("*.wav"))
            assert len(wavs) == 1


def test_batch_process_folder(sr):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i, freq in enumerate((440.0, 554.0)):
            y = _sine_burst(sr, freq, 0.05, 0.3, 0.1)
            sf.write(root / f"note_{i}.wav", y, sr)
        cfg = core.SegmentConfig(use_smart=True, min_sustain_duration=0.05)
        results = core.batch_process_folder(root, cfg, fade_ms=25.0)
        assert len(results) == 2
        assert all("error" not in r for r in results)


def test_segment_config_from_preset_unknown():
    with pytest.raises(ValueError, match="Unknown preset"):
        core.SegmentConfig.from_preset("Not A Preset")


def test_long_sound_sustain_not_cropped_to_one_second(sr):
    """6 s note: sustain export must stay long enough for STFT (not ~1 s pitch crop)."""
    y = _sine_burst(sr, 440.0, 0.35, 4.5, 0.65, gap_s=0.25)
    cfg = core.SegmentConfig.from_preset("Very Long (> 6.0s)")
    result = core.detect_segments(y, sr, cfg, file_path=Path("Violin_A4.wav"))
    sustain_dur = result.t_dec - result.t_att
    assert sustain_dur >= 2.8, f"sustain only {sustain_dur:.2f}s"


def test_annotate_mode_keeps_energy_sustain(sr):
    y = _sine_burst(sr, 440.0, 0.08, 1.2, 0.25)
    cfg_energy = core.SegmentConfig(use_smart=True, use_pitch_refine=False, min_sustain_duration=0.1)
    cfg_annotate = core.SegmentConfig(
        use_smart=True,
        use_pitch_refine=True,
        pitch_refine_mode="annotate",
        min_sustain_duration=0.1,
    )
    r0 = core.detect_segments(y, sr, cfg_energy)
    r1 = core.detect_segments(y, sr, cfg_annotate, file_path=Path("A4.wav"))
    assert abs((r1.t_dec - r1.t_att) - (r0.t_dec - r0.t_att)) < 0.05
    assert r1.pitch_refine.get("kept_energy_boundaries") is True


def test_min_fraction_guard_keeps_long_sustain(sr):
    """If pitch crop would shrink below 70% of energy sustain, keep energy boundaries."""
    y = _sine_burst(sr, 440.0, 0.08, 2.5, 0.3)
    cfg = core.SegmentConfig(
        use_smart=True,
        pitch_refine_mode="crop",
        pitch_refine_min_fraction=0.70,
        pitch_stability_cents=50.0,
        min_sustain_duration=0.1,
    )
    result = core.detect_segments(y, sr, cfg)
    energy_dur = result.pitch_refine.get("energy_sustain_duration") or 0
    actual_dur = result.t_dec - result.t_att
    if energy_dur > 0:
        assert actual_dur >= energy_dur * 0.68
