# SPLIT Audio Segments — Technical Documentation

**Version:** 3.1 (Optimized Audio Segment Splitter)  
**Audience:** Acoustics researchers, sound designers, and software engineers  
**License / authorship:** As per project owner.

---

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [System Architecture](#2-system-architecture)
3. [Acoustic Model: ADSR Segmentation](#3-acoustic-model-adsr-segmentation)
4. [Signal Processing Pipeline](#4-signal-processing-pipeline)
5. [Detection Algorithms](#5-detection-algorithms)
6. [Pitch-Based Sustain Refinement](#6-pitch-based-sustain-refinement)
7. [Boundary Editing and Click-Free Export](#7-boundary-editing-and-click-free-export)
8. [Output Layout and Metadata](#8-output-layout-and-metadata)
9. [Configuration Reference](#9-configuration-reference)
10. [API Reference (`audio_segment_core`)](#10-api-reference-audio_segment_core)
11. [GUI Application (`split_audio_segments.py`)](#11-gui-application-split_audio_segmentspy)
12. [Tutorial](#12-tutorial)
13. [Testing](#13-testing)
14. [Troubleshooting](#14-troubleshooting)
15. [Dependencies](#15-dependencies)

---

## 1. Purpose and Scope

This project **automatically splits monophonic or quasi-monophonic audio files** into four classical envelope regions—**Attack**, **Sustain**, **Decay**, and **Release**—plus optional composites:

| Output folder           | Content |
|-------------------------|---------|
| `_Attacks/`             | Onset → attack boundary |
| `_Sustains/`            | Attack boundary → decay boundary |
| `_Decays/`              | Decay boundary → end of active sound |
| `_Release_Silence/`     | Tail after active sound (no fades applied) |
| `_Full_Active_Sound/`   | Entire trimmed active region (attack through decay end) |

Typical use cases:

- Building sample libraries for samplers (SFZ, Kontakt, etc.)
- Separating bow/noise attack from stable pitched sustain in orchestral recordings
- Batch-processing instrument one-shots with consistent segment ratios
- Research pipelines that need reproducible ADSR labels and CSV/JSON metadata

**Supported input formats:** `.wav`, `.mp3`, `.flac`, `.aif`, `.aiff`, `.ogg`, `.m4a`, `.wma`, `.mp4`, `.mka` (via librosa / backend codecs).

**Not in scope:** polyphonic source separation, beat slicing, or semantic phrase detection.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  split_audio_segments.py  (Tkinter GUI + batch orchestration)     │
│  • folder picker, presets, review UI, metadata export           │
└────────────────────────────┬────────────────────────────────────┘
                             │ SegmentConfig, detect/extract calls
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  audio_segment_core.py  (pure DSP — no GUI, unit-testable)      │
│  trim → detect (smart/advanced/proportional) → pitch refine     │
│  → zero-crossing snap → fades → segment dict                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
    librosa              numpy              soundfile
  (load, trim, RMS,     (arrays, math)     (write WAV/FLAC/…)
   STFT, YIN, YIN)
```

| Module | Role |
|--------|------|
| `audio_segment_core.py` | All detection and extraction logic; safe to import from scripts or notebooks. |
| `split_audio_segments.py` | Desktop app: parameters, batch run, manual review, JSON/CSV export. |
| `split_audio_cli.py` | Headless batch CLI (no GUI): presets, metadata export, CI-friendly. |
| `tests/test_segment_detection.py` | Regression tests on synthetic sine bursts. |
| `tests/test_advanced_features.py` | Vibrato, articulation presets, batch I/O, boundary accuracy. |

**Coordinate systems (important):**

- **File time:** seconds from sample 0 of the loaded file (`t_att`, `t_dec`, `t_end` in `SegmentResult`).
- **Trim-relative time:** seconds from the start of the **active** region after `librosa.effects.trim` (used internally during detection, then offset by `trim.t_start`).

---

## 3. Acoustic Model: ADSR Segmentation

The tool implements a **single-note / single-event** ADSR model:

```
Amplitude
    │     ┌──────── sustain plateau ────────┐
    │    /│                                  │\
    │   / │                                  │ \
    │  /  │                                  │  \ decay
    │ /   │                                  │   \
────┴─────┴──────────────────────────────────┴────\────► time
  silence  attack end              decay start   end   release
           (t_att)                  (t_dec)      (t_end)
```

**Definitions used in code:**

| Symbol | Meaning |
|--------|---------|
| `t_start` | Start of active audio after trim (`trim.idx_start / sr`) |
| `t_att` | End of attack / start of sustain (boundary sample after zero-crossing snap) |
| `t_dec` | Start of decay / end of sustain |
| `t_end` | End of musically active energy (trim end or energy tail) |
| Release | Samples from `t_end` to EOF (often silence or room tail) |

**Preset percentages** (`attack_pct`, `sustain_pct`, `decay_pct`) do not define fixed durations in seconds; they define **target proportions of active length** used as anchors, especially in Smart mode blending.

---

## 4. Signal Processing Pipeline

End-to-end processing for one file:

### 4.1 Load

```python
y, sr = librosa.load(path, sr=None)  # native sample rate preserved
```

### 4.2 Active-region trim

`trim_active_region()` calls `librosa.effects.trim(y, top_db=trim_db)` (default **60 dB** below peak). This removes leading/trailing silence so detection runs on the **musical body** only.

`TrimInfo` stores:

- `idx_start`, `idx_end` — sample indices in the original file
- `t_start`, `t_end` — corresponding times
- `active_len` — `t_end - t_start`

### 4.3 Short-time analysis frame

Default STFT/RMS parameters (configurable via `SegmentConfig`):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `frame_length` | 1024 | ~23 ms @ 44.1 kHz |
| `hop_length` | 512 | 50% overlap |
| RMS | `librosa.feature.rms` | Short-time energy envelope |
| Spectral flux | magnitude STFT frame difference | Onset emphasis (Advanced mode) |

Time axis for envelope frames: `librosa.times_like(rms, sr=sr, hop_length=hop_length)`.

### 4.4 Boundary detection

One of three modes (see [§5](#5-detection-algorithms)); produces trim-relative `t_att_rel`, `t_dec_rel`.

### 4.5 Pitch refinement (optional)

Sliding-window **YIN** F0 analysis may shrink sustain to the most pitch-stable sub-window (see [§6](#6-pitch-based-sustain-refinement)).

### 4.6 Clamp and convert to file time

`_clamp_segment_rel()` enforces minimum sustain/decay tails. Boundaries are shifted by `trim.t_start`.

### 4.7 Extract, zero-cross, fade

`extract_and_fade_segments()`:

1. Converts times → sample indices.
2. Snaps cut points with `find_zero_crossing()` (±100 ms search, linear interpolation between samples).
3. Slices five arrays; applies **cosine** (default), Hann, or linear fades on all segments except `_Release_Silence`.
4. Re-applies longer fades if `verify_no_clicks()` fails at segment edges.

---

## 5. Detection Algorithms

Mode selection in `detect_segments()` (`audio_segment_core.py`):

```text
if cfg.use_advanced:     → detect_segments_advanced_rel()
elif cfg.use_smart:      → detect_segments_smart_rel()   ← default
else:                     → detect_segments_proportional()
```

### 5.1 Proportional mode (`use_smart=False`, `use_advanced=False`)

Purely duration-based on `active_len`:

- `t_attack_end = active_len × attack_pct`
- `t_decay_start = active_len × (attack_pct + sustain_pct)`
- Enforces `min_sustain_duration` and a 5% end margin.

**Use when:** envelopes are irregular but you want deterministic, preset-driven splits (e.g. batch normalizing a homogenous library).

### 5.2 Smart mode (default, recommended)

**Energy-guided detection blended with proportional anchors.**

1. Compute proportional anchors `prop_att`, `prop_dec`.
2. Compute RMS envelope; `peak_idx = argmax(rms)`.
3. **Attack end:** `detect_attack_energy()` — first frame **≥ attack_threshold × peak RMS** while scanning from onset toward peak (default threshold **0.90**).
4. **Decay start:** `detect_decay_energy()` — first frame **≤ decay_threshold × peak RMS** after attack/peak (default **0.50**).
5. **Blend:**  
   `t_att = 0.7 × energy_att + 0.3 × prop_att`  
   `t_dec = 0.7 × energy_dec + 0.3 × prop_dec`  
6. `_clamp_segment_rel()` guarantees minimum sustain/decay.

**Rationale:** Energy tracks the physical envelope; proportional terms prevent pathological placements on noisy or reversed dynamics.

### 5.3 Advanced mode (`use_advanced=True`)

Uses **derivative + spectral flux** heuristics:

| Stage | Method |
|-------|--------|
| Attack | `detect_attack_combined(..., use_derivative=True)` — minimum of energy-based time and max positive ΔRMS before peak; capped at 85% of peak time; flux onset can pull earlier |
| Decay | `detect_decay_derivative()` — first run of **≥3 consecutive** negative ΔRMS frames after a delay past peak; fallback to energy decay |
| Sustain plateau | If RMS coefficient of variation in sustain region `< sustain_variance_threshold` (0.2) and duration OK, boundaries snap to plateau edges |

**Use when:** attacks are spectrally rich but energetically subtle (bow noise, breath, plucks with slow RMS rise).

**Mutual exclusion in GUI:** enabling Advanced disables Smart (`use_smart` forced false in `_config_from_ui()`).

### 5.4 Attack / decay energy detectors (detail)

**Attack (`detect_attack_energy`):**  
Scans frames `[0 … peak_idx]` for first `rms[i] >= threshold * rms[peak]`.

**Decay (`detect_decay_energy`):**  
Scans from `max(attack_idx, peak_idx)` for first `rms[i] <= threshold * max(rms)`.

These are **relative to the trimmed signal’s peak**, not absolute dBFS.

---

## 6. Pitch-Based Sustain Refinement

After energy boundaries are set, `refine_sustain_by_pitch()` may **replace** `(t_att_rel, t_dec_rel)` with a shorter window inside the sustain region if pitch is sufficiently stable.

### 6.1 Algorithm

1. Extract sustain slice `y_trimmed[start:end]`.
2. Estimate F0 with `librosa.yin()` (A0–C8 range, same `frame_length` / `hop_length`).
3. Express each valid frame as **cents deviation from median F0**:  
   `cents = 1200 × log2(f0 / median_f0)`.
4. Slide a window of length `max(pitch_window_duration, effective_min_sustain_duration)`.
5. Score each window: `score = std(cents) + mean_abs_cents_from_note` (latter term only if note parsed from filename).
6. If best `std(cents) ≤ pitch_stability_cents` (default **5¢**), set attack/decay boundaries to that window (in trim-relative time).

### 6.2 Filename note parsing

`parse_note_hz_from_filename()` matches patterns like `Violin_A4_test.wav` → **440 Hz** via `librosa.note_to_hz()`.  
Supports accidentals: `C#4`, `Bb3`, etc.

When a note is known, windows closer to the expected pitch rank higher even if internal deviation is similar.

### 6.3 Metadata fields (`pitch_refine` dict)

| Field | Description |
|-------|-------------|
| `used` | Whether refinement changed boundaries |
| `std_cents` | Pitch stability of chosen window |
| `window_start`, `window_end` | Absolute file times (after offset) |
| `expected_note_hz` | Parsed fundamental |
| `mean_abs_cents_from_note` | Mean deviation from expected pitch |

---

## 7. Boundary Editing and Click-Free Export

### 7.1 Zero-crossing alignment

Cuts are moved to nearest **sign-change** sample within `DEFAULT_ZERO_CROSSING_SEARCH_MS` (100 ms), with linear interpolation for sub-sample placement. Reduces discontinuity clicks at splices.

### 7.2 Fades

| `fade_type` | Shape |
|-------------|--------|
| `cosine` (default) | Raised cosine in/out |
| `hann` | Same implementation as cosine in code |
| `linear` | Linear ramp |

Fade length: `fade_ms` (default 50 ms), clamped to at least **50 ms** equivalent (`sr/20`) and at most **half** the segment length.

### 7.3 Manual review (GUI)

After batch processing, the **Review Segmentation** window allows:

- Drag **green** (attack) and **orange** (decay) vertical lines on the waveform.
- Keyboard nudge: arrow keys (**Shift** = 5× step, default 5 ms).
- Manual edits stored in `manual_overrides` and re-export segments without re-running auto-detection.

---

## 8. Output Layout and Metadata

### 8.1 File naming

For input `Violin_A4.wav`:

```
source_folder/
├── _Attacks/Violin_A4_Attack.wav
├── _Sustains/Violin_A4_Sustain.wav
├── _Decays/Violin_A4_Decay.wav
├── _Release_Silence/Violin_A4_Release.wav
├── _Full_Active_Sound/Violin_A4_FullActive.wav
├── segmentation_metadata.json
└── segmentation_metadata.csv
```

Files ending with `_backup` in the stem are skipped.

### 8.2 JSON metadata schema

Top-level keys: `export_date`, `parameters`, `files[]`.

Per file: `file_path`, `sample_rate`, `segments.attack_end`, `decay_start`, `end`, `durations.*`, `pitch_stability`.

### 8.3 CSV columns

File name, sample rate, boundary times, segment durations, and pitch-refinement columns (see `_export_metadata()` in GUI module).

---

## 9. Configuration Reference

### 9.1 `SegmentConfig` dataclass

| Field | Default | Description |
|-------|---------|-------------|
| `trim_db` | 60.0 | Trim threshold (dB below peak) |
| `attack_threshold` | 0.90 | Fraction of peak RMS for attack end |
| `decay_threshold` | 0.50 | Fraction of peak RMS for decay start |
| `attack_pct` | 0.15 | Proportional attack share |
| `sustain_pct` | 0.60 | Proportional sustain share |
| `decay_pct` | 0.25 | Proportional decay share |
| `min_sustain_duration` | 0.35 s | Minimum sustain length (may shrink for short sounds) |
| `pitch_window_duration` | 0.5 s | YIN analysis window |
| `pitch_stability_cents` | 5.0 | Max σ for pitch refinement |
| `use_advanced` | False | Derivative + flux mode |
| `use_smart` | True | Energy + proportional blend |
| `sustain_variance_threshold` | 0.2 | Plateau detector (advanced) |
| `frame_length` | 1024 | Analysis frame |
| `hop_length` | 512 | Hop size |
| `min_sustain_frames` | 40 | Frame-based minimum sustain |

`effective_min_sustain_duration()` returns the maximum of: `min_sustain_duration`, `pitch_window_duration`, frame-based minimum, and (for very short sounds) 25% of `active_len`.

### 9.2 Built-in presets (`PRESETS`)

| Preset | Typical length | attack% | sustain% | decay% | fade ms | min sustain |
|--------|----------------|---------|----------|--------|---------|-------------|
| Very Short (< 0.5s) | < 0.5 s | 0.20 | 0.50 | 0.30 | 30 | 0.06 s |
| Short (0.5–1.5s) | 0.5–1.5 s | 0.15 | 0.60 | 0.25 | 40 | 0.15 s |
| Medium (1.5–3.0s) | 1.5–3.0 s | 0.12 | 0.65 | 0.23 | 50 | 0.35 s |
| Long (3.0–6.0s) | 3–6 s | 0.10 | 0.70 | 0.20 | 60 | 0.60 s |
| Very Long (> 6.0s) | > 6 s | 0.08 | 0.75 | 0.17 | 70 | 1.00 s |
| Custom | user-defined | 0.15 | 0.60 | 0.25 | 50 | 0.35 s |

GUI **Auto-Detect Mean Length** scans up to 100 files, trims each at 60 dB, averages active duration, and selects a matching preset.

---

## 10. API Reference (`audio_segment_core`)

### 10.1 Primary entry points

```python
from pathlib import Path
import librosa
import audio_segment_core as core

y, sr = librosa.load("note.wav", sr=None)
cfg = core.SegmentConfig(use_smart=True, attack_threshold=0.9)

result = core.detect_segments(y, sr, cfg, file_path=Path("Violin_A4.wav"))
# result.t_att, result.t_dec, result.t_end  — absolute times (seconds)
# result.trim — TrimInfo
# result.pitch_refine — dict

parts, idx_att, idx_dec, idx_end = core.extract_and_fade_segments(
    y, sr,
    result.t_att, result.t_dec, result.t_end,
    result.trim,
    fade_ms=50.0,
    fade_type="cosine",
)
# parts["_Attacks"], parts["_Sustains"], ...
```

### 10.2 Key functions

| Function | Returns | Purpose |
|----------|---------|---------|
| `trim_active_region(y, sr, trim_db)` | `(y_trimmed, TrimInfo)` | Silence gate |
| `compute_rms_envelope(y, sr, …)` | `(rms, times)` | Energy envelope |
| `compute_spectral_flux(y, sr, …)` | `(flux, times)` | Onset-sensitive flux |
| `detect_segments(y, sr, cfg, file_path)` | `SegmentResult` | Full detection pipeline |
| `validate_segments(t_att, t_dec, t_end)` | `bool` | Ordering / min duration check |
| `extract_and_fade_segments(…)` | `(dict, idx_att, idx_dec, idx_end)` | Slice + ZC + fade |
| `find_zero_crossing(y, idx, sr, search_ms)` | `int` | Nearest ZC sample index |
| `apply_fades(audio, sr, fade_ms, fade_type)` | `np.ndarray` | Edge ramps |
| `parse_note_hz_from_filename(path)` | `float \| None` | Expected F0 from name |

### 10.3 Validation rules

`validate_segments()` requires:

- `t_att < t_dec < t_end`
- Sustain duration ≥ `min_duration` (default 10 ms)
- Decay tail ≥ `min_duration`

---

## 11. GUI Application (`split_audio_segments.py`)

### 11.1 Launch

**GUI:**
```bash
cd "C:\...\SPLIT_audio_segments"
pip install -r requirements.txt
python split_audio_segments.py
```

**Headless CLI:**
```bash
python split_audio_cli.py --folder "D:/Samples/Violin" --preset "Legato / Bow" --export-metadata
python split_audio_cli.py --folder . --preset "Staccato / Pluck" --advanced --fade-ms 25
```

**Windows note:** MP3/M4A require a working **ffmpeg** install on PATH for librosa/audioread.

### 11.2 Workflow summary

1. **Source Folder** — directory containing audio files (outputs written beside inputs).
2. **Preset Configuration** — choose duration class; optionally **Auto-Detect Mean Length** or set **Mean Sound Length** manually; **Apply Preset**.
3. **Segmentation Parameters** — thresholds, fades, Smart/Advanced, pitch controls, optional thread pool.
4. **► RUN OPTIMIZED SPLIT** — batch process; progress bar and log.
5. **Review Segmentation** — opens automatically; adjust boundaries per file.
6. **Clear** — reset state for a new folder (does not delete output files).

### 11.3 Parallel processing

**Parallel batch (experimental)** uses `ThreadPoolExecutor` (not separate processes) to avoid pickling the Tkinter app. Default is **sequential** for thread safety with GUI state.

---

## 12. Tutorial

### Tutorial A — First batch split (GUI)

**Goal:** Split a folder of violin one-shots into ADSR stems.

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Install [ffmpeg](https://ffmpeg.org/) if you use compressed formats.

2. **Prepare files**

   - Place all `.wav` files in one folder, e.g. `D:\Samples\Violin\`.
   - Name files with pitch for better sustain refinement, e.g. `Violin_A4_01.wav`, `Violin_G3_02.wav`.

3. **Start the application**

   ```bash
   python split_audio_segments.py
   ```

4. **Configure source**

   - Click **Browse** → select `D:\Samples\Violin\`.
   - Click **Auto-Detect Mean Length** (optional). Note the detected seconds and auto-selected preset.

5. **Choose preset**

   - For ~2 s notes, keep **Medium (1.5–3.0s)**.
   - Click **Apply Preset** if you changed selection.

6. **Verify parameters**

   - Leave **Smart Mode** enabled (Energy-guided).
   - **Attack Threshold** 0.90, **Decay Threshold** 0.50 are good starting points.
   - **Fade (ms)** 50, **Fade Type** cosine.
   - **Pitch Stability** 5¢, **Pitch Window** 0.5 s — tighten to 3¢ for solo strings if sustains are shortened too aggressively.

7. **Run**

   - Click **► RUN OPTIMIZED SPLIT**.
   - Watch the log for lines like:  
     `Att: 0.18s | Sus: 1.42s | Dec: 0.35s | PitchWin: 0.52-1.05s (σ=3.21¢)`

8. **Review**

   - In **Review Segmentation**, select a file.
   - Drag the **green** line (attack end) and **orange** line (decay start).
   - Use arrow keys for 5 ms nudges; **Shift** for 25 ms.
   - Edits auto-save new WAVs into `_Attacks`, `_Sustains`, etc.

9. **Use metadata**

   - Open `segmentation_metadata.csv` in Excel or pandas for QA.
   - Import JSON into downstream tooling.

---

### Tutorial B — Scripting without GUI

**Goal:** Process one file from a Python script.

```python
from pathlib import Path
import librosa
import soundfile as sf
import audio_segment_core as core

input_path = Path("Violin_A4.wav")
out_dir = Path("output")
out_dir.mkdir(exist_ok=True)

y, sr = librosa.load(input_path, sr=None)
cfg = core.SegmentConfig(
    use_smart=True,
    attack_threshold=0.9,
    decay_threshold=0.5,
    min_sustain_duration=0.2,
)

result = core.detect_segments(y, sr, cfg, file_path=input_path)
parts, _, _, _ = core.extract_and_fade_segments(
    y, sr,
    result.t_att, result.t_dec, result.t_end,
    result.trim,
    fade_ms=40.0,
    fade_type="cosine",
)

for folder, audio in parts.items():
    if len(audio) == 0:
        continue
    tag = folder.strip("_").replace("Release_Silence", "Release")
    if folder == "_Full_Active_Sound":
        tag = "FullActive"
    target = out_dir / folder.strip("_")
    target.mkdir(parents=True, exist_ok=True)
    sf.write(target / f"{input_path.stem}_{tag}.wav", audio, sr)

print(f"Attack ends at {result.t_att:.3f}s, decay at {result.t_dec:.3f}s")
if result.pitch_refine.get("used"):
    print("Pitch-refined sustain:", result.pitch_refine)
```

---

### Tutorial C — When to switch detection modes

| Symptom | Suggested change |
|---------|------------------|
| Attack segment includes too much steady tone | Lower **Attack Threshold** (e.g. 0.85) or enable **Advanced Mode** |
| Decay starts too early on long bows | Lower **Decay Threshold** (e.g. 0.40) or increase **Min Sustain** |
| Very short plucks get empty sustain | Use **Very Short** preset; reduce `min_sustain_duration` to ~0.06 s |
| Noisy recordings, unstable splits | Disable pitch refinement (set **Pitch Stability** very low is ineffective — instead set a large **Pitch Stability** value like 20¢ so refinement rarely applies) or use proportional-only: uncheck Smart and Advanced |
| Envelope has multiple peaks | Manual review is required; consider pre-trimming files |

---

### Tutorial D — Running tests

```bash
cd "C:\...\SPLIT_audio_segments"
pytest tests/ -v
```

Tests synthesize sine bursts with known attack/sustain/decay timing and assert boundary ordering and trim alignment.

---

## 13. Testing

| Test | Validates |
|------|-----------|
| `test_trim_active_region` | Trim shortens signal, positive active length |
| `test_energy_attack_before_peak` | Attack time precedes RMS peak |
| `test_detect_segments_smart_ordering` | `t_att < t_dec < t_end` |
| `test_detect_segments_short_sound` | Minimum sustain on brief sounds |
| `test_extract_starts_at_trim_not_file_start` | Attack index ≥ trim start |
| `test_parse_note_from_filename` | A4 → 440 Hz |
| `test_proportional_percentages_sum` | Proportional mode ordering |

---

## 14. Troubleshooting

| Issue | Cause | Remedy |
|-------|-------|--------|
| `NoBackendError` / backend errors | Missing ffmpeg or soundfile backend | Install ffmpeg; use WAV; `pip install soundfile` |
| No files found | Wrong folder or extension | Check `SUPPORTED_AUDIO_FORMATS`; files must be directly in folder (not nested unless you change code) |
| Clicks at segment edges | Cut away from zero crossing | Increase **Fade (ms)**; use cosine; manual nudge in review |
| Sustain too short after processing | Pitch refinement accepted a tight window | Increase **Pitch Stability (¢)** or **Pitch Window**; remove note from filename to disable note-weighted scoring |
| All segments similar length | Proportional-only path | Ensure Smart Mode is checked |
| MP3 load slow/fails | Codec / path | Convert to WAV for batch jobs |
| GUI frozen during batch | Long CPU work | Normal; wait for progress; do not close window |

---

## 15. Dependencies

| Package | Role |
|---------|------|
| `librosa` ≥ 0.10 | Load, trim, RMS, STFT, YIN, note names |
| `numpy` ≥ 1.23 | Numerical arrays |
| `soundfile` ≥ 0.12 | WAV/FLAC/AIFF/OGG write |
| `matplotlib` ≥ 3.7 | Review waveform plots |
| `pytest` ≥ 7.0 | Unit tests |

Standard library: `tkinter`, `threading`, `json`, `csv`, `pathlib`, `logging`, `concurrent.futures`.

---

## Appendix A — Time index cheat sheet

```text
File:     |---- leading silence ----|==== ACTIVE (trimmed) ====|---- release ----|
          0                    idx_start              idx_end              len(y)

Segments (sample indices after ZC snap):
          [idx_start : idx_att)     → Attack
          [idx_att : idx_dec)       → Sustain
          [idx_dec : idx_end)       → Decay
          [idx_end : len(y))        → Release
```

---

## Appendix B — Smart mode blend (equation)

Let \(E_{\mathrm{att}}\) be energy-based attack time, \(P_{\mathrm{att}}\) proportional attack time, active length \(L\):

\[
t_{\mathrm{att}} = 0.7\,E_{\mathrm{att}} + 0.3\,P_{\mathrm{att}}, \quad
t_{\mathrm{dec}} = 0.7\,E_{\mathrm{dec}} + 0.3\,P_{\mathrm{dec}}
\]

Then clamped so \(t_{\mathrm{dec}} - t_{\mathrm{att}} \geq t_{\mathrm{sustain,min}}\).

---

*Document generated for SPLIT_audio_segments v3.0. For code changes, update this file alongside `SegmentConfig` defaults and `PRESETS`.*
