# Quick Guide — Sound Split ADSR

For users who want to split instrument one-shots without reading the full technical manual.

## 1. Install and open

**Easiest:** use the one-click installer for your system (see [installers/README.md](../installers/README.md)).

**Already have Python 3.10+:**

```bash
pip install -e .
python split_audio_segments.py
```

## 2. Prepare files

- Put all audio files in **one folder** (`.wav` recommended; MP3 needs ffmpeg).
- Name files with pitch when possible, e.g. `Violin_A4_01.wav` (helps sustain detection).

## 3. Run a batch split

1. **Browse** → select your folder.
2. Choose a **Preset** matching average note length (or **Auto-Detect Mean Length**).
3. Leave **Smart Mode** on for most orchestral material.
4. For **spectral analysis / STFT**, set **Pitch Refine** to **annotate** (keeps long sustains).
5. Click **► RUN OPTIMIZED SPLIT**.
6. Use **Review Segmentation** to drag attack (green) and decay (orange) lines if needed.

## 4. Outputs

Next to your source files:

- `_Attacks/`, `_Sustains/`, `_Decays/`, `_Release_Silence/`, `_Full_Active_Sound/`
- `segmentation_metadata.json` and `.csv`

## 5. Presets at a glance

| Preset | Typical use |
|--------|-------------|
| Very Short | Plucks, staccato |
| Short / Medium | Most single notes |
| Long / Very Long | Sustained bowed notes (5–7 s) |
| Legato / Bow | Long notes with vibrato |
| Staccato / Pluck | Short attacks, advanced detection |

## 6. Need help?

See [docs/TECHNICAL_MANUAL.md](TECHNICAL_MANUAL.md) §14 Troubleshooting.

## Copyright

Copyright © 2026 Luís Raimundo. Proprietary research material — see [# Copyright and Use Notice.md](../#%20Copyright%20and%20Use%20Notice.md).
