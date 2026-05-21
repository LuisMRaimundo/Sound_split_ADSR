# Sound Split ADSR

**Repository:** [github.com/LuisMRaimundo/Sound_split_ADSR](https://github.com/LuisMRaimundo/Sound_split_ADSR)

Desktop tool for **automatic ADSR segmentation** of monophonic or quasi-monophonic audio (orchestral one-shots, sample-library prep, spectral-analysis pipelines). Splits each file into **Attack**, **Sustain**, **Decay**, and **Release** regions with optional manual review, JSON/CSV metadata, and a boundary-error benchmark.

Related research tooling: [Interval-Homogeneity-Analyser](https://github.com/LuisMRaimundo/Interval-Homogeneity-Analyser).

---

## No Python installed? (one-click)

See **[installers/README.md](installers/README.md)**:

| Platform | Launcher |
|----------|----------|
| **Windows 10/11** | Double-click `installers\windows\Install and Run.bat` |
| **macOS** | Double-click `installers/macos/Install and Run.command` (after `chmod +x`) |
| **Linux** | `./installers/linux/install-and-run.sh` |

First run downloads a private Python and libraries (~150–250 MB), then opens the **graphical ADSR splitter**. No system Python, pip, or conda required.

---

## Developers (Python already installed)

```bash
pip install -e ".[dev]"
python split_audio_segments.py          # GUI
python split_audio_cli.py -f ./samples  # headless batch
python run_benchmark.py --generate-corpus && python run_benchmark.py
pytest
```

**MP3/M4A:** install [ffmpeg](https://ffmpeg.org/) on your PATH.

---

## What it does

| Output folder | Content |
|---------------|---------|
| `_Attacks/` | Onset → attack boundary |
| `_Sustains/` | Attack → decay boundary |
| `_Decays/` | Decay → end of active sound |
| `_Release_Silence/` | Tail after active energy |
| `_Full_Active_Sound/` | Full trimmed active region |

Detection modes: **smart** (energy + proportional anchors, default), **advanced** (spectral flux + derivatives), **proportional**. Pitch refinement: **expand** (default), **annotate** (full sustain for STFT + metadata), **crop** (tight stable window).

---

## Documentation

| Document | Description |
|----------|-------------|
| [QUICK_GUIDE.md](QUICK_GUIDE.md) | Non-specialist workflow |
| [docs/TECHNICAL_MANUAL.md](docs/TECHNICAL_MANUAL.md) | Full DSP specification, API, tutorials |
| [installers/README.md](installers/README.md) | Autonomous installers (Windows / macOS / Linux) |
| [# Copyright and Use Notice.md](#%20Copyright%20and%20Use%20Notice.md) | Proprietary terms |
| [docs/ACKNOWLEDGEMENTS.md](docs/ACKNOWLEDGEMENTS.md) | Funding and thanks |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Tests and CI for authorised contributors |

---

## Benchmark

Reproducible synthetic corpus (40 labeled one-shots) and per-mode/per-preset mean boundary error (ms):

```bash
python run_benchmark.py --generate-corpus
python run_benchmark.py
```

Reports: `benchmark/results/benchmark_report.txt`. Label your own recordings with `python run_benchmark.py --template my_labels.csv`.

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

GitHub Actions runs `pytest` on push (see `.github/workflows/ci.yml`).

---

## Copyright and use

Copyright © 2026 Luís Raimundo. All rights reserved.

This repository and its contents are proprietary research material. **No open-source licence is granted.** No permission to copy, redistribute, modify, publish, or derive works without prior written permission from the copyright holder.

**Contact:** lmr.2020@outlook.pt

---

## Acknowledgements

This project was developed by **Luís Raimundo** with the support and funding of the **Fundação para a Ciência e a Tecnologia (FCT)** and **Universidade NOVA de Lisboa**.

**Funding DOI:** [https://doi.org/10.54499/2020.08817.BD](https://doi.org/10.54499/2020.08817.BD)

The author also gratefully acknowledges **Isabel Pires** for her support throughout the development of this work.
