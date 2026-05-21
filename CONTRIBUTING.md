# Contributing (authorised collaborators)

This is **proprietary research software**. Contributions are accepted only from authorised collaborators under explicit agreement with the copyright holder.

## Setup

```bash
pip install -e ".[dev]"
pytest
```

## Before pushing

- Run **`pytest`** from the repository root.
- Do **not** commit `installers/runtime/`, `__pycache__`, `.pytest_cache`, or generated `benchmark/corpus/*.wav`.
- Regenerate benchmark corpus locally if needed: `python run_benchmark.py --generate-corpus`.

## CI

`.github/workflows/ci.yml` runs tests on Python 3.10–3.11.

## Contact

lmr.2020@outlook.pt
