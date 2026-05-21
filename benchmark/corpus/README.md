# Benchmark corpus

Synthetic WAV files are **not** stored in git (see `.gitignore`).

Generate locally:

```bash
python run_benchmark.py --generate-corpus
```

This writes 40 labeled one-shots and `annotations.json` here. Then run:

```bash
python run_benchmark.py
```

For your own recordings, create labels with:

```bash
python run_benchmark.py --template my_labels.csv
```
