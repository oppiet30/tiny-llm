# Keras 3 transformer experiment

This experimental benchmark leaves `train.py`, checkpoints, and MariaDB untouched.

From the repository root, in a virtual environment with PyTorch installed:

```bash
python -m pip install "keras>=3,<4"
python experiments/keras_benchmark.py --steps 100 --batch-size 4 --threads 2
```

On a Raspberry Pi 5, use a compatible CPU-only PyTorch wheel. Keras uses the **PyTorch backend**, configured before import in this script. This does **not** benchmark TensorFlow.

It reads existing `data/meta.json`, `data/train.bin`, and `data/val.bin`, trains a separate 4-layer, 4-head, 128-embedding character-level causal transformer, and prints training time, throughput, and a one-batch validation loss. It saves no checkpoints or SQL records.

**Comparison caveats:** The Keras model uses separate token and output projection weights, while `model.py` ties them; attention projections and training implementations also differ. Its parameter count, initial weights, sampling, and loss evaluation are not identical. Consequently this is an exploratory Keras workflow test, **not** a controlled PyTorch-versus-Keras performance claim. A fair benchmark requires matching parameterization, dataset batches, optimizer settings, evaluation procedure, and warmup.

Start with 2 steps to confirm compatibility before running 100. Do not compare this short benchmark's validation batch with the PyTorch trainer's 10-batch validation estimate.

## Optional MariaDB upload

Once your existing `benchmark-upload.local.json` and `TINY_LLM_DB_PASSWORD`
are configured and this host is authorized, run from the repository root:

```bash
python experiments/keras_benchmark.py --data-dir data/huckleberry --steps 100 --batch-size 4 --threads 2 --upload
```

You may add `--db-host HOST` to override the configured database server.
Uploads use the existing `benchmark_upload.upload_benchmark` function,
including host allowlisting and deduplication by upload ID. Keras models are
stored under distinct names containing `keras3-torch`, with framework/backend
and single-batch validation caveats in the run notes. No schema migration is
required. The existing `pytorch_version` field records the actual PyTorch
backend version; Keras version is stored in notes.

Unlike `train.py`, this experimental runner does not generate a SQL fallback
file. If upload fails, training still completes, but you must rerun with
`--upload` to record another benchmark. The earlier benchmark run is not
backfilled automatically.
