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
