# Google Colab training guide

This guide explains how to run **tiny-llm** in [Google Colab](https://colab.research.google.com/) using the repository's [tiny-llm-colab.ipynb](tiny-llm-colab.ipynb) notebook.

Colab provides a hosted Python environment. GPU access (including T4) is **not guaranteed**, particularly on the free tier; available accelerators, session lengths, and usage quotas can change. You need a Google account and internet access.

## 1. Open the notebook

1. Open [Google Colab](https://colab.research.google.com/).
2. Choose **File → Open notebook → GitHub**.
3. Enter `https://github.com/oppiet30/tiny-llm` and select `tiny-llm-colab.ipynb`.
4. If the notebook is only on a feature branch, select that branch in Colab, or wait until [PR #4](https://github.com/oppiet30/tiny-llm/pull/4) is merged.

You can also open the notebook on GitHub and select **Open in Colab** when offered.

## 2. Choose the GPU runtime

1. In Colab, select **Runtime → Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** or another available GPU.
3. Save the setting and connect to the runtime.
4. Run the first notebook cell to check the device:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

If CUDA is unavailable, Colab has not assigned a usable NVIDIA GPU. You can continue on CPU or try again when a GPU is available.

## 3. Run the notebook in order

Use **Runtime → Run all**, or run each cell individually:

1. **GPU check:** reports PyTorch version and GPU availability.
2. **Clone:** downloads `oppiet30/tiny-llm` to `/content/tiny-llm` and uses Colab's installed PyTorch.
3. **Dataset:** prepares a character-level dataset with `prepare_data.py`. If `data/input.txt` is absent, Colab asks you to upload a UTF-8 text file. Use only data you are permitted to train on.
4. **Google Drive:** asks you to authorize access and creates `MyDrive/tiny-llm-checkpoints`.
5. **Training:** invokes `train.py` with an initial target of **250 steps**, saving intermediate and final checkpoints.
6. **Results:** lists saved checkpoint files and benchmark SQL files.

**Important:** Do not install the repository's pinned `requirements.txt` in Colab unless you have checked compatibility with the runtime's preinstalled CUDA/PyTorch stack. The notebook relies on the Colab environment.

## 4. Adjust training length

In the training cell, edit:

```python
steps = 250
interval = 100
```

For example, change `steps` to `1000` after a successful short test. The step count is a **total target**, not additional steps on top of a resumed checkpoint. The notebook resumes automatically when it finds a checkpoint with a lower step number.

The training command is equivalent to:

```bash
python train.py --steps 250 --checkpoint-interval 100 --checkpoint-dir /content/drive/MyDrive/tiny-llm-checkpoints
```

When a prior checkpoint exists and its step is below the requested target, the notebook adds `--resume /content/drive/MyDrive/tiny-llm-checkpoints/model.pt`.

Only load checkpoints you created or trust. PyTorch checkpoint loading may deserialize Python objects.

## 5. Save and recover results

Checkpoints are stored in **Google Drive → My Drive → tiny-llm-checkpoints**:

- `model.pt` — latest model checkpoint.
- `checkpoint-XXXXXXXX.pt` — step-numbered checkpoint files.
- `checkpoint.txt` — latest saved step and target.

Training also generates SQL benchmark files under `/content/tiny-llm/benchmarks/`. **These SQL files are not automatically backed up to Drive.** Download or copy any you need before Colab disconnects.

The notebook does **not** enable MariaDB uploads and does not require database credentials. Avoid putting passwords, tokens, or private datasets in shared notebooks.

## 6. Troubleshooting

- **No GPU:** confirm the runtime accelerator setting, reconnect if appropriate, or use CPU. Free GPU capacity may be exhausted.
- **Out of memory:** reduce the training model/batch configuration in `train.py` or choose a higher-memory runtime if available. A T4 commonly has 16 GB GPU memory, but assignment varies.
- **Dataset missing or too small:** upload a larger UTF-8 text file; `train.py` requires both training and validation splits to exceed its 128-token block size.
- **Colab disconnects:** reconnect, rerun setup and dataset preparation, mount the same Drive, then set a higher `steps` target to resume.
- **Package conflict:** restart with a clean Colab runtime and use its preinstalled PyTorch rather than forcing the project's pinned CUDA dependencies.
- **Unexpected CPU training:** ensure the version of `train.py` includes CUDA device selection (`torch.cuda.is_available()`), introduced with this notebook.

## 7. Limitations and next steps

The initial workflow is intended as a reproducible trial, not a promise of uninterrupted long training. Google Colab runtimes are temporary, and training speed depends on the assigned accelerator and data. To continue training, increase the target steps and reuse the same Google Drive checkpoints.

Project: [tiny-llm on GitHub](https://github.com/oppiet30/tiny-llm) · [GitHub Sponsors](https://github.com/sponsors/oppiet30)
