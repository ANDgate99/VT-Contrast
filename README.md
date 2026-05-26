# VT-Contrast

Core training code for VT-Contrast.

## Directory Structure

```text
VT-Contrast/
├── configs/
│   └── training_config.yaml
├── datasets/
│   ├── train.json
│   └── validation.json
├── scripts/
│   ├── __init__.py
│   └── train.py
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── distributed_utils.py
│   ├── model_utils.py
│   ├── train_utils.py
│   └── wandb_utils.py
├── requirements.txt
├── train.sh
└── README.md
```

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

If `flash-attn` does not install cleanly through `requirements.txt`, install it separately for your CUDA/PyTorch environment:

```bash
pip install flash-attn --no-build-isolation
```

## Data

The files under `datasets/` are synthetic schema examples only:

- `datasets/train.json`
- `datasets/validation.json`

They do not include official Something-Something V2 annotations or raw videos. To train on SSv2, obtain the dataset from the official source according to its license, prepare local annotation files with the same JSON schema, and set the local video directory in `configs/training_config.yaml`:

Fake rows are committed intentionally: they document the expected schema without redistributing licensed SSv2 annotation content.

```yaml
data_config:
  video_root: "/path/to/ssv2/videos"
```

Each video is resolved as:

```text
<video_root>/<id><video_ext>
```

With the default config, this means:

```text
/path/to/ssv2/videos/<id>.webm
```

The committed sample rows use placeholder IDs and labels only. Do not treat them as training data.

## Training

Run with the default config:

```bash
./train.sh
```

Select one GPU:

```bash
./train.sh 0
```

Run multi-GPU training:

```bash
./train.sh 0,1 configs/training_config.yaml
```

Resume from the latest checkpoint under `training_config.output_dir`:

```bash
./train.sh 0 configs/training_config.yaml --resume_from_checkpoint latest
```

For multi-GPU runs, `train.sh` launches `torchrun` with one process per selected GPU. Logs and checkpoints are written under `training_config.output_dir`.
