# EulerCAD

EulerCAD pretrains a decoder-only transformer on 2D CAD drawings. Each drawing is a graph of points and
straight segments, and the model reads it as a random Euler walk: an ordered sequence of rows that crosses
every segment once. It learns to predict the next row. A drawing's embedding is the mean of one layer's
hidden states over the walk.

The drawings are 2D views of the 3D objects of the [ABC dataset](https://deep-geometry.github.io/abc-dataset/).
The recipe adapts [GraphGPT](https://github.com/alibaba/graph-gpt) to drawings. The published model,
**Deep**, has 14 layers of width 384 (36.9M parameters) and was trained on one RTX 4060 Ti (16 GB) for
208,200 updates of 160 drawings, on 246,108 training drawings. Its best validation loss is 0.7144, at
update 208,000.

- Model: [NicoloLocatelli/EulerCAD-Deep](https://huggingface.co/NicoloLocatelli/EulerCAD-Deep)
- Data: [NicoloLocatelli/EulerCAD-ABC-Preprocessed](https://huggingface.co/datasets/NicoloLocatelli/EulerCAD-ABC-Preprocessed)

## Install

You need Python 3.12, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU with a CUDA 13 driver for
training. The analysis also runs on a CPU, slowly.

```bash
git clone https://github.com/NicoloLocatelli/EulerCAD.git
cd EulerCAD
uv sync
```

To rebuild the dataset from the raw ABC files you also need the Open CASCADE kernel:
`uv sync --extra abc`.

## Use

Run every command from the repository folder. Data go to `data/`, runs to `runs/`, and analyses to
`outputs/`. To keep them elsewhere, set `EULERCAD_HOME`.

**1. Get the data.** Download the preprocessed point graphs (about 3.3 GB) from
[EulerCAD-ABC-Preprocessed](https://huggingface.co/datasets/NicoloLocatelli/EulerCAD-ABC-Preprocessed):

```bash
uv run eulercad data download
```

Or rebuild them from raw ABC. This downloads 5 archives of 10,000 STEP files, projects every object
along 20 directions, cleans the drawings, splits them by object into train, validation and test, and
builds the point graphs. It takes hours and about 120 GB of disk.

```bash
uv sync --extra abc
uv run eulercad data build-abc --chunks 0-4
```

**2. Get the model.** Download [Deep](https://huggingface.co/NicoloLocatelli/EulerCAD-Deep) into
`runs/deep/checkpoints/`:

```bash
uv run eulercad model download
```

**3. Train.** Pretrain a model with the recipe of Deep. Any other option goes to the trainer, so you can
change it, for example `--workers 8` or `--total-steps 40000`. `uv run eulercad train --help` lists them.

```bash
uv run eulercad train --preset deep --experiment my_run
```

The run writes `runs/my_run/`: `run.json` (what the run is), `training.jsonl` (the losses), and
`checkpoints/`. Run the same command with `--resume` to continue a stopped run.

**4. Analyze a checkpoint.** Measure every layer with the seven measures of
[Layer by Layer](https://arxiv.org/abs/2502.02013) (prompt entropy, curvature, dataset entropy, effective
rank, InfoNCE, LiDAR, DiME), then embed the whole split and score retrieval of the other views of each
3D object, per layer:

```bash
uv run eulercad analyze runs/deep/checkpoints/model.safetensors --split val
uv run eulercad analyze runs/my_run/checkpoints/best.ckpt --split val
```

The results go to `outputs/<run>/`: `layers_<checkpoint>_<split>/` holds `metrics.json` and one plot per
measure, and `embeddings_<checkpoint>/` holds the embeddings and `retrieval.json`. Each step also runs
alone, with all its options: `uv run eulercad measures --help`, `embed --help`, `retrieval --help`.

## Layout

```
src/eulercad/
├── abc/            ABC download, 2D projection, cleaning, splits, point graphs
├── representation/ drawing graph and Euler walk
├── tokenization/   rows of the walk and their vocabulary
├── data/           datasets, batches, duplicate tables
├── model/          the transformer
├── losses/         next-row loss, SIGReg
├── training/       optimizer, schedule, Lightning module, run folders
├── commands/       the trainer's command line
├── analysis/       layer measures, embeddings, retrieval
├── hub.py          published weights and Hugging Face downloads
└── cli.py          the `eulercad` command
```

Run the tests with `uv run pytest`.

## License and data

The code is under the MIT License (`LICENSE`). Parts of it derive from GraphGPT, also under the MIT
License; see `THIRD_PARTY_NOTICES.md`.

The ABC dataset is not ours. Its CAD models belong to their creators and are distributed under the
[Onshape Terms of Use](https://www.onshape.com/legal/terms-of-use), section 1.g.ii. If you use the data,
please cite ABC:

```bibtex
@inproceedings{koch2019abc,
  author    = {Koch, Sebastian and Matveev, Albert and Jiang, Zhongshi and Williams, Francis and Artemov, Alexey and Burnaev, Evgeny and Alexa, Marc and Zorin, Denis and Panozzo, Daniele},
  title     = {{ABC}: A Big {CAD} Model Dataset for Geometric Deep Learning},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019}
}
```
