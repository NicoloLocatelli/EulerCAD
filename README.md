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
- Model fine-tuned with SIGReg: [NicoloLocatelli/EulerCAD-Deep-SIGReg](https://huggingface.co/NicoloLocatelli/EulerCAD-Deep-SIGReg)
- Data: [NicoloLocatelli/EulerCAD-ABC-Preprocessed](https://huggingface.co/datasets/NicoloLocatelli/EulerCAD-ABC-Preprocessed)

This is the code of the MSc thesis of Nicolò Locatelli at Politecnico di Milano. For an overview of the work and
its results, read the six-page [executive summary](docs/Locatelli_Nicolo_Executive_Summary.pdf).

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

**Deep-SIGReg** is Deep fine-tuned for 20,000 more updates with [SIGReg](https://arxiv.org/abs/2511.08544),
a loss that pushes the drawing embeddings of every layer toward an isotropic Gaussian. Its next-row loss is
slightly worse (0.7191), but its embeddings retrieve the other views of an object better on mean average
precision. Download it into `runs/deep-sigreg/checkpoints/`:

```bash
uv run eulercad model download deep-sigreg
```

**3. Train.** Pretrain a model with the recipe of Deep. Any other option goes to the trainer, so you can
change it, for example `--workers 8` or `--total-steps 40000`. `uv run eulercad train --help` lists them.

```bash
uv run eulercad train --preset deep --experiment my_run
```

For a lighter test of the whole pipeline, `deep-small-test-run` uses the same recipe at width 128 (4.3M
parameters) for 20,000 updates. On one RTX 4060 Ti it took about 1 hour 20 minutes and reached a
validation loss of 1.0033.

```bash
uv run eulercad train --preset deep-small-test-run
```

To repeat the SIGReg fine-tune, download Deep first (step 2), then run:

```bash
uv run eulercad train --preset deep-sigreg
```

The run writes `runs/my_run/`: `run.json` (what the run is), `training.jsonl` (the losses), and
`checkpoints/`. Run the same command with `--resume` to continue a stopped run.

**4. Analyze a checkpoint.** Measure every layer with the seven measures of
[Layer by Layer](https://arxiv.org/abs/2502.02013) (prompt entropy, curvature, dataset entropy, effective
rank, InfoNCE, LiDAR, DiME), then embed the whole split and score retrieval of the other views of each
3D object, per layer:

```bash
uv run eulercad analyze runs/deep/checkpoints/model.safetensors --split val
uv run eulercad analyze runs/deep-sigreg/checkpoints/model.safetensors --split val
uv run eulercad analyze runs/my_run/checkpoints/best.ckpt --split val
```

The results go to `outputs/<run>/`: `layers_<checkpoint>_<split>/` holds `metrics.json` and one plot per
measure, and `embeddings_<checkpoint>/` holds the embeddings and `retrieval.json`. Each step also runs
alone, with all its options: `uv run eulercad measures --help`, `embed --help`, `retrieval --help`.

## The `eulercad` command

`uv sync` installs one command, `eulercad`. Run it as `uv run eulercad <command>`, from the repository
folder. Add `--help` to any command to see its options.

| Command | What it does |
|---|---|
| `data download` | Download the preprocessed point graphs into `data/ABC/`. |
| `data build-abc --chunks 0-4` | Rebuild them from raw ABC instead. Needs `uv sync --extra abc`. Other options go to the projection step. |
| `model download [deep\|deep-sigreg]` | Download a published model into `runs/<name>/checkpoints/`. The default is `deep`. |
| `model export <checkpoint> <folder>` | Write a training checkpoint as `model.safetensors` and `config.json`, the form of the published models. |
| `train [--preset <name>] [options]` | Pretrain or fine-tune a model. |
| `analyze <checkpoint> [--split val] [--no-retrieval]` | Measure every layer, then embed the split and score retrieval. |
| `measures`, `embed`, `retrieval` | Run one step of `analyze` alone. |

### Training

A preset is a full recipe. Any option you add after it replaces the preset's value, so
`train --preset deep --total-steps 40000` is Deep's recipe with 40,000 updates.

| Preset | What it trains |
|---|---|
| `deep` | Deep: width 384, 14 layers, 208,200 updates. |
| `deep-small-test-run` | The same recipe at width 128 for 20,000 updates, to test the pipeline. |
| `deep-sigreg` | Deep fine-tuned with SIGReg for 20,000 updates. Needs `model download` first. |

Without a preset you give every option yourself, and `--experiment` and `--total-steps` are required.
`uv run eulercad train --help` lists all the options. These are the ones you will most likely change:

- `--experiment <name>`: the run folder, `runs/<name>/`. The default is the preset's name.
- `--workers <n>`: processes that build the batches (default 14). Use fewer on a machine with fewer
  CPU cores. The batches stay the same.
- `--max-steps <n>`: stop after `n` updates. The learning-rate schedule still follows `--total-steps`.
- `--cell-budget <n>`: the batch runs in groups of at most `n` rows times drawings. If the GPU runs out of
  memory, lower it. The presets have no dropout, so the loss and the gradients stay the same.
- `--resume`: continue a stopped run from `last.ckpt`. Repeat the original command and add `--resume`.
- `--print-config`: print the settings the run would use, and exit.

A run writes `runs/<experiment>/`:

- `run.json`: the settings of the run.
- `training.jsonl`: the training and validation losses.
- `checkpoints/`: `best.ckpt` (the lowest validation loss so far), `last.ckpt` (the latest), and the last
  few step checkpoints.

### Analysis

`analyze` takes either a downloaded `model.safetensors` or a `.ckpt` of your own run. `--split` picks
`train`, `val` (the default) or `test`. `--no-retrieval` skips the embeddings and the retrieval.
Any other option goes to the measures step, for example `--drawings 500` to measure fewer
drawings. Every step uses the GPU when there is one, and the CPU otherwise.

### Paths

Every relative path, such as `data/`, `runs/` and `outputs/`, is resolved under the folder in
`EULERCAD_HOME`. Without it, they are resolved under the current folder. For example, to keep the data and
runs on another disk:

```bash
export EULERCAD_HOME=/mnt/big_disk/eulercad
```

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
