# Third-party notices

## GraphGPT

Some files of `src/eulercad/` are cleaned copies of a faithful translation of GraphGPT v0.5.0
that was written for the thesis behind this repository and is not shipped here; the docstrings
call it `core/graph_gpt_baseline/`. GraphGPT itself is:

```text
GraphGPT: Generative Pre-trained Graph Eulerian Transformer
https://github.com/alibaba/graph-gpt
commit 314e7034928c8910d07e86ec1361a22b40194876
```

GraphGPT is distributed under the MIT License below. This notice must stay with the files
derived from it:

| File | Derived from (in `core/graph_gpt_baseline/`) |
|---|---|
| `representation/graph.py` | baseline `representation/graph.py` (edge checks) |
| `representation/euler.py` | baseline `representation/euler.py` (randomized Euler walk) |
| `tokenization/identity.py` | baseline `representation/identity.py` |
| `tokenization/rows.py` | baseline `representation/tokenizer.py` (short row layout) |
| `data/sample.py` | baseline `data/next_token.py` and the seeding of `data/examples.py` |
| `data/collate.py` | baseline `data/collate.py` (`DataCollatorForGSTCausal`) |
| `data/stream.py` | baseline `data/examples.py` (`PretrainingBatches`) |
| `model/embedding.py` | baseline `model/backbone.py` (token embedding) |
| `model/merge.py` | baseline `model/aggregation.py` (`StackedFeatAggregation`) |
| `model/backbone.py` | baseline `model/backbone.py` (decoder layers, masks) |
| `model/head.py` | baseline `model/heads.py` (`n_token_proj`, `lm_head`) |
| `losses/next_row.py` | baseline `model/losses.py` (`_get_ce_loss`) |
| `training/schedule.py` | baseline `training/module.py` (`learning_rate`, `WarmupDecay`) |
| `training/microbatch.py` | baseline `training/microbatch.py` |
| `training/module.py` | baseline `training/module.py` (optimizer, compiled forward, generator states) |
| `training/artifacts.py` | baseline `training/artifacts.py` (run records, checkpoints) |
| `commands/pretrain.py` | baseline `commands/cli.py` (pretraining command) |

`model/config.py` takes GraphGPT's Tiny and Mini sizes and dropout values, and
`training/config.py` its pretraining optimizer values, as settings.

```text
MIT License

Copyright (c) 2023 Alibaba Group Holding Limited

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The MIT text covers the GraphGPT source repository only. It grants no rights to the GraphGPT
paper, its figures, datasets, or checkpoints. PyTorch, Hugging Face Transformers, NetworkX,
PyTorch Geometric, and the other dependencies keep their own licenses; they are used as
installed packages, not copied here.
