# Nemotron3-114M SSA

This repo contains the Baby Luciole / Nemotron3-114M training setup used for
the SSA Triton v4 experiment. The main run is 22k steps on FineWeb-Edu with the
local Luciole 50k tokenizer.

## Environment

The Slurm scripts expect the NeMo container:

```text
nemo_25.04.03_arm.sif
```

Download it with Apptainer:

```bash
apptainer pull nemo_25.04.03_arm.sif docker://nvcr.io/nvidia/nemo:25.04
```

That container provides Python 3.12, PyTorch, NeMo, Megatron-Core, Triton, and
CUDA libraries. Extra local dependencies are listed in `requirements.txt`;
`datasets` is for the Hugging Face downloader and `nemo-run` is installed from
source:

```bash
pip install -r requirements.txt
```

## Data


The training datamix expects the FineWeb indexed dataset prefix
`data/fineweb_edu_text_document`. You can either generate it from Hugging Face
or copy the already-built files from the cluster.

Download FineWeb-Edu text from Hugging Face:


```bash
python3 data/download_fineweb_edu.py \
  --config sample-10BT \
  --output data/fineweb_edu_text_document.txt
```

Tokenize it into NeMo/Megatron indexed files:

```bash
sbatch scripts/tokenize_fineweb_edu.sh
```

Direct tokenization command:

```bash
python3 data/tokenize_text_to_indexed_dataset.py \
  --input-txt data/fineweb_edu_text_document.txt \
  --tokenizer tokenizer/luciole_50k \
  --output-prefix data/fineweb_edu_text_document \
  --max-tokens 10000000000
```

`data/datamix.json` is included and points to:

```json
{
  "tokenizer": "tokenizer/luciole_50k",
  "data_path": "data",
  "total_tokens": 10000000000,
  "train": [
    {
      "name": "fineweb_edu_text_document",
      "weight": 1.0
    }
  ]
}
```

## Experiment Defaults

The SSA experiment defaults are:

```text
Architecture:       baby_luciole
Sequence length:    1024
Global batch size:  768
Micro batch size:   8
Max steps:          22000
This-run max steps: 22000
Seed:               1234
SSA n:              1.5, trainable initial value
SSA b:              0.8, fixed
Kernel:             SSA Triton v4
```

## Build Index

Build the NeMo index mappings before training:

```bash
sbatch scripts/build_index.sh
```

Equivalent torchrun command for a single GPU:

```bash
torchrun --nproc_per_node=1 train/train_ssa_triton.py \
  --datamix data/datamix.json \
  --output_dir outputs \
  --name baby_luciole-ssa-triton-v4 \
  --arch baby_luciole \
  --max_steps 22000 \
  --this_run_max_steps 1 \
  --seq_length 1024 \
  --batch_size 768 \
  --micro_batch_size 1 \
  --num_nodes 1 \
  --gpus_per_node 1 \
  --save_every_n_steps 999999 \
  --skip_triton_warmup \
  --force_contiguous_qkv
```

## Train

Submit the SSA Triton v4 run:

```bash
sbatch scripts/train_ssa_triton.sh
```

The script defaults to 22k steps. To override explicitly:

```bash
GLOBAL_MAX_STEPS=22000 \
THIS_RUN_MAX_STEPS=22000 \
DATAMIX=data/datamix.json \
OUTPUT_DIR=outputs \
NAME=baby_luciole-ssa-triton-v4 \
sbatch scripts/train_ssa_triton.sh
```

Direct `torchrun` version:

```bash
torchrun --nproc_per_node=1 train/train_ssa_triton.py \
  --datamix data/datamix.json \
  --output_dir outputs \
  --name baby_luciole-ssa-triton-v4 \
  --arch baby_luciole \
  --max_steps 22000 \
  --this_run_max_steps 22000 \
  --seq_length 1024 \
  --batch_size 768 \
  --micro_batch_size 8 \
  --num_nodes 1 \
  --gpus_per_node 1 \
  --save_every_n_steps 6000 \
  --log_ssa_every_n_steps 1000 \
  --warmup_steps 500 \
  --force_contiguous_qkv \
  --seed 1234
```

Softmax baseline:

```bash
sbatch scripts/train_softmax.sh
```

## Evaluate

SSA perplexity on FineWeb + Wikipedia:

```bash
sbatch scripts/eval_perplexity_triton_v4.sh
```

Override the checkpoint if needed:

```bash
CHECKPOINT=outputs/baby_luciole-ssa-triton-v4/checkpoints/<checkpoint-name> \
sbatch scripts/eval_perplexity_triton_v4.sh
```

Softmax perplexity:

```bash
sbatch scripts/eval_perplexity.sh
```

## Benchmarks

Run LM Evaluation Harness benchmarks:

```bash
TASKS=leaderboard MODEL_SELECTION=both sbatch scripts/run_benchmarks_triton_v4.sh
```

Useful `TASKS` values:

```text
quick
standard
leaderboard
superglue_core
superglue
all
```

Results are written under `benchmark_results/`.
