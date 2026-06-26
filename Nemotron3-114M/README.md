# Nemotron3-114M SSA Training

This repository contains training, evaluation, and benchmark launchers for a
Baby Luciole / Nemotron-style 114M parameter language model. The main
experiment path replaces standard softmax attention with SSA
(Softmax-Substituted Attention) implemented as a Triton v4 fused attention
kernel.

The code is organized for Slurm + Apptainer GPU runs, with Python entry points
kept under `train/` for direct execution or adaptation.


## Requirements

The launchers assume a GPU training environment with:

- Slurm (`sbatch`, `srun`) for the scripts in `scripts/`.
- Apptainer with an NVIDIA-enabled NeMo container.
- CUDA-capable GPUs.
- PyTorch, PyTorch Lightning, NeMo, Megatron-Core, Triton, and Fiddle.
- `lm-eval` and cached Hugging Face datasets for benchmark runs.
- A repository-compatible `utils.py` on `PYTHONPATH`; the training scripts
  import `process_datamix_file`, `check_tokenizer`, `save_config`,
  `read_datamix_file`, and `get_data_paths` from that module.

Run all commands from the repository root unless noted otherwise.

## Data And Path Templates

Use repo-relative paths for local artifacts whenever possible:

```text
data/datamix.json
data/fineweb_edu_text_document
data/wikipedia_en_text_document
tokenizers/luciole_50k
outputs/<run-name>/
outputs/<run-name>/checkpoints/<checkpoint-name>
outputs/eval/<result-name>.json
benchmark_results/<result-name>.json
```

For indexed NeMo/Megatron datasets, pass the dataset prefix without `.bin` or
`.idx`. For example, use `data/fineweb_edu_text_document` when the actual files
are:

```text
data/fineweb_edu_text_document.bin
data/fineweb_edu_text_document.idx
```

### Datamix Template


```json
{
  "tokenizer": "tokenizers/luciole_50k",
  "train": [
    {
      "weight": 1.0,
      "path": "data/fineweb_edu_text_document",
      "num_tokens": 50000000000
    }
  ],
  "validation": [],
  "test": []
}
```

If your parser uses flattened NeMo blend paths, the effective training paths
should resolve to this form:

```json
{
  "train": [
    "1.0",
    "data/fineweb_edu_text_document"
  ]
}
```

## Training

### SSA Triton v4

Submit the SSA Triton v4 training job with explicit paths:

```bash
DATAMIX=data/datamix.json \
OUTPUT_DIR=outputs \
NAME=baby_luciole-ssa-triton-v4 \
GLOBAL_MAX_STEPS=60000 \
THIS_RUN_MAX_STEPS=30000 \
LR_WARMUP_STEPS=500 \
SEED=1234 \
sbatch scripts/train_ssa_triton.sh
```

Important knobs:

- `DATAMIX`: datamix JSON used by `train/train_ssa_triton.py`.
- `OUTPUT_DIR`: parent directory for run outputs.
- `NAME`: run name; checkpoints are written under
  `outputs/<run-name>/checkpoints/`.
- `GLOBAL_MAX_STEPS`: absolute training horizon.
- `THIS_RUN_MAX_STEPS`: optional per-job step cap for Slurm time slicing.
- `LR_WARMUP_STEPS`: scheduler warmup steps.
- `SKIP_TRITON_WARMUP=1`: skip pre-compiling Triton kernels.
- `DISABLE_COMPILED_BDA=1`: use eager bias-dropout-add.
- `FORCE_CONTIGUOUS_QKV=1`: materialize contiguous Q/K/V before Triton attention.
- `SSA_KERNEL_VERSION`: must be `v4`.

The SSA policy is fixed in the launcher: `ssa_n=1.5`, `ssa_b=0.8`, `n` is
learnable, and `b` is fixed.

### Softmax Baseline

Submit the softmax baseline job:

```bash
DATAMIX=data/datamix.json \
OUTPUT_DIR=outputs \
NAME=baby_luciole_softmax \
SEED=1234 \
sbatch scripts/train_softmax.sh
```

The softmax launcher uses `train/train_sftmax.py` and writes checkpoints under
`outputs/<run-name>/checkpoints/`.

### Direct Python Entry Points

Use direct Python execution when adapting the launchers outside Slurm:

```bash
torchrun --nproc_per_node=1 train/train_ssa_triton.py \
  --datamix data/datamix.json \
  --output_dir outputs \
  --name baby_luciole-ssa-triton-v4 \
  --arch baby_luciole \
  --max_steps 60000 \
  --this_run_max_steps 30000 \
  --seq_length 1024 \
  --batch_size 768 \
  --micro_batch_size 8 \
  --num_nodes 1 \
  --gpus_per_node 1 \
  --save_every_n_steps 6000 \
  --warmup_steps 500 \
  --force_contiguous_qkv
```

```bash
torchrun --nproc_per_node=1 train/train_softmax.py \
  --datamix data/datamix.json \
  --output_dir outputs \
  --name baby_luciole_softmax \
  --arch baby_luciole \
  --max_steps 10500 \
  --global_max_steps 60000 \
  --seq_length 1024 \
  --batch_size 768 \
  --micro_batch_size 8 \
  --num_nodes 1 \
  --gpus_per_node 1 \
  --save_every_n_steps 5000
```

## Perplexity Evaluation

### SSA Triton v4 On FineWeb And Wiki

```bash
CHECKPOINT=outputs/baby_luciole-ssa-triton-v4/checkpoints/<checkpoint-name> \
FW_DATA_PATH=data/fineweb_edu_text_document \
WIKI_DATA_PATH=data/wikipedia_en_text_document \
TOKENIZER=tokenizers/luciole_50k \
NUM_SAMPLES=1000 \
SEQ_LENGTH=1024 \
BATCH_SIZE=8 \
OUTPUT=outputs/eval/perplexity_triton_v4.json \
sbatch scripts/eval_perplexity_triton_v4.sh
```

To resolve evaluation data from datamix files instead of explicit dataset
prefixes:

```bash
CHECKPOINT=outputs/baby_luciole-ssa-triton-v4/checkpoints/<checkpoint-name> \
FW_DATA_PATH="" \
FW_DATAMIX=data/datamix_fineweb.json \
WIKI_DATA_PATH="" \
WIKI_DATAMIX=data/datamix_wiki.json \
TOKENIZER=tokenizers/luciole_50k \
OUTPUT=outputs/eval/perplexity_triton_v4.json \
sbatch scripts/eval_perplexity_triton_v4.sh
```

If both `FW_DATA_PATH` and `FW_DATAMIX` are set, the explicit data path wins.
The same rule applies to `WIKI_DATA_PATH` and `WIKI_DATAMIX`.

### Softmax Baseline

```bash
CHECKPOINT=outputs/baby_luciole_softmax/checkpoints/<checkpoint-name> \
DATA_PATH=data/wikipedia_en_text_document \
TOKENIZER=tokenizers/luciole_50k \
NUM_SAMPLES=1000 \
SEQ_LENGTH=1024 \
BATCH_SIZE=8 \
OUTPUT=outputs/eval/perplexity_softmax.json \
sbatch scripts/eval_perplexity.sh
```

## Benchmarks

Run LM Evaluation Harness benchmarks through the Slurm wrapper:

```bash
MODEL_SELECTION=both \
SSA_CHECKPOINT=outputs/baby_luciole-ssa-triton-v4/checkpoints/<checkpoint-name> \
SOFTMAX_CHECKPOINT=outputs/baby_luciole_softmax/checkpoints/<checkpoint-name> \
TOKENIZER=tokenizers/luciole_50k \
TASKS=leaderboard \
BATCH_SIZE=16 \
MAX_LENGTH=2048 \
OUTPUT_DIR=benchmark_results \
sbatch scripts/run_benchmarks_triton_v4.sh
```

`MODEL_SELECTION` can be `both`, `ssa`, or `softmax`.

Supported task groups in `train/run_benchmarks_triton_v4.py`:

```text
quick
standard
leaderboard
superglue_core
superglue
all
```

You can also pass comma-separated task names such as:

```text
arc_easy,arc_challenge,hellaswag,winogrande,truthfulqa,gsm8k
```

Direct benchmark example:

```bash
python train/run_benchmarks_triton_v4.py \
  --model "ssa_triton_v4|ssa_triton_v4|outputs/baby_luciole-ssa-triton-v4/checkpoints/<checkpoint-name>" \
  --model "softmax|softmax|outputs/baby_luciole_softmax/checkpoints/<checkpoint-name>" \
  --tasks leaderboard \
  --tokenizer tokenizers/luciole_50k \
  --batch_size 16 \
  --max_length 2048 \
  --output benchmark_results/leaderboard.json
```

## Outputs

Training writes:

```text
outputs/<run-name>/checkpoints/
outputs/job_<slurm-job-id>/
outputs/index_mapping/
slurm/<job-name>_<job-id>.out
```

Perplexity evaluation writes JSON files to the path provided by `OUTPUT`, for
example:

```text
outputs/eval/perplexity_triton_v4.json
outputs/eval/perplexity_softmax.json
```

Benchmarks write JSON files under `benchmark_results/` by default.

## Notes For Porting

- The Slurm scripts in `scripts/` contain cluster-specific Apptainer and
  container paths. Keep the repo-relative command structure, but adapt those
  environment paths for a different cluster.
- `train/train_ssa_triton.py` is pinned to `SSA_KERNEL_VERSION=v4`.
- `train/eval_perplexity_triton_v4.py` expects checkpoints to be directories
  containing a `weights/` subdirectory.
- Benchmark runs default to offline Hugging Face datasets through
  `HF_DATASETS_OFFLINE=1`; make sure the required datasets are cached or set the
  environment appropriately for your cluster.
- The repository currently documents and exposes Baby Luciole through
  `recipes/baby_luciole.py`; other architectures listed in
  `recipes/recipe_utils.py` require their corresponding recipe modules or NeMo
  built-in recipes to be available.