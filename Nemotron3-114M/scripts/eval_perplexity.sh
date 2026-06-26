#!/bin/bash
#SBATCH -J eval_bbyluc_ppl
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x_%j.out

# Ensure output directory exists
mkdir -p slurm

# Checkpoint path - use latest checkpoint from SSA training
REPO_DIR=${REPO_DIR:-"$PWD"}
CHECKPOINT_PATH=${CHECKPOINT:-"outputs/baby_luciole_softmax/checkpoints/baby_luciole_softmax-step=0022000-last"}


# Data path - preprocessed FineWeb data
DATA_PATH=${DATA_PATH:-"data/wikipedia_en_text_document"}


# Tokenizer
TOKENIZER_PATH=${TOKENIZER:-"tokenizer/luciole_50k"}

# Evaluation parameters
NUM_SAMPLES=${NUM_SAMPLES:-1000}
SEQ_LENGTH=${SEQ_LENGTH:-1024}
BATCH_SIZE=${BATCH_SIZE:-8}
SEED=${SEED:-42}

# Output file for results
OUTPUT_FILE=${OUTPUT:-"outputs/eval/perplexity_softmax_${SLURM_JOB_ID}.json"}
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "=========================================="
echo "Evaluating Perplexity: Baby Luciole SSA"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data:       $DATA_PATH"
echo "Tokenizer:  $TOKENIZER_PATH"
echo "Samples:    $NUM_SAMPLES"
echo "Seq Length: $SEQ_LENGTH"
echo "Batch Size: $BATCH_SIZE"
echo "Output:     $OUTPUT_FILE"
echo "=========================================="

# Run evaluation with apptainer
apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --bind "${REPO_DIR}:${REPO_DIR}" \
    --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    python3 "${REPO_DIR}/train/eval_perplexity.py" \
        --checkpoint "$CHECKPOINT_PATH" \
        --tokenizer "$TOKENIZER_PATH" \
        --data_path "$DATA_PATH" \
        --num_samples $NUM_SAMPLES \
        --seq_length $SEQ_LENGTH \
        --batch_size $BATCH_SIZE \
        --no_ssa \
        --seed $SEED \
        --output "$OUTPUT_FILE"

status=$?
echo "=========================================="
echo "Perplexity evaluation finished with status $status"
echo "=========================================="
exit $status
