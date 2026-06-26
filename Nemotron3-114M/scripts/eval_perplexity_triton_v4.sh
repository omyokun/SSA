#!/bin/bash
#SBATCH -J eval_bbyluc_ppl_triton_v4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

REPO_DIR=${REPO_DIR:-"$PWD"}
CHECKPOINT_PATH=${CHECKPOINT:-"outputs/baby_luciole-ssa-triton-v4/checkpoints/baby_luciole-ssa-triton-v4-step=0022000-last"}

FW_DATA_PATH=${FW_DATA_PATH:-"data/fineweb_edu_text_document"}
WIKI_DATA_PATH=${WIKI_DATA_PATH:-"data/wikipedia_en_text_document"}

FW_DATAMIX=${FW_DATAMIX:-""}
WIKI_DATAMIX=${WIKI_DATAMIX:-""}

TOKENIZER_PATH=${TOKENIZER:-"tokenizer/luciole_50k"}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
SEQ_LENGTH=${SEQ_LENGTH:-1024}
BATCH_SIZE=${BATCH_SIZE:-8}
SEED=${SEED:-42}

COMPILED_BDA=${COMPILED_BDA:-0}
FORCE_CONTIGUOUS_QKV=${FORCE_CONTIGUOUS_QKV:-1}

OUTPUT_FILE=${OUTPUT:-"outputs/eval/perplexity_triton_v4_fw_wiki_${SLURM_JOB_ID}.json"}
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "=========================================="
echo "Evaluating Perplexity: Baby Luciole SSA Triton v4"
echo "Checkpoint:    $CHECKPOINT_PATH"
echo "FineWeb data:  $FW_DATA_PATH"
echo "Wiki data:     $WIKI_DATA_PATH"
echo "Tokenizer:     $TOKENIZER_PATH"
echo "Samples/data:  $NUM_SAMPLES"
echo "Seq length:    $SEQ_LENGTH"
echo "Batch size:    $BATCH_SIZE"
echo "Compiled BDA:  $COMPILED_BDA"
echo "Contig QKV:    $FORCE_CONTIGUOUS_QKV"
echo "Output JSON:   $OUTPUT_FILE"
echo "=========================================="

EXTRA_ARGS=()

if [[ -n "${FW_DATAMIX}" ]]; then
    EXTRA_ARGS+=(--fw-datamix "${FW_DATAMIX}")
fi

if [[ -n "${WIKI_DATAMIX}" ]]; then
    EXTRA_ARGS+=(--wiki-datamix "${WIKI_DATAMIX}")
fi

if [[ "${COMPILED_BDA}" == "0" ]]; then
    EXTRA_ARGS+=(--no-compiled-bda)
fi

if [[ "${FORCE_CONTIGUOUS_QKV}" == "0" ]]; then
    EXTRA_ARGS+=(--no-force-contiguous-qkv)
fi

apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --bind "${REPO_DIR}:${REPO_DIR}" \
    --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    python3 "${REPO_DIR}/train/eval_perplexity_triton_v4.py" \
        --checkpoint "$CHECKPOINT_PATH" \
        --tokenizer "$TOKENIZER_PATH" \
        --fw-data-path "$FW_DATA_PATH" \
        --wiki-data-path "$WIKI_DATA_PATH" \
        --num-samples $NUM_SAMPLES \
        --seq-length $SEQ_LENGTH \
        --batch-size $BATCH_SIZE \
        --seed $SEED \
        --output "$OUTPUT_FILE" \
        "${EXTRA_ARGS[@]}"

status=$?
echo "=========================================="
echo "Perplexity evaluation finished with status $status"
echo "=========================================="
exit $status
