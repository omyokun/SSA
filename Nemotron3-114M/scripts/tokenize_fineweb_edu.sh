#!/bin/bash
#SBATCH -J tokenize_fwe
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -p small
#SBATCH --time=06:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

REPO_DIR=${REPO_DIR:-"$PWD"}
INPUT_TXT=${INPUT_TXT:-"data/fineweb_edu_text_document.txt"}
TOKENIZER=${TOKENIZER:-"tokenizer/luciole_50k"}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"data/fineweb_edu_text_document"}
MAX_TOKENS=${MAX_TOKENS:-10000000000}

echo "=========================================="
echo "Tokenizing FineWeb-Edu"
echo "Input:      $INPUT_TXT"
echo "Tokenizer:  $TOKENIZER"
echo "Output:     ${OUTPUT_PREFIX}.{bin,idx}"
echo "Max tokens: $MAX_TOKENS"
echo "=========================================="

apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --bind "${REPO_DIR}:${REPO_DIR}" \
    --bind /tmpdir,/work /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    python3 "${REPO_DIR}/data/tokenize_text_to_indexed_dataset.py" \
        --input-txt "$INPUT_TXT" \
        --tokenizer "$TOKENIZER" \
        --output-prefix "$OUTPUT_PREFIX" \
        --max-tokens "$MAX_TOKENS"

status=$?
echo "=========================================="
echo "Tokenization finished with status $status"
echo "=========================================="
exit $status
