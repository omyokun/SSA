#!/bin/bash
#SBATCH -J tr_bbyluc_softmax
#SBATCH -N 6
#SBATCH -n 6
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

# Defaults
REPO_DIR=${REPO_DIR:-"$PWD"}
DATAMIX=${DATAMIX:-"data/datamix.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs"}
NAME=${NAME:-"baby_luciole_softmax"}
SEED=${SEED:-1234}
MAX_STEPS=${MAX_STEPS:-22000}
GLOBAL_MAX_STEPS=${GLOBAL_MAX_STEPS:-${MAX_STEPS}}
mkdir -p "$OUTPUT_DIR"

# Multi-node coordination
export MASTER_PORT=$(echo "${SLURM_JOB_ID:-0} % 100000 % 50000 + 10001" | bc)
export MASTER_ADDR=$(hostname --ip-address)

# Convert SBATCH time to DD:HH:MM:SS for StatelessTimer
SBATCH_TIME=$(grep -E '^#SBATCH --time=' "$0" | head -n1 | sed -E 's/^#SBATCH --time=//')
if [[ "$SBATCH_TIME" == *-* ]]; then
    SLURM_DURATION=$(echo "$SBATCH_TIME" | awk -F'[-:]' '{printf "%02d:%02d:%02d:%02d", $1, $2, $3, $4}')
else
    SLURM_DURATION=$(echo "$SBATCH_TIME" | awk -F':' '{printf "00:%02d:%02d:%02d", $1, $2, $3}')
fi

echo "=========================================="
echo "Starting Baby Luciole (Softmax) Training"
echo "Datamix:     $DATAMIX"
echo "Output:      $OUTPUT_DIR"
echo "Nodes:       $SLURM_NNODES"
echo "Duration:    ${SLURM_DURATION}"
echo "=========================================="

# TODO: Calculate --global_max_steps for final dataset (tokens / tokens_per_batch) 

srun apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --env "MASTER_ADDR=${MASTER_ADDR}" \
    --env "MASTER_PORT=${MASTER_PORT}" \
    --env "SLURM_NNODES=${SLURM_NNODES}" \
    --env "NVTE_DEBUG=1" \
    --env "NVTE_DEBUG_LEVEL=2" \
    --bind "${REPO_DIR}:${REPO_DIR}" \
    --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    torchrun \
        --nnodes=${SLURM_NNODES} \
        --nproc_per_node=2 \
        --rdzv_id=${SLURM_JOB_ID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        "${REPO_DIR}/train/train_softmax.py" \
        --datamix "$DATAMIX" \
        --output_dir "$OUTPUT_DIR" \
        --name "$NAME" \
        --arch baby_luciole \
        --max_steps ${MAX_STEPS} \
        --seq_length 1024 \
        --batch_size 768 \
        --micro_batch_size 8 \
        --num_nodes ${SLURM_NNODES} \
        --gpus_per_node 2 \
        --tensor_parallelism 1 \
        --pipeline_parallelism 1 \
        --context_parallelism 1 \
        --duration "${SLURM_DURATION}" \
        --global_max_steps ${GLOBAL_MAX_STEPS} \
        --save_every_n_steps 5000 \
        --seed $SEED

status=$?
echo "=========================================="
echo "Baby Luciole (Softmax) Training finished with status $status"
echo "=========================================="
exit $status
