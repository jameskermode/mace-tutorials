#!/usr/bin/env bash
#SBATCH --job-name=mace-tutorial
#SBATCH --account=p201433
#SBATCH --qos=default
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/project/home/p201433/logs/%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/project/home/p201433}"
ENV_DIR="${ENV_DIR:-${PROJECT_DIR}/.venv-mace-tutorials}"
TUTORIAL_SCRIPTS="${TUTORIAL_SCRIPTS:-${PROJECT_DIR}/mace-tutorials}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/Tutorials}"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/.cache/matplotlib"
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"
export MPLCONFIGDIR="${PROJECT_DIR}/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

source "${ENV_DIR}/bin/activate"

# Override after sbatch, for example:
#   TUTORIAL=T04_MACE_Finetuning.py STAGE=mp0 \
#     TUTORIAL_EXTRA_ARGS="--epochs 1" sbatch run_tutorial.slurm
TUTORIAL="${TUTORIAL:-T01_MACE_Practice_I.py}"
if [[ "${TUTORIAL}" == "T04_MACE_Finetuning.py" ]]; then
    default_stage="prepare"
else
    default_stage="all"
fi
STAGE="${STAGE:-${default_stage}}"

arguments=(--work-dir "${DATA_DIR}")
if [[ "${TUTORIAL}" != "T03_MACE_Theory.py" ]]; then
    arguments+=(--device cuda --stage "${STAGE}")
fi

# This is intentionally simple shell-style whitespace splitting. It is suitable
# for numeric and flag options; paths containing spaces should use a direct call.
if [[ -n "${TUTORIAL_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_arguments <<< "${TUTORIAL_EXTRA_ARGS}"
    arguments+=("${extra_arguments[@]}")
fi

srun python -u "${TUTORIAL_SCRIPTS}/${TUTORIAL}" "${arguments[@]}"
