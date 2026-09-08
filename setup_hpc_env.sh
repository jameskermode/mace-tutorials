#!/usr/bin/env bash
set -euo pipefail

# Override these variables if the HPC allocation or Python module differs.
PROJECT_DIR="${PROJECT_DIR:-/project/home/p201433}"
ENV_DIR="${ENV_DIR:-${PROJECT_DIR}/.venv-mace-tutorials}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
DOWNLOAD_TUTORIAL_DATA="${DOWNLOAD_TUTORIAL_DATA:-1}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/Tutorials}"
UV_INSTALL_DIR="${UV_INSTALL_DIR:-${PROJECT_DIR}/.local/bin}"
LOCK_FILE="${LOCK_FILE:-${PROJECT_DIR}/.cache/mace-tutorials-requirements.lock}"

if ! command -v uv >/dev/null 2>&1; then
    if [[ ! -x "${UV_INSTALL_DIR}/uv" ]]; then
        printf 'uv was not found; installing it under %s\n' "${UV_INSTALL_DIR}"
        mkdir -p "${UV_INSTALL_DIR}"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${UV_INSTALL_DIR}" sh
    fi
    export PATH="${UV_INSTALL_DIR}:${PATH}"
fi
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

mkdir -p "${PROJECT_DIR}" "${PROJECT_DIR}/.cache"
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    uv venv --python "${PYTHON_VERSION}" "${ENV_DIR}"
fi
uv pip compile \
    --quiet \
    --python "${ENV_DIR}/bin/python" \
    --output-file "${LOCK_FILE}" \
    "${SCRIPT_DIR}/requirements-hpc.txt"
uv pip sync --python "${ENV_DIR}/bin/python" "${LOCK_FILE}"

# CuEq is optional because the correct wheel depends on the cluster CUDA version.
# Example for CUDA 12:
# uv pip install --python "${ENV_DIR}/bin/python" \
#   'cuequivariance-torch>=0.2' 'cuequivariance-ops-torch-cu12>=0.2'

if [[ "${DOWNLOAD_TUTORIAL_DATA}" == "1" ]]; then
    if [[ ! -d "${DATA_DIR}/data" ]]; then
        if [[ -e "${DATA_DIR}" ]]; then
            printf 'Cannot download data: %s exists but has no data/ directory.\n' "${DATA_DIR}" >&2
            exit 1
        fi
        git clone --depth 1 https://github.com/imagdau/Tutorials.git "${DATA_DIR}"
    fi
fi

mkdir -p "${PROJECT_DIR}/.cache/matplotlib" "${PROJECT_DIR}/logs"
cat <<EOF

Environment ready:
  source ${ENV_DIR}/bin/activate

Tutorial data:
  ${DATA_DIR}

Example:
  python ${SCRIPT_DIR}/T03_MACE_Theory.py --work-dir ${DATA_DIR}
EOF
