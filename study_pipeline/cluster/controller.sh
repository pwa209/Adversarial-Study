#!/usr/bin/env bash
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
module load StdEnv/2023 python/3.11.5
plan=$1
source_root=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$plan")
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$source_root/study_pipeline/cluster/controller.py" --plan "$plan"
