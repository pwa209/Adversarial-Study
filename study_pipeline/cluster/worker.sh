#!/usr/bin/env bash
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
plan=$1
stage=$2
module load StdEnv/2023 python/3.11.5 scipy-stack/2026a arrow/19.0.1
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONDONTWRITEBYTECODE=1
source_root=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$plan")
environment_root=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment"])' "$plan")
gpu_count=$(python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(next(s for s in p["stages"] if s["id"]==sys.argv[2]).get("gpus",0))' "$plan" "$stage")
export PYTHONPATH="$source_root/study_pipeline${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$stage" != environment ]]; then
  source "$environment_root/bin/activate"
fi
if [[ "$gpu_count" -gt 0 ]]; then
  module load cuda/12.6
  if [[ "$stage" != centaur_dependencies ]]; then
    source "$environment_root-centaur/bin/activate"
  fi
fi
if [[ "$stage" == *qualify_detector* || "$stage" == *score* || "$stage" == clinical_human_finetune || "$stage" == adapted_clinical_geometry ]]; then
  module load matlab/2026a.3
fi
python "$source_root/study_pipeline/cluster/stage_worker.py" --plan "$plan" --stage "$stage"
