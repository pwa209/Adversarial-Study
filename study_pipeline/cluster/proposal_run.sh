#!/usr/bin/env bash
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
test "$(id -un)" = pwa209
case "$(hostname -f)" in *.narval.calcul.quebec) ;; *) exit 10;; esac
proposal_source=$1
proposal_run=$2
proposal_phase=$3
module load StdEnv/2023 python/3.11.5 scipy-stack/2026a arrow/19.0.1
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TMPDIR="$SLURM_TMPDIR"
export PYTHONPATH="$proposal_source/study_pipeline/bootstrap:$proposal_source/study_pipeline${PYTHONPATH:+:$PYTHONPATH}"
parent_environment=/scratch/pwa209/adversarial-study/environments/20261004T210516Z-53593b22ca18
test "$(readlink -f "$parent_environment")" = /scratch/pwa209/adversarial-study/environments/20261004T193709Z-545c74b08e71
test "$(readlink -f "$parent_environment-centaur")" = /scratch/pwa209/adversarial-study/environments/20261004T193709Z-545c74b08e71-centaur
if [[ "$proposal_phase" == cpu ]]; then
    source "$parent_environment/bin/activate"
    module load matlab/2026a.3
    export ACS_REPAIR_OVERLAY="$SLURM_TMPDIR/mne-overlay"
    python /home/pwa209/adversarial-study/maintenance/conversion-repair-20261005/paris_runtime/prepare_overlay.py --wheel /home/pwa209/adversarial-study/maintenance/conversion-repair-20261005/wheels/mne-1.10.2-py3-none-any.whl --target "$ACS_REPAIR_OVERLAY"
    export PYTHONPATH="$ACS_REPAIR_OVERLAY:$PYTHONPATH"
elif [[ "$proposal_phase" != breadth_compare ]]; then
    module load cuda/12.6
    source "$parent_environment-centaur/bin/activate"
else
    source "$parent_environment/bin/activate"
fi
python "$proposal_source/study_pipeline/cluster/proposal_runtime.py" --source "$proposal_source" --run "$proposal_run" --phase "$proposal_phase"
