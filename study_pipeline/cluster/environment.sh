#!/usr/bin/env bash
set -euo pipefail
environment_root=$1
source_root=$2
run=$3
cache_config=${4:-}
test -n "${SLURM_JOB_ID:-}"
test ! -e "$environment_root"
if [[ -n "$cache_config" ]]; then
  python "$source_root/study_pipeline/cluster/runtime_cache.py" --kind cpu --destination "$environment_root" --run "$run" --config "$cache_config" --phase link
else
  python -m venv --system-site-packages "$environment_root"
fi
source "$environment_root/bin/activate"
if [[ -z "$cache_config" ]]; then
  python -m pip install --no-index 'mne>=1.6,<2' h5py pandas
fi
python -m pip freeze > "$run/environment-resolved.txt"
python -c 'import sys,numpy,scipy,mne,h5py,pyarrow,json; print(json.dumps({"python":sys.version,"numpy":numpy.__version__,"scipy":scipy.__version__,"mne":mne.__version__,"h5py":h5py.__version__,"pyarrow":pyarrow.__version__,"arrow_module":"arrow/19.0.1"},indent=2))' > "$run/environment-runtime.json"
if [[ -n "$cache_config" ]]; then
  python "$source_root/study_pipeline/cluster/runtime_cache.py" --kind cpu --destination "$environment_root" --run "$run" --config "$cache_config" --phase recheck
fi
cd "$source_root/study_pipeline"
python -m unittest discover -s tests -v
python run_study.py demo --output "$run/synthetic_runtime_check"
