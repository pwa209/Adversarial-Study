#!/usr/bin/env bash
set -euo pipefail
environment_root=$1
run=$2
cache_config=${3:-}
cache_helper="$(dirname "${BASH_SOURCE[0]}")/runtime_cache.py"
test -n "${SLURM_JOB_ID:-}"
test ! -e "$environment_root"
if [[ -n "$cache_config" ]]; then
  python "$cache_helper" --kind centaur --destination "$environment_root" --run "$run" --config "$cache_config" --phase link
else
  python -m venv "$environment_root"
fi
source "$environment_root/bin/activate"
if [[ -z "$cache_config" ]]; then
  python -m pip install --no-index 'torch>=2.5' 'transformers>=4.49,<5' 'peft>=0.13,<1' 'accelerate>=1,<2' 'bitsandbytes>=0.45,<1' safetensors sentencepiece numpy
fi
python -m pip freeze > "$run/centaur-environment-resolved.txt"
python - "$run" <<'PY'
import datetime,importlib.metadata,json,pathlib,sys,torch
versions={name:importlib.metadata.version(name) for name in ['torch','transformers','peft','accelerate','bitsandbytes','safetensors','sentencepiece']}
devices=[{'name':torch.cuda.get_device_properties(i).name,'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
value={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'versions':versions,'cuda':torch.version.cuda,'devices':devices,'status':'native_dependencies_available' if len(devices)>=2 and sum(d['total_memory_bytes'] for d in devices)>=76*1024**3 else 'insufficient_allocated_native_devices'}
with (pathlib.Path(sys.argv[1])/'centaur-environment-runtime.json').open('x') as f:json.dump(value,f,indent=2)
if value['status']!='native_dependencies_available':raise SystemExit(2)
import transformers,peft,accelerate,bitsandbytes
PY
if [[ -n "$cache_config" ]]; then
  python "$cache_helper" --kind centaur --destination "$environment_root" --run "$run" --config "$cache_config" --phase recheck
fi
