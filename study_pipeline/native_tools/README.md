# Native runtime stages

Run Python modules from `study_pipeline`, or set that directory on `PYTHONPATH`.
Python 3.11+ and NumPy are sufficient for these helpers. They do not submit jobs.
Source acquisition may run on the login node; compilation, simulation and MATLAB
workers require an actual scheduled `SLURM_JOB_ID`. No hypothesis, significance,
effect-size or expected-result threshold controls execution.

## Pinned source acquisition

```sh
python -m native_tools.figshare_source \
  --metadata ../source_records/toker_v2/figshare_v2_metadata.json \
  --destination /absolute/new/release \
  --include-detector --include-config-bank
```

Without `--metadata`, the helper reads the version-2 Figshare API endpoint.
It verifies article/version DOI, actual Figshare file URLs, sizes and computed
MD5 values, then records SHA256 values. All 63 published C++ source/header files
are reconstructed into `src/` with their bytes unchanged. The conscious baseline
and both license files are always included. The 280 disorder configurations and
native detector are optional. An existing verified manifest is never replaced.
Do not interpret file acquisition as executable or biological qualification.

## Pristine NFTsim executable and diagnostics

```sh
python -m native_tools.nftsim_runtime \
  --release-dir /absolute/release --output-dir /absolute/new/qualification \
  --compiler g++ --timeout 600
```

The helper verifies all acquired files and rejects unrecorded C++ sources. It
builds directly with GCC, C++11 and portable `-O2` flags, avoiding the released
Makefile's final `cat conditions.txt` instruction because that file is absent
from the published payload. Compiler version, command, binary hash, logs and
source hashes are recorded. The original 20-second conscious baseline runs twice
in separate processes. Outputs must be finite, match exactly across repeats and
contain 5,000 samples with 2-ms spacing. Its printed time is relative to the
published output start at 10 s; the output is a model population quantity rather
than a validated scalp-EEG measurement.

Separate diagnostic configurations change only Dendrite 1 `hsyn` by a factor of
2 and Coupling 1 `nu` by a factor of 1.05, respectively. These diagnose parameter
functionality. Their results do not impose a scientific-outcome gate and a failed
diagnostic does not erase successful baseline runtime evidence. An unchanged
`hsyn` trace with a changed coupling trace is explicitly reported as a failed
gain-functionality diagnostic for this baseline. The published scaled-dendrite
class shadows base rates and integrators and does not override `step`; do not
assume that the advertised gain operates until numerical evidence establishes
it. This helper never patches the source. No observation map or human mechanism
is qualified by these tests.

## Genuine empirical detector reference

```sh
python -m native_tools.detector_worker \
  --signal-csv /absolute/subject01_Cz_10s.csv --sfreq 500 \
  --release-dir /absolute/release \
  --reference-json /absolute/reference.json \
  --output-json /absolute/new/detector_qualification.json \
  --matlab matlab --timeout 1800
```

The reference JSON must contain:

```json
{
  "reference_kind": "empirical_eeg",
  "synthetic": false,
  "continuity_verified": true,
  "reference_id": "source_participant_channel_epoch",
  "participant_id": "global_participant_id",
  "source_id": "source_id",
  "channel": "Cz",
  "source_recording_path": "/absolute/actual/raw/recording",
  "source_recording_sha256": "64 lowercase hexadecimal characters",
  "signal_sha256": "64 lowercase hexadecimal characters",
  "start_sample": 0,
  "stop_sample": 5000,
  "sfreq_hz": 500
}
```

The raw recording and exported CSV must be distinct files. SHA256 values and
the declared continuous, exclusive-stop sample span are verified. Empirical
identity comes from the source/adapter provenance; hash checking alone cannot
prove it. A signal must be univariate, finite, nonconstant, ten seconds long and
sampled above 90 Hz. The worker masks CUDA devices and invokes the unchanged
published MATLAB wrapper. Required runtime dependencies are MATLAB with its
native `SeriesNetwork` and Deep Learning Toolbox, Signal Processing functions
`fir1`, `filtfilt`, `resample`, an available `zscore`, the custom bounded layer
and Java for hashing. Receipts record toolbox versions and license-test booleans
without server/license contents. License-test success is not an assurance that a
subsequent checkout succeeds; actual inference is tested.

The pre-bound predecessor is always included in repeated activation extraction.
Additional actual layer names can be supplied with `--layers name1,name2`.
Reconstructed preprocessing must reproduce the unchanged wrapper before
activations are accepted, and clipping the pre-bound scalar must reproduce GCS.
Forward outputs and activations must repeat. The loaded layer inventory, input
hashes, helper hashes, environment, timings and bounded-score flag are recorded.
The score is remapped GCS, not a calibrated awareness probability. A clipped tie
does not establish biological equivalence. No synthetic reference substitutes
for an empirical recording; failure retains available evidence without a false
qualification claim.

## Frozen native batch inference

```sh
python -m native_tools.detector_worker --mode batch \
  --release-dir /absolute/release --jobs-csv /absolute/jobs.csv \
  --output-csv /absolute/new/scores.csv \
  --qualification-json /absolute/detector_qualification.json --matlab matlab
```

This uses the existing `matlab/batch_predict.m` export/import contract and
requires the empirical repeated-activation receipt. Existing outputs are never
overwritten. The wrapper/checkpoint/helpers and input hashes are checked in the
MATLAB batch. Python import should subsequently validate its native receipt.
