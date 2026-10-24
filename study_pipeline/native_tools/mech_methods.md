# Native biological simulation and interrogation bank

The biological branch uses the exact published EIRS–basal ganglia NFTsim source
and 281 configurations in Figshare DOI `10.6084/m9.figshare.30906557.v2`:
the conscious-brain reference and all 280 numbered DoC configurations. Source
acquisition verifies the published MD5, records SHA-256, and preserves the source
cache. Neither the clinical predictor nor its preprocessing is replaced.

The executable bank produces actual neural-population trajectories. A one-node
cortical population potential, `Pop.1.V`, supplies a ten-second, 500 Hz input for
the separately scheduled frozen detector. All native population potentials,
firing rates, dendrites and emitted input channels remain in native output files.
This population voltage is an explicit model-space observation. Its scalp EEG
forward map, reference, spatial mixture and physiological calibration remain
unqualified; no synthetic population is called a human scalp electrode.

## Explicit source implementation comparison

The pristine source is compiled and run unchanged. Source inspection finds that
`DendriteScaled` redeclares `alpha`/`beta`, leaving its copies uninitialized by
`Dendrite::init`, and inherits `Dendrite::step`, which advances the base integrator
rather than the scaled equation's integrator. The earlier Narval baseline test
observed identical output after an hsyn change and different output after a
coupling change. That observation concerns this released baseline/runtime.

A separate implementation named `scaled_dendrite_v1_explicit_patch` copies the
initialized base-class time constants, advances the derived scaled integrator,
and publishes its voltage through the inherited output vector. It also uses the
equal-rate limit of the same alpha/beta response kernel to avoid an optional
numerical singularity. Only the exact hash-pinned two source files may receive
this change. A unified diff, before/after hashes, compiler log, executable hash
and every generated configuration are preserved. The pristine reference remains
available, and the same entire published configuration bank runs under each
implementation. This named numerical repair is an explicitly distinct account,
not a silent claim to reproduce the published study's implementation or results.

Each implementation separately retains a repeated reference, a gain diagnostic
and a coupling diagnostic. The records distinguish failed execution, nonfinite
output, finite but constant output, standardizable trajectories, repeatability
and parameter sensitivity. A finite execution is not a biological validation.

## Interpretable family exploration and input responses

Forty-eight deterministic proposals explore three families relative to the
published conscious reference: cortical input synaptic gain (`hsyn` of
dendrites 1, 2, 3), corticothalamic recurrent coupling (`nu` of connections 1, 2,
3, 5, 6, 7, 10, 11, 14, 15), and synaptic response history (alpha and beta jointly
scaled for dendrites 1, 2, 3). Axis proposals use factors 0.5, 0.8, 1.25 and 2;
the remaining proposals form a deterministic Latin hypercube in log factors
between 0.5 and 2. Lower history factors extend the kernel's time scale. Signs
of inhibitory/excitatory couplings, neural population definitions, thresholds,
noise source, conduction delays and integration step remain explicit in the
retained configurations.

Each proposal runs with both zero input and a genuine, source-supported pulse
through the published Perturb population and its existing connections. Onset is
12 s, width 20 ms and amplitudes cycle through 1, 10 and 100 **model units**.
These are diagnostic simulation interventions. They are not calibrated human
stimulation intensities. The actual `Pop.11.Q` output must contain the input,
and paired trajectories must agree before it. Failed checks preserve an
unqualified response rather than an attributed intervention effect.

Matched input-minus-control trajectories provide response peak, integral and
recovery below 5% of peak for at least 250 ms. No detectable response and
right-censored recovery remain separate outcomes. Human perturbation predictions
require an appropriate independent intervention and qualified observation map.

The default bank contains 664 runs: 281 configurations × two implementations,
six runtime diagnostics and 48 × two input conditions. At most four native
workers run inside the actual Slurm CPU allocation. Each run has a 180-second
cap; the pessimistic simulation bound is about 8.3 hours, with two compilation
caps adding at most 20 minutes. Each output has a 20 MB bound, giving at most
13.3 GB for the default bank's native outputs. All candidates are retained in
proposal order, including failed or constant trajectories. Eligible detector
inputs contain one actual population channel per run, so the maximum detector
job count is 664, rather than four artificial channels per simulation.

## Fitting and interpretation boundary

Temporal descriptors use the same `guided_univariate_v1` implementation as the
empirical branch: centered ACF at 100/500 ms, 0.5–45 Hz Hann-periodogram centroid
and normalized spectral entropy, Hjorth mobility and standard deviation.
Absolute amplitude is recorded but excluded from cross-domain fitting while
the observation map is unqualified. The frozen clinical score and actual hidden
activations are computed in a subsequent native MATLAB stage.

A matching bank trajectory constrains a computational account, not uniquely a
human physiological parameter. Synaptic gain and connection strength can
compensate multiplicatively; source bank factors equal to one mean no added
change relative to **that configuration**, not identical absolute parameters
across the 280 published configurations. Parameter recovery, sensitivity,
nonuniqueness, held-out prediction and diagnostic intervention predictions must
therefore be reported separately. The biological explanation remains open.

`bank_receipt.json` binds the inventory, detector job table, source manifest,
executables, repaired-source diff and actual native output hashes. Complete bank
status denotes bounded execution and preservation of outcomes. It does not
denote a completed human mechanism study.
