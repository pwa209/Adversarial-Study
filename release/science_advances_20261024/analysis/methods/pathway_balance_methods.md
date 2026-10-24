# Saved native pathway balance and timing

The question is whether models with similar clinical readouts and conventional
cortical features conceal different balances of their incoming postsynaptic
voltage contributions. This retrospective diagnostic uses saved outputs. It is
not a new human experiment or a selective pathway intervention.

## Source and configuration identity

The pinned native-bank receipt binds both named implementations and the hashes
of their compiled sources. The analysis reads and rechecks `population.cpp`,
`firing_response.cpp`, `firing_response.h`, `dendrite.cpp`, `dendrite.h`, and
`dendrite_scaled.cpp`, and `dumpf.cpp` in each recorded build's `src` directory. Source anchors
require Population::V to delegate to FiringResponse, whose step resets the soma
voltage and adds `dendrites[i]->V()[j]`. Both relevant getters return the saved
voltage vector. The saved Dendrite.V is thus an already filtered, coupled
postsynaptic voltage contribution. No nu or hsyn weighting is applied again.
It is not reconstructed electric current. The pristine and explicit scaled-
dendrite patch retain separate results and bootstrap strata.

The corrected native v2 cache qualifies all nine endogenous population voltages
and firing rates, their clock, actual noise, observed pulse, full configuration,
and per-output SHA256. This stage rechecks each selected raw-output and config
hash and parses all 41 saved Dendrite.V columns. It verifies exact agreement
with the cached V/Q, noise, pulse, and original labels `.002, .004, ..., 10 s`.
There is no clock rebasing, resampling, recoding of failed simulations, or
substitution of cortex voltage for missing pathways.

## Frozen coverage

The selected set is the union of the 96 members of the existing 48 pulse/control
pairs and the 185 zero-input configurations retained by the already completed
human training/validation bridge. This is at most 281 raw files, read once each
in the main stage, with a 64 MiB per-file bound. The stage preserves an explicit
disposition for every native inventory row. It does not rerun the 664 simulation
bank or download it locally.

The compatibility JSON must explicitly state training/validation selection with
no human test-target selection. A supported model is a fitted candidate; the
label does not establish physiological equivalence or observational adequacy.
The original five-person neural test set is not enlarged by simulations or by
the separate 42-person behavioral target. Root orchestration enforces the
42-person target and verifies both actual human-data fine-tunings and clinical
adaptation before comparison.

## Equation reconstruction

For each destination population, configured source/destination indices specify
the incoming saved dendrites, including external inputs. Their sum must
reconstruct the recorded soma voltage. The proof reports the maximum absolute
error and its fixed tolerance, `1e-12 * largest original absolute pathway/population
voltage scale` (with only a machine-underflow absolute allowance). The pinned
Dumpf source sets `file.precision(14)` with scientific notation: fourteen digits
after the decimal imply at most approximately 5e-15 relative rounding per field.
The fixed 1e-12 relative sum tolerance conservatively includes multiple incoming
fields, subtraction and mean-centering arithmetic. For pulse-minus-control
differences and centered fluctuations, the rounding reference
includes the original control and pulse trajectories, since subtraction error
originates in those printed values. This tolerance is independent of baseline
standard deviation and scientific effect direction.

The zero-input stage reports tonic-plus-fluctuation balance and separately
mean-centered fluctuation balance. A constant voltage reference can be removed
for the latter only. The pulse stage subtracts the actual matched control from
the pulse before interpreting pathway balance. Any failed sum is retained and
excluded from inferential cancellation summaries. A bounded qualification probe
checks the compiled-source equation and a real paired response before the full
stage; it requires a valid identity, never a preferred scientific result.

## Physical-unit measurements

For incoming voltage contributions d_j(t), the total absolute pathway mass is
A = integral(sum_j |d_j(t)|) dt, and net voltage mass is
N = integral(|sum_j d_j(t)|) dt. Relative voltage cancellation is `1 - N/A` when
A is nonzero. The triangle inequality bounds it between zero and one. Zero
response keeps an undefined relative metric; no epsilon is inserted.

The analysis retains A and N in V*s, positive and negative response masses,
individual-pathway and net energy in V^2*s, and signed cross energy. Positive
cross energy indicates reinforcement and is not mislabeled cancellation. The
fraction of observations with both response signs is descriptive. Raw and
endogenous-only incoming sets are reported separately; the reconstruction proof
always uses all actual incoming dendrites. Endogenous-only balance does not
identify a selectively manipulated recurrent loop.

For every pulse-response edge, signed area, peak voltage and latency, and the
times containing 10%, 50%, and 90% of cumulative response energy are reported on
the observed clock. These characterize a common global input, not selective
pathway transmission or a new within-configuration amplitude sweep.

No channel is divided by its own baseline standard deviation. Relative
comparisons retain their common raw voltage scale and absolute RMS differences.
The common scale is the largest absolute raw voltage in either configuration's
pathway or population signature. Every signature distance and the hidden-minus-
cortical comparison uses this same scale, rather than separately normalizing a
nearly zero cortical component and a large hidden component.
Values below a fixed 1e-12 V numerical floor remain physically reported but are
excluded from cancellation inference and normalized-distance interpretation.
This prevents a nearly zero denominator from becoming evidence for a large
hidden response. It does not assert that 1e-12 V is a physiological detection
threshold.

## Matched contrasts and uncertainty

The existing frozen matching rules are reused: same implementation, native
input, and complete observed background-noise identity, with clinical score
distance <=.025. The strict sensitivity also matches the published fixed ACF,
spectral-centroid, entropy, mobility, and cortical-amplitude tolerances from the
completed perturbation stage. All failed matching reasons and all available
strict matches are counted. Comparisons retain whether both configurations are
human training/validation supported and whether scores are clipped. No minimum
match count is manufactured by relaxing tolerances.

Zero-input fluctuation signatures are physical-unit per-edge/per-population RMS
vectors. Paired pulse signatures preserve the full post-input trajectories.
Matched differences include cancellation, raw pathway mass, absolute RMS
differences, and relative signature distances. Configuration comparisons share
models and are dependent. They are not subjected to an ordinary independent-
pair t-test or a contrast-row bootstrap.

The only bootstrap here resamples configuration clusters, separately within
mode, implementation, account family, native pulse amplitude, destination and
incoming scope. It gives a fixed-seed, 2,000-replicate percentile interval for
mean cancellation as a descriptive sensitivity to this enumerated bank. A
single configuration receives no confidence interval. Duplicating time points,
edge rows or configuration records cannot increase the cluster count. These
intervals and the simulation bank do not supply human population significance
or turn null human predictive findings into positive results. No P-values are
computed by this module.

## Outputs and limits

Aggregate CSVs record all configuration dispositions, population balance, edge
timing, paired sum proofs, configuration summaries and matched comparisons.
JSONs bind source-equation proofs, descriptive cluster bootstrap and summary
counts; a receipt records hashes. No raw simulations are copied to the result
directory.

The model has one spatial node and nine named populations. These outputs may
show different balances and timings among modeled systems, but cannot establish
regional whole-brain propagation, human causal necessity, or a physiological
explanation of low clinical-model scores in healthy people. Positive, null,
weak-response and unsupported-match results all remain valid stage outcomes.
