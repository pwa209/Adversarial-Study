# Full-population finite-input response analysis

Implementation: `population_perturbations.py`, version `full_population_finite_perturbations_v1`. These definitions are frozen before the new full-population response results are inspected. The source bank and previous cortical response results are already development evidence; this is a retrospective diagnostic analysis, not an independent preregistered replication.

## Question and diagnostic comparison

Can configurations with similar frozen clinical scores and conventional cortical features have different nine-population responses to the same input? A positive result would identify information discarded by the cortical readout within the native generator. A null result would leave this particular observation-bottleneck explanation unsupported over the tested bank. Neither outcome establishes a human causal mechanism.

The published generator has one spatial node. Its nine endogenous populations are excitatory cortex, inhibitory cortex, thalamic reticular, thalamic relay, striatal D1, striatal D2, GPi, GPe and STN. Population 10 is the stochastic background input, and population 11 is the perturbation. Inputs are not counted as neural populations. Delays, dendritic states and propagators are not captured by a nine-by-nine connectivity eigenvalue calculation; no such stability claim is made here.

## Actual inputs and pairing qualification

The analysis reuses all 48 declared native control/pulse pairs. It launches no new simulation and computes no new clinical score. The aggregate nine-population cache is produced by `native_population.py`: the original inventory, configuration and source-output hashes are verified; failed and nonfinite native runs are retained in the disposition table. Its `qualify_pair` independently verifies the following before subtraction:

- Exactly one zero-input control and one positive-input pulse per declared pair, with identical account family, implementation variant and gain/coupling/history factors.
- Identical native time grids, the complete physically observed background-noise stream and all nine voltage/rate trajectories before input. The default numerical tolerance on the latter is `1e-10 * max(abs(control baseline), 1)`. Changed WhiteSeed metadata is permissible only if the complete observed input stream is exactly equal.
- The only allowed dynamical configuration difference is the documented population-11 pulse amplitude; all other parameter text must match after comment/whitespace normalization and the explicitly qualified WhiteSeed exception.
- Population 11 is physically retained in both outputs; the control has zero input, and the observed pulse agrees with its configured amplitude and window.

The native configuration applies the pulse at absolute time 12 s, with output beginning at 10 s. Thus its onset on the retained clock is **2 s**, its width is **20 ms**, and the native output interval is 2 ms. A changed clock is an error, not a silently adjusted window. Missing, duplicate or numerically unqualified pair members fail the all-48 analysis rather than change its denominator.

The existing amplitudes 1, 10 and 100 occur in different parameter configurations. They are preserved as separate strata. These are finite-amplitude responses to a common all-population input, not a within-configuration amplitude sweep, selective circuit intervention, validated linear impulse response or controllability measurement. Pristine and explicitly patched implementations are never pooled for matched contrasts.

## Per-population response measurements

For each voltage and firing-rate trace separately, compute the signed pulse-minus-control trajectory on its actual clock. The baseline is all retained samples earlier than onset minus 10 ms. The response interval starts at onset and ends at the last recorded sample; it is not extrapolated.

Record the absolute peak, sign at peak, peak latency and time integral of squared response. Report peak and integrated energy in native units and relative to the population's own pre-input standard deviation. A baseline standard deviation at or below a numerical floor is explicitly unqualified: no epsilon denominator creates a large standardized effect. The floor is 100 machine epsilons times the larger of one and the largest baseline magnitude.

Record finite peak/input-amplitude and energy/input-amplitude-squared ratios, labelled in native model units. These do not imply an infinitesimal gain or linearity, and remain stratified by actual input amplitude.

Recruitment is the first response exceeding three baseline standard deviations for at least 10 ms. Also report the fixed one-standard-deviation sensitivity result. Recruitment is a descriptive threshold crossing, not a significance test. Peak latency is available even if the three-SD criterion is not met.

Recovery is the first interval after both the pulse and global response peak in which the absolute response stays below 5% of its finite peak for at least 250 ms. Responses with no qualifying return are right-censored at the observed horizon; recovery time remains missing. A zero response is labelled no detectable response. A truncated final segment is never treated as recovered. The peak-relative recovery definition and alternative recruitment threshold have distinct jobs and are not interchangeable.

## Collective response organization

Within voltage and firing rate separately, scale each pulse-minus-control population trace by its own qualified baseline standard deviation. Compute the uncentered singular spectrum of the response trajectory and its participation ratio, `(sum singular^2)^2 / sum singular^4`. This describes the number of participating temporal population modes in the recorded finite response. It is not the full delayed system's eigenvalue spectrum or evidence of recurrent necessity.

Report the fraction of integrated standardized response energy in the original readout population (excitatory cortical voltage/rate) and in the other eight populations, the number recruited, and joint-norm recovery. Full nine-population distances and energy fractions require qualified baseline scales in all nine populations; otherwise report the missingness explicitly. Voltage and rate are never mixed in one dimensional energy calculation.

## Matching cortical observables while testing hidden response differences

Compare configurations only within identical implementation variant, input amplitude, onset, width and **complete actual background-noise stream**. A pair is score-matched when the actual already-frozen clinical scores of its spontaneous controls differ by at most 0.025. These scores are the original scalar detector outputs, not the adapted clinical instrument, GCS itself or a probability that a healthy person is conscious.

A more restrictive match additionally requires absolute differences at most 0.05 for each 100-ms and 500-ms ACF, 1 Hz spectral centroid, 0.05 spectral entropy, 10% symmetric relative Hjorth-mobility difference and log standard-deviation ratio at most log(1.10). All features belong to the actual spontaneous control. Scores at the floor/ceiling are reported in separate clipping strata; a clipped score match does not establish preserved informative score resolution. No outcome-driven tolerance expansion is allowed if there are few or zero matches.

For each eligible configuration contrast, compare the complete signed standardized response trajectories. Relative distance is `||response_a-response_b|| / ((||response_a||+||response_b||)/2)` on the identical clock. Compute it for readout population 1, other populations 2–9 and the complete nine-population system, then report other-population distance minus readout distance. Positive differences indicate that the tested readout conceals part of the particular model-response distinction. These quantities are kept separate from the control-score match itself.

The same configurations recur in multiple contrasts. Consequently counts are labelled dependent configuration contrasts, and descriptive minima/medians/maxima replace pseudoreplicated confidence intervals. There are no participants, human population prevalence estimates or clinical conclusions in this finite-bank analysis. Family-specific summaries preserve gain, recurrent-coupling, history and combined accounts, implementation and amplitude rather than attributing a pooled gradient to one mechanism.

## Link to the human-constrained bridge

The root driver first checks target 42 (26/8/8) and both actual human-data fine-tuning receipts before any cross-system comparison, then builds the cache and executes the human bridge. The optional `bridge_compatibility` role is the bridge's immediate `compatibility.json`, containing explicit supported native configuration identities, `selection_uses_test_targets: false` and `selection_unit: training/validation`.

All 48 native pairs remain in the output regardless of this support. Each matched contrast receives an additional flag stating whether both spontaneous controls were supported by human training/validation assignments. This is **fit support**, not demonstrated observational adequacy: poor accounts can retain supported configurations. The bridge support rule is fixed by that module before test evaluation. A configuration supported during training/validation does not become a physiologically identified mechanism or a formal statistical-equivalence class. Test performance never chooses the input, account or response contrast. If the bridge supports published configurations but none of the 48 pulse-candidate controls, the supported subset is empty and is reported as such; no family label substitutes for exact configuration identity.

## Outputs and scientific interpretation

The stage writes `pair_qualification.csv` (48 proofs), `population_responses.csv` (48 × 9 × 2 responses), `system_responses.csv` (48 × 2 summaries), `matched_response_contrasts.csv`, `summary.json` and hashed `receipt.json`. No dense intermediate response files or per-population output directories are created.

The analysis can establish a model-internal dissociation between spontaneous cortical readout and finite response organization, or show that it is not supported in this bank. It cannot identify which loop is causally necessary because the existing pulse is common input and the parameter candidates are not selective loop knockouts. New selective rescue simulations and a diagnostic human perturbation remain separate future tests. A positive result is not a healthy-person unconsciousness diagnosis; a negative result does not refute all circuit-level accounts.

## Validation

Synthetic tests check a known exponential response's integrated energy and analytic 5%-recovery time, persistent-response censoring, absent-response/constant-baseline handling, one-mode collective responses, exact all-population baseline/noise matching, full-pair denominator preservation, input-amplitude stratification, fixed match boundaries, clipping strata, rejection of test-selected bridge support, and a deliberately identical cortical response with different hidden-population response. These are scientific behavior checks; the native loader separately verifies actual hashes, configuration parsing and aggregate-cache integrity.
