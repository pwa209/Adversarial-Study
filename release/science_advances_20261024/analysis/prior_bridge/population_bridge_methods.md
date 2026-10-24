# Full-population observation-bottleneck test

This analysis asks whether useful predictive information was discarded when a nine-population native corticothalamic–basal-ganglia system was reduced to its cortical `Pop.1.V` trace. The diagnostic comparison is a full-system readout versus an equally searched, nominally equal-capacity cortical-delay readout on the **same future human EEG states**. A better full-system readout supports an observation-bottleneck interpretation within these effective mappings; it does not identify human thalamic or basal-ganglia activity.

The implementation follows scientific-nature's question → design → operation → diagnostic comparison → warranted inference trace. It reuses actual bank trajectories and the original DREAM neural-state archive. It does not retrain the clinical instrument, regenerate a human PCA basis, change the original participant split or choose a preferred result direction.

## Inputs, exposure and independent units

The caller verifies both real changed-weight human-data fine-tunings before this comparative stage. The behavioural target remains **42 people, split 26/8/8**; this number is not the neural sample size. The original DREAM mechanistic evaluation retains **five held-out participants and 32 episodes**, subject to exact reconstruction of its published receipt. Previously inspected public data remain development evidence, and this extension is retrospective.

All original mechanism-selection, training, evaluation, prediction, state, split and bank inputs require explicit SHA256 pins. The native cache independently qualifies each raw output and configuration before population analysis. Native paths, configuration hashes, pristine/patched variants and all dispositions remain traceable. No unknown or failed configuration is silently converted into a successful observation.

The same original ten account/variant combinations are retained, including published reference and runtime diagnostic configurations. Only spontaneous zero-input configurations enter observation-map fitting. All qualifying configurations for an account/variant are candidates for every representation. No clinical score, report direction or test error prunes this bank.

## Frozen coordinates and common forecast clock

The human target is the original training-derived multivariate neural-state trajectory, not invented anatomical coordinates or a reconstructed awareness label. Each episode is interpolated onto the exact saved mechanism grid using its original endpoint alignment. Conditioning stops at the recorded early-state boundary. All original accounts must share identical validation and test episode sets, timestamps and conditioning counts; a silent intersection is an error.

Interpolation never extrapolates, pads a trace or joins episodes. The saved native trajectories retain their observed output-relative labels, **0.002 through 10.000 seconds**. The first saved sample is not treated as a new time origin. The fixed human forecast grid and all original episodes remain unchanged, including its final timestamp of 9.998 seconds. Configured pulse onset remains 12 seconds minus the configured output start of 10 seconds, giving a relative onset of 2 seconds; this bridge does not rewrite native or pulse timestamps.

The cortical delay range is fixed from clock support before fitting any observation map. For each human source, every spontaneous candidate from all original account/variant groups contributes to one shared bound: `used_max_delay = min(requested_max_delay, first_frozen_target_time − latest_native_first_sample)`. All candidates retain the same range. With the original requested maximum of 0.2 seconds, a first target at 0.2 seconds and a first observed native sample at 0.002 seconds, the available maximum is **0.198 seconds**. Its earliest delayed query is the actually observed 0.002-second sample. Nonpositive common history is an error. Requested and used ranges, clock endpoints, candidate counts and the support reason are recorded in the training/validation freeze, every validation-grid row and the final report. This bound uses source clocks and candidate membership only, with no human test targets or performance-based range selection.

Nuisance baseline support must end within the conditioning period. Frozen source neural models retain their original sample interval when forecasting; their predictions, rather than later observed states, are interpolated onto the comparison clock. The reconstructed original scalar forecast is still read from the unchanged original prediction arrays.

## Four native representations

1. **Cortical voltage:** the directly observed `Pop.1.V` trace.
2. **Cortical delay18:** 18 equally spaced delays of that same voltage, from zero through the shared available-history maximum described above, normally 0.198 seconds from an original 0.2-second request.
3. **Population voltage9:** voltages of all nine endogenous populations.
4. **Population voltage/rate18:** voltage and firing rate from every endogenous population, exactly 18 measured columns. External noise and pulse inputs are not represented as extra neural populations.

Each configuration's native channel origin and scale use only its conditioning segment. The later simulated trajectory does not set those normalization quantities. This retains the same time axis and candidate configurations for every view.

## Training and validation

Each participant receives equal total training weight; episodes divide that person's weight equally. Configuration assignment is initialized using normalized temporal Gram geometry of conditioning states. Two alternating assignment/readout updates and one final readout fit use only **training participants' conditioning states**. Configurations offered to validation and test episodes are restricted to support retained by this training fit. Configuration identity and an optional constant state offset for a new episode are inferred from its conditioning observations only.

Feature means, scales and PCA bases are fitted only to assigned training conditioning blocks, using the same equal-person weights. The predeclared feature-rank grid is 2/4/8, with rank 1 for the cortical scalar. Ridge penalties are 0.01/0.1/1 and constant-offset strengths are 0/1. A ridge regression predicts the original human coordinate vector. The primary future loss is mean squared coordinate error divided by that episode's conditioning-state mean-square magnitude, floored at 1e-9 as in the original bridge.

For each view, validation selects the setting with the lowest equal-person future-state loss. The primary **matched-capacity** comparison instead selects one shared rank/ridge/offset setting by minimizing the mean validation loss across population voltage/rate18 and cortical delay18. Both have 18 observed native predictors and identical nominal readout coefficient counts after rank restriction. Their realized effective degrees of freedom can differ because feature covariance differs; these quantities are reported and are not declared identical. Configuration search and feature-basis estimation add complexity beyond the reported conditional ridge degrees of freedom, equally subject to the held-person design.

The account/variant used for the selected-account summary is selected using this joint validation criterion. All account results remain visible. The entire choice record, fitted transforms, weights, support identities, validation grid and selected account are written to an immutable `training_validation_freeze.json` **before the code loads test target arrays**.

## Comparators and endpoints

Every test case is evaluated with the selected and jointly matched readouts, the original scalar-map forecast, a new train-fitted/validation-selected multivariate autoregression, the frozen source neural transition and persistence. The original scalar prediction loss is reconstructed from the saved prediction array and actual target coordinates; recorded aggregate losses are not accepted without this equality check.

All forecasts use the same observed last conditioning state or identical conditioning interval. The autoregression is fitted on training conditioning transitions, with ridge selected on validation future loss, and recurses without later measured inputs. The frozen neural transition retains its fitted native clock. Failed or weak forecasts are retained; divergence is an explicit failure, not silently clipped.

The predeclared paired contrasts are: matched full18 versus matched delay18; individually selected full18 versus selected delay18; selected voltage9 versus selected cortical scalar; matched full18 versus original scalar map, autoregression, frozen neural transition and persistence. Losses are averaged within each participant and then across participants. Reported intervals resample participants conditional on the fitted maps. Exact two-sided sign-flip references assume independent symmetric participant differences; they are not randomized-human causal tests. Holm references are reported across all account-by-contrast comparisons and separately for the validation-selected account. Percentile intervals are exploratory, not simultaneous, and do not include training or model-selection uncertainty. With five people, even uniformly signed differences have a minimum two-sided exact sign-flip reference of 0.0625.

## Link to perturbation analysis

`compatibility.json` exports configurations retained as support by the joint matched-capacity maps. It uses only training fits and validation conditioning assignment, never human test outcomes. It is **fit support, not an observational-adequacy threshold or proof of physiological equivalence**. The perturbation branch must retain all 48 actual pulse/control pairs and separately report coverage and any fit-supported subset. Empty overlap is a missingness result, not permission to change the fit or select on test outcomes.

## Warranted conclusions

An advantage over the matched cortical-delay representation and statistical comparators would show that the full native population state retains useful information for this human state-prediction task. An advantage shared equally by flexible cortical readouts would instead implicate readout flexibility. Failure of both does not identify which biological account is true; it constrains these candidate generators and observation mappings. The one-node simulator supports a population-circuit interpretation. It does not establish anatomical whole-brain connectivity, native causal necessity in humans, a qualified scalp forward map or independent human validation.

The branch writes selected fitted models, the full validation grid, support provenance, every held-person prediction, per-case losses, participant contrasts and immutable source/input/output hashes. Synthetic contract tests cover split leakage, future-target access, native normalization, PCA provenance, exact clock alignment, matched settings, original-loss reconstruction and end-to-end output custody. Clock regression tests use the actual 0.002–10.000-second support convention, verify all 18 delay queries remain within observed history, retain the exact fixed human grid, enforce a global candidate bound and reject nonpositive history.
