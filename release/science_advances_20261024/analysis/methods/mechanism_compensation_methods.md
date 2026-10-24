# Effective dynamics and cross-mechanism compensation

This retrospective analysis asks whether the existing gain, coupling and synaptic-history simulator accounts can be distinguished from their effective dynamics, and whether those dynamics predict a held configuration's finite response to an observed common input. It makes no prediction about which account should win. It uses saved, verified simulations and does not fit human test data or generate new simulations.

## Inputs and provenance

`run(inputs, output, seed=20261024)` accepts a role map or `inputs['files']`. The required `population_cache` role (alias `native_cache`) is a cache directory string or an object containing `path`/`cache_dir` and an optional exact `receipt_sha256` pin. The root driver verifies both human-data fine-tuning receipts before comparison, preserves behavioral target42 with26/8/8 people, and supplies the repaired source-pinned `native_population` v2 loader. The loader checks the aggregate cache hashes, column identities, metadata and actual2-ms labels before returning any observations. No human sample size is inferred from simulator trajectories.

The bank contains664 finite traces:616 with zero configured pulse amplitude and48 corresponding pulse traces. All616 zero-input traces contribute to description of system modes. Inversion of gain/coupling/history factors uses only actual configurations from those three sweep families and their combined family. Published-bank configurations carry nominal factor values of1 despite changes in other physiological parameters, so they are excluded from inverse-factor ground truth. The pristine and explicit numerical-patch implementations are analyzed separately; absence of an actual factor sweep for a variant is recorded as unsupported.

## Configuration units and separation of fitting and evaluation

The unit is a physical model configuration. A canonical configuration hash removes comments, the external pulse design, noise seed and recording declarations, while preserving all endogenous dynamical parameters and the implementation identity. The input masking serves only to keep the same physical system together across input conditions. It never qualifies a pulse subtraction. Actual pulse/control subtraction separately calls the strict existing `qualify_pair`, which requires matched complete noise streams, pre-input states, clocks and configurations outside the known pulse amplitude or observationally identical noise seed change.

Within each implementation and actual account-family stratum, whole configuration groups receive deterministic approximately60/20/20 training/validation/test roles from the fixed seed. Groups with fewer than three members remain training-only and cannot support a test result. Pulse/control partners, repeated runs and otherwise identical physical configurations share a role. Repeated physical configurations are not treated as separate inverse-prediction units. The split is written out for every source trace. No test value selects a representation or ridge penalty.

## Shared modes and effective temporal features

For each implementation, a common reference is fitted using unique training zero-input configurations. Per-channel scales are the root mean within-configuration variance over the training systems, with a physical-level numerical floor. The reference covariance and principal axes use these common scales; test configurations do not define their own normalization. Voltages and firing rates are also analyzed separately in native units. The covariance participation ratio, fraction captured by the common training mode, channel means and standard deviations, cortical and common-mode autocorrelations at100 and500ms, lag-one regression slope, spectral centroid and spectral entropy are retained.

There is no division by a configuration's nearly zero baseline standard deviation. Fixed physical-level floors protect constant channels; their treatment is explicit. A low covariance rank describes the sampled spontaneous trajectories. It does not establish anatomical necessity, the number of all possible dynamical modes, or controllability. The lag-one regression slope is a descriptive sampled-trajectory statistic. No nine-by-nine matrix is called the Jacobian of the delayed synaptic system.

## Inverse-factor recovery and ambiguity

Training-only ridge regressions predict log gain, log coupling and log history from cortical features, common effective-mode features or full-population features. A training-mean predictor is retained. Penalties0.01,1 and100 are selected only by validation standardized mean squared error. Feature standardization and target scales use training configurations. Fixed test configurations are scored once, with predictions, true targets, losses, selected penalties and effective degrees of freedom written to tables.

Training factor coverage includes observed factor values, largest sampled log gaps and the parameter-design rank. A fixed-ridge linear map from standardized log factors to standardized dynamical features supplies a finite-bank sensitivity spectrum and directions. Its relative5% rank is a descriptive index; it is neither structural identifiability nor a full-system Jacobian. Nearest training configurations from the same or another mechanism family are reported in each representation, alongside physical log-factor distances and the local gain-plus-coupling distance. A nearest neighbor is not declared signal-equivalent. Overlapping hsyn and nu effects may compensate locally; no exact equivalence of the complete mechanism families is asserted.

## Held finite-response prediction

Each of the48 real pulse/control pairs is qualified before subtraction. Prediction features use only the control history before pulse onset minus10ms. They cannot include any subsequent control activity or evoked trajectory. Six fixed targets describe the first2s after onset: peak and integrated root-mean-square response across nine voltages, the corresponding two firing-rate targets, and the first response-mode energy fractions for voltage and rate. Native timestamps remain unchanged, and absolute pulse onset is converted using the configured output start.

Every dynamical response predictor receives the actually observed pulse amplitude and configured width. Comparators include a training mean, observed input alone, the three actual factor values plus observed input, cortical dynamics plus observed input, effective mode plus observed input, full-population dynamics plus observed input, and factors plus effective mode plus observed input. No amplitude is divided out as if the nonlinear finite-input bank were a linear impulse-response design. The amplitudes1/10/100 occur on different configurations; they do not form a within-configuration dose-response series. Finite responses remain model-internal observations from a common all-population input, not selective anatomical interventions.

## Bootstrap interpretation

The default is2000 draws (`bootstrap_repetitions`,100–10000 allowed). Within each implementation and account family, the bootstrap samples held physical configuration groups with replacement; every compared model's prediction remains paired within its group. It summarizes equal-configuration loss and paired loss contrasts. The factor-plus-input model is the response comparator, and the training mean is the inverse-factor comparator. Outputs include percentile ranges, exact group counts, family stratum sizes and singleton strata.

These ranges describe sensitivity to the deliberately sampled finite simulator bank, conditional on the fixed training/validation fits. They omit fit uncertainty. They are not population confidence intervals for biological parameters, human-participant uncertainty, significance tests or independent replication. Time points, populations and repeated pairwise comparisons are never counted as independent bootstrap units.

## Qualification and outputs

`qualification_probe(inputs, seed=20261024)` runs only in the qualification compute allocation. It verifies the actual v2 cache,664/616/48 counts, actual configuration grouping, small real training/held feature probes in both variants and one genuinely matched input pair. It does not run the full model grid or bootstrap. No preferred effect size, p-value, mechanistic direction or improvement is required for qualification.

Outputs are aggregate CSV tables for source roles, all zero-input features, held inverse-factor predictions, held finite-response predictions and cross-family neighbors, together with parameter coverage, conditional configuration bootstrap, the training reference, model selection and a hash-bearing completion receipt. No per-epoch files or duplicated native arrays are created.
