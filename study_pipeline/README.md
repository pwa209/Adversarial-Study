# Adversarial consciousness study — computational pipeline

Revised 4 October 2026 for the original clinical-model-guided mechanistic programme on Narval. The immutable release is deployed through the project's existing authenticated bridge; a scheduled Slurm controller queues the declared graph. Current receipts and qualification outcomes are in ../Deployment_Status.md.

The programme keeps the frozen clinical detector central while separating measured EEG, detector activations, simulator populations and cognitive-model representations. Software checks and synthetic fixtures are not human findings. The unresolved phenomenon, biological account, manipulation and primary outcome must be determined through discovery rather than supplied by the code.

## Original programme execution

`cluster/make_plan.py` constructs the complete available-data graph, rather than only descriptive score–report measurements. Its branches include:

1. Pinned clinical source, true EEG reference, native MATLAB predictions and activations.
2. Monash/Paris attention development, all18 inventoried tactile participants and dense Tononi DREAM sleep39people/287episodes; yield follows actual qualification. Sparse Oudiette63people/246episodes is acquired/audited but detector scoring waits for documented physical units.
3. `adversarial_study.model_guided`: natural clinical-model-guided episode comparisons with unselected denominators, fullscalp human trajectories, leave-person/held-measurement generative fits and outcome comparison against clinical/conventional features.
4. `native_tools.mechanistic_native`: original281configurations in pristine and explicitly repaired implementations, runtime controls and48bounded gain/coupling/history/combined input–control pairs, for664native runs. Failed simulations remain in the inventory. Frozen MATLAB scores their actual population signals separately.
5. Empirically constrained adversarial contrasts across supported accounts, with unsupported whole-bank contrasts retained separately; diagnostic artifacts preserve weak, equivalent and absent accounts without forcing a result.
6. Psych-101/201 exposure/task audits and sequential comparators, then matched human fine-tuning of the original public 4-bit Centaur/base models before held-out likelihood and representation comparisons. The clinical EEG model never scores behavioural transcripts.
7. Protected replication and new randomized human data collection are explicit external inputs after candidate qualification and prospective scientific freezing.

The controller bounds concurrent accepted allocations at16CPUcores (including itself) and2A100GPUs, including pending jobs. Transfers are a serial8MiB/s pinned public-file helper; numerical work runs only in Slurm. The study scratch allowance is1TB/150,000entries, within the separately measured personal quota. Source, plan, input, native outputs and stage receipts are hash-bound. Failed stages cannot automatically rerun or replace evidence. A failure in one transfer group does not abandon unrelated branches.

The current biological observation mapping is an effective standardized single-population temporal proxy, tested against actual human measurements with participant grouping. Its uncertainty, out-of-domain cases and equivalence classes are retained. It does not infer a specific anatomical synapse or establish a scalp forward model. Functional predictions are evaluated from measured human outcomes with grouped training; simulated human function remains unqualified until a separate validated mapping exists. The programmed human-experiment artifact is a candidate protocol, not recruited people or a completed prospective preregistration.

## Start here

Use Python 3.10 or later with NumPy. MNE is optional for native EEG file formats; the core interchange format is NPZ. MATLAB with the original checkpoint and required toolboxes is needed for empirical detector inference. Published NFTsim and the explicitly named source repair are compiled and compared in scheduled native runtime stages. Simulator execution does not qualify a human observation mapping.

From this directory:

```sh
python run_study.py --help
python -m unittest discover -s tests -v
python run_study.py demo --output examples/my_synthetic_check
```

Each demo destination must be new or empty. The demo generates 16 synthetic participants, genuine continuous synthetic arrays, original synthetic labels and an explicitly arbitrary mock score. It exercises preparation, participant contrasts, grouped prediction and simulation diagnostics. It exports native input jobs but never executes or impersonates the released detector. `demo_summary.json` records this distinction.

For a standalone installation, use `python -m pip install -e .` in the eventual environment. Optional MNE readers can be installed with `python -m pip install -e '.[eeg]'`. No installation was performed during construction.

## Configure before using human data

Copy `configs/study.template.json` to a project-specific configuration. Paths resolve relative to the configuration directory plus its `path_base`, so the pipeline can move without editing Python. The template deliberately leaves the scientific contrast unset. Populate source metadata and role-specific episode files using `data_templates/` and the [data contract](docs/DATA_CONTRACT.md).

Globally consistent, deidentified participant IDs are required across visits and sources. Whole participants and acquisition cohorts cannot cross development, evaluation, diagnostic and clinical-reference roles. Unknown detector or analyst exposure disqualifies a source from protected roles. Metadata auditing never opens report tables or signal arrays.

`mode=development` is reserved for synthetic software checks. Human discovery uses `mode=empirical` together with `role=development`; the two fields have different purposes. Synthetic and empirical recordings cannot be mixed.

```sh
python run_study.py --config configs/study.json census
python run_study.py --config configs/study.json verify-instrument
python run_study.py --config configs/study.json prepare --role development
```

Preparation verifies the declared recording metadata, sample grid, genuine ten-second windows, bad intervals, channel labels, units and configurable quality rules. It computes interpretable descriptive features and exports one native signal per selected channel. The amplitude and flat-channel defaults are development choices requiring source qualification. They do not constitute comprehensive artifact rejection. `--features-only` is available for measurement development; it does not complete the detector-centred study.

Original report instructions and timing basis are retained. The SART paper's five-second labelling convention is an analyst-defined interval with uncertain experiential timing; point reports and explicitly interval-referenced reports are distinct. A computed overlap records temporal bookkeeping, without certifying experience for that interval. The pipeline never pads, concatenates trials, interpolates artifacts or silently changes the EEG reference.

## Qualify and run the frozen detector

The release is pinned to Figshare version 2. Checkpoint, wrapper, remapping and output-layer byte hashes are checked. The single released checkpoint's identity relative to the paper's variants remains unresolved.

In MATLAB, add `study_pipeline/matlab` to the path, then use explicit absolute paths:

```matlab
addpath('/chosen/project/study_pipeline/matlab');
qualify_detector('/chosen/reference/ten_second_channel.csv', 250, ...
    '/chosen/project/source_records/toker_v2', ...
    '/chosen/project/study_pipeline/qualification/native_detector.json', ...
    'reference_recording_provenance_id');
batch_predict('/chosen/prepare_run/native/native_jobs.csv', ...
    '/chosen/native_outputs/scores.csv', ...
    '/chosen/project/source_records/toker_v2', ...
    '/chosen/project/study_pipeline/qualification/native_detector.json');
```

Create the qualification/output directories beforehand. Qualification requires a declared empirical reference. It tests repeated native inference, records runtime versions and attempts optional pre-clipping extraction with forward correspondence. Supply a sixth cell-array argument of exact layer names to request additional activations. Their names are not guessed. Extracted arrays are saved separately from scalar scores. Repeated activation checks are necessary before layers are labelled qualified.

Qualification establishes runtime operation on the reference input. It does not establish clinical calibration, source transport validity, or biological meaning. The primary score always comes from the unmodified native wrapper. Its GCS scale and linear remapping are retained; neither is labelled an awareness probability.

```sh
python run_study.py --config configs/study.json import-scores --prepared /chosen/prepare_run --scores /chosen/native_outputs/scores.csv
python run_study.py --config configs/study.json discover --table /chosen/scored_run/analysis.csv
```

Native import requires a complete successful batch, per-signal hashes and matching qualification/checkpoint receipts. Missing, duplicate, failed, changed, out-of-range or inconsistently remapped outputs stop the stage. Channel aggregation is declared in configuration. Saturation and unavailable pre-clipping outputs remain explicit.

## Discover and explain

`discover` reports the full qualified score distribution and unselected within-person relationships before complementary exploratory registries: similar score/different capacity, similar capacity/different score, original report-category differences, and timed neural transitions. Each pair registry uses a participant and recording with declared tolerances and temporal separation; different registries can reuse observations. Saturated episodes are excluded from matching by default while remaining in distribution and relationship summaries. Unknown saturation stays unknown. Gaps and unavailable report timing remain explicit. These comparisons support open discovery, with no required result, direction, significance or mechanism.

After identifying a reproducible question, specify the analysis contract:

```sh
python run_study.py --config configs/study.json contrast --table /chosen/scored_run/analysis.csv
python run_study.py --config configs/study.json predict --table /chosen/scored_run/analysis.csv
python run_study.py --config configs/study.json precision
python run_study.py --config configs/study.json recovery
```

Contrasts average repeated observations within episode and participant before inference. Confidence intervals resample independent participants within observed source strata. Equivalence uses a separate 90% percentile-bootstrap interval inside a prespecified margin; this is an approximate interval criterion, not an exact TOST implementation. Singleton source strata withhold inferential intervals. Grouped ridge prediction fits scaling on training participants and compares baseline and augmented models on identical complete cases. Cross-validation remains development evidence.

Precision simulations expose informative yield, missingness, participant/site variation and episode autocorrelation as assumptions. They do not report achieved power. The recovery example is a transparent two-state linear simulator for checking recovery and diagnostic predictions. It is not NFTsim, a validated scalp forward map, or evidence for a human pathway.

Use `nftsim-bank` with a JSON object containing `parameter_sets` and explicit `bounds` to prepare constrained native configuration candidates. The reviewed adapter currently addresses scalar `Coupling N.nu` fields only. It rejects comments, ambiguous/multivalue fields, unsupported parameters and out-of-bound values. The original template is retained. Its one-node baseline cannot establish regional propagation. A separate external runner API records executable hashes and logs; it does not submit jobs.

Generative observation-space banks can be compared with `fit-mechanisms` and `evaluate-mechanisms` once their simulator-to-observation mapping is qualified. The associated API retains near-equivalent candidates and evaluates held-out people without refitting. It cannot identify a unique biological parameter from an ambiguous bank. See [implementation coverage](docs/IMPLEMENTATION.md).

Set `mechanisms.enabled=true`, the named phenotype condition, feature columns and ambiguity tolerance before fitting. The candidate table preserves the mapping and simulator references. Include the fitted artifact in the freeze before protected transport:

```sh
python run_study.py --config configs/study.json fit-mechanisms --table /chosen/development_scored_run/analysis.csv --bank /chosen/qualified_candidates.csv
python run_study.py --config configs/study.json evaluate-mechanisms --table /chosen/evaluation_scored_run/analysis.csv
```

## Protect independent evaluation

Before opening protected signals or outcomes, finalize the contrast, transformations, quality rules, exclusions, inference procedure and cohort assignment. If transporting a predictor, set `analysis.enable_prediction=true`, fit it on development data and retain its artifact:

```sh
python run_study.py --config configs/study.json fit-predictor --table /chosen/development_scored_run/analysis.csv
python run_study.py --config configs/study.json freeze
python run_study.py --config configs/study.json prepare --role evaluation
python run_study.py --config configs/study.json import-scores --role evaluation --prepared /chosen/evaluation_prepare_run --scores /chosen/evaluation_scores.csv
python run_study.py --config configs/study.json contrast --role evaluation --table /chosen/evaluation_scored_run/analysis.csv
python run_study.py --config configs/study.json evaluate-predictor --table /chosen/evaluation_scored_run/analysis.csv
```

A freeze records configuration, code including MATLAB helpers, cohort manifest, qualified instrument and fitted model hashes. A changed artifact stops protected evaluation. The receipt is not preregistration and cannot prove absence of prior exposure outside this programme. A repair after evaluation needs an explicit new version and an appropriate fresh-evidence plan; do not overwrite the original receipt.

Frozen prediction applies development scaling and coefficients unchanged and rejects all exposed training participants, including those excluded from fitting. It reports source overlap and participant-weighted errors. Shared sources do not establish source-superpopulation generality.

## Diagnostic experiment and conditional branches

The actual manipulation is deliberately unspecified until candidate explanations make differing feasible predictions. Once fixed, populate `diagnostic` in configuration before assigning diagnostic cohorts and freezing. The `diagnostic` command requires the frozen outcome/arms and checks that the completed eligibility window precedes assignment, intervention and post-intervention observation. It retains all randomized eligible trials in counts and reports missingness. Its current effect estimator concerns participants with observed outcomes in both arms; it does not claim a complete intention-to-treat estimate under nonrandom missingness.

```sh
python run_study.py --config configs/study.json diagnostic --trials /chosen/diagnostic_trials.csv --outcome capacity_value
```

DREAM, waking collections and clinical sources use the same contracts but retain their own report definitions, sampling frames and context-specific analyses. Source-specific raw-file/event conversion requires an audited adapter for the selected release. The current code does not guess archive schemas or automatically pool sleep with waking.

Psych-101/201 and Centaur provide a native task-specific human-training branch. `extension_finetune` freezes participant splits, performs actual adapter updates for both Centaur and the same-base comparator on identical human rows/budgets, evaluates only after both trained artifacts pass verification, and then compares matched held-out choices and genuine activations. The initial fixed feedback-task branch qualifies native training and task comparison; it does not establish discovery-selected behavioural generality. Source eligibility and actual two-GPU gradients/memory remain runtime checks.

`native_tools.human_finetune` exports genuine tactile inputs and adapts a separate copy of the released clinical network to original human detection responses. The original GCS discovery instrument remains fixed. `human_generative_adaptation` uses fixed participant partitions for train-only temporal calibration and constrained native-bank parameter selection. `cross_system` verifies the completed adaptation artifacts and records the further shared-assay requirements. Cross-modal inference remains unavailable until a qualified common human task and protected matched evaluation exist. Behavioural sample sizes do not increase neural participant N.

## Connected neural-state and mechanistic analysis

The connected analysis uses participant-disjoint training, validation and test groups. `neural_state_space` learns fixed sensor coordinates and predictive state dynamics; `native_tools.adapted_geometry` exports actual activations from the human-adapted clinical checkpoint; `representation_bridge` compares those activations with human trajectory geometry and original human outcomes; `manifold_mechanisms` fits the native biological traces to the full multivariate human trajectories and freezes distinguishing computational predictions. These stages run automatically in the Narval graph after their actual training prerequisites. The previous scalar-summary fits remain descriptive references. See [the full methods](docs/neural_manifold_methods.md) for commands, reliability controls, missingness, parameter ambiguity and the limits of the effective observation map.

`manifold_diagnostic freeze` binds those connected outputs and all development participants. Its future `evaluate` command requires a qualified human intervention, prospective registration, independent participants, source-bound measured outcomes and a complete original assignment registry. It retains missing outcomes through declared bounds and estimates the randomized contrast with equal participant weights. A computational candidate alone cannot authorize human diagnostic analysis.

## Audit outputs

Each stage creates a new UTC-stamped run directory. Inputs, generated outputs, normalized recording files, configuration and code are hashed. Completed tables verify their upstream provenance before further use. Failed preparation preserves exclusions and a failure record. Important scientific defaults and qualifications are described in docstrings and the data contract.

Local build checks cover Python algorithms, source adapters and synthetic integration. Destination-specific MATLAB, MNE and published NFTsim execution is qualified by the Narval graph in `cluster/`, with actual job/runtime receipts kept separately from local tests. See `../Scientific_Revision_20261004.md` for the revised scientific design and `cluster/README.md` for source-pinned deployment and recovery.

`BUILD_MANIFEST.json` records the checked source hashes, exact Python/NumPy versions, test log, synthetic demonstration and static release verification. `python tools/record_build.py` regenerates that record while preserving the previous manifest and creating fresh validation/demo directories.
