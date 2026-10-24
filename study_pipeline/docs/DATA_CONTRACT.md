# Data and inference contract

## Recording metadata

`recordings.csv` has one row per continuous recording. Required columns:

```text
recording_id,participant_id,source_id,cohort_id,site_id,session_id,role,signal_path,signal_format,modality,units,duration_s,sfreq_hz,channel_count,detector_exposure,analyst_exposure,synthetic
```

IDs are deidentified. `participant_id` is globally reconciled across visits and sources; the programme cannot detect an undocumented alias. `cohort_id` identifies an acquisition cohort and stays in one role. Roles are `development`, `evaluation`, `diagnostic` or `clinical_reference`. Exposure values are `yes`, `no`, or `unknown`; protected roles require confirmed `no`. `synthetic` is an explicit boolean. Empirical runs reject synthetic recordings. EEG amplitudes must be normalized to microvolts with provenance preserved. Signal paths stay inside the configured data root.

The metadata-only census reports declared counts. It does not certify compatible signals, accessible dense EEG, informative-person yield or awareness labels. The original Monash subset must use the same global IDs when appearing in another archive.

## Continuous signal interchange

NPZ archives contain:

| Key | Contract |
|---|---|
| `data` | Finite real array, channels × continuous samples, microvolts |
| `sfreq_hz` | One finite positive sampling frequency |
| `channel_names` | Unique nonempty Unicode array, one label per channel |
| `units` | Optional scalar `uV`; omission asserts the interchange contract |
| `reference` | Optional provenance string, never automatically applied |
| `sample_times_s` | Optional time per sample relative to recording onset; exposes discontinuities |
| `gap_intervals_s` | Optional N × 2 invalid intervals in seconds |
| `bad_channels` | Optional Unicode names requiring explicit upstream handling |

Without timestamps or gap intervals, the producer explicitly asserts continuity. Source conversion must preserve meaningful gaps and BAD/EDGE/boundary annotations. Native file readers require optional MNE, select EEG channels and normalize MNE volts to microvolts without re-referencing. Test coverage uses a mocked MNE interface; selected real-source releases remain unqualified.

## Episodes and reports

Use separate `episodes_development.csv`, `episodes_evaluation.csv` and `episodes_diagnostic.csv` files. The code must not read a mixed report table merely to select development rows. Required columns:

```text
episode_id,recording_id,start_s,stop_s,report_start_s,report_stop_s,report_category,capacity_value
```

`start_s`/`stop_s` address genuine continuous ten-second intervals on the sample grid. There is no padding or concatenation. Episode IDs are unique within the role and should be globally namespaced by source/recording. Report endpoints are supplied together or both left blank. Their overlap with the EEG interval is a timing quantity, not a consciousness label.

Retain original report categories. Mind blanking, no recall, white dream, reported no experience, responsiveness, environmental connectedness and a named capacity are distinct measurements. Add `report_timing_basis`, `report_instruction_reference`, and `report_point_s` when source-supported. Distinguish analyst-assigned intervals from explicit interval instructions and point reports; overlap alone cannot validate experiential duration. The SART five-second analysis convention is not an independently verified five-second experience report. A numerical `capacity_value` has task-specific units and missingness. The actual construct, assay, timed support and comparison belong in the analysis contract before independent evaluation.

## Descriptive features

Features use the available multichannel scalp signals; native detector jobs use explicitly selected individual channels. They are different representations.

- Relative and absolute band powers use a documented Hann Welch calculation, segment-wise mean removal and one-sided density normalization. Relative powers normalize over the declared spectral range.
- Effective rank summarizes covariance eigenvalue dispersion. It is a covariance statistic, not a recovered nonlinear manifold dimension.
- Autocorrelation features preserve each channel's actual sample lag. They are conventional temporal comparators.
- VAR(1) fits and chronological within-window holdouts are descriptive diagnostics. They do not replace participant-level or acquisition-cohort evaluation.
- Phase-surrogate APIs preserve Fourier magnitudes. Shared phases preserve cross spectra; independent phases alter cross-channel phase relationships. Neither is a biological intervention or necessarily preserves amplitude distributions.

Quality rules are configurable development screens. Source-specific artifact exclusion, reference choice, and any alternative preprocessing require qualification and freezing; the code never repairs them implicitly.

## Native scoring

The exporter writes one real channel vector per job. Inputs pass unchanged to the published MATLAB wrapper: ten seconds, native filtering/resampling/z-scoring, frozen weights, bounded GCS and linear remapping. `preclip_gcs` is retained only after forward correspondence. Selected layer activations have their own artifact files.

Every batch must carry matching signal, job-manifest, qualification and score-table hashes. The import rejects incomplete batches. Channel aggregation is prespecified, while saturation fraction remains visible. Equal clipped scores cannot be interpreted as equivalent underlying representations.

## Statistical units and uncertainty

The independent human unit is the participant. Repeated windows, episodes, visits, model seeds and simulator candidates do not increase participant N. Paired contrasts average windows within episode, episodes within condition and conditions within participant. Bootstrap intervals resample participants within fixed observed source strata. Estimates concern informative participants; population prevalence requires the complete eligible sampling frame.

No-outcome observations remain in counts and exclusions. Capacity preservation requires a justified equivalence margin. The supplied equivalence routine is an approximate bootstrap-interval criterion rather than an exact TOST. Source-superpopulation uncertainty is outside the current bootstrap estimand.

Grouped and frozen prediction compare baseline/augmented models on identical complete cases with equal participant weighting. Scaling and coefficients are learned only in development. Held-out predictions do not establish biological use or causal relevance.

## Randomized diagnostic trials

Required scalar fields are participant/episode IDs, original eligibility and assignment flags, a treatment arm and the frozen outcome. For empirical eligible randomized trials also record:

```text
eligibility_start_s,eligibility_stop_s,assignment_s,intervention_s,post_window_start_s,post_window_stop_s
```

Times share one documented session clock. Eligibility must use a completed ten-second window ending before assignment. Assignment precedes intervention, which precedes the outcome observation. Original eligibility cannot depend on a successful post-intervention outcome. Faster evoked responses need separately qualified timing/artifact handling.

Missing outcomes remain missing. The current completed-outcome paired contrast is not a full intention-to-treat analysis under arbitrary missingness, nor does it identify the causal effect of the selected starting state. A final missing-data estimand, allocation protocol, manipulation controls and power calculation are scientific inputs to finalize before recruitment.

## Generative candidate banks

Candidate observation-space tables identify `candidate_id`, `mechanism_family`, parameter values, a qualified observation mapping and the named feature predictions. Fitting only the clinical score is prohibited. Fit features, scaling, candidate hierarchy and ambiguity tolerance must be determined in development. Similar fitting errors retain an equivalence class rather than invent a unique mechanism.

Linear recovery diagnostics are labelled synthetic. Native simulator preparation alone does not validate a scalp forward map. A specific synapse, deep pathway, experiential outcome or functional restoration requires appropriate human/biological evidence.
