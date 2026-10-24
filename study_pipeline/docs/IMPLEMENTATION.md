# Implementation trace and current coverage

The authoritative scientific design remains the proposal and decision record one directory above this pipeline. This document links the planned comparisons to operations actually implemented.

| Study operation | Implementation | Evidential boundary |
|---|---|---|
| Source/exposure census | `core.census`, role validation, metadata CSV | Declared counts; original-source schema and identity reconciliation remain inputs |
| Fixed clinical instrument | `native.py`; MATLAB qualification, native wrapper and batch receipts | Python static hashes verified; actual MATLAB execution pending |
| Genuine EEG observations | `data.py`, `cli.prepare` | Portable NPZ executed synthetically; MNE mocked; source-specific preprocessing pending |
| Complete score distribution and natural comparisons | `discovery.py`, `discover` | Descriptive matching; selected differences do not identify mechanisms |
| Conventional dynamics/geometry comparators | `signals.py` | Spectra, covariance rank, ACF and regularized VAR diagnostics; no claim of nonlinear manifold recovery |
| Independent-unit contrasts | `statistics.paired_contrast` | Informative-participant paired estimand and source-conditional bootstrap |
| Predictive increment | `models.grouped_prediction` | Development cross-validation, identical complete cases, training-only scaling |
| Protected prediction | `models.fit_prediction_model`, `evaluate_prediction_model`; CLI/freeze | Frozen parameters, no refit, all exposed participants excluded |
| Competing generative accounts | `mechanisms.py`; candidate bank fit/transport | Qualified observation mapping required; equivalent candidates retained |
| Native simulator candidates | `simulation.prepare_nftsim_bank`, external runner API | Bounded scalar configurations; NFTsim runtime and human forward map unqualified |
| Recovery and diagnostic-input tests | `models.effective_recovery` | Two-state linear synthetic harness, not the published biological model |
| Scale planning | `statistics.simulate_precision` | Explicit yield, fixed sites, independent missingness and normal-CI assumptions |
| Randomized human comparison | `statistics.randomized_contrast`; diagnostic timing/contract gates | Completed-outcome effect with missingness counts; final intervention protocol remains a discovery product |
| Behavioural generality | Grouped/frozen numeric-feature APIs | Task-family and native cognitive-model adapters conditional on a specified computation |
| Provenance and evidence freezing | `core.py`, recursive parent integrity checks | Records code/data identity; not preregistration or proof of outside analyst non-exposure |

The implementation avoids silently inventing the discovered phenotype, anatomical mechanism, clinical ground truth, Centaur prompt interface, stimulation parameters or publication outcome. These are scientifically consequential inputs, not deployment path choices.

The Python core depends on NumPy and uses `unittest`. Optional native EEG reading depends on MNE. MATLAB scripts retain the author-released primary inference; optional activation extraction uses the checkpoint's existing SeriesNetwork API and checks forward correspondence. No model conversion has been introduced.

Relevant interface references: [NumPy archive loading](https://numpy.org/doc/stable/reference/generated/numpy.load.html), [MATLAB SeriesNetwork activations](https://www.mathworks.com/help/deeplearning/ref/seriesnetwork.activations.html), [pinned author release](https://doi.org/10.6084/m9.figshare.30906557.v2). The published source files remain unchanged.

Construction validation uses known synthetic spectral/covariance structure, missingness and independent-unit cases, frozen transport, byte-hash tampering, clipping, source/participant separation, gaps and full synthetic integration. Tests verify those operations; they do not establish the study's empirical finding. Destination-specific MATLAB/toolbox, NFTsim executable and real-source checks are required before human-data execution.
