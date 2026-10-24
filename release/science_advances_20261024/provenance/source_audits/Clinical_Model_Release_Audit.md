# Toker 2026 clinical detector and simulator release audit

Audited 4 October 2026. Public web records and downloaded small code files were read; no MATLAB prediction or simulator run was executed.

## Primary records

- Final paper: https://doi.org/10.1038/s41593-026-02220-4
- Final supplement: https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-026-02220-4/MediaObjects/41593_2026_2220_MOESM1_ESM.pdf
- Pinned release: https://doi.org/10.6084/m9.figshare.30906557.v2
- Release metadata: https://api.figshare.com/v2/articles/30906557/versions/2
- README: https://ndownloader.figshare.com/files/60570617
- Wrapper: https://ndownloader.figshare.com/files/60459695
- GCS mapping: https://ndownloader.figshare.com/files/60459671
- Output-bound layer: https://ndownloader.figshare.com/files/60459692
- Checkpoint: https://ndownloader.figshare.com/files/60459821
- Conscious baseline config: https://ndownloader.figshare.com/files/60460013

## Safe claims and qualification

| Item | Verified | Qualification for proposal |
|---|---|---|
| Clinical foundation | Final paper reports deep convolutional detectors, neural-field generators and biological tests. | The headline training/validation totals describe the published framework. Do not automatically attribute every total to the particular released checkpoint. |
| Available detector | Pinned v2 has exactly one .mat checkpoint, trained_consciousness_detector.mat, 2,101,046 bytes, MD5 03a9c7f599c5630cd71bddfd3a542600. | Final supplement compares three cortical variants. Only one checkpoint was found in v2; exact variant and training manifest need confirmation. |
| Inputs | Wrapper accepts a single cortical vector and sampling rate, checks ten-second duration within 0.01 s, filters 0.5–45 Hz, resamples 250 Hz and z-scores. | Apply per channel with a specified aggregation. Do not concatenate multiple channels into a vector, provide five-second windows, or directly provide behaviour/LLM/fMRI activations. Input orientation and runtime must be tested. |
| Output | Wrapper calls predict(trainedNet, features), describes predicted GCS 3–15, then (GCS–3)/12. Custom output layer clips to [3,15]. | Continuous readout is a remapped clinical-scale prediction, not a calibrated probability of awareness. Floor/ceiling ties can result from clipping. Preserve pre-bound output and check saturation before matched-score inference or sensitivity analysis. |
| Internal network | Static MAT inspection finds trainedNet, SeriesNetwork and CNN layer names/classes, including bounded_output and regressionoutput. | MATLAB supports SeriesNetwork layer activation extraction. Feasible in principle, pending actual network load, layer enumeration, input-shape and forward-pass checks. No extracted activations have been produced. |
| Simulator | Modified NFTsim source, one conscious baseline and 280 DoC configurations are present. | Final supplement discusses 1,680 simulated cases; the release is not verified as the complete set or full optimisation pipeline. |
| Spatial scope | Conscious config has Nodes: 1 and nine neural populations, with external stimulus/perturbation drivers. | Baseline provides lumped population dynamics, not a validated region-resolved multichannel scalp-EEG manifold. A spatial extension and observation model require separate constraints and validation. |

MATLAB activation documentation: https://www.mathworks.com/help/deeplearning/ref/seriesnetwork.activations.html

## Training and sample overlap

The final supplement lists acute TBI coma EEG (22 people), chronic DoC EEG (14), healthy dataset 1 (10), and cortical recordings from essential tremor, Parkinson disease and waking monkeys among training/validation sources. Two additional cortical variants use human EEG only, predicting GCS or CRS-R. These variants are described in Supplementary Figure 6, but are not separately identified in the v2 file list.

The healthy resting source cited by Toker is Torkamani-Azar/SPIS, whose author repository describes ten Sabanci participants with pre-SART rest: https://github.com/mastaneht/SPIS-Resting-State-Dataset . Andrillon's Monash SART cohort is a separate 26-person sample with Monash ethics approval: https://doi.org/10.1038/s41467-021-23890-7 . Thus the shared task name SART does not establish participant overlap. Their documented institutions, authors and experiments differ. Verify against the final training manifest before declaring complete provenance independence.

The PNAS 62-person mind-blanking analysis reuses the Andrillon Monash cohort, so those 26 must not be added to 62 as a new replication: https://pmc.ncbi.nlm.nih.gov/articles/PMC12771569/ .

## Strong design implication

Make the frozen released clinical detector the initiating empirical probe. Retain its layer activations and pre-bound readout after runtime qualification. Use the provided simulator as a constrained starting model, fit competing physiological explanations to measured human dynamics and held-out perturbation responses, and use clinically anchored same-score/different-mechanism comparisons. Clinical scale scores initiate discovery; independent experience, performance and stimulation responses adjudicate human interpretation. Added spatial modelling, missing detector variants or independently accessed clinical recordings remain proposed resources, not achieved capabilities.
