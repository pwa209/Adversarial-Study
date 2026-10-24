# Submission figure and supplementary-table crosswalk

This map links the **two-main-figure, 14-supplementary-plate** draft to existing dated archival figure sources. These display items are called **Supplementary Figs. S1–S14**, not publisher-approved Extended Data figures. The original release is preserved as it was generated.

## Main figures

| Current display | Scientific job | Input source | Status |
| --- | --- | --- | --- |
| Fig. 1a | Held-configuration parameter-recovery error (four fitted pipelines) | `plot_data/recovery_summary.csv`; archive Fig. 2 | Model-based, unequal fitted capacity |
| Fig. 1b | Paired cortical versus full-population error over 10 held configurations | `data/mechanism_compensation/parameter_predictions.csv`; archive Fig. 2 | Model-based, no new test |
| Fig. 2a | Native dendritic pathway summation and illustrative cancellation | `data/pathway_balance/source_equation_proof.json`, associated native source; archive Fig. 5 | **Schematic**; no measured waveforms |
| Fig. 2b–c | Absolute/net pathway response and cancellation across nine populations | `data/pathway_balance/population_pathway_balance.csv`; archive Fig. 5 | 48 configuration pairs, 432 population observations |

All paths above are relative to `release/science_advances_20261024/`.

## Supplemental display plates

| New plate | Topic | Earlier preparation name |
| --- | --- | --- |
| Supplementary Fig. S1 | Model construction and analysis workflow | ED1A_model_and_protocol |
| Supplementary Fig. S2 | Manipulated configurations and native coverage | ED1B_configuration_coverage |
| Supplementary Fig. S3 | Recovery calibration and fitted complexity | ED2A_recovery_calibration |
| Supplementary Fig. S4 | Recovery residuals across mechanisms | ED2B_recovery_residuals |
| Supplementary Fig. S5 | Ridge validation of recovery | ED2C_recovery_validation |
| Supplementary Fig. S6 | Spontaneous model geometry | ED3A_geometry |
| Supplementary Fig. S7 | Full finite-response prediction comparisons | ED4A_pulse_prediction |
| Supplementary Fig. S8 | Pulse-prediction calibration | ED4B_pulse_calibration |
| Supplementary Fig. S9 | Ridge validation of perturbation prediction | ED4C_pulse_validation |
| Supplementary Fig. S10 | Pathway balance and observational matching | ED5A_pathway_and_matching |
| Supplementary Fig. S11 | Numerical reconstruction and eligibility | ED5B_numerical_qualification |
| Supplementary Fig. S12 | Human sleep-report comparisons | ED6A_human_report_contrasts |
| Supplementary Fig. S13 | Human geometry and adjusted tests | ED6B_human_geometry_and_tests |
| Supplementary Fig. S14 | Held-person forecasting comparisons | ED7A_human_forecasting |

These 14 plates are editorial rearrangements of the archived figure panels, not new experiments. They are included in the separate author-controlled submission package; the complete numerical sources, original five main figures, seven supplementary figures, R scripts and provenance remain under the dated archival release. The original source directory does not itself use S1–S14 filenames.

## Source statistical tables

`release/science_advances_20261024/statistics/S01_parameter_recovery.csv` through `S09_descriptive_summaries.csv` retain all analyses:

S01 parameter recovery; S02 finite response; S03 corrected human spatial tests; S04 human participant effects; S05 configuration bootstrap; S06 forecast account comparisons; S07 forecast participant effects; S08 strict matching dispositions; S09 pathway descriptive summaries. The `statistics/README.txt` describes estimands, sources, denominators and interval conventions.

The full source archive preserves null results and reasons that strict human-supported evoked-pathway matching could not be established. No equivalence claim is inferred from a nonsignificant test.
