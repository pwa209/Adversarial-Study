STATISTICAL RESULTS IN CSV

S01_parameter_recovery.csv: all four inverse-factor predictors; means, existing
2.5th/97.5th configuration-bootstrap percentiles, baseline differences, family
strata, singleton strata and the actual 2,000-draw source settings.

S02_finite_response_prediction.csv: all seven predictors and all saved contrasts
with factors plus observed input, including negative and null outcomes.

S03_human_spatial_all_tests.csv: all four primary and ten sensitivity tests;
effect definitions, person counts, paired episode counts, conditional intervals,
uncorrected sign-flip and Holm-adjusted p-values. Spatial bootstrap: 10,000.

S04_human_participant_effects_and_leave_one_out.csv: saved spatial participant
effects and leave-one-person-out records where present in the original output.

S05_pathway_configuration_bootstrap.csv: all 576 saved descriptive cluster
summaries. These have no p-values and no human-population interpretation.

S06_human_forecast_all_account_comparisons.csv: all saved account comparisons,
plus the separately identified validation-selected seven-comparison family;
effect estimates, conditional intervals and corrected reference tests.
Forecast bootstrap: 20,000. These five people differ from the six-person spatial
test set. Repeated account rows are not independent participants.

S07_human_forecast_participant_effects.csv: original participant differences and
leave-one-out records, retaining the analysis-set and comparator identities.

S08_pathway_matching_dispositions.csv: all saved score and strict-feature match
counts, failure reasons and human-fit support. The zero count of strict,
supported pulse comparisons is retained.

S09_descriptive_summaries.csv: newly calculated means, medians, inclusive
quartiles, minima/maxima and finite/total row counts for native participation
ratios and cancellation, directly from pinned CSVs. No inferential test is added.

source_tables/: byte-identical copies of all 22 original result CSV files,
including 1,968 edge timing rows and 7,524 population-balance rows, all held
predictions, matching/disposition tables, features and validation grids.

display_*.csv: the 11 display tables used by the R script.

all_saved_numerical_results.csv: every numeric JSON leaf from all 14 source
JSON files (7,712 values), with exact source path and RFC 6901 JSON pointer.
all_saved_result_values.csv additionally preserves strings, booleans and nulls
(42,612 primitive values) so labels, interval meanings and unqualified statuses
remain resolvable. Numeric IDs/counts/coefficients are included; a numeric leaf
is not automatically a hypothesis test. Empty p-value fields mean no p-value
was computed; they never mean p = 0. Original CSV numeric fields are preserved
in source_tables and are not redundantly counted as JSON leaves.

Column names with __ delimiters flatten a nested JSON field. Numbered interval
fields __0 and __1 retain the exact source endpoints; they are not symmetric
standard errors. Lists of strata or metadata remain JSON strings when needed
to preserve their structure within a CSV cell. All quoted numbers have their
original independent unit, scope and missingness in the source record.

Run scripts/export_statistics.py to verify all 37 source hashes and regenerate
these CSVs. It performs no fitting, resampling or significance testing.
