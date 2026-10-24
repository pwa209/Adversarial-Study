# Before making Adversarial-Study public

This checklist is for the repository owner; it is not a claim that institutional, ethics or licensing approvals have been completed. Repository visibility has **not** been changed by this documentation update.

## Verify release identity and scientific traceability

- [ ] Confirm the current default-branch commit, pin it in the submitted manuscript/SI, and update the citation if it differs from the earlier research snapshot `944982414275c2c340da539fc5b4a42ad56dae9d`.
- [ ] Keep the dated `release/science_advances_20261024` source files and hash-pinned manifests unchanged; correction requires a new, independently labeled release.
- [ ] Run `python release/science_advances_20261024/scripts/prepare_plot_data.py` and source-export checks in the documented environment, or state clearly if the provenance audit was not rerun.
- [ ] Run the new two-figure renderer and visually inspect `Figure_1_Recovery` and `Figure_2_Integrated` at publication size; the schematic waveforms are illustrative.
- [ ] Confirm all 14 supplementary plates and source tables S01–S09 are mapped by the crosswalk and that relevant null results are retained.

## Privacy and data rights

- [ ] Manually review both committed history and current files for credentials, tokens, internal endpoints, passwords, private URLs and user-identifying metadata. A simple local pattern scan found no obvious secret-like strings in the supplied reviewer archive; **this is not a full historical secret scan**.
- [ ] Review `statistics/S04_human_participant_effects_and_leave_one_out.csv`, `S07_human_forecast_participant_effects.csv`, and their duplicated source tables. They contain **pseudonymized participant labels**. Verify the original DREAM data's consent terms, permitted redistribution and re-identification risk before public release.
- [ ] Review retained HPC usernames, filesystem paths and environment identifiers, including `study_pipeline/cluster/README.md`, `study_pipeline/configs/dependency_runtime_cache.json`, and `provenance/completed_analysis_inputs.json`. These are provenance details, not access credentials, but may expose infrastructure identifiers.
- [ ] Confirm the third-party Toker model, NFTsim components, DREAM/other public data and any borrowed code are appropriately attributed; do not imply this repository redistributes third-party weights or grants rights it does not possess.
- [ ] Check SVG/PDF metadata and generated ZIPs for embedded author details or local paths. Manuscript, cover letter, correspondence, local secrets and raw human recordings should stay outside this repository.
- [ ] Review the full Git **history**, not just the current tree, before switching visibility. Deleting a file from the current revision does not remove it from old commits.

## Licensing, citation and release process

- [ ] Make an explicit rights decision for **original code** and **derived datasets/figures**. Do not paste a blanket LICENSE over third-party assets without a rights review. Until a license is chosen, public visibility alone confers no general permission to reuse.
- [ ] Review `CITATION.cff` names and affiliations; update with an actual article DOI **only when assigned**.
- [ ] Create an immutable Git tag/release and consider Zenodo DOI archiving **after** resolving rights and the appropriate publication timing.
- [ ] Update the manuscript Data Availability and Code Availability declarations to the final public commit/archival DOI; do not claim a persistent DOI in advance.
- [ ] Confirm both coauthors approve public posting and that planned dissemination is compatible with the journal submission/publishing policy.

**Visibility change is intentionally left to the repository owner.**
