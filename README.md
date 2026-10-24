# Adversarial Study

**Reproducible analyses of observation-dependent parameter recovery and hidden pathway contributions in a computational model of consciousness.**

This repository accompanies the ongoing two-author *Matters Arising* project by **Peng Wang and Steven Laureys** concerning the computational framework of [Toker et al. (2026)](https://doi.org/10.1038/s41593-026-02220-4). The article has **not been represented here as accepted or published**. The manuscript, cover letter and author correspondence remain outside the repository during editorial consideration.

## Start here

| Goal | Location |
| --- | --- |
| Understand the current two-figure research presentation | [Nature Neuroscience figure guide](release/nature_neuroscience_matters_arising_20261024/README.md) |
| Regenerate its quantitative figures from the existing tables | [Figure-generation instructions](release/nature_neuroscience_matters_arising_20261024/README.md#rebuild-the-two-figures) |
| Read the original dated computational release | [Frozen analysis release](release/science_advances_20261024/README.txt) |
| Inspect all statistical results, including null outcomes | [Statistics dictionary](release/science_advances_20261024/statistics/README.txt) |
| Inspect the full staged research pipeline | [Study pipeline](study_pipeline/README.md) |
| Match the new display numbers to the original evidence | [Figure crosswalk](release/nature_neuroscience_matters_arising_20261024/FIGURE_CROSSWALK.md) |
| Review public-release and reuse issues | [Public-release checklist](PUBLIC_RELEASE_CHECKLIST.md) |
| Cite the repository | [Citation metadata](CITATION.cff) |

## Main findings from the existing release

- In 48 manipulated model configurations (28/10/10 training, validation and test split), full-population features yielded lower *descriptive* held-configuration parameter-recovery error than a cortical-feature pipeline: approximately **0.169 versus 0.336** standardized squared error across the 10 held configurations. Fitted model capacities differed.
- Source-verified reconstruction of coupled and filtered postsynaptic voltage contributions exposes signed pathway cancellation. Across 48 paired pulses and nine modeled populations (432 population-by-configuration observations), mean cancellation indices included **0.413 in population 4** and **0.333 in population 6**.
- Recovering known parameters, predicting finite perturbation responses, and explaining human observations are **distinct evaluation tasks**. The release also retains response-prediction limitations, insufficient strict matching support and corrected null human results.

These results characterize a **nine-population, single-spatial-node computational model**, not directly demonstrated physiological pathways in human brains.

## Repository layout

```text
release/
  science_advances_20261024/            # Immutable original aggregate research release
    analysis/                            # Native reanalysis code, unit tests and methods
    data/                                # Hash-pinned source result files (no raw EEG)
    plot_data/                           # Original figure display tables
    statistics/                          # S01-S09 and all saved result exports
    exports/                             # Original 5 main + 7 supplementary figures
    scripts/                             # Source checks and R figure generation
    provenance/                          # Analysis plan, receipts and source checks
  nature_neuroscience_matters_arising_20261024/
    README.md                            # Current 2-figure publication-facing crosswalk
    FIGURE_CROSSWALK.md                  # Main and Supplementary Fig. S1-S14 mapping
    requirements.txt                     # Python figure-rendering dependencies
    scripts/                             # Replot from frozen tables (no new analyses)
study_pipeline/                           # Full original staged acquisition / model workflow
PUBLIC_RELEASE_CHECKLIST.md              # Final checks before switching visibility
CITATION.cff                              # Provisional software citation metadata
```

The original `science_advances_20261024` directory name identifies a **dated working release**, not publication or acceptance by that journal. It is retained unchanged to preserve existing pinned manifests, source hashes and figure provenance.

## Reproduction

The completed original figure release can be regenerated on a compatible system:

```bash
python release/science_advances_20261024/scripts/prepare_plot_data.py
Rscript --vanilla release/science_advances_20261024/scripts/draw_figures.R "$(pwd)/release/science_advances_20261024"
python release/science_advances_20261024/scripts/audit_exports.py
python release/science_advances_20261024/scripts/export_statistics.py
```

The new two-figure presentation uses existing aggregate data and does not rerun the neural simulator, refit estimators or conduct new tests. Follow the [specific renderer instructions](release/nature_neuroscience_matters_arising_20261024/README.md#rebuild-the-two-figures) for that version. Full native replication additionally requires third-party model assets and specific computing environments, as described in the source release.

## Data access, licensing and transparency

The original model resources are archived at [Figshare v2](https://doi.org/10.6084/m9.figshare.30906557.v2), with underlying sources identified in the dated research release. Raw human recordings, proprietary checkpoints, local authentication/deployment state and submission correspondence are **not** included. Some result tables contain pseudonymized source participant labels; inspect consent/data-sharing terms before making this repository public.

**Making the repository public does not automatically grant a reuse license.** The authors have not yet supplied a definitive software/data licensing decision. Consult [PUBLIC_RELEASE_CHECKLIST.md](PUBLIC_RELEASE_CHECKLIST.md) before changing visibility. No GitHub release, DOI or peer-reviewed publication status is claimed here.
