# Nature Neuroscience *Matters Arising*: computational figure release

**Research presentation:** *Neural signals conceal opposing pathways in a model of consciousness*  
**Authors:** Peng Wang and Steven Laureys  
**Purpose:** reproduce the **two-figure presentation** and trace all complementary analyses back to the dated original release.

This directory is a presentation-layer addition. It neither replaces the archived five-main/seven-supplement release nor claims a new model fit, simulator run, hypothesis test or human validation. The manuscript and cover letter remain outside GitHub during consideration. Formal article acceptance, a public DOI and publisher-controlled figure numbering are not presumed.

## Rebuild the two figures

Run from the repository root (Python 3.10+):

```bash
python -m pip install -r release/nature_neuroscience_matters_arising_20261024/requirements.txt
python release/nature_neuroscience_matters_arising_20261024/scripts/render_submission_figures.py --repo .
```

The renderer saves `Figure_1_Recovery.{svg,pdf,png}` and `Figure_2_Integrated.{svg,pdf,png}` to this directory's `figures/` folder. Its only inputs are the frozen `release/science_advances_20261024/` aggregate data:

- `plot_data/recovery_summary.csv`
- `data/mechanism_compensation/parameter_predictions.csv`
- `data/pathway_balance/population_pathway_balance.csv`

The new output is **replotted from saved values**. No new data fitting, bootstrapping or simulator execution is performed. Figures use a white canvas, the established red/taupe palette and vector text where supported. Panel 2a contains only **illustrative waveforms** whose algebraic summation is computed from the shown curves; panels 2b–c show saved source-derived measurements.

Before publication, compare figure exports against the submitted Word/PDF artwork, assess legibility at the final placed size and confirm image-use requirements with the publisher.

## Main results

**Figure 1** compares four fitted parameter-recovery pipelines over 10 held-out manipulated configurations. Full-population and cortical models have different feature counts/effective degrees of freedom, so the figure establishes a difference between **fitted observation pipelines**, not matched-capacity biological information.

**Figure 2** combines the source-checked voltage identity with model-specific pulse-minus-control cancellation. Four sources into destination population 1 are illustrated; the same source-accounting identity is verified at modeled destinations. The aggregate analysis covers 48 physical pulse/control pairs × nine populations = 432 population-by-configuration observations. Configuration observations are not independent human participants.

The cancellation definition is:

```text
N = integral_t abs(sum_j d_j(t)) dt
A = integral_t sum_j abs(d_j(t)) dt
C = 1 - N / A               when A > 0
```

These are already coupled and filtered dendritic voltage contributions. Do not multiply by coupling weights a second time. A zero-denominator response has undefined relative cancellation, not `C=0`.

## Supporting analyses and file map

See [FIGURE_CROSSWALK.md](FIGURE_CROSSWALK.md). The original release's `statistics/` directory contains the nine primary table CSVs and source-table originals, including negative/uncorrected/corrected results and all matching failures. The new paper's 14 supplementary display plates are mapped to the original figure panels in the crosswalk; the underlying numerical results remain in the original immutable directory.

## Scope

The neural model is a nine-effective-population system at **one spatial node**; parameter inversion uses the explicitly labeled repaired implementation. The available human EEG analyses were secondary and did not provide corrected positive human-mechanism validation. The original independent biological evidence of Toker et al. is a separate scientific contribution; this release does not claim to refute it.

Original source paper: [Toker et al., *Nature Neuroscience* (2026)](https://doi.org/10.1038/s41593-026-02220-4). Model source: [Figshare v2](https://doi.org/10.6084/m9.figshare.30906557.v2).
