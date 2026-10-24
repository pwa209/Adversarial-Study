OBSERVATION AND MECHANISTIC INFERENCE

This release contains five main figures, seven supplementary figures, all
saved aggregate statistics underlying their scientific storyline, and the
analysis and figure code. The manuscript is deliberately excluded from this
public release and is kept locally by the project owner.

The study distinguishes known-parameter recovery, finite-input response
prediction, internal pathway decomposition and human report prediction.
Population features have lower descriptive parameter-recovery error in the
current bank, while model capacity differs. Pulse-prediction and human
comparisons include null and negative results. No independent human mechanism,
matched-capacity validation or randomized intervention is claimed.

Contents
exports/: individual PDF, SVG and 400 dpi PNG files for all 12 figures, plus
          five-page main and seven-page supplementary PDF collections.
data/: 37 hash-pinned source result files, including all original aggregate CSVs.
plot_data/: the 11 derived tables used by the R renderer.
statistics/: complete CSV exports, including nulls and fixed sensitivities.
scripts/: R plotting, source verification, export audit and statistics export.
analysis/: exact completed cross-mechanistic analysis modules, tests and methods.
provenance/: source hashes, analysis plans, export checks and operation receipts.
Figure_captions.txt: complete legends, denominators and interpretation limits.

Reproduce the figures
Run Run_Figures.ps1 on the original workstation. On another machine, update
its R and Python executable paths, or run these stages directly:
  python scripts/prepare_plot_data.py
  Rscript --vanilla scripts/draw_figures.R <absolute release directory>
  python scripts/audit_exports.py
  python scripts/export_statistics.py

The R backend is base R 4.6.1 with Cairo PDF/SVG; no extension R packages are
required. Python requires Pillow for export audits and standard-library-only
code for statistical export. Poppler must be on PATH. PNGs are rasterized from
the canonical R PDFs at 400 dpi. All text is black, canvases are white, and
comparable inner plotting regions are aligned. Palette: #f5f2ec, #d8d4cd,
#756458 and #b54a3c for non-text elements.

The complete original study pipeline is provided separately in the repository's
study_pipeline directory. Native simulations and raw human preprocessing need
their identified source assets and runtimes. This aggregate release reproduces
the figures without downloading raw EEG, model weights or simulation arrays.
No model fitting, new hypothesis testing or bootstrap runs are performed by
the plotting and statistical-export scripts.

Study constraints
The behavioural target remains 42, split 26/8/8. Both actual human-data
fine-tunings and the separate clinical adaptation qualify before cross-system
comparison. These behavioural counts are not EEG sample sizes: the fixed sensor
analysis has 183 eligible episodes from 31 people and six held test people;
the earlier forecasting bridge has five held people and 32 common episodes.

The simulation model has nine effective populations at one spatial node.
The pristine release and the explicit scaled-dendrite numerical repair are
separate implementations. Parameter inversion uses 48 actually manipulated
configurations, split 28/10/10, and is supported only in the repaired bank.
Configuration-bootstrap ranges are conditional finite-bank stability summaries,
not human confidence intervals. The spatial human bootstrap used 10,000 draws;
the earlier forecasting bootstrap used 20,000; simulation summaries used 2,000.
No number of bootstrap draws increases the independent sample size.

Source records
Toker et al., Nature Neuroscience (2026):
  https://doi.org/10.1038/s41593-026-02220-4
Released model assets:
  https://doi.org/10.6084/m9.figshare.30906557.v2
Siclari et al., Nature Neuroscience (2017):
  https://doi.org/10.1038/nn.4545
Tononi Serial Awakenings, DREAM version 2:
  https://bridges.monash.edu/articles/dataset/Tononi_Serial_Awakenings/23306054/2
NFTsim:
  https://doi.org/10.1371/journal.pcbi.1006387

Original asset licenses and consent/access conditions remain attached to their
sources. Raw source assets and third-party publisher artwork are not included.
