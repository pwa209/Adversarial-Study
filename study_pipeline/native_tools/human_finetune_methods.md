# Native human-data fine-tuning of the clinical-model copy

The released Figshare v2 SeriesNetwork remains the frozen GCS discovery
instrument. This implementation creates a separate human-task classifier. It
does not turn a detection response or thought report into clinical GCS or an
awareness diagnosis, and a fitted frozen-feature readout cannot qualify it.

`python -m native_tools.human_finetune export --pointer P --scores S --qualification Q --output-dir E --assay detection`
verifies genuine empirical EEG and native frozen scores, preserves original
labels and source hashes, and exports every qualified fixed-channel example.
The named assay is `ds001785_tactile_detection_v1`, predicting the original
subsequent detected/not-detected response from genuine preceding ten-second EEG.
The alternate `report_category` assay preserves original nominal SART reports.
Missing original labels retain exclusions. Source mismatch, unsupported classes,
exceeded budgets and overlapping participants fail explicitly. No selection uses
clinical scores or the desired training/evaluation outcome.

People receive a stable label-blind SHA256 split, approximately 70/15/15 percent.
With eighteen eligible people the allocation is 12/3/3. Every session, episode
and channel stays with that person. A declared existing split may be supplied.
The exporter requires at least six eligible people; it preserves unsupported
held-out classes rather than resplitting after inspecting test outcomes.
Previously analysed public data remain development evaluation and unknown
pretrained exposure remains unknown.

`python -m native_tools.human_finetune train --export-dir E --release-dir R --output-dir T`
runs only within Slurm. Native MATLAB loads the pinned SeriesNetwork, inspects
its actual image-input shape (2500 samples) and scalar clinical head, retains
the pretrained prefix, and replaces the scalar FC/BoundedOutput/regression tail
with a nominal classification head. Original convolutional/normalization
parameter learning-rate factors are nonzero (.1 by default); the new head uses
factor1. Native FIR 0.5–45Hz filtering, zero-phase filtering, resampling to250Hz
and row standardization match the released input contract. Actual signal bytes
are rehashed when loaded. Adam uses learning rate1e-4, at most8epochs by default,
minibatch32 and CPU execution with at most4allocated threads. Validation loss
selects the returned checkpoint, with patience3. Test signals are loaded after
optimization returns. Default Python timeout is4hours; bounded CLI maxima are
30epochs,128batch and12hours.

The checkpoint retains native training information. Receipts record actual
optimizer steps, completed epochs, layer names, changed pretrained parameter
hashes, original/adapted checkpoint hashes, people lists/hashes, fixed split and
source-input hashes, exact assay and actual held-out probabilities. At least one
returned backbone parameter must have changed to qualify actual fine-tuning;
zero change retains its checkpoint but does not emit a completed qualification.
Fixed selected-channel probabilities are averaged within an actual episode;
channels are not independent people. Accuracy/log-loss are descriptive
development evaluation, and no predictive performance threshold grants a human
mechanistic interpretation.

The public `verify_adaptation(T)` audit executes no MATLAB. It verifies actual
training/changed-weight evidence, source/export/EEG hashes, participant splits,
directory-bound adapted checkpoint, same checkpoint used for evaluation and
the exact held-out episode identities, original labels and probability vectors.
Cross-system readiness requires corresponding completed adaptations for the
biological account and both Centaur/base models. Human mechanistic correspondence
further requires the same genuinely measured assay and paired held-out people.
Tactile detection, lottery Left/Right choice and a temporal-dynamics proxy remain
different assays; numeric recoding cannot create the missing bridge.

The source-native SeriesNetwork approach follows the primary MathWorks
[trainNetwork API](https://www.mathworks.com/help/deeplearning/ref/trainnetwork.html),
[transfer-learning procedure](https://www.mathworks.com/help/deeplearning/ug/retrain-neural-network-to-classify-new-images.html)
and [classification-layer contract](https://www.mathworks.com/help/deeplearning/ref/classificationlayer.html).
The existing native API is retained for the inspected released architecture;
actual MATLAB/Deep Learning/Signal Processing compatibility must qualify in a
scheduled runtime. Python unit fixtures test artifact and split integrity and
do not establish actual native optimization or any human result.
