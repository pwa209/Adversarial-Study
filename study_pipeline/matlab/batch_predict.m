function batch_predict(jobs_csv, output_csv, release_dir, qualification_json)
% Native inference on exported individual channels; empirical provenance receipt.
% Exceptions are retained per job. Python import rejects any failed batch job.
    jobs_csv = char(java.io.File(jobs_csv).getCanonicalPath());
    output_csv = char(java.io.File(output_csv).getCanonicalPath());
    release_dir = char(java.io.File(release_dir).getCanonicalPath());
    qualification_json = char(java.io.File(qualification_json).getCanonicalPath());
    addpath(release_dir);
    q = jsondecode(fileread(qualification_json));
    assert(strcmp(q.status, 'native_forward_pass_reproducible'), 'Runtime is not qualified');
    checkpoint_hash = study_hash(fullfile(release_dir, 'trained_consciousness_detector.mat'));
    wrapper_hash = study_hash(fullfile(release_dir, 'predict_consciousness_level.m'));
    assert(strcmp(q.checkpoint_sha256, checkpoint_hash), 'Qualified checkpoint changed');
    assert(strcmp(q.wrapper_sha256, wrapper_hash), 'Qualified wrapper changed');
    assert(strcmp(q.mapping_sha256, study_hash(fullfile(release_dir, 'remap_gcs_scale.m'))), ...
        'Qualified mapping changed');
    assert(strcmp(q.bound_layer_sha256, study_hash(fullfile(release_dir, 'BoundedOutputLayer.m'))), ...
        'Qualified bounded layer changed');
    assert(strcmp(q.helper_sha256, study_hash(which('qualified_predict'))), 'Extraction helper changed');
    assert(~isfile(output_csv), 'Preserve existing batch output; use a new filename');
    % CSV inference misidentified real DREAM/NFTsim headers on Narval. The
    % exported contract is one comma-delimited header followed by data rows.
    jobs = readtable(jobs_csv, 'FileType', 'text', 'Delimiter', ',', ...
        'NumHeaderLines', 0, 'ReadVariableNames', true, ...
        'VariableNamingRule', 'preserve', 'TextType', 'string');
    required_columns = {'job_id','episode_id','channel_name','participant_id','source_id','input_path','sfreq_hz','signal_sha256'};
    assert(all(ismember(required_columns, jobs.Properties.VariableNames)), 'Actual native job CSV schema differs');
    count = height(jobs);
    assert(count > 0 && numel(unique(jobs.job_id)) == count, 'Empty/duplicate native jobs');
    gcs = nan(count, 1); score = nan(count, 1); preclip_gcs = nan(count, 1);
    status = repmat("failed", count, 1); errors = strings(count, 1);
    signal_sha256 = strings(count, 1); extraction_status = strings(count, 1);
    activation_receipts = struct('job_id', {}, 'path', {}, 'sha256', {});
    layer_names = q.qualified_layers;
    if ischar(layer_names), layer_names = {layer_names}; end
    if isstring(layer_names), layer_names = cellstr(layer_names); end
    for index = 1:count
        try
            input_path = char(jobs.input_path(index));
            signal_sha256(index) = string(study_hash(input_path));
            assert(signal_sha256(index) == jobs.signal_sha256(index), 'Input hash changed');
            x = readmatrix(input_path);
            prediction = qualified_predict(x, jobs.sfreq_hz(index), release_dir, layer_names);
            score(index) = prediction.score; gcs(index) = prediction.gcs;
            if ~isempty(prediction.preclip_gcs), preclip_gcs(index) = prediction.preclip_gcs; end
            extraction_status(index) = string(prediction.extraction_status);
            if ~isempty(fieldnames(prediction.activations))
                activation_values = prediction.activations; %#ok<NASGU>
                activation_path = [output_csv '.' char(jobs.job_id(index)) '.activations.mat'];
                save(activation_path, 'activation_values', '-v7');
                activation_receipts(end+1) = struct('job_id', char(jobs.job_id(index)), ...
                    'path', activation_path, 'sha256', study_hash(activation_path)); %#ok<AGROW>
            end
            status(index) = "ok";
        catch prediction_error
            errors(index) = string(prediction_error.message);
        end
    end
    output = table(jobs.job_id, gcs, score, preclip_gcs, status, errors, signal_sha256, extraction_status, ...
        'VariableNames', {'job_id','gcs','score','preclip_gcs','status','error','signal_sha256','extraction_status'});
    writetable(output, output_csv);
    % MATLAB writes missing pre-bound activations as NaN. Retain these as empty
    % CSV cells so Python never mistakes missing extraction for a real number.
    text = fileread(output_csv);
    text = regexprep(text, ',NaN,', ',,');
    fid = fopen(output_csv, 'w'); assert(fid >= 0, 'Cannot write score CSV');
    fprintf(fid, '%s', text); fclose(fid);
    receipt = struct('status', 'native_batch_completed', 'jobs', count, ...
        'successful_jobs', sum(status == "ok"), 'checkpoint_sha256', checkpoint_hash, ...
        'wrapper_sha256', wrapper_hash, 'scores_sha256', study_hash(output_csv), ...
        'jobs_sha256', study_hash(jobs_csv), 'qualification_sha256', study_hash(qualification_json), ...
        'matlab_version', version, 'score_semantics', 'remapped_GCS_not_awareness_probability');
    receipt.activation_outputs = activation_receipts;
    fid = fopen([output_csv '.receipt.json'], 'w'); assert(fid >= 0, 'Cannot write native receipt');
    closer = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s\n', jsonencode(receipt, 'PrettyPrint', true));
end
