function adapted_geometry(output_dir, release_dir)
% Export genuine post-human-finetuning hidden activations on original EEG.
    assert(~isempty(getenv('SLURM_JOB_ID')), 'Scheduled native worker required');
    cpu_count = str2double(getenv('SLURM_CPUS_PER_TASK'));
    assert(isfinite(cpu_count) && cpu_count >= 1, 'Missing allocated CPUs');
    maxNumCompThreads(min(4, floor(cpu_count))); %#ok<MAXNCT>
    addpath(release_dir);
    plan_path = fullfile(output_dir, 'export_plan.json');
    plan = jsondecode(fileread(plan_path));
    assert(strcmp(plan.status, 'verified_adapted_native_export_plan'), 'Unverified export plan');
    assert(strcmp(study_hash(plan.adapted_checkpoint.path), plan.adapted_checkpoint.sha256), 'Actual adapted checkpoint changed');
    assert(strcmp(study_hash(fullfile(output_dir, 'adapted_jobs.csv')), plan.jobs_sha256), 'Actual adapted jobs changed');
    assert(strcmp(study_hash(which('adapted_geometry')), plan.helper_sha256), 'Native exporter changed');
    loaded = load(plan.adapted_checkpoint.path, 'adaptedNet', 'input_size', 'classes');
    net = loaded.adaptedNet;
    assert(isa(net, 'SeriesNetwork') && isa(net.Layers(end), 'nnet.cnn.layer.ClassificationOutputLayer'), 'Saved human-adapted native network unsupported');
    shape = double(loaded.input_size);
    assert(prod(shape) == 2500, 'Actual native ten-second input shape differs');
    layers = cellstr(string(plan.layers));
    fields = cellfun(@matlab.lang.makeValidName, layers, 'UniformOutput', false);
    assert(numel(unique(fields)) == numel(fields) && all(ismember(layers, {net.Layers.Name})), 'Adapted hidden layer inventory differs');
    assert(~any(ismember(layers, {'human_task_head','human_task_softmax','human_task_classification'})), 'Hidden geometry must precede human task head');
    jobs = readtable(fullfile(output_dir, 'adapted_jobs.csv'), 'FileType', 'text', 'Delimiter', ',', ...
        'NumHeaderLines', 0, 'ReadVariableNames', true, 'TextType', 'string', 'VariableNamingRule', 'preserve');
    required_columns = {'job_id','episode_id','channel_name','participant_id','source_id','input_path','sfreq_hz','signal_sha256','clinical_split'};
    assert(all(ismember(required_columns, jobs.Properties.VariableNames)), 'Actual adapted job CSV schema differs');
    assert(height(jobs) > 0 && height(jobs) <= 50000 && numel(unique(jobs.job_id)) == height(jobs), 'Invalid actual job registry');
    status = repmat("failed", height(jobs), 1); errors = strings(height(jobs), 1);
    probabilities = cell(height(jobs), 1); bindings = struct('job_id', {}, 'path', {}, 'sha256', {});
    qualified = false;
    for j = 1:height(jobs)
        try
            assert(~isempty(regexp(char(jobs.job_id(j)), '^[A-Za-z0-9_-]+$', 'once')), 'Unsafe native job ID');
            assert(strcmp(study_hash(char(jobs.input_path(j))), char(jobs.signal_sha256(j))), 'Actual source EEG bytes changed');
            x = readmatrix(jobs.input_path(j)); Fs = double(jobs.sfreq_hz(j));
            assert(isvector(x) && all(isfinite(x)) && Fs > 90 && abs(numel(x)/Fs-10) <= .01, 'Genuine continuous ten-second input required');
            x = double(x(:)');
            b = fir1(2*round(Fs), [0.5,45]/(Fs/2), 'bandpass'); x = filtfilt(b, 1, x);
            if Fs ~= 250, x = resample(x, 250, Fs); end
            x = zscore(x); assert(numel(x) == 2500 && all(isfinite(x)), 'Native preprocessing produced invalid input');
            X = reshape(single(x), shape);
            p = double(predict(net, X, 'ExecutionEnvironment', 'cpu'));
            assert(all(isfinite(p), 'all') && all(p >= 0, 'all') && abs(sum(p, 'all')-1) < 1e-5, 'Actual human-task probabilities invalid');
            activation_values = struct();
            for k = 1:numel(layers)
                value = activations(net, X, layers{k}, 'ExecutionEnvironment', 'cpu');
                assert(isnumeric(value) && ~isempty(value) && all(isfinite(value), 'all'), 'Invalid actual hidden activation');
                if ~qualified
                    repeat = activations(net, X, layers{k}, 'ExecutionEnvironment', 'cpu');
                    assert(isequal(size(value), size(repeat)) && max(abs(double(value(:))-double(repeat(:)))) <= 1e-6*max(1,max(abs(double(value(:))))), 'Adapted hidden activation is not repeatable');
                end
                activation_values.(fields{k}) = value;
            end
            if ~qualified
                head = double(activations(net, X, 'human_task_softmax', 'ExecutionEnvironment', 'cpu'));
                assert(numel(head) == numel(p) && max(abs(head(:)-p(:))) <= 1e-6, 'Adapted exported head does not match actual full forward pass');
                qualified = true;
            end
            filename = ['activation_' char(jobs.job_id(j)) '.mat'];
            path = fullfile(output_dir, filename);
            assert(~isfile(path), 'Never overwrite actual activation output');
            save(path, 'activation_values', '-v7');
            bindings(end+1) = struct('job_id', char(jobs.job_id(j)), 'path', filename, 'sha256', study_hash(path)); %#ok<AGROW>
            probabilities{j} = jsonencode(p); status(j) = "ok";
        catch error
            errors(j) = string(error.message); probabilities{j} = '';
        end
    end
    assert(qualified, 'No actual adapted input qualified forward correspondence/repeats');
    result = table(jobs.job_id, status, errors, string(probabilities), 'VariableNames', {'job_id','status','error','human_class_probabilities'});
    predictions_path = fullfile(output_dir, 'adapted_predictions.csv'); writetable(result, predictions_path);
    assert(strcmp(study_hash(plan.adapted_checkpoint.path), plan.adapted_checkpoint.sha256), 'Adapted checkpoint was changed');
    receipt = struct('status', 'adapted_native_activation_export_completed', 'forward_correspondence_verified', true, ...
        'activation_repeats_verified', true, 'adapted_checkpoint_sha256', plan.adapted_checkpoint.sha256, ...
        'jobs_sha256', plan.jobs_sha256, 'export_plan_sha256', study_hash(plan_path), 'helper_sha256', study_hash(which('adapted_geometry')), ...
        'predictions_sha256', study_hash(predictions_path), 'activation_outputs', bindings, 'layers', {layers}, ...
        'successful_jobs', nnz(status == "ok"), 'failed_jobs', nnz(status ~= "ok"), 'matlab_version', version, ...
        'clinical_training_receipt_sha256', plan.clinical_training_receipt_sha256, 'states_receipt_sha256', plan.states_receipt_sha256, ...
        'native_preprocessing', 'Actual source FIR bandpass/filtfilt/resample250Hz/zscore; same original human-adaptation native input contract', ...
        'semantics', 'Human-task copy probabilities and genuine hidden activations, not released clinical GCS or brain manifold coordinates');
    fid = fopen(fullfile(output_dir, 'native_receipt.json'), 'w'); assert(fid >= 0, 'Cannot write actual native receipt');
    closer = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s\n', jsonencode(receipt));
end
