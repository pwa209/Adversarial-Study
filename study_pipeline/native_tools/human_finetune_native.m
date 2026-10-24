function human_finetune_native(export_dir, release_dir, output_dir, epochs, minibatch, learning_rate, backbone_factor)
% Actual MATLAB fine-tuning of a COPY of the released cortical CNN.
% The human classifier has a new nominal-task head. No GCS/awareness relabeling.
    assert(~isempty(getenv('SLURM_JOB_ID')), 'Native training requires a scheduled job');
    cpu_count = str2double(getenv('SLURM_CPUS_PER_TASK'));
    assert(isfinite(cpu_count) && cpu_count >= 1, 'Missing allocated CPU budget');
    maxNumCompThreads(min(4, floor(cpu_count))); %#ok<MAXNCT>
    assert(epochs >= 1 && epochs <= 30 && minibatch >= 1 && minibatch <= 128, 'Training budget invalid');
    assert(learning_rate > 0 && learning_rate <= .01 && backbone_factor > 0 && backbone_factor <= 1, 'Training factors invalid');
    assert(isfolder(output_dir), 'Python worker must create a new output directory');
    assert(~isfile(fullfile(output_dir, 'adapted_human_task_net.mat')), 'Never overwrite an adapted checkpoint');
    export_path = fullfile(export_dir, 'export_receipt.json');
    e = jsondecode(fileread(export_path));
    assert(strcmp(e.status, 'human_finetuning_inputs_verified_not_trained') && e.human_data, 'Unverified human input');
    assert(strcmp(study_hash(fullfile(export_dir, 'dataset_split.json')), e.data_split_sha256), 'Split changed');
    assert(strcmp(study_hash(fullfile(export_dir, 'human_examples.csv')), e.examples_sha256), 'Human examples changed');
    checkpoint = fullfile(release_dir, 'trained_consciousness_detector.mat');
    original_sha = study_hash(checkpoint);
    assert(strcmp(original_sha, 'd8b5f89e9c958bddf0804e445a21ca8160a04f23494c1a94bd3e0e9bd134dd17'), 'Released checkpoint changed');
    addpath(release_dir);
    loaded = load(checkpoint, 'trainedNet');
    original = loaded.trainedNet;
    assert(isa(original, 'SeriesNetwork'), 'Only the inspected released SeriesNetwork is supported');
    layers = original.Layers;
    assert(isa(layers(1), 'nnet.cnn.layer.ImageInputLayer'), 'Expected actual native image input');
    input_size = double(layers(1).InputSize);
    assert(numel(input_size) == 3 && prod(input_size) == 2500, 'Unexpected released ten-second input shape');
    bound = find(arrayfun(@(layer) isa(layer, 'BoundedOutputLayer'), layers));
    assert(numel(bound) == 1 && bound > 2 && isa(layers(bound-1), 'nnet.cnn.layer.FullyConnectedLayer') && layers(bound-1).OutputSize == 1, 'Original scalar clinical head is not identifiable');
    assert(isa(layers(end), 'nnet.cnn.layer.RegressionOutputLayer'), 'Expected original regression objective');
    prefix = layers(1:bound-2);
    original_prefix = prefix;
    factor_properties = {'WeightLearnRateFactor', 'BiasLearnRateFactor', 'ScaleLearnRateFactor', 'OffsetLearnRateFactor'};
    learnable_properties = {'Weights', 'Bias', 'Scale', 'Offset'};
    trained_layer_names = {};
    for k = 1:numel(prefix)
        has_parameters = false;
        for p = 1:numel(factor_properties)
            if isprop(prefix(k), factor_properties{p})
                prefix(k).(factor_properties{p}) = backbone_factor;
                has_parameters = true;
            end
        end
        if has_parameters, trained_layer_names{end+1} = prefix(k).Name; end %#ok<AGROW>
    end
    assert(~isempty(trained_layer_names), 'No pretrained backbone parameters can be trained');
    jobs = readtable(fullfile(export_dir, 'human_examples.csv'), 'FileType', 'text', 'Delimiter', ',', ...
        'NumHeaderLines', 0, 'ReadVariableNames', true, 'TextType', 'string', 'VariableNamingRule', 'preserve');
    required_columns = {'example_id','episode_id','participant_id','source_id','recording_id','channel_name','input_path','sfreq_hz','signal_sha256','label','split'};
    assert(all(ismember(required_columns, jobs.Properties.VariableNames)), 'Actual human fine-tuning CSV schema differs');
    assert(height(jobs) >= 3 && height(jobs) <= 50000, 'Actual example budget invalid');
    train_idx = find(jobs.split == "train");
    validation_idx = find(jobs.split == "validation");
    test_idx = find(jobs.split == "test");
    assert(~isempty(train_idx) && ~isempty(validation_idx) && ~isempty(test_idx), 'Empty actual partition');
    train_people = unique(jobs.participant_id(train_idx));
    validation_people = unique(jobs.participant_id(validation_idx));
    test_people = unique(jobs.participant_id(test_idx));
    assert(isempty(intersect(train_people, validation_people)) && isempty(intersect(train_people, test_people)) && isempty(intersect(validation_people, test_people)), 'Participant leakage');
    classes = sort(unique(jobs.label(train_idx)));
    assert(numel(classes) >= 2 && all(ismember(jobs.label, classes)), 'Unsupported held-out nominal class');
    class_names = cellstr(classes);
    categorical_classes = categorical(class_names, class_names);
    adapted_layers = [prefix; fullyConnectedLayer(numel(classes), 'Name', 'human_task_head', 'WeightLearnRateFactor', 1, 'BiasLearnRateFactor', 1); softmaxLayer('Name', 'human_task_softmax'); classificationLayer('Name', 'human_task_classification', 'Classes', categorical_classes)];
    % Only train/validation signals are loaded before the optimizer returns.
    Xtrain = inputs(jobs, train_idx, input_size);
    Xvalidation = inputs(jobs, validation_idx, input_size);
    Ytrain = categorical(cellstr(jobs.label(train_idx)), class_names);
    Yvalidation = categorical(cellstr(jobs.label(validation_idx)), class_names);
    rng(20261004, 'twister');
    iterations_per_epoch = ceil(numel(train_idx)/minibatch);
    options = trainingOptions('adam', 'MaxEpochs', epochs, 'MiniBatchSize', minibatch, ...
        'InitialLearnRate', learning_rate, 'Shuffle', 'every-epoch', 'ExecutionEnvironment', 'cpu', ...
        'ResetInputNormalization', false, 'ValidationData', {Xvalidation, Yvalidation}, ...
        'ValidationFrequency', max(1, iterations_per_epoch), 'ValidationPatience', 3, ...
        'OutputNetwork', 'best-validation', 'Verbose', false, 'Plots', 'none');
    [adaptedNet, trainingInfo] = trainNetwork(Xtrain, Ytrain, adapted_layers, options);
    optimizer_steps = nnz(isfinite(trainingInfo.TrainingLoss));
    assert(optimizer_steps > 0, 'No actual optimizer iterations recorded');
    changed = struct('layer', {}, 'parameter', {}, 'before_sha256', {}, 'after_sha256', {});
    for k = 1:numel(original_prefix)
        for p = 1:numel(learnable_properties)
            name = learnable_properties{p};
            if isprop(original_prefix(k), name) && ~isempty(original_prefix(k).(name))
                before = numeric_hash(original_prefix(k).(name));
                after = numeric_hash(adaptedNet.Layers(k).(name));
                if ~strcmp(before, after)
                    changed(end+1) = struct('layer', original_prefix(k).Name, 'parameter', name, 'before_sha256', before, 'after_sha256', after); %#ok<AGROW>
                end
            end
        end
    end
    checkpoint_path = fullfile(output_dir, 'adapted_human_task_net.mat');
    save(checkpoint_path, 'adaptedNet', 'trainingInfo', 'classes', 'input_size', '-v7.3');
    adapted_sha = study_hash(checkpoint_path);
    assert(~strcmp(adapted_sha, original_sha), 'Adapted checkpoint is the original checkpoint');
    % A zero-change backbone is retained as an unqualified outcome, never head-only fine-tuning.
    assert(~isempty(changed), 'No returned pretrained backbone parameter changed; checkpoint retained, fine-tuning not qualified');
    assert(strcmp(study_hash(checkpoint), original_sha), 'Original discovery instrument changed');
    write_json(fullfile(output_dir, 'backbone_updates.json'), changed);
    clear Xtrain Xvalidation;
    Xtest = inputs(jobs, test_idx, input_size);
    [~, probabilities] = classify(adaptedNet, Xtest, 'MiniBatchSize', minibatch, 'ExecutionEnvironment', 'cpu');
    probabilities = double(probabilities);
    assert(all(isfinite(probabilities), 'all') && all(probabilities >= 0, 'all'), 'Invalid actual held-out probabilities');
    output_classes = string(adaptedNet.Layers(end).Classes);
    assert(numel(output_classes) == numel(classes) && all(ismember(classes, output_classes)), 'Actual class order differs');
    [~, order] = ismember(classes, output_classes);
    probabilities = probabilities(:, order);
    held = episode_predictions(jobs(test_idx, :), probabilities, classes);
    predictions_path = fullfile(output_dir, 'heldout_predictions.csv');
    writetable(held, predictions_path);
    evaluation = struct('schema_version', 1, 'status', 'completed_actual_heldout_human_evaluation', ...
        'system_id', 'clinical_human_task_copy', 'human_data', true, 'human_assay', e.human_assay, 'assay_id', e.assay_id, ...
        'trained_checkpoint', struct('path', checkpoint_path, 'sha256', adapted_sha), ...
        'data_split_sha256', e.data_split_sha256, 'split', e.split, 'test_observation_ids', {cellstr(held.episode_id)}, ...
        'test_predictions', struct('path', predictions_path, 'sha256', study_hash(predictions_path)), ...
        'test_participant_ids', {cellstr(test_people)}, 'test_examples', numel(test_idx), 'test_episodes', height(held), ...
        'aggregation', 'Arithmetic mean of genuine fixed selected-channel class probabilities within each actual episode; channels are not independent people', ...
        'metrics', struct('episode_accuracy', mean(held.correct), 'episode_log_loss', mean(held.log_loss), 'participant_macro_accuracy', participant_accuracy(held)), ...
        'scope', 'Development held-person nominal task prediction; no awareness diagnosis or prospective replication claim');
    write_json(fullfile(output_dir, 'evaluation_receipt.json'), evaluation);
    training = struct('schema_version', 1, 'status', 'completed_actual_human_finetune', 'system_id', 'clinical_human_task_copy', ...
        'adaptation_kind', 'native_pretrained_backbone_and_new_human_task_classification_head', ...
        'native_training_executed', true, 'human_data', true, 'adaptation_completed', true, 'backbone_training', true, ...
        'optimizer', 'Adam', 'optimizer_steps', optimizer_steps, 'actual_optimizer_steps', optimizer_steps, ...
        'actual_epochs_completed', floor(optimizer_steps/iterations_per_epoch), 'iterations_per_epoch', iterations_per_epoch, ...
        'maximum_epochs', epochs, 'minibatch', minibatch, 'learning_rate', learning_rate, 'backbone_learning_rate_factor', backbone_factor, ...
        'explicit_trained_backbone_layers', {trained_layer_names}, 'changed_backbone_parameter_count', numel(changed), ...
        'backbone_updates', struct('path', fullfile(output_dir, 'backbone_updates.json'), 'sha256', study_hash(fullfile(output_dir, 'backbone_updates.json'))), ...
        'original_checkpoint_sha256', original_sha, 'adapted_checkpoint_sha256', adapted_sha, ...
        'trained_checkpoint', struct('path', checkpoint_path, 'sha256', adapted_sha), ...
        'data_split_sha256', e.data_split_sha256, 'split', e.split, 'train_examples_sha256', e.train_examples_sha256, ...
        'train_participant_ids', {cellstr(train_people)}, 'validation_participant_ids', {cellstr(validation_people)}, 'test_participant_ids', {cellstr(test_people)}, ...
        'train_participant_ids_sha256', text_hash(train_people), 'validation_participant_ids_sha256', text_hash(validation_people), 'test_participant_ids_sha256', text_hash(test_people), ...
        'human_assay', e.human_assay, 'assay_id', e.assay_id, 'pretrained_exposure', e.pretrained_exposure, ...
        'test_observation_ids', {cellstr(held.episode_id)}, 'test_predictions', evaluation.test_predictions, ...
        'selection', 'Best native validation loss checkpoint; test arrays loaded only after optimizer returns', ...
        'input', struct('actual_native_shape', input_size, 'native_preprocessing', 'fir1 order 2*round(Fs), 0.5-45Hz; filtfilt; resample to250Hz; row zscore; exact native 2500sample shape'), ...
        'export_dir', export_dir, 'export_receipt', struct('path', export_path, 'sha256', study_hash(export_path)), ...
        'original_release_dir', release_dir, 'export_receipt_sha256', study_hash(export_path), 'trainer_sha256', study_hash(which('human_finetune_native')), ...
        'slurm_job_id', getenv('SLURM_JOB_ID'), 'execution_environment', 'cpu', ...
        'interpretation', 'Adapted copy predicts the specified original human task response. It is not a GCS or awareness predictor. Fixed original clinical instrument remains discovery instrument.');
    write_json(fullfile(output_dir, 'training_receipt.json'), training);
end

function X = inputs(jobs, indexes, shape)
    X = zeros([shape, numel(indexes)], 'single');
    for k = 1:numel(indexes)
        j = indexes(k);
        assert(strcmp(study_hash(char(jobs.input_path(j))), char(jobs.signal_sha256(j))), 'Actual EEG bytes changed');
        x = readmatrix(jobs.input_path(j));
        assert(isvector(x) && all(isfinite(x)), 'Human input must be one finite actual channel');
        Fs = double(jobs.sfreq_hz(j));
        assert(Fs > 90 && abs(numel(x)/Fs - 10) <= .01, 'Actual native ten-second input required');
        x = double(x(:)');
        b = fir1(2 * round(Fs), [0.5, 45] / (Fs / 2), 'bandpass');
        x = filtfilt(b, 1, x);
        if Fs ~= 250, x = resample(x, 250, Fs); end
        x = zscore(x);
        assert(numel(x) == 2500 && all(isfinite(x)), 'Native standardized input invalid');
        X(:,:,:,k) = reshape(single(x), shape);
    end
end

function held = episode_predictions(jobs, probabilities, classes)
    identities = unique(jobs.episode_id, 'stable');
    names = cellstr("probability_" + string(matlab.lang.makeValidName(cellstr(classes))));
    assert(numel(unique(names)) == numel(names), 'Distinct nominal labels collide in CSV column names');
    held = table('Size', [numel(identities), 8], 'VariableTypes', {'string','string','string','string','string','double','double','double'}, ...
        'VariableNames', {'episode_id','participant_id','source_id','label','prediction','channel_count','correct','log_loss'});
    for c = 1:numel(classes), held.(names{c}) = zeros(numel(identities),1); end
    for i = 1:numel(identities)
        selected = jobs.episode_id == identities(i);
        part = unique(jobs.participant_id(selected)); labels = unique(jobs.label(selected)); sources = unique(jobs.source_id(selected));
        assert(numel(part) == 1 && numel(labels) == 1 && numel(sources) == 1, 'Episode identities conflict');
        average = mean(probabilities(selected, :), 1);
        [~, predicted] = max(average);
        actual = find(classes == labels);
        held(i,1:8) = {identities(i),part, sources, labels, classes(predicted), nnz(selected), double(predicted == actual), -log(max(average(actual), realmin))};
        for c = 1:numel(classes), held.(names{c})(i) = average(c); end
    end
end

function value = participant_accuracy(held)
    people = unique(held.participant_id); values = zeros(numel(people),1);
    for i = 1:numel(people), values(i) = mean(held.correct(held.participant_id == people(i))); end
    value = mean(values);
end

function value = numeric_hash(array)
    array = gather(array);
    engine = java.security.MessageDigest.getInstance('SHA-256');
    engine.update(int8(unicode2native(class(array), 'UTF-8')));
    engine.update(typecast(typecast(array(:), 'uint8'), 'int8'));
    digest = typecast(engine.digest(), 'uint8');
    value = lower(reshape(dec2hex(digest, 2).', 1, []));
end

function value = text_hash(text)
    engine = java.security.MessageDigest.getInstance('SHA-256');
    engine.update(typecast(unicode2native(strjoin(sort(text), newline), 'UTF-8'), 'int8'));
    digest = typecast(engine.digest(), 'uint8');
    value = lower(reshape(dec2hex(digest, 2).', 1, []));
end

function write_json(path, value)
    fid = fopen(path, 'w'); assert(fid >= 0, 'Cannot write native receipt');
    closer = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s\n', jsonencode(value));
end
