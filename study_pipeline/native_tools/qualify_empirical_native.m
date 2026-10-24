function qualify_empirical_native(signal_csv, Fs, release_dir, output_json, reference_json, layer_names)
% Native empirical reference correspondence. This is not clinical validation.
    if nargin < 6, layer_names = {}; end
    reference = jsondecode(fileread(reference_json));
    assert(strcmp(reference.reference_kind, 'empirical_eeg') && ~reference.synthetic ...
        && reference.continuity_verified, 'Empirical reference provenance is required');
    assert(strcmp(study_hash(signal_csv), reference.signal_sha256), 'Export hash changed');
    assert(strcmp(stream_hash(reference.source_recording_path), reference.source_recording_sha256), ...
        'Source recording hash changed');
    assert(reference.sfreq_hz == Fs, 'Reference sampling rate changed');
    assert(~isfile(output_json), 'Use a new qualification receipt filename');
    addpath(release_dir);
    environment = struct('matlab_version', version, 'toolboxes', ver, ...
        'deep_learning_toolbox_installed', ~isempty(ver('nnet')), ...
        'signal_processing_toolbox_installed', ~isempty(ver('signal')), ...
        'statistics_toolbox_installed', ~isempty(ver('stats')), ...
        'deep_learning_license_test', logical(license('test', 'Neural_Network_Toolbox')), ...
        'signal_processing_license_test', logical(license('test', 'Signal_Toolbox')), ...
        'statistics_license_test', logical(license('test', 'Statistics_Toolbox')), ...
        'fir1_available', exist('fir1', 'file') ~= 0, ...
        'filtfilt_available', exist('filtfilt', 'file') ~= 0, ...
        'resample_available', exist('resample', 'file') ~= 0, ...
        'zscore_available', exist('zscore', 'file') ~= 0, ...
        'java_available', usejava('jvm'), 'cuda_visible_devices', getenv('CUDA_VISIBLE_DEVICES'));
    write_json([output_json '.environment.json'], environment);
    assert(environment.deep_learning_toolbox_installed && environment.signal_processing_toolbox_installed ...
        && environment.fir1_available && environment.filtfilt_available ...
        && environment.resample_available && environment.zscore_available && environment.java_available, ...
        'Native dependencies are missing; see environment receipt');
    loaded = load(fullfile(release_dir, 'trained_consciousness_detector.mat'), 'trainedNet');
    assert(isfield(loaded, 'trainedNet') && isa(loaded.trainedNet, 'SeriesNetwork'), ...
        'Pinned checkpoint must load as its native SeriesNetwork');
    net = loaded.trainedNet;
    bounded_index = find(arrayfun(@(v) isa(v, 'BoundedOutputLayer'), net.Layers));
    assert(numel(bounded_index) == 1 && bounded_index > 1, 'Unique pre-bound predecessor missing');
    predecessor = net.Layers(bounded_index - 1).Name;
    if isempty(layer_names)
        layer_names = {};
        for index = 1:numel(net.Layers)
            native_class = class(net.Layers(index));
            if contains(native_class, 'Convolution') || contains(native_class, 'FullyConnected')
                layer_names{end + 1, 1} = net.Layers(index).Name; %#ok<AGROW>
            end
        end
        selection_rule = 'Actual native SeriesNetwork layers whose classes contain Convolution or FullyConnected, in architecture order, plus the pre-bound predecessor';
    else
        selection_rule = 'Explicit supplied actual layer names, in supplied order, plus the pre-bound predecessor';
    end
    layer_names = unique([layer_names(:); {predecessor}], 'stable');
    assert(all(ismember(layer_names, {net.Layers.Name})), 'Requested layer name unavailable');
    safe_names = cellfun(@matlab.lang.makeValidName, layer_names, 'UniformOutput', false);
    assert(numel(unique(safe_names)) == numel(safe_names), 'Activation field names would collide');
    inventory = struct('index', {}, 'name', {}, 'class', {});
    for index = 1:numel(net.Layers)
        inventory(index) = struct('index', index, 'name', net.Layers(index).Name, 'class', class(net.Layers(index)));
    end
    write_json([output_json '.layers.json'], inventory);
    qualify_detector(signal_csv, Fs, release_dir, output_json, reference.reference_id, layer_names);
    record = jsondecode(fileread(output_json));
    record.reference_kind = 'empirical_eeg';
    record.reference_provenance = reference;
    record.reference_provenance_sha256 = study_hash(reference_json);
    record.native_network_class = class(net);
    record.native_environment = environment;
    record.activation_interpretation = 'Intermediate network activations; no validated awareness or biological-state labels';
    record.activation_layer_selection_rule = selection_rule;
    record.clipping_interpretation = 'Score is remapped GCS; endpoint ties may result from the bounded layer and are not biological equivalence';
    prediction = qualified_predict(readmatrix(signal_csv), Fs, release_dir, layer_names);
    assert(abs(prediction.score - record.score) < 1e-8, 'Audit prediction differs from repeated reference');
    record.preclip_gcs = prediction.preclip_gcs;
    record.score_at_bound = record.gcs <= 3 || record.gcs >= 15;
    record.preclip_correspondence_verified = strcmp(prediction.extraction_status, 'forward_correspondence_verified');
    activation_dimensions = struct('name', {}, 'class', {}, 'architecture_index', {}, ...
        'is_hidden_layer', {}, 'shape', {}, 'elements', {}, 'status', {});
    hidden_vector_available = false;
    for index = 1:numel(layer_names)
        name = layer_names{index};
        architecture_index = find(strcmp({net.Layers.Name}, name));
        assert(numel(architecture_index) == 1, 'Native layer names must identify unique architecture positions');
        is_hidden = architecture_index > 1 && architecture_index < bounded_index - 1;
        field = safe_names{index};
        if isfield(prediction.activations, field)
            values = prediction.activations.(field);
            shape = size(values);
            elements = numel(values);
            status = 'finite_native_activation_available';
            if ~all(isfinite(values(:)))
                status = 'nonfinite_native_activation';
            elseif all(values(:) == 0)
                status = 'finite_zero_activation_available';
            end
            hidden_vector_available = hidden_vector_available || ...
                (is_hidden && elements > 1 && all(isfinite(values(:))));
        else
            shape = [];
            elements = 0;
            status = 'native_extraction_unavailable';
        end
        activation_dimensions(index) = struct('name', name, ...
            'class', class(net.Layers(architecture_index)), ...
            'architecture_index', architecture_index, 'is_hidden_layer', is_hidden, ...
            'shape', shape, 'elements', elements, 'status', status);
    end
    record.sampled_activation_dimensions = activation_dimensions;
    record.hidden_vector_geometry_capability = hidden_vector_available;
    record.geometry_capability_criterion = 'At least one selected layer between input and pre-bound scalar has more than one finite element; zero/collinear values allowed; no score or result-direction threshold';
    record.activation_layer_inventory_sha256 = study_hash([output_json '.layers.json']);
    record.worker_helper_sha256 = study_hash(which('qualify_empirical_native'));
    % The original helper wrote this fresh receipt in this invocation; enrich it
    % before reporting success. Existing receipt replacement remains forbidden.
    fid = fopen(output_json, 'w');
    assert(fid >= 0, 'Cannot enrich qualification receipt');
    fprintf(fid, '%s\n', jsonencode(record, 'PrettyPrint', true));
    fclose(fid);
    assert(record.activation_repeats_verified && record.preclip_correspondence_verified, ...
        'Primary native output ran, but activation qualification failed; inspect receipt');
    assert(hidden_vector_available, ...
        'Primary native output ran, but no selected hidden vector has >1 finite elements; inspect layer/dimension receipts');
end

function write_json(path, value)
    assert(~isfile(path), 'Preserve existing audit file');
    fid = fopen(path, 'w');
    assert(fid >= 0, 'Cannot create audit file');
    closer = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s\n', jsonencode(value, 'PrettyPrint', true));
end

function value = stream_hash(path)
% Keep raw-recording hashing memory bounded for long EEG acquisitions.
    fid = fopen(path, 'rb');
    assert(fid >= 0, 'Cannot open empirical source recording');
    closer = onCleanup(@() fclose(fid)); %#ok<NASGU>
    engine = java.security.MessageDigest.getInstance('SHA-256');
    while true
        bytes = fread(fid, 1024 * 1024, '*uint8');
        if isempty(bytes), break; end
        engine.update(typecast(bytes, 'int8'));
    end
    digest = typecast(engine.digest(), 'uint8');
    value = lower(reshape(dec2hex(digest, 2).', 1, []));
end
