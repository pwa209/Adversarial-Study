function result = qualified_predict(signal, Fs, release_dir, layer_names)
% Native wrapper owns the primary output. Reconstructed inputs are used ONLY
% for activation extraction, and must reproduce that native output first.
    if nargin < 4, layer_names = {}; end
    original_dir = pwd;
    restore = onCleanup(@() cd(original_dir)); %#ok<NASGU>
    cd(release_dir);
    score = predict_consciousness_level(signal, Fs);
    assert(isscalar(score) && isfinite(score) && score >= 0 && score <= 1, ...
        'Native wrapper must return one finite bounded score');
    result = struct('score', double(score), 'gcs', 3 + 12 * double(score), ...
        'preclip_gcs', [], 'extraction_status', 'not_available', ...
        'extraction_error', '', 'activations', struct());
    try
        loaded = load('trained_consciousness_detector.mat', 'trainedNet');
        net = loaded.trainedNet;
        x = double(signal(:)');
        order = 2 * round(Fs);
        b = fir1(order, [0.5, 45] / (Fs / 2), 'bandpass');
        x = filtfilt(b, 1, x);
        if Fs ~= 250, x = resample(x, 250, Fs); end
        x = zscore(x);
        reconstructed = double(predict(net, x));
        assert(isscalar(reconstructed) && abs(reconstructed - result.gcs) <= 1e-5, ...
            'Reconstructed preprocessing differs from native wrapper');
        bounded_index = find(arrayfun(@(v) isa(v, 'BoundedOutputLayer'), net.Layers));
        assert(numel(bounded_index) == 1 && bounded_index > 1, ...
            'Unique bounded output predecessor unavailable');
        predecessor = net.Layers(bounded_index - 1).Name;
        raw = double(activations(net, x, predecessor, 'OutputAs', 'rows'));
        assert(isscalar(raw) && isfinite(raw), 'Pre-bound layer is not one finite scalar');
        assert(abs(min(max(raw, 3), 15) - result.gcs) <= 1e-5, ...
            'Pre-bound activation does not reproduce clipped prediction');
        result.preclip_gcs = raw;
        for index = 1:numel(layer_names)
            name = char(layer_names{index});
            result.activations.(matlab.lang.makeValidName(name)) = ...
                activations(net, x, name, 'OutputAs', 'rows');
        end
        result.extraction_status = 'forward_correspondence_verified';
    catch extraction_error
        result.preclip_gcs = [];
        result.activations = struct();
        result.extraction_error = extraction_error.message;
        % A qualified primary native output remains available if optional
        % extraction fails. Never substitute a made-up pre-bound value.
    end
end
