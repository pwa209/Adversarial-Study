function value = study_hash(path, algorithm)
% Hash bytes exactly as Python receipts do. Java-enabled MATLAB required.
    if nargin < 2, algorithm = 'SHA-256'; end
    fid = fopen(path, 'rb');
    assert(fid >= 0, 'Cannot read hash input: %s', path);
    closer = onCleanup(@() fclose(fid)); %#ok<NASGU>
    bytes = fread(fid, Inf, '*uint8');
    engine = java.security.MessageDigest.getInstance(algorithm);
    engine.update(typecast(bytes, 'int8'));
    digest = typecast(engine.digest(), 'uint8');
    value = lower(reshape(dec2hex(digest, 2).', 1, []));
end
