from studio.uploads import move_uploaded_files_to_cache


def test_duplicate_upload_does_not_truncate_a_file_being_played(tmp_path):
    cached = tmp_path / 'cached.wav'
    upload = tmp_path / 'upload.wav'
    data = b'speech data' * 10000
    cached.write_bytes(data)
    upload.write_bytes(data)
    modified = cached.stat().st_mtime_ns
    with cached.open('rb') as playing:
        prefix = playing.read(100)
        move_uploaded_files_to_cache([str(upload)], [str(cached)])
        assert prefix + playing.read() == data
    assert cached.stat().st_mtime_ns == modified
    assert not upload.exists()


def test_first_upload_is_moved_into_the_cache(tmp_path):
    upload, cached = tmp_path / 'upload.wav', tmp_path / 'cached.wav'
    upload.write_bytes(b'speech')
    move_uploaded_files_to_cache([str(upload)], [str(cached)])
    assert cached.read_bytes() == b'speech'
    assert not upload.exists()
