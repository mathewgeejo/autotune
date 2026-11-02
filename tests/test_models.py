from pathlib import Path

import pytest

from studio import models, paths


def test_native_asr_bundle_resolves_relative_cache_symlinks(tmp_path, monkeypatch):
    """Moving a relative HF snapshot symlink makes a model unreadable."""
    cache = tmp_path/'cache'
    blobs = cache/'blobs'
    snapshot = cache/'snapshots'/'pinned'
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    names = ['model.bin', 'config.json', 'vocabulary.txt', 'README.md', 'tokenizer.json']
    for name in names:
        (blobs/name).write_bytes(('asset '+name).encode())
        try:
            (snapshot/name).symlink_to(Path('../../blobs')/name)
        except OSError:
            pytest.skip('Creating symlinks requires Windows developer mode or equivalent permission')
    monkeypatch.setattr(models, 'snapshot_download', lambda *args, **kwargs: str(snapshot))
    monkeypatch.setattr(models, 'hf_hub_download', lambda *args, **kwargs: str(snapshot/'tokenizer.json'))
    monkeypatch.setattr(paths, 'MODELS', tmp_path/'bundles')
    bundle = Path(models.download_asr(language='ml'))
    for name in names:
        assert (bundle/name).read_bytes() == (blobs/name).read_bytes()
        assert not (bundle/name).is_symlink()
