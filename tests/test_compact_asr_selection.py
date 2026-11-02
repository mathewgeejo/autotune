import pytest

from studio import models, compact_recognition


def test_opt_in_verified_checkpoint_avoids_original_model_download(tmp_path, monkeypatch):
    monkeypatch.setenv('VOICE_STUDIO_COMPACT_ASR', '1')
    monkeypatch.setattr(compact_recognition, 'verified_malayalam_cache', lambda: tmp_path)
    def unexpected(*args, **kwargs):
        pytest.fail('A verified compact cache must not download the original weights')
    monkeypatch.setattr(models, 'snapshot_download', unexpected)
    assert models.download_asr(language='ml') == str(tmp_path)


@pytest.mark.parametrize('enabled', ['0', '1'])
def test_disabled_or_unavailable_compact_cache_keeps_original_loading(monkeypatch, enabled):
    monkeypatch.setenv('VOICE_STUDIO_COMPACT_ASR', enabled)
    def compact():
        assert enabled == '1', 'Opt-out must not inspect the optional checkpoint'
        return None
    monkeypatch.setattr(compact_recognition, 'verified_malayalam_cache', compact)
    class OriginalLoading(Exception):
        pass
    def original(*args, **kwargs):
        raise OriginalLoading
    monkeypatch.setattr(models, 'snapshot_download', original)
    with pytest.raises(OriginalLoading):
        models.download_asr(language='ml')
