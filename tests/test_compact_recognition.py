import hashlib
import json
import os

from studio import compact_recognition as compact


def cached_model(tmp_path, monkeypatch):
    monkeypatch.setattr(compact, 'MODELS', tmp_path)
    monkeypatch.setattr(compact, 'version', lambda package: compact.RUNTIME_VERSION)
    root = tmp_path/'experimental-asr-int8-cpu'
    root.mkdir()
    weights = b'known checkpoint'
    digest = hashlib.sha256(weights).hexdigest()
    monkeypatch.setattr(compact, 'MODEL_SHA256', digest)
    monkeypatch.setattr(compact, 'MODEL_BYTES', len(weights))
    (root/'model.bin').write_bytes(weights)
    assets = {'config.json': b'{"pinned":true}', 'tokenizer.json': b'pinned tokenizer'}
    monkeypatch.setattr(compact, 'ASSET_SHA256', {name: hashlib.sha256(data).hexdigest() for name, data in assets.items()})
    for name, data in assets.items():
        (root/name).write_bytes(data)
    manifest = {'source_sha256': compact.SOURCE_SHA256, 'model_sha256': digest,
                'model_bytes': len(weights), 'ctranslate2_version': compact.RUNTIME_VERSION,
                'source_header': [6, 'WhisperSpec', 3], 'keep_convolutions_fp32': True}
    (root/'conversion.json').write_text(json.dumps(manifest), encoding='utf-8')
    return root


def test_known_cache_is_hashed_once_until_a_file_changes(tmp_path, monkeypatch):
    root = cached_model(tmp_path, monkeypatch)
    original = compact._sha256
    calls = []
    def counted(path):
        calls.append(path.name)
        return original(path)
    monkeypatch.setattr(compact, '_sha256', counted)
    assert compact.verified_malayalam_cache() == root
    assert compact.verified_malayalam_cache() == root
    assert sorted(calls) == ['config.json', 'model.bin', 'tokenizer.json']
    file = root/'model.bin'
    before = file.stat()
    file.write_bytes(b'wrong checkpoint')
    os.utime(file, ns=(before.st_atime_ns, before.st_mtime_ns+1000000))
    assert compact.verified_malayalam_cache() is None


def test_modified_tokenizer_is_rejected_even_with_matching_weight_manifest(tmp_path, monkeypatch):
    root = cached_model(tmp_path, monkeypatch)
    (root/'tokenizer.json').write_bytes(b'different tokenizer')
    assert compact.verified_malayalam_cache() is None


def test_rejected_convolution_policy_and_wrong_runtime_are_not_selected(tmp_path, monkeypatch):
    root = cached_model(tmp_path, monkeypatch)
    manifest = json.loads((root/'conversion.json').read_text())
    manifest['keep_convolutions_fp32'] = False
    (root/'conversion.json').write_text(json.dumps(manifest))
    assert compact.verified_malayalam_cache() is None
    monkeypatch.setattr(compact, 'version', lambda package: 'different-runtime')
    assert compact.verified_malayalam_cache() is None


def test_missing_cache_is_optional_and_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(compact, 'MODELS', tmp_path)
    assert compact.verified_malayalam_cache() is None
    assert not list(tmp_path.iterdir())
    def missing_runtime(package):
        raise compact.PackageNotFoundError(package)
    monkeypatch.setattr(compact, 'version', missing_runtime)
    assert compact.verified_malayalam_cache() is None
