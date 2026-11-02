"""Verify the pinned, locally derived Malayalam recognizer before optional use."""
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path

from studio.paths import MODELS

SOURCE_SHA256 = 'c9df6bd4180f81b5bf663bc6a99236996116378112b17f3d2a8f0e00310afbcd'
MODEL_SHA256 = 'c4583310147882c7d25f7e02caad65ddb37b7ada649587b5b3b1e2919eb751f7'
MODEL_BYTES = 784897268
RUNTIME_VERSION = '4.6.0'
ASSET_SHA256 = {
    'config.json': 'ec76623a3714736341506ae939822bcb5e4d90dc21c0e6a6525a56af73596734',
    'tokenizer.json': '27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566',
    'vocabulary.txt': '5db7e94b7fc75da2e2b2ccb196eceb52b549c7e8e5e32bf0eabff3ce233de27b',
    'README.md': 'b8f6a48546e6257bdd97177619a99b68e5325c41bfc7dd7508f71b75e0823366'}
_checked_signature = None
_checked_valid = False


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verified_malayalam_cache():
    """Fail closed; cache byte verification only while every file stat is unchanged.

    This verifies known checkpoint bytes, not universal recognition accuracy.
    Selection remains the caller's responsibility; this module downloads nothing.
    """
    global _checked_signature, _checked_valid
    root = MODELS/'experimental-asr-int8-cpu'
    try:
        runtime = version('ctranslate2')
        if runtime != RUNTIME_VERSION:
            return None
        names = ['conversion.json', 'model.bin', *ASSET_SHA256]
        stats = [(name, (root/name).stat()) for name in names]
        signature = (str(root.resolve()), runtime,
                     tuple((name, s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) for name, s in stats))
        if signature == _checked_signature:
            return root if _checked_valid else None
        _checked_signature, _checked_valid = signature, False
        manifest = json.loads((root/'conversion.json').read_text(encoding='utf-8'))
        if (manifest.get('source_sha256') != SOURCE_SHA256
                or manifest.get('model_sha256') != MODEL_SHA256
                or manifest.get('model_bytes') != MODEL_BYTES
                or manifest.get('ctranslate2_version') != RUNTIME_VERSION
                or manifest.get('source_header') != [6, 'WhisperSpec', 3]
                or manifest.get('keep_convolutions_fp32') is not True
                or (root/'model.bin').stat().st_size != MODEL_BYTES):
            return None
        for name, expected in {'model.bin': MODEL_SHA256, **ASSET_SHA256}.items():
            if _sha256(root/name) != expected:
                return None
        _checked_valid = True
        return root
    except (OSError, ValueError, TypeError, AttributeError, KeyError, PackageNotFoundError):
        return None


def checkpoint_metadata(root):
    if Path(root).resolve() != (MODELS/'experimental-asr-int8-cpu').resolve():
        return None
    if verified_malayalam_cache() is None:
        return None
    return {'weights_sha256': MODEL_SHA256, 'source_weights_sha256': SOURCE_SHA256,
            'storage_precision': 'int8-with-fp32-convolutions'}
