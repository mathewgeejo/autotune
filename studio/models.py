"""Pinned, project-local models for reference-conditioned speech synthesis."""
import os
from pathlib import Path
import shutil
import uuid
from huggingface_hub import snapshot_download, hf_hub_download
from studio.runtime import keep_awake

MODEL_ID = 'ResembleAI/chatterbox-turbo-ONNX'
REVISION = 'd21799bd0354adb85e348b8a0442a8405110a2cf'
GRAPHS = ('speech_encoder', 'embed_tokens', 'language_model', 'conditional_decoder')
PRECISION = {name: 'q4' if name == 'language_model' else 'quantized' for name in GRAPHS}
INDIC_ID = 'Bbkblo/DhVaani-0.5-ONNX'
INDIC_REVISION = '597cd20e44e85910fdb086d04c074cdc313a1a80'
SPEAKER_ID = 'Xenova/wavlm-base-plus-sv'
SPEAKER_REVISION = 'e61029603001bd11295c36d878698708bf59190f'
ASR_ID = 'mobiuslabsgmbh/faster-whisper-large-v3-turbo'
ASR_REVISION = '0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf'
MALAYALAM_ASR_ID = 'smcproject/vegam-whisper-medium-ml'
MALAYALAM_ASR_REVISION = '21334192f79d87cc05fe4253d7d839b5a42c33bf'
ASR_TOKENIZER_ID = 'openai/whisper-tiny'
ASR_TOKENIZER_REVISION = '169d4a4341b33bc18d8881c4b69c2e104e1cc0af'


@keep_awake()
def download_models(progress=lambda text: None):
    files = ['tokenizer.json']
    for name in GRAPHS:
        filename = f'onnx/{name}_{PRECISION[name]}.onnx'
        files.extend((filename, filename + '_data'))
    progress('Checking voice-cloning models (first download: about 0.9 GB)')
    root = Path(snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=files, max_workers=3))
    paths = {name: str(root / f'onnx/{name}_{PRECISION[name]}.onnx') for name in GRAPHS}
    paths['tokenizer'] = str(root / 'tokenizer.json')
    return paths


@keep_awake()
def download_indic_models(quality='Detailed', progress=lambda message: None):
    suffix = '' if quality == 'Detailed' else '_int8'
    files = [f'text_encoder{suffix}.onnx', f'fm_decoder{suffix}.onnx',
             'vocoder_backbone.onnx', 'vocos_head.npz', 'mel_fb.npz', 'tokens.txt', 'model.json', 'README.md']
    progress('Checking Malayalam/Hindi voice models (downloaded once)')
    return Path(snapshot_download(INDIC_ID, revision=INDIC_REVISION, allow_patterns=files, max_workers=3))


@keep_awake()
def download_asr(progress=lambda message: None, language=None):
    if language == 'ml':
        from studio.paths import MODELS
        if os.environ.get('VOICE_STUDIO_COMPACT_ASR', '0') == '1':
            from studio.compact_recognition import verified_malayalam_cache
            compact = verified_malayalam_cache()
            if compact is not None:
                progress('Using the verified compact Malayalam recognizer')
                return str(compact)
        progress('Checking native Malayalam recognition (first download: about 3.1 GB)')
        root = Path(snapshot_download(MALAYALAM_ASR_ID, revision=MALAYALAM_ASR_REVISION,
            allow_patterns=['model.bin', 'config.json', 'vocabulary.txt', 'README.md'], max_workers=2))
        tokenizer = Path(hf_hub_download(ASR_TOKENIZER_ID, 'tokenizer.json', revision=ASR_TOKENIZER_REVISION))
        # The older multilingual Whisper vocabulary must use its matching
        # tokenizer. Bundle pinned assets without editing the immutable cache.
        bundle = MODELS/'malayalam-asr'/MALAYALAM_ASR_REVISION
        bundle.mkdir(parents=True, exist_ok=True)
        for source, name in [(root/name, name) for name in ('model.bin', 'config.json', 'vocabulary.txt', 'README.md')]+[(tokenizer, 'tokenizer.json')]:
            # HF snapshots use relative symlinks. Hard-link the real blob;
            # linking the symlink itself would break it in the new directory.
            source = source.resolve(strict=True)
            destination = bundle/name
            if destination.is_file() and destination.stat().st_size == source.stat().st_size:
                continue
            temporary = bundle/(name+'.'+uuid.uuid4().hex+'.tmp')
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copyfile(source, temporary)
            temporary.replace(destination)
        return str(bundle)
    progress('Checking multilingual speech recognition (first download: about 1.6 GB)')
    return snapshot_download(ASR_ID, revision=ASR_REVISION,
        allow_patterns=['model.bin', 'config.json', 'preprocessor_config.json', 'tokenizer.json', 'vocabulary.json'],
        max_workers=3)


@keep_awake()
def download_speaker(progress=lambda message: None):
    progress('Checking independent speaker verifier')
    root = Path(snapshot_download(SPEAKER_ID, revision=SPEAKER_REVISION,
        allow_patterns=['onnx/model_quantized.onnx', 'config.json', 'preprocessor_config.json'], max_workers=2))
    return root / 'onnx/model_quantized.onnx'
