"""Pinned, project-local models for reference-conditioned speech synthesis."""
from pathlib import Path
from huggingface_hub import snapshot_download
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
def download_asr(progress=lambda message: None):
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
