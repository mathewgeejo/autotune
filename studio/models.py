"""Pinned, project-local models for reference-conditioned speech synthesis."""
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = 'ResembleAI/chatterbox-turbo-ONNX'
REVISION = 'd21799bd0354adb85e348b8a0442a8405110a2cf'
GRAPHS = ('speech_encoder', 'embed_tokens', 'language_model', 'conditional_decoder')
PRECISION = {name: 'q4' if name == 'language_model' else 'quantized' for name in GRAPHS}


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
