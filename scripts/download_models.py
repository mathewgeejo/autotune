"""Optional one-time download for offline singer conversion and separation."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure, SEED
configure()
os.environ['HF_HUB_CACHE'] = str(SEED / 'checkpoints/hf_cache')
from huggingface_hub import hf_hub_download


def main():
    # Download vocoder and Whisper first: the voice checkpoint is already cached
    # when running this after an interrupted first conversion.
    files = [
        ('nvidia/bigvgan_v2_44khz_128band_512x', 'config.json', None),
        ('nvidia/bigvgan_v2_44khz_128band_512x', 'bigvgan_generator.pt', None),
        ('openai/whisper-small', 'config.json', None),
        ('openai/whisper-small', 'preprocessor_config.json', None),
        ('openai/whisper-small', 'model.safetensors', None),
        ('Plachta/Seed-VC', 'DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth', SEED/'checkpoints'),
        ('Plachta/Seed-VC', 'config_dit_mel_seed_uvit_whisper_base_f0_44k.yml', SEED/'checkpoints'),
        ('lj1995/VoiceConversionWebUI', 'rmvpe.pt', SEED/'checkpoints'),
        ('funasr/campplus', 'campplus_cn_common.bin', SEED/'checkpoints'),
    ]
    for repo, filename, cache in files:
        print(f'Downloading {repo} / {filename}', flush=True)
        hf_hub_download(repo, filename, cache_dir=str(cache) if cache else None)
    from studio.worker import separation_weights
    folder = ROOT / 'data/model-download'
    folder.mkdir(exist_ok=True)
    separation_weights(folder)
    print('All models are cached for offline use.', flush=True)


if __name__ == '__main__':
    main()
