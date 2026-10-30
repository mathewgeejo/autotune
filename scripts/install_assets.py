"""Download local runtime assets; rerunning resumes an incomplete setup."""
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SEED_REVISION = '51383efd921027683c89e5348211d93ff12ac2a8'


def download(url, target):
    target = Path(target)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    print(f'Downloading {target.name}', flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'VoiceStudio/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as out:
        while block := response.read(1024 * 1024):
            out.write(block)
    partial.replace(target)


def main():
    vendor = ROOT / 'vendor'
    seed = vendor / 'seed-vc'
    if not (seed / 'inference.py').exists():
        archive = ROOT / '.tools/seed-vc.zip'
        download(f'https://github.com/Plachtaa/seed-vc/archive/{SEED_REVISION}.zip', archive)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(vendor)
        (vendor / f'seed-vc-{SEED_REVISION}').rename(seed)
        archive.unlink()
    # Upstream forces Whisper to fp16, including on CPU. Keep CPU inference float32.
    inference = seed / 'inference.py'
    text = inference.read_text(encoding='utf-8')
    text = text.replace('torch_dtype=torch.float16).to(device)',
                        'torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32)).to(device)')
    # Upstream's 30-second context is <2600 frames. 4096 accommodates it without
    # allocating an unnecessarily large 8192-frame transformer KV cache on CPU.
    text = text.replace('max_seq_length=8192', 'max_seq_length=4096')
    inference.write_text(text, encoding='utf-8')
    # This singing configuration does not use vector quantization. Lazy-loading
    # that optional branch avoids installing the unrelated DAC training stack.
    regulator = seed / 'modules/length_regulator.py'
    regulator_text = regulator.read_text(encoding='utf-8')
    regulator_text = ''.join(line for line in regulator_text.splitlines(keepends=True)
                             if line.strip() != 'from dac.nn.quantize import VectorQuantize')
    regulator_text = regulator_text.replace('if vector_quantize:\n                self.vq',
        'if vector_quantize:\n                from dac.nn.quantize import VectorQuantize\n                self.vq')
    regulator.write_text(regulator_text, encoding='utf-8')
    (seed / 'VOICE_STUDIO_PATCH.txt').write_text(
        f'Upstream revision: {SEED_REVISION}\nCPU Whisper dtype changed to float32.\n'
        'Transformer inference cache reduced to 4096 frames (30-second context fits).\n'
        'Optional DAC vector quantizer import is lazy; singing configuration does not use it.\n', encoding='utf-8')
    fluid_dir = vendor / 'fluidsynth'
    if not list(fluid_dir.rglob('fluidsynth.exe')):
        archive = ROOT / '.tools/fluidsynth.zip'
        download('https://github.com/FluidSynth/fluidsynth/releases/download/v2.6.1/fluidsynth-v2.6.1-win10-x64-cpp11.zip', archive)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(fluid_dir)
        archive.unlink()
    sf = ROOT / 'data/soundfonts/GeneralUser-GS.sf2'
    download('https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/GeneralUser-GS.sf2', sf)
    download('https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/README.md', sf.parent / 'GeneralUser-README.md')
    # Fetch the soundfont documentation, including its redistribution license.
    download('https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/documentation/LICENSE.txt',
             sf.parent / 'LICENSE.txt')
    print('Runtime assets ready. AI weights download on first use.', flush=True)


if __name__ == '__main__':
    main()
