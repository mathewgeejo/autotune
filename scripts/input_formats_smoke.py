"""Real minute-long stereo WAV, FLAC, MP3, and M4A input boundary checks."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.audio import AudioError, decode, ffmpeg
from studio.processes import HIDDEN_WINDOW
from studio.speech import prepare_sample
import numpy as np
import soundfile as sf


def main():
    output = ROOT/'data/smoke/minute-inputs'
    output.mkdir(parents=True, exist_ok=True)
    decoded = decode(ROOT/'data/smoke/dave.wav', output/'original.wav', rate=44100, channels=2)
    original, rate = sf.read(decoded, dtype='float32', always_2d=True)
    minute = np.tile(original, (int(np.ceil(60*rate/len(original))), 1))[:60*rate]
    source = output/'minute.wav'
    sf.write(source, minute, rate, subtype='PCM_16')
    checks = []
    for extension, codec in [('wav', None), ('flac', 'flac'), ('mp3', 'libmp3lame'), ('m4a', 'aac')]:
        path = output/('minute.'+extension)
        if codec:
            subprocess.run([ffmpeg(), '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                            '-i', str(source), '-c:a', codec, str(path)], check=True, creationflags=HIDDEN_WINDOW)
        folder = output/extension
        folder.mkdir(exist_ok=True)
        samples, preview, _ = prepare_sample(path, folder)
        assert samples.shape[0] == 1 and 3*24000 <= samples.shape[1] <= 10*24000
        assert np.isfinite(samples).all() and sf.info(preview).samplerate == 24000
        section = json.loads((folder/'reference-section.json').read_text(encoding='utf-8'))
        assert abs(section['uploaded_selection_seconds']-60) <= 0.02
        checks.append({'format': extension, **section, 'passed': True})
    overlong = output/'over-limit.wav'
    sf.write(overlong, np.r_[minute, minute[:rate]], rate)
    try:
        prepare_sample(overlong, output)
    except AudioError as exc:
        assert '60 seconds' in str(exc)
        checks.append({'61_seconds_rejected': True})
    else:
        raise AssertionError('61-second upload was accepted')
    (output/'checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
