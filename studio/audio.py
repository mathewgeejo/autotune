"""Small audio operations shared by the UI and isolated workers."""
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf
from studio.processes import HIDDEN_WINDOW


class AudioError(ValueError):
    pass


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def decode(source, target, start=0, duration=None, rate=44100, channels=1):
    args = [ffmpeg(), '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
            '-ss', str(start), '-i', str(source)]
    if duration is not None:
        args += ['-t', str(duration)]
    args += ['-vn', '-ar', str(rate), '-ac', str(channels), '-c:a', 'pcm_s16le', str(target)]
    result = subprocess.run(args, capture_output=True, text=True, check=False,
                            creationflags=HIDDEN_WINDOW, timeout=120)
    if result.returncode:
        raise AudioError('Could not read this audio. Try a WAV, MP3, FLAC, or M4A file.')
    return Path(target)


def validate_signal(path, minimum=0.5):
    samples, rate = sf.read(path, dtype='float32', always_2d=True)
    if len(samples) / rate < minimum:
        raise AudioError(f'Record at least {minimum} seconds of audio.')
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) < 0.002:
        raise AudioError('This recording is silent or too quiet. Move closer to the microphone.')
    return samples, rate


def best_reference(path, seconds=10):
    samples, rate = validate_signal(path, minimum=2)
    mono = samples.mean(axis=1)
    # Rank windows by sustained vocal energy, so a transient alone cannot win.
    frame = rate // 5
    count = len(mono) // frame
    energy = np.sqrt(np.mean(mono[:count*frame].reshape(count, frame)**2, axis=1))
    width = min(len(energy), int(seconds * 5))
    scores = np.convolve(np.minimum(energy, np.percentile(energy, 85)), np.ones(width)/width, mode='valid')
    start = int(np.argmax(scores)) / 5
    return start, min(seconds, len(mono)/rate-start), len(mono)/rate
