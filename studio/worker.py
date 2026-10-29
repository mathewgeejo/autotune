"""Isolated audio engines. Invoked by JobManager, never in the UI process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from urllib.parse import urlparse

from studio.paths import configure, ROOT, REFERENCES, SEED

configure()


def stage(folder, text):
    print(text, flush=True)
    temporary = folder / 'status.tmp'
    temporary.write_text(json.dumps({'stage': text}), encoding='utf-8')
    temporary.replace(folder / 'status.json')


def command(args, cwd=None):
    subprocess.run([str(a) for a in args], cwd=cwd, check=True, creationflags=0x08000000)


def import_song(request, folder):
    from studio.audio import AudioError, ffmpeg
    if request.get('file'):
        return Path(request['file'])
    url = request.get('url', '').strip()
    if urlparse(url).scheme not in ('https', 'http'):
        raise AudioError('Paste a song URL starting with https://, or upload a downloaded song.')
    stage(folder, 'Downloading reference song')
    import yt_dlp
    def limits(info, *, incomplete):
        if (info.get('duration') or 0) > 600:
            return 'Choose a reference song under 10 minutes.'
        if (info.get('filesize') or 0) > 150 * 1024**2:
            return 'Reference download exceeds 150 MB. Upload a smaller audio file.'
    options = {
        'format': 'bestaudio/best', 'noplaylist': True,
        'outtmpl': str(folder / 'download.%(ext)s'), 'max_filesize': 150 * 1024**2,
        'match_filter': limits, 'ffmpeg_location': ffmpeg(),
        'socket_timeout': 30, 'retries': 2,
        'js_runtimes': {'node': {}},
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            if not info or info.get('_type') == 'playlist':
                raise AudioError('Use a single song link, or upload its downloaded audio.')
            path = Path(downloader.prepare_filename(info))
            if not path.exists():
                raise AudioError('The song could not be downloaded. Upload its audio file instead.')
            return path
    except yt_dlp.utils.DownloadError as exc:
        raise AudioError('This link could not be imported. It may require login or restrict downloads. '
                         'Upload the downloaded song instead.') from exc


def prepare(request, folder):
    import numpy as np
    import soundfile as sf
    from studio.audio import decode, validate_signal, best_reference, AudioError
    source = import_song(request, folder)
    if source.stat().st_size > 150 * 1024**2:
        raise AudioError('Upload an audio file smaller than 150 MB.')
    stage(folder, 'Reading reference song')
    decoded = decode(source, folder / 'song.wav', duration=600.1, channels=2)
    info = sf.info(decoded)
    if info.duration > 600:
        raise AudioError('Choose a reference song under 10 minutes.')
    validate_signal(decoded, 2)
    digest = hashlib.sha256()
    with decoded.open('rb') as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    cache = REFERENCES / digest.hexdigest()
    vocals = cache / 'vocals.wav'
    if vocals.exists():
        stage(folder, 'Reusing cached reference vocals')
    else:
        stage(folder, 'Separating vocals locally · first use downloads the separation model')
        command([sys.executable, '-m', 'demucs', '--two-stems', 'vocals', '-n', 'htdemucs',
                 '-d', 'cpu', '--shifts', '0', '--segment', '5', '--overlap', '0.1',
                 '-j', '1', '-o', folder / 'separated', decoded])
        extracted = folder / 'separated/htdemucs/song/vocals.wav'
        validate_signal(extracted, 2)
        cache.mkdir(parents=True, exist_ok=True)
        # Compare separated energy to the source to catch a substantially empty vocal stem.
        vocal_audio, _ = sf.read(extracted, dtype='float32')
        song_audio, _ = sf.read(decoded, dtype='float32')
        if np.sqrt(np.mean(vocal_audio**2)) < max(0.001, 0.015*np.sqrt(np.mean(song_audio**2))):
            raise AudioError('No clear vocals were found. Try a song section with a solo singer.')
        partial = cache / 'vocals.part.wav'
        shutil.copy2(extracted, partial)
        partial.replace(vocals)
        shutil.rmtree(folder / 'separated')
    stage(folder, 'Selecting a clear vocal reference')
    start, duration, total = best_reference(vocals)
    preview = decode(vocals, folder / 'reference-preview.wav', start=start, duration=duration)
    # Persist only vocals in reference cache; raw downloaded songs are temporary.
    decoded.unlink(missing_ok=True)
    if source.parent == folder:
        source.unlink(missing_ok=True)
    return {'reference': str(vocals), 'preview': str(preview), 'start': start,
            'duration': duration, 'total': total}


def instrument(request, folder):
    from studio.audio import trim_source, normalize, AudioError
    stage(folder, 'Preparing your recording')
    source = trim_source(request['source'], folder / 'input.wav', request.get('start'), request.get('end'))
    stage(folder, 'Listening for notes and expression')
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict, Model
    model_path = Path(ICASSP_2022_MODEL_PATH).with_suffix('.onnx')
    model = Model(model_path)
    sensitivity = float(request.get('sensitivity', 0.5))
    _, midi, _ = predict(str(source), model_or_model_path=model,
                         onset_threshold=0.65 - 0.4*sensitivity,
                         frame_threshold=0.5 - 0.3*sensitivity,
                         minimum_note_length=100, minimum_frequency=65, maximum_frequency=1200,
                         multiple_pitch_bends=False)
    programs = {'Piano': 0, 'Guitar': 24, 'Violin': 40, 'Synth': 80}
    name = request.get('instrument', 'Piano')
    if name not in programs:
        raise AudioError('Choose one of the available instruments.')
    octave = int(request.get('octave', 0))
    if octave not in (-2, -1, 0, 1, 2):
        raise AudioError('Octave shift must be between -2 and +2.')
    notes = 0
    for track in midi.instruments:
        track.program = programs[name]
        for note in track.notes:
            note.pitch = max(0, min(127, note.pitch + 12*octave))
            notes += 1
        # A discrete instrument renderer handles pitch bends poorly when notes overlap.
        # Keep MIDI pitch bends for bowed/synth sounds; piano and guitar use discrete notes.
        if name in ('Piano', 'Guitar'):
            track.pitch_bends.clear()
    if not notes:
        raise AudioError('No stable notes were detected. Try sustained humming or increase note sensitivity.')
    midi_path = folder / 'melody.mid'
    midi.write(str(midi_path))
    stage(folder, f'Playing your melody as {name.lower()}')
    fluids = list((ROOT / 'vendor/fluidsynth').rglob('fluidsynth.exe'))
    if not fluids:
        raise AudioError('Instrument renderer is missing. Run setup.cmd again.')
    output = folder / 'instrument.wav'
    command([fluids[0], '-ni', '-q', '-R', '0', '-C', '0', '-g', '0.8',
             '-F', output, '-T', 'wav', '-r', '44100',
             ROOT / 'data/soundfonts/GeneralUser-GS.sf2', midi_path])
    normalize(output)
    return {'audio': str(output), 'midi': str(midi_path), 'notes': notes}


def singer(request, folder):
    from studio.audio import trim_source, decode, validate_signal, normalize, AudioError
    import torch
    torch.set_num_threads(2)
    stage(folder, 'Preparing your singing')
    source = trim_source(request['source'], folder / 'input.wav', request.get('start'), request.get('end'))
    reference = Path(request['reference']).resolve()
    if not reference.is_relative_to(REFERENCES.resolve()) or not reference.is_file():
        raise AudioError('Prepare a song reference first. Its cached vocals may have been deleted.')
    ref_start = float(request.get('ref_start', 0))
    ref_duration = float(request.get('ref_duration', 10))
    if ref_start < 0 or not 2 <= ref_duration <= 15:
        raise AudioError('Choose a reference between 2 and 15 seconds long.')
    target = decode(reference, folder / 'target.wav', start=ref_start, duration=ref_duration)
    validate_signal(target, 2)
    steps = int(request.get('steps', 20))
    if steps not in (10, 20, 30):
        raise AudioError('Choose Preview, Standard, or Detailed quality.')
    if not (SEED / 'inference.py').exists():
        raise AudioError('Voice model code is missing. Run setup.cmd again.')
    stage(folder, 'Loading singer model · first use downloads several model files')
    # Run upstream in its own working directory for relative config paths.
    # It sets HF_HUB_CACHE to checkpoints/hf_cache; all weights remain project-local.
    sys.path.insert(0, str(SEED))
    os.chdir(SEED)
    import inference
    inference.device = torch.device('cpu')
    original_load = inference.load_models
    def loading(args):
        models = original_load(args)
        stage(folder, 'Converting your singing on CPU · this can take several minutes')
        return models
    inference.load_models = loading
    args = argparse.Namespace(source=str(source), target=str(target), output=str(folder / 'converted'),
                              diffusion_steps=steps, length_adjust=1.0, inference_cfg_rate=0.7,
                              f0_condition=True, auto_f0_adjust=False, semi_tone_shift=0,
                              checkpoint=None, config=None, fp16=False)
    inference.main(args)
    outputs = list((folder / 'converted').glob('*.wav'))
    if not outputs:
        raise AudioError('The voice engine did not produce audio. See the job log for details.')
    output = folder / 'singing.wav'
    shutil.move(str(outputs[0]), output)
    normalize(output)
    return {'audio': str(output)}


def main():
    request_path = Path(sys.argv[1]).resolve()
    folder = request_path.parent
    request = json.loads(request_path.read_text(encoding='utf-8'))
    try:
        result = {'prepare': prepare, 'instrument': instrument, 'singer': singer}[request['mode']](request, folder)
        (folder / 'result.json').write_text(json.dumps(result), encoding='utf-8')
        stage(folder, 'Complete')
    except Exception as exc:
        traceback.print_exc()
        from studio.audio import AudioError
        message = str(exc) if isinstance(exc, AudioError) else (
            'The local audio engine could not finish. Check the job log below for details. '
            'If a download failed, check your connection and retry.')
        if isinstance(exc, MemoryError) or 'not enough memory' in str(exc).lower():
            message = ('This conversion ran out of memory. Close other apps, use a shorter recording '
                       'and reference section, then retry with Preview quality.')
        stage(folder, message)
        (folder / 'error.txt').write_text(str(exc), encoding='utf-8')
        sys.exit(1)


if __name__ == '__main__':
    main()
