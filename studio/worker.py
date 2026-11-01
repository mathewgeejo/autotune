"""Persistent, cancellable ONNX worker; never loads AI into the interface."""
import contextlib
import json
from pathlib import Path
import sys
import traceback

from studio.paths import configure
configure()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value), encoding='utf-8')
    temporary.replace(path)


def run(request_path, engine=None):
    folder = request_path.parent
    latest_preview = None
    def progress(message, preview=None):
        nonlocal latest_preview
        if preview:
            latest_preview = preview
        print(message, flush=True)
        atomic_json(folder / 'status.json', {'stage': message, 'preview': latest_preview})
    with (folder / 'worker.log').open('a', encoding='utf-8', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                from studio.audio import AudioError
                from studio.speech import SpeechEngine, split_text, prepare_sample
                request = json.loads(request_path.read_text(encoding='utf-8'))
                split_text(request.get('text'))
                progress('Checking your voice sample')
                prepared = prepare_sample(request['source'], folder,
                                          request.get('start'), request.get('end'))
                if engine is None:
                    progress('Starting local voice cloning (first use downloads about 0.9 GB)')
                    engine = SpeechEngine(progress)
                else:
                    progress('Reusing the loaded speech model')
                result = engine.synthesize(request, folder, progress, prepared=prepared)
                progress('Your speech is ready to play and download.')
                atomic_json(folder / 'result.json', result)
            except Exception as exc:
                traceback.print_exc()
                from studio.audio import AudioError
                message = str(exc) if isinstance(exc, AudioError) else (
                    'Speech generation failed. Open Processing details for the error. '
                    'For a first-use download failure, check the connection and retry.')
                if isinstance(exc, MemoryError) or 'memory' in str(exc).lower():
                    message = 'The speech model ran out of memory. Close other apps and retry with shorter text.'
                progress(message)
                atomic_json(folder / 'failure.json', {'error': str(exc), 'message': message})
    return engine


def main():
    if sys.argv[1:] == ['--serve']:
        engine = None
        for line in sys.stdin:
            if line.strip():
                engine = run(Path(json.loads(line)).resolve(), engine)
    else:
        request = Path(sys.argv[1]).resolve()
        run(request)
        if (request.parent / 'failure.json').exists():
            sys.exit(1)


if __name__ == '__main__':
    main()
