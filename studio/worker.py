"""Persistent, cancellable ONNX worker; never loads AI into the interface."""
import contextlib
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback
import zipfile

from studio.paths import configure, REFERENCES
configure()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value), encoding='utf-8')
    temporary.replace(path)


def reference_transcript(preview, folder, language, progress):
    from studio.models import ASR_REVISION
    key = hashlib.sha256(preview.read_bytes()+ASR_REVISION.encode()+str(language).encode()+b'level-v1').hexdigest()
    target = REFERENCES/'transcripts'/f'{key}.json'
    if target.is_file():
        data = json.loads(target.read_text(encoding='utf-8'))
        if data.get('text', '').strip():
            progress('Reusing the selected sample transcript')
            return data['text']
    from studio.recognition import recognize
    result = recognize(preview, folder, language, progress)
    if not result['text'].strip():
        from studio.audio import AudioError
        raise AudioError('The sample could not be transcribed. Use clearer speech or enter its exact words.')
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, result)
    return result['text']


def analyze(request, folder, progress):
    from studio.analysis import analyze_files
    progress('Measuring pitch, loudness, clipping, and spectral characteristics')
    report, _, json_path, csv_path = analyze_files(request['source'], request['comparison'], folder,
                                                  same_words=bool(request.get('same_words')))
    if request.get('full_audit'):
        from studio.recognition import full_audit
        report.update(full_audit(request['source'], request['comparison'], request.get('text', ''), folder,
                                 request.get('language') if request.get('language') != 'auto' else None, progress))
        Path(json_path).write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    archive = folder/'audio-diagnostics.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in folder.glob('analysis*'):
            if path.is_file():
                zipped.write(path, path.name)
        if (folder/'quality-audit.json').is_file():
            zipped.write(folder/'quality-audit.json', 'quality-audit.json')
    return {'analysis_json': json_path, 'analysis_csv': csv_path,
            'analysis_tracks': str(folder/'analysis-tracks.npz'), 'analysis_archive': str(archive),
            'analysis': report}


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
                if request['mode'] == 'analyze':
                    if request.get('full_audit'):
                        engine = None
                        gc.collect()
                    result = analyze(request, folder, progress)
                    progress('Audio comparison ready. Open the analysis charts and reports.')
                else:
                    from studio.languages import resolve_language
                    split_text(request.get('text'))
                    language = resolve_language(request.get('text', ''), request.get('language', 'auto'))
                    request['language'] = language
                    progress('Checking your voice sample')
                    prepared = prepare_sample(request['source'], folder,
                                              request.get('start'), request.get('end'))
                    quality = request.get('quality', 'Detailed')
                    key = ('en',) if language == 'en' else ('indic', quality)
                    if engine is not None and getattr(engine, 'engine_key', None) != key:
                        engine = None
                        gc.collect()
                    if language != 'en' and not request.get('reference_text', '').strip():
                        engine = None
                        gc.collect()
                        reference_language = request.get('reference_language')
                        if reference_language in ('auto', ''):
                            reference_language = None
                        request['reference_text'] = reference_transcript(prepared[1], folder,
                                                                        reference_language, progress)
                    if engine is None:
                        if language == 'en':
                            progress('Starting the English cloning engine')
                            engine = SpeechEngine(progress)
                        else:
                            from studio.indic import IndicEngine
                            progress('Starting the Malayalam/Hindi cloning engine')
                            engine = IndicEngine(progress, quality)
                        engine.engine_key = key
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
