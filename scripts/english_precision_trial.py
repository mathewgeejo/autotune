"""Compare cached Q4/Q8 English language models with unchanged voice/decoder.

This isolated experiment does not download weights or change app defaults.
Fresh sessions give each variant the same decoder random-state starting point;
different speech tokens can still change random consumption and waveform noise.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio import speech
from studio.analysis import analyze_files, normalize_text
from studio.models import PRECISION
from studio.recognition import full_audit
from studio.runtime import active_time, keep_awake

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--precisions', choices=['q4', 'q8'], nargs='+', default=['q4', 'q8'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path to preserve prior measurements.')
    baseline = json.loads(args.baseline_report.read_text(encoding='utf-8'))
    if baseline['language'] != 'en':
        parser.error('An English baseline is required.')
    original_download = speech.download_models
    paths = original_download()
    variants = {}
    for precision in args.precisions:
        graph = Path(paths['language_model']).with_name(
            'language_model_q4.onnx' if precision == 'q4' else 'language_model_quantized.onnx')
        if not graph.is_file() or not Path(str(graph)+'_data').is_file():
            parser.error(f'The {precision} weights are not cached; this trial does not download them.')
        variants[precision] = {**paths, 'language_model': str(graph)}
    folder = args.output.parent/args.output.stem
    folder.mkdir(parents=True, exist_ok=False)
    reference_folder = folder/'reference'
    reference_folder.mkdir()
    prepared = speech.prepare_sample(baseline['source_reference'], reference_folder)
    if hashlib.sha256(Path(baseline['result']['reference_preview']).read_bytes()).digest() != hashlib.sha256(prepared[1].read_bytes()).digest():
        raise ValueError('The conditioning reference changed; this is not a controlled comparison.')
    report = {'baseline_report': str(args.baseline_report), 'status': 'running',
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'scope': 'Only language-model weight precision changes; automated scores do not prove indistinguishability.',
              'decoder_randomness': 'Fresh sessions at corresponding run index; different speech tokens can change random consumption.',
              'trials': []}
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        temporary.replace(args.output)
    progress = lambda message, **kwargs: print(message, flush=True)
    with keep_awake():
        for precision, trial_paths in variants.items():
            destination = folder/precision
            destination.mkdir()
            trial = {'precision': precision, 'state': 'running'}
            report['trials'].append(trial)
            save()
            engine = None
            try:
                speech.download_models = lambda progress=None: trial_paths
                started = active_time()
                engine = speech.SpeechEngine(progress)
                engine.precision = {**PRECISION, 'language_model': 'q4' if precision == 'q4' else 'quantized'}
                trial['load_seconds'] = active_time()-started
                trial['worker_rss_after_load_mb'] = psutil.Process().memory_info().rss/2**20
                trial['result'] = engine.synthesize({'source': baseline['source_reference'],
                    'text': baseline['expected'], 'language': 'en'}, destination, progress, prepared=prepared)
                trial['audio_sha256'] = hashlib.sha256(Path(trial['result']['audio']).read_bytes()).hexdigest()
                trial['state'] = 'generated'
            except Exception as exc:
                trial.update(state='failed', error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                speech.download_models = original_download
                engine = None
                gc.collect()
                save()
        for trial in report['trials']:
            destination = folder/trial['precision']
            trial['audit'] = full_audit(str(prepared[1]), trial['result']['audio'], baseline['expected'],
                                        destination, 'en', progress)
            if normalize_text(baseline.get('reference_text', '')) == normalize_text(baseline['expected']):
                trial['same_words_acoustics'], _, _, _ = analyze_files(
                    str(prepared[1]), trial['result']['audio'], destination/'same-words', same_words=True)
            trial['state'] = 'measured'
            save()
        report['status'] = 'measured'
        save()
        for trial in report['trials']:
            print(json.dumps({'precision': trial['precision'],
                'seconds': trial['result']['synthesis_seconds'],
                'speaker': trial['audit']['speaker_verification'].get('cosine_similarity'),
                'content': trial['audit']['content_comparison']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
