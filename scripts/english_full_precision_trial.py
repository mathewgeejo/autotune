"""Run the pinned publisher FP32 pipeline in memory-bounded model stages."""
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
from studio.models import download_models, GRAPHS, REVISION, MODEL_ID
from studio.runtime import active_time, keep_awake
from studio.recognition import full_audit
from studio.speech import SpeechEngine, prepare_sample, split_english_text, english_model_text, RATE
from scripts.english_decoder_trial import save_audio
import numpy as np
import onnxruntime as ort
import psutil
import soundfile as sf
from tokenizers import Tokenizer


def load_session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def verified_models(reports):
    paths = {}
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report['status'] != 'verified' or report['revision'] != REVISION or report['model'] != MODEL_ID:
            raise ValueError('Every FP32 component must have a verified pinned download.')
        for item in report['files']:
            path = Path(item['path'])
            checksum = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024**2), b''):
                    checksum.update(block)
            if path.stat().st_size != item['bytes'] or checksum.hexdigest() != item['sha256']:
                raise ValueError('A verified FP32 component changed.')
            if path.suffix == '.onnx':
                if path.stem not in GRAPHS or path.stem in paths:
                    raise ValueError('Expected exactly one original FP32 graph per component.')
                paths[path.stem] = path
    if set(paths) != set(GRAPHS):
        raise ValueError('All four pinned FP32 components are required.')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--model-reports', type=Path, nargs=4, required=True)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path.')
    baseline = json.loads(args.baseline_report.read_text(encoding='utf-8'))
    paths = verified_models(args.model_reports)
    folder = args.output.parent/args.output.stem
    folder.mkdir(parents=True, exist_ok=False)
    prepared = prepare_sample(baseline['source_reference'], folder)
    if hashlib.sha256(prepared[1].read_bytes()).digest() != hashlib.sha256(Path(baseline['result']['reference_preview']).read_bytes()).digest():
        raise ValueError('The selected natural recording changed.')
    chunks = split_english_text(baseline['expected'])
    engine = SpeechEngine.__new__(SpeechEngine)
    engine.sessions = {}
    engine.decoder_threads = 4
    engine.tokenizer = Tokenizer.from_file(download_models()['tokenizer'])
    report = {'status': 'running', 'baseline_report': str(args.baseline_report), 'seed': args.seed,
              'model_revision': REVISION, 'precision': {name: 'fp32' for name in GRAPHS},
              'scope': 'Publisher FP32 components with existing inference contract. Encoder, text generation and waveform rendering run sequentially, with sessions released between stages. No app default change or perceptual guarantee.',
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'stages': [], 'chunk_timings': []}
    process = psutil.Process()
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        temporary.replace(args.output)
    def progress(message, **kwargs):
        print(message, flush=True)
    def stage(message):
        report['stage'] = message
        report['stages'].append({'stage': message, 'rss_mb': process.memory_info().rss/2**20,
                                 'elapsed_seconds': active_time()-started})
        progress(message)
        save()
    started = active_time()
    save()
    with keep_awake():
        try:
            ort.set_seed(args.seed)
            stage('Loading the full-precision reference encoder')
            engine.sessions['speech_encoder'] = load_session(paths['speech_encoder'], 2)
            voice = engine.sessions['speech_encoder'].run(None, {'audio_values': prepared[0]})
            if len(voice) != 4 or not all(value.size and np.isfinite(value).all() for value in voice):
                raise ValueError('Invalid full-precision voice conditioning.')
            np.savez(folder/'conditioning.npz', **dict(zip(('cond', 'prompt', 'embedding', 'features'), voice)))
            stage('Reference encoded; releasing encoder')
            engine.sessions.clear()
            gc.collect()
            stage('Encoder released; loading full-precision text models')
            for name in ('embed_tokens', 'language_model'):
                engine.sessions[name] = load_session(paths[name], 2)
            stage('Full-precision text models loaded')
            composed = []
            for index, text in enumerate(chunks, 1):
                stage(f'Composing full-precision passage {index}/{len(chunks)}')
                beginning = active_time()
                tokens, count = engine.compose_tokens(english_model_text(text), voice, progress)
                composed.append(tokens)
                report['chunk_timings'].append({'language_model_seconds': active_time()-beginning,
                                                'speech_tokens': count})
                np.savez(folder/f'passage-{index}-feeds.npz', speech_tokens=tokens,
                         speaker_embeddings=voice[2], speaker_features=voice[3])
                save()
            engine.sessions.clear()
            gc.collect()
            stage('Text models released; loading full-precision waveform decoder')
            engine.sessions['conditional_decoder'] = load_session(paths['conditional_decoder'], 4)
            stage('Full-precision waveform decoder loaded')
            waveforms, first_audio = [], None
            for index, tokens in enumerate(composed, 1):
                stage(f'Rendering full-precision passage {index}/{len(chunks)}')
                beginning = active_time()
                waveform = engine.render_tokens(tokens, voice, progress)
                report['chunk_timings'][index-1].update(waveform_decoder_seconds=active_time()-beginning,
                                                       samples=len(waveform))
                waveforms.append(waveform)
                save_audio(folder/f'passage-{index}.wav', [waveform])
                if first_audio is None:
                    first_audio = active_time()-started
                save()
            audio = save_audio(folder/'speech.wav', waveforms)
            report['result'] = {'audio': audio, 'reference_preview': str(prepared[1]),
                'duration': sf.info(audio).duration, 'text': baseline['expected'], 'language': 'en',
                'passages': len(chunks), 'passage_texts': chunks, 'model_revision': REVISION, 'model': MODEL_ID,
                'precision': report['precision'], 'cached_voice': False, 'first_audio_seconds': first_audio,
                'synthesis_seconds': active_time()-started, 'chunk_timings': report['chunk_timings']}
            report['status'] = 'generated'
            save()
        except Exception as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            engine.sessions.clear()
            gc.collect()
            save()
        stage('All synthesis sessions released; checking voice and words')
        audit_folder = folder/'audit'
        audit_folder.mkdir()
        try:
            report['audit'] = full_audit(prepared[1], audio, baseline['expected'], audit_folder, 'en', progress)
            report['status'] = 'measured'
        except Exception as exc:
            report.update(status='audit-failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            save()
    print('Staged publisher FP32 pipeline measured.', flush=True)


if __name__ == '__main__':
    main()
