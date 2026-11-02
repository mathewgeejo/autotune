"""Isolated full-precision Dhee-Indic-F5 trial using the fixed baseline reference.

Run with .tools/f5-runtime/Scripts/python.exe after the verified download. This
does not import Torch into the app or change its production voice engine.
"""
import argparse
import gc
import hashlib
import importlib.machinery
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure, MODELS
configure()
from studio.runtime import active_time, keep_awake

MODEL_REVISION = 'b169ad1af92d659f9b85ac393510f8960e4a0690'
SOURCE_REVISION = '13f7c4d627cc10111aea8fe9c0039462cacacdc7'
VOCAB_REVISION = '2b922175f2b4727b2415ed5cf636cd5ed2a2f8a9'


def inference_modules(source):
    # The upstream model __init__ eagerly imports training classes. Load the
    # unmodified inference submodules without those unrelated training imports.
    for name, directory in [('f5_tts', source/'f5_tts'), ('f5_tts.model', source/'f5_tts/model')]:
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, choices=[16, 32], default=32)
    parser.add_argument('--guidance', type=float, default=2.0)
    parser.add_argument('--speed', type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path to preserve prior trials.')
    if not 0.25 <= args.speed <= 4 or not 0 <= args.guidance <= 5:
        parser.error('Unsupported trial speed or guidance.')
    baseline = json.loads(args.baseline_report.read_text(encoding='utf-8'))
    source = ROOT/'.tools/indic-f5-source'
    checkpoint = MODELS/'experimental-f5'/MODEL_REVISION/'model.safetensors'
    download = json.loads(checkpoint.with_name('download.json').read_text(encoding='utf-8'))
    if download['status'] != 'verified' or not checkpoint.is_file():
        raise RuntimeError('Wait for the checkpoint download and full SHA256 verification.')
    vocab_path = MODELS/'huggingface/hub/models--Anjan9320--IndicF5/snapshots'/VOCAB_REVISION/'checkpoints/vocab.txt'
    folder = args.output.parent/args.output.stem
    folder.mkdir(parents=True, exist_ok=False)
    reference = folder/'voice-sample.wav'
    shutil.copyfile(baseline['result']['reference_preview'], reference)
    report = {'status': 'running', 'baseline_report': str(args.baseline_report),
              'source_reference': baseline['source_reference'], 'expected': baseline['expected'],
              'reference_text': baseline['result']['reference_text'], 'language': baseline['language'],
              'model': 'dheeyantra/dhee-indic-f5', 'model_revision': MODEL_REVISION,
              'source_revision': SOURCE_REVISION, 'checkpoint_sha256': download['sha256'],
              'vocab_source': 'Anjan9320/IndicF5', 'vocab_revision': VOCAB_REVISION,
              'vocab_sha256': hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
              'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'scope': 'Experimental public checkpoint, original inference architecture and matching checkpoint vocoder; no production engine change.'}
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        temporary.replace(args.output)
    save()
    with keep_awake():
        try:
            import numpy as np
            import soundfile as sf
            import torch
            from safetensors import safe_open
            inference_modules(source)
            from f5_tts.model.backbones.dit import DiT
            from f5_tts.model.cfm import CFM
            from f5_tts.model.modules import precompute_freqs_cis
            from f5_tts.model.utils import get_tokenizer, convert_char_to_pinyin
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            vocab, vocab_size = get_tokenizer(str(vocab_path), 'custom')
            print('Loading FP32 F5 transformer from verified SafeTensors', flush=True)
            started = active_time()
            with torch.device('meta'):
                transformer = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
                                  conv_layers=4, text_num_embeds=vocab_size, mel_dim=100)
            with safe_open(checkpoint, framework='pt', device='cpu') as archive:
                state = {key.removeprefix('ema_model.transformer.'): archive.get_tensor(key)
                         for key in archive.keys() if key.startswith('ema_model.transformer.')}
            transformer.load_state_dict(state, strict=True, assign=True)
            del state
            transformer.text_embed.freqs_cis = precompute_freqs_cis(512, transformer.text_embed.precompute_max_pos)
            if any(value.is_meta for value in transformer.parameters()) or any(value.is_meta for value in transformer.buffers()):
                raise RuntimeError('Some transformer tensors were not materialized on CPU')
            transformer.eval()
            model = CFM(transformer=transformer, mel_spec_kwargs={'n_fft': 1024, 'hop_length': 256,
                'win_length': 1024, 'n_mel_channels': 100, 'target_sample_rate': 24000,
                'mel_spec_type': 'vocos'}, vocab_char_map=vocab).eval()
            report['load_seconds'] = active_time()-started
            report['torch_version'] = torch.__version__
            report['transformer_parameters'] = sum(value.numel() for value in transformer.parameters())
            save()
            audio, rate = sf.read(reference, dtype='float32')
            if rate != 24000 or audio.ndim != 1:
                raise ValueError('The fixed baseline must supply mono 24 kHz reference audio')
            rms = float(np.sqrt(np.mean(audio*audio)))
            audio = torch.from_numpy(audio[None])
            if rms < 0.1:
                audio = audio*0.1/max(rms, 1e-8)
            prompt = unicodedata.normalize('NFC', report['reference_text']).strip()
            text = unicodedata.normalize('NFC', report['expected']).strip()
            prompt += ' ' if prompt.endswith(('.', '!', '?')) else '. '
            if not text.endswith(('.', '!', '?')):
                text += '.'
            tokenized = convert_char_to_pinyin([prompt+text])
            missing = sorted({char for char in tokenized[0] if char not in vocab})
            if missing:
                raise ValueError('The supplied public vocabulary lacks input characters: '+repr(missing))
            reference_frames = audio.shape[-1]//256
            duration = reference_frames+int(reference_frames/len(prompt.encode('utf-8'))*len(text.encode('utf-8'))/args.speed)
            report.update(reference_frames=reference_frames, total_frames=duration,
                          precision={'transformer': 'fp32', 'steps': args.steps, 'guidance': args.guidance,
                                     'speed': args.speed, 'seed': 666, 'sway_sampling_coefficient': -1.0},
                          actual_synthesis_text=text, actual_prompt_text=prompt)
            save()
            evaluations = [0]
            def progress(module, inputs):
                evaluations[0] += 1
                print(f'F5 flow evaluation {evaluations[0]}', flush=True)
            handle = transformer.register_forward_pre_hook(progress)
            started = active_time()
            with torch.inference_mode():
                generated, trajectory = model.sample(cond=audio, text=tokenized, duration=duration,
                    steps=args.steps, cfg_strength=args.guidance, sway_sampling_coef=-1.0, seed=666)
            report['flow_seconds'] = active_time()-started
            handle.remove()
            mel = generated[:, reference_frames:, :].permute(0, 2, 1).contiguous().clone()
            np.save(folder/'generated-mel.npy', mel.numpy())
            del generated, trajectory, model, transformer, audio, handle
            gc.collect()
            save()
            print('Loading matching checkpoint Vocos after releasing F5 transformer', flush=True)
            from vocos import Vocos
            from vocos.feature_extractors import MelSpectrogramFeatures
            from vocos.models import VocosBackbone
            from vocos.heads import ISTFTHead
            vocoder = Vocos(feature_extractor=MelSpectrogramFeatures(sample_rate=24000, n_fft=1024,
                hop_length=256, n_mels=100), backbone=VocosBackbone(input_channels=100, dim=512,
                intermediate_dim=1536, num_layers=8), head=ISTFTHead(dim=512, n_fft=1024, hop_length=256, padding='same'))
            with safe_open(checkpoint, framework='pt', device='cpu') as archive:
                state = {key.removeprefix('vocoder.'): archive.get_tensor(key)
                         for key in archive.keys() if key.startswith('vocoder.')}
            vocoder.load_state_dict(state, strict=True)
            vocoder.eval()
            started = active_time()
            with torch.inference_mode():
                waveform = vocoder.decode(mel).reshape(-1).numpy()
            report['vocoder_seconds'] = active_time()-started
            if rms < 0.1:
                waveform *= rms/0.1
            if not np.isfinite(waveform).all() or not len(waveform) or np.max(np.abs(waveform)) < 0.001:
                raise ValueError('F5 returned invalid or empty speech')
            waveform *= min(1, 0.98/np.max(np.abs(waveform)))
            destination = folder/'speech.wav'
            sf.write(destination, waveform, 24000, subtype='PCM_16')
            report.update(status='generated', result={'audio': str(destination), 'reference_preview': str(reference),
                          'duration': len(waveform)/24000, 'model': report['model'], 'model_revision': MODEL_REVISION,
                          'reference_text': report['reference_text'], 'precision': report['precision'],
                          'synthesis_seconds': report['flow_seconds']+report['vocoder_seconds']})
            save()
            print('Generated experimental F5 audio:', destination, flush=True)
        except Exception as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            save()
            raise


if __name__ == '__main__':
    main()
