"""Optional DNSMOS P.835 noise-quality proxy, calibrated against natural controls.

Microsoft DNS-Challenge, CC-BY-4.0, pinned below. This evaluates signal/background
quality; it does not prove voice identity, naturalness, or human equivalence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure, MODELS
configure()
from studio.audio import decode
from studio.recognition import verifier_input
import numpy as np
import soundfile as sf

REVISION = '591184a9fcb2cbdec02520fed81a32bbbf9d73ff'
SHA256 = '269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd'


def proxy_audio(source, folder):
    source, folder = Path(source), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    path = decode(source, folder/(fingerprint[:12]+'.wav'), rate=16000, floating=True, duration=60)
    audio, _ = sf.read(path, dtype='float32')
    audio, preprocessing = verifier_input(audio)
    return fingerprint, audio, preprocessing


class SpeechQualityProxy:
    """UTMOS22 synthetic-speech quality prediction; one unpadded clip at a time."""
    model_id = 'TigreGotico/utmos-onnx'
    revision = 'ff41b8f440cb12ecda18261f9ff7326d058275ce'

    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(self.model_id, 'utmos22_strong.onnx', revision=self.revision)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, options, providers=['CPUExecutionProvider'])

    def score(self, source, folder):
        fingerprint, audio, preprocessing = proxy_audio(source, folder)
        window = 8*16000
        if len(audio) < 16000:
            raise ValueError('Use at least one second for the speech-quality proxy')
        starts = list(range(0, max(1, len(audio)-window+1), window))
        if len(audio) > window and starts[-1] != len(audio)-window:
            starts.append(len(audio)-window)
        scores = []
        for start in starts:
            clip = audio[start:start+window]
            value = self.session.run(None, {'wave':clip[None].astype(np.float32)})[0]
            scores.append({'start_seconds':start/16000,'duration_seconds':len(clip)/16000,
                           'predicted_mos':float(np.asarray(value).reshape(-1)[0])})
        return {'predicted_mos':float(np.mean([item['predicted_mos'] for item in scores])),
                'windows':scores, 'model':self.model_id,'revision':self.revision,'input_sha256':fingerprint,
                'preprocessing':preprocessing,
                'scope':'Learned synthetic-speech quality proxy, not human MOS or proof of naturalness/identity. Calibrate against natural clips and degradations.'}


class NoiseQualityProxy:
    def __init__(self):
        import onnxruntime as ort
        folder = MODELS/'dnsmos'/REVISION
        folder.mkdir(parents=True, exist_ok=True)
        model = folder/'sig_bak_ovr.onnx'
        url = f'https://raw.githubusercontent.com/microsoft/DNS-Challenge/{REVISION}/'
        for path, remote in [(model, 'DNSMOS/DNSMOS/sig_bak_ovr.onnx'), (folder/'LICENSE', 'LICENSE')]:
            if not path.exists():
                with urllib.request.urlopen(url+remote, timeout=40) as response:
                    content = response.read()
                path.write_bytes(content)
        if hashlib.sha256(model.read_bytes()).hexdigest() != SHA256:
            raise ValueError('DNSMOS model checksum mismatch')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])

    def score(self, source, folder):
        fingerprint, audio, preprocessing = proxy_audio(source, folder)
        original_samples = len(audio)
        repeat = 1
        size = int(9.01*16000)
        if not original_samples:
            raise ValueError('Noise-quality scoring requires nonempty audio')
        while len(audio) < size:
            audio = np.tile(audio, 2)
            repeat *= 2
        # Repeat-to-length and one-second hops follow the upstream short-clip
        # protocol. Repetition does not create additional independent evidence.
        hops = int(np.floor(len(audio)/16000)-9.01)+1
        raw = []
        for index in range(hops):
            segment = audio[index*16000:index*16000+size]
            if len(segment) == size:
                raw.append(self.session.run(None, {'input_1':segment[None].astype(np.float32)})[0][0])
        raw = np.asarray(raw)
        coefficients = [[-0.08397278, 1.22083953, 0.0052439],
                        [-0.13166888, 1.60915514, -0.39604546],
                        [-0.06766283, 1.11546468, 0.04602535]]
        values = {name:float(np.mean(np.polyval(coefficient, raw[:,i])))
                  for i,(name,coefficient) in enumerate(zip(('signal', 'background', 'overall'), coefficients))}
        return {**values, 'model':'Microsoft DNSMOS P.835', 'revision':REVISION, 'sha256':SHA256,
            'input_sha256':fingerprint, 'preprocessing':preprocessing, 'repeat_factor':repeat,
            'scored_windows':len(raw), 'original_seconds':original_samples/16000,
            'scope':'Noise-quality proxy, not a human listening score or proof of cloning naturalness/identity.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--metric', choices=['dnsmos','utmos'], default='dnsmos')
    args = parser.parse_args()
    proxy = SpeechQualityProxy() if args.metric == 'utmos' else NoiseQualityProxy()
    report = {str(path):proxy.score(path, args.output.parent/'proxy-inputs') for path in args.audio}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({name:{k:v for k,v in result.items() if k in ('signal','background','overall','predicted_mos')}
                      for name,result in report.items()}, indent=2))


if __name__ == '__main__':
    main()
