"""Collect a deterministic FLEURS subset across available gender labels."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.audio import reference_activity
from studio.runtime import keep_awake
import numpy as np
import soundfile as sf

REVISION = '70bb2e84b976b7e960aa89f1c648e09c59f894dd'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['ml_in', 'en_us'], default='ml_in')
    parser.add_argument('--per-gender', type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    folder = ROOT/f'data/quality/corpus/{args.language}-{args.per_gender}'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder/'manifest.json'
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if previous.get('status') == 'ready' and all((ROOT/clip['audio']).is_file() for clip in previous['clips']):
            print(f'Corpus already present: {manifest}')
            return
    base = f'https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/data/{args.language}/'
    with urllib.request.urlopen(base+'test.tsv', timeout=40) as response:
        rows = csv.reader(response.read().decode('utf-8').splitlines(), delimiter='\t', quoting=csv.QUOTE_NONE)
        eligible = {row[1]: row for row in rows if len(row) >= 6 and row[-2].isdigit()
                    and 5 <= int(row[-2])/16000 <= 9.8 and row[-1] in ('MALE', 'FEMALE')}
    available = {gender: sum(row[-1] == gender for row in eligible.values()) for gender in ('MALE', 'FEMALE')}
    targets = {gender: min(args.per_gender, count) for gender, count in available.items()}
    if not sum(targets.values()):
        raise RuntimeError('The pinned metadata has no recordings matching the declared duration range.')
    report = {'dataset': 'google/fleurs', 'revision': REVISION, 'split': 'test', 'language': args.language,
        'license': 'CC-BY-4.0', 'source': 'https://huggingface.co/datasets/google/fleurs',
        'selection': f'First {args.per_gender} recordings per available gender in archive order, duration 5–9.8s; no selection based on clone results.',
        'available_eligible_gender_counts': available,
        'missing_gender_labels': [gender for gender, count in available.items() if not count],
        'speaker_identity_note': 'Gender labels are not unique speaker IDs; no claim that all clips are different people.',
        'status': 'collecting', 'clips': []}
    counts = {'MALE': 0, 'FEMALE': 0}
    request = urllib.request.Request(base+'audio/test.tar.gz', headers={'User-Agent': 'VoiceStudio-Quality/1.0'})
    with keep_awake(), urllib.request.urlopen(request, timeout=90) as response:
        with tarfile.open(fileobj=response, mode='r|gz') as archive:
            for member in archive:
                name = Path(member.name).name
                if not member.isfile() or name not in eligible:
                    continue
                row = eligible[name]
                gender = row[-1]
                if counts[gender] >= targets[gender]:
                    continue
                with archive.extractfile(member) as stream:
                    content = stream.read()
                audio, rate = sf.read(io.BytesIO(content), dtype='float32', always_2d=True)
                mono = audio.mean(axis=1)
                _, transients = reference_activity(mono, rate)
                destination = folder/name
                destination.write_bytes(content)
                caption = destination.with_suffix('.txt')
                caption.write_text(row[2], encoding='utf-8')
                rms = float(np.sqrt(np.mean(mono*mono)))
                peak = float(np.max(np.abs(mono)))
                report['clips'].append({'id': name.removesuffix('.wav'), 'gender': gender,
                    'audio': destination.relative_to(ROOT).as_posix(), 'caption': caption.relative_to(ROOT).as_posix(),
                    'text': row[2], 'sha256': hashlib.sha256(content).hexdigest(), 'duration_seconds': len(mono)/rate,
                    'original_rms': rms, 'original_peak': peak, 'crest_db': float(20*np.log10(max(peak, 1e-9)/max(rms, 1e-9))),
                    'boundary_transients': transients})
                counts[gender] += 1
                manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f'{gender}: {name}, {len(mono)/rate:.2f}s, RMS {rms:.4f}, boundary transients {len(transients)}', flush=True)
                if all(counts[gender] >= target for gender, target in targets.items()):
                    report['status'] = 'ready'
                    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                    print(f'Corpus ready: {manifest}', flush=True)
                    return
    raise RuntimeError('Archive ended before the declared sample quotas were met.')


if __name__ == '__main__':
    main()
