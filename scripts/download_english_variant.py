"""Fetch and verify one pinned English FP32 component for isolated quality trials."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.models import MODEL_ID, REVISION
from studio.runtime import keep_awake
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=['speech_encoder', 'conditional_decoder', 'language_model', 'embed_tokens'], required=True)
    parser.add_argument('--manifest', type=Path, required=True, help='Pinned repository file metadata including LFS SHA256 hashes.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path.')
    entries = {entry['path']: entry for entry in json.loads(args.manifest.read_text(encoding='utf-8'))}
    names = [f'onnx/{args.component}.onnx', f'onnx/{args.component}.onnx_data']
    for name in names:
        if not entries.get(name, {}).get('lfs', {}).get('oid'):
            parser.error(f'Pinned LFS hash is missing for {name}.')
    report = {'status': 'downloading', 'model': MODEL_ID, 'revision': REVISION,
              'scope': 'Isolated quality experiment; application defaults unchanged.', 'files': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(args.output)
    save()
    try:
        with keep_awake():
            for name in names:
                print('Checking pinned component:', name, flush=True)
                path = Path(hf_hub_download(MODEL_ID, name, revision=REVISION))
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(8*1024**2), b''):
                        digest.update(block)
                entry = entries[name]
                if path.stat().st_size != entry['size'] or digest.hexdigest() != entry['lfs']['oid']:
                    raise ValueError(f'Pinned component size/hash mismatch: {name}')
                report['files'].append({'name': name, 'path': str(path), 'bytes': path.stat().st_size,
                                        'sha256': digest.hexdigest()})
                save()
        report['status'] = 'verified'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    print('Pinned component verified.', flush=True)


if __name__ == '__main__':
    main()
