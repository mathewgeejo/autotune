"""Download the pinned ONNX candidate without unrelated precision variants."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.runtime import keep_awake
from huggingface_hub import snapshot_download

MODEL = 'wavekat/Qwen3-TTS-0.6B-Base-ONNX'
REVISION = '177cd39e2ddab0d4d7ce5152a7abc2d382621158'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path.')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if manifest.get('sha') != REVISION or manifest.get('id') != MODEL:
        parser.error('The candidate manifest must match the reviewed model revision.')
    roots = {'config.json', 'README.md', 'requirements.txt', 'generate_clone_onnx.py',
             'speaker_encoder.onnx', 'speaker_encoder.onnx.data',
             'tokenizer_encoder.onnx', 'tokenizer_encoder.onnx.data'}
    selected = [item for item in manifest['siblings'] if item['rfilename'] in roots or
                item['rfilename'].startswith(('int4/', 'embeddings/', 'tokenizer/'))]
    if any(item['rfilename'].startswith('fp32/') for item in selected):
        raise ValueError('An unselected precision variant entered the download.')
    report = {'status': 'downloading', 'model': MODEL, 'revision': REVISION, 'variant': 'int4',
              'selected_bytes': sum(item['size'] for item in selected), 'files': [],
              'scope': 'Separate quality candidate; app engine and runtime are unchanged.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(args.output)
    save()
    try:
        with keep_awake():
            root = Path(snapshot_download(MODEL, revision=REVISION,
                allow_patterns=[item['rfilename'] for item in selected], max_workers=2))
            report['root'] = str(root)
            for item in selected:
                path = root/item['rfilename']
                checksum = hashlib.sha256()
                git_blob = hashlib.sha1(b'blob '+str(item['size']).encode()+b'\0')
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(8*1024**2), b''):
                        checksum.update(block)
                        if not item.get('lfs'):
                            git_blob.update(block)
                expected = item.get('lfs', {}).get('sha256')
                if (path.stat().st_size != item['size'] or
                        expected and checksum.hexdigest() != expected or
                        not expected and git_blob.hexdigest() != item['blobId']):
                    raise ValueError(f'Pinned file verification failed: {item["rfilename"]}')
                report['files'].append({'name': item['rfilename'], 'bytes': item['size'],
                                        'sha256': checksum.hexdigest()})
                save()
        report['status'] = 'verified'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    print('Pinned candidate files verified.', flush=True)


if __name__ == '__main__':
    main()
