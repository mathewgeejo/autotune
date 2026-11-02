"""Fetch a tiny, attributed FLEURS evaluation subset without a full dataset download."""
import csv
import json
from pathlib import Path
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()


def fetch(language):
    target = ROOT/'data/smoke'/f'fleurs-{language}'
    target.mkdir(parents=True, exist_ok=True)
    if (target/'reference.json').is_file():
        return
    base = f'https://huggingface.co/datasets/google/fleurs/resolve/main/data/{language}/'
    with urllib.request.urlopen(base+'test.tsv', timeout=60) as response:
        rows = list(csv.reader(response.read().decode('utf-8').splitlines(), delimiter='\t', quoting=csv.QUOTE_NONE))
    # Some language revisions include an extra phoneme column. Frame count and
    # gender are the final two columns in both formats; quotes are literal text.
    eligible = {row[1]: row for row in rows if len(row) >= 6 and row[-2].isdigit()
                and 3 <= int(row[-2])/16000 <= 9.8}
    print(f'{language}: looking for one of {len(eligible)} short native reference clips', flush=True)
    request = urllib.request.Request(base+'audio/test.tar.gz', headers={'User-Agent': 'VoiceStudio-Evaluation/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response:
        with tarfile.open(fileobj=response, mode='r|gz') as archive:
            for member in archive:
                name = Path(member.name).name
                if member.isfile() and name in eligible:
                    row = eligible[name]
                    with archive.extractfile(member) as source:
                        (target/'reference.wav').write_bytes(source.read())
                    (target/'reference.txt').write_text(row[2], encoding='utf-8')
                    (target/'reference.json').write_text(json.dumps({
                        'dataset': 'google/fleurs', 'config': language, 'split': 'test',
                        'file': name, 'duration_seconds': int(row[-2])/16000,
                        'text': row[2], 'license': 'CC-BY-4.0',
                        'source': 'https://huggingface.co/datasets/google/fleurs'},
                        ensure_ascii=False, indent=2), encoding='utf-8')
                    print(f'{language}: saved {name}, {int(row[-2])/16000:.2f}s native speech', flush=True)
                    return
    raise RuntimeError('No eligible reference was found in the archive.')


if __name__ == '__main__':
    for language in ('hi_in', 'ml_in'):
        fetch(language)
