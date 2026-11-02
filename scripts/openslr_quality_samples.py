"""Fetch small attributed Malayalam samples using ZIP ranges, not the full corpus."""
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.runtime import keep_awake
import soundfile as sf

BASE = 'https://www.openslr.org/resources/63/'


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url, self.position, self.transferred = url, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
            self.length = int(response.headers['Content-Length'])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position+offset if whence == 1 else self.length+offset
        if position < 0:
            raise ValueError('Negative ZIP position')
        self.position = position
        return position

    def read(self, size=-1):
        size = min(size if size >= 0 else self.length-self.position, self.length-self.position)
        if size <= 0:
            return b''
        if size > 4*1024**2:
            raise ValueError('Refusing an unexpectedly large ZIP range')
        start, end = self.position, self.position+size-1
        request = urllib.request.Request(self.url, headers={'Range':f'bytes={start}-{end}', 'User-Agent':'VoiceStudio-Quality/1.0'})
        with urllib.request.urlopen(request, timeout=40) as response:
            expected = f'bytes {start}-{end}/{self.length}'
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                raise ValueError('Server did not honor the requested bounded byte range')
            content = response.read(size+1)
        if len(content) != size:
            raise ValueError('ZIP range response was truncated')
        self.position += size
        self.transferred += size
        return content


def main():
    output = ROOT/'data/quality/corpus/openslr63-male'
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'manifest.json'
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if previous.get('status') == 'ready':
            print('OpenSLR quality samples already present.')
            return
    with urllib.request.urlopen(BASE+'line_index_male.tsv', timeout=30) as response:
        rows = dict(line.split('\t', 1) for line in response.read().decode('utf-8').splitlines() if '\t' in line)
    with urllib.request.urlopen(BASE+'LICENSE', timeout=30) as response:
        (output/'LICENSE').write_bytes(response.read())
    report = {'dataset':'OpenSLR SLR63', 'source':'https://www.openslr.org/63/', 'license':'CC-BY-SA-4.0',
              'copyright':'2018, 2019 Google, Inc.', 'status':'collecting', 'clips':[],
              'selection':'First two distinct filename groups with audio duration 5–9.8 seconds; no clone-result filtering.',
              'identity_note':'Filename groups are inferred from anonymized IDs, not independently verified speaker identities.'}
    groups = set()
    with keep_awake(), RemoteZip(BASE+'ml_in_male.zip') as remote:
        with zipfile.ZipFile(remote) as archive:
            for info in archive.infolist():
                name = Path(info.filename).name
                stem = Path(name).stem
                parts = stem.split('_')
                if stem not in rows or len(parts) != 3 or parts[1] in groups or not 150000 <= info.file_size <= 500000:
                    continue
                content = archive.read(info)  # zipfile validates the stored CRC.
                samples, rate = sf.read(io.BytesIO(content), dtype='float32')
                duration = len(samples)/rate
                if not 5 <= duration <= 9.8:
                    continue
                path = output/name
                path.write_bytes(content)
                path.with_suffix('.txt').write_text(rows[stem], encoding='utf-8')
                groups.add(parts[1])
                report['clips'].append({'id':stem,'filename_group':parts[1],'audio':path.relative_to(ROOT).as_posix(),
                    'caption':path.with_suffix('.txt').relative_to(ROOT).as_posix(),'text':rows[stem],
                    'duration_seconds':duration,'sample_rate':rate,'sha256':hashlib.sha256(content).hexdigest()})
                print(f'Selected {stem}, {duration:.2f}s', flush=True)
                if len(groups) == 2:
                    break
        report.update(status='ready' if len(groups) == 2 else 'incomplete', downloaded_bytes=remote.transferred,
                      full_archive_bytes=remote.length)
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if report['status'] != 'ready':
        raise RuntimeError('Could not collect both declared filename groups')
    print(f"Downloaded {report['downloaded_bytes']} bytes of a {report['full_archive_bytes']}-byte archive", flush=True)


if __name__ == '__main__':
    main()
