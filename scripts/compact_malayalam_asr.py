"""Create an isolated INT8 recognizer candidate from cached FP32 CTranslate2 weights.

Uses the installed official quantizer in bounded row chunks. Source files and app
selection are untouched. A generated candidate still requires recognition parity
and timing checks; this script does not make it the app's recognizer.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
import ctranslate2
from ctranslate2.specs import model_spec, whisper_spec
from studio.runtime import active_time, keep_awake

DTYPES = {0: '<f4', 1: 'i1', 2: '<i2', 3: '<i4', 4: '<f2'}


class Reader:
    def __init__(self, file, digest=None):
        self.file, self.digest = file, digest

    def read(self, count):
        data = self.file.read(count)
        if len(data) != count:
            raise ValueError('Truncated CTranslate2 model')
        if self.digest is not None:
            self.digest.update(data)
        return data

    def number(self, fmt):
        return struct.unpack('<'+fmt, self.read(struct.calcsize('<'+fmt)))[0]

    def string(self):
        data = self.read(self.number('H'))
        if not data.endswith(b'\0'):
            raise ValueError('Invalid model string')
        return data[:-1].decode('utf-8')


def write_string(file, value):
    raw = value.encode('utf-8')+b'\0'
    file.write(struct.pack('<H', len(raw))+raw)


def write_tensor(file, name, shape, dtype, count):
    write_string(file, name)
    file.write(struct.pack('<B', len(shape)))
    for dimension in shape:
        file.write(struct.pack('<I', dimension))
    file.write(struct.pack('<BI', dtype, count))


def index_model(path):
    tensors, aliases = [], []
    with path.open('rb') as file:
        reader = Reader(file)
        header = (reader.number('I'), reader.string(), reader.number('I'))
        if header != (model_spec.CURRENT_BINARY_VERSION, 'WhisperSpec', 3) or header[0] != 6:
            raise ValueError('Only the inspected binary-v6 WhisperSpec-revision-3 layout is supported')
        for _ in range(reader.number('I')):
            name = reader.string()
            shape = [reader.number('I') for _ in range(reader.number('B'))]
            dtype, size = reader.number('B'), reader.number('I')
            if dtype not in DTYPES or size != math.prod(shape)*np.dtype(DTYPES[dtype]).itemsize:
                raise ValueError(f'Unsupported tensor layout: {name}')
            offset = file.tell()
            tensors.append({'name': name, 'shape': shape, 'dtype': dtype, 'bytes': size, 'offset': offset})
            file.seek(size, 1)
        for _ in range(reader.number('I')):
            aliases.append((reader.string(), reader.string()))
        if file.tell() != path.stat().st_size:
            raise ValueError('Unexpected trailing data')
    return header, tensors, aliases


def quantizable_tensors(path, tensors):
    def heads(side):
        tensor = next(item for item in tensors if item['name'] == side+'/num_heads')
        with path.open('rb') as file:
            file.seek(tensor['offset'])
            return int(np.frombuffer(file.read(tensor['bytes']), dtype=DTYPES[tensor['dtype']])[0])
    def layers(side):
        indices = [int(match[1]) for item in tensors
                   if (match := re.match(side+r'/layer_(\d+)/', item['name']))]
        return max(indices, default=-1)+1
    spec = whisper_spec.WhisperSpec(layers('encoder'), heads('encoder'), layers('decoder'), heads('decoder'))
    selected = set()
    for tensor in tensors:
        scope, key = tensor['name'].rsplit('/', 1)
        owner = model_spec.index_spec(spec, scope)
        if len(tensor['shape']) > 1 and hasattr(owner, key+'_scale'):
            if tensor['dtype'] != 0:
                raise ValueError('The source must contain original FP32 quantizable weights')
            selected.add(tensor['name'])
    if not selected:
        raise ValueError('No quantizable weights found')
    return selected


class RowWeights(model_spec.LayerSpec):
    def __init__(self, rows):
        # The official quantizer updates FP32 values in place. Own this bounded
        # chunk; source bytes and read-only frombuffer views must stay intact.
        self.weight = model_spec.NumpyVariable(rows.copy())
        self.weight_scale = model_spec.OPTIONAL


def convert(source, output, expected_sha256, chunk_rows=128):
    if not 1 <= chunk_rows <= 1024 or not re.fullmatch(r'[a-f0-9]{64}', expected_sha256):
        raise ValueError('A valid source SHA256 and 1-1024 chunk rows are required')
    source = source.resolve(strict=True)
    model = source/'model.bin'
    header, tensors, aliases = index_model(model)
    selected = quantizable_tensors(model, tensors)
    output.mkdir(parents=True, exist_ok=False)
    report = {'state': 'converting', 'source': str(model), 'expected_source_sha256': expected_sha256,
              'ctranslate2_version': ctranslate2.__version__, 'source_header': header,
              'quantized_tensors': len(selected), 'chunk_rows': chunk_rows,
              'algorithm': 'Official LayerSpec._quantize(int8), independent row chunks',
              'scope': 'Experimental derived checkpoint; no app adoption or recognition parity claim.'}
    manifest = output/'conversion.json'
    def save():
        manifest.write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    started = active_time()
    digest = hashlib.sha256()
    temporary = output/'model.bin.partial'
    try:
        with model.open('rb') as original, temporary.open('wb') as target:
            reader = Reader(original, digest)
            target.write(struct.pack('<I', reader.number('I')))
            write_string(target, reader.string())
            target.write(struct.pack('<II', reader.number('I'), reader.number('I')+len(selected)))
            for tensor in tensors:
                name = reader.string()
                shape = [reader.number('I') for _ in range(reader.number('B'))]
                dtype, size = reader.number('B'), reader.number('I')
                if (name, shape, dtype, size) != (tensor['name'], tensor['shape'], tensor['dtype'], tensor['bytes']):
                    raise ValueError('Source tensor metadata changed')
                if name in selected:
                    write_tensor(target, name, shape, 1, size//4)
                    scales = []
                    row_size = math.prod(shape[1:])
                    for row in range(0, shape[0], chunk_rows):
                        count = min(chunk_rows, shape[0]-row)
                        values = np.frombuffer(reader.read(count*row_size*4), dtype='<f4').reshape(count, *shape[1:])
                        layer = RowWeights(values)
                        layer._quantize('int8')
                        target.write(layer.weight.to_bytes())
                        scales.append(layer.weight_scale.numpy())
                    scale = np.concatenate(scales).astype('<f4', copy=False)
                    write_tensor(target, name+'_scale', [shape[0]], 0, scale.nbytes)
                    target.write(scale.tobytes())
                else:
                    write_tensor(target, name, shape, dtype, size)
                    remaining = size
                    while remaining:
                        block = reader.read(min(4*1024*1024, remaining))
                        target.write(block)
                        remaining -= len(block)
            count = reader.number('I')
            target.write(struct.pack('<I', count))
            for alias, variable in aliases:
                if (reader.string(), reader.string()) != (alias, variable):
                    raise ValueError('Source aliases changed')
                write_string(target, alias)
                write_string(target, variable)
            if count != len(aliases) or original.read(1):
                raise ValueError('Unexpected source aliases or trailing data')
        report['source_sha256'] = digest.hexdigest()
        if report['source_sha256'] != expected_sha256:
            raise ValueError('Source SHA256 does not match the pinned expected checkpoint')
        target_digest = hashlib.sha256()
        with temporary.open('rb') as file:
            for block in iter(lambda: file.read(4*1024*1024), b''):
                target_digest.update(block)
        report.update(model_sha256=target_digest.hexdigest(), model_bytes=temporary.stat().st_size,
                      elapsed_seconds=active_time()-started, state='generated-unvalidated')
        for name in ('config.json', 'tokenizer.json', 'vocabulary.txt', 'README.md'):
            if (source/name).is_file():
                shutil.copyfile(source/name, output/name)
        temporary.replace(output/'model.bin')
    except Exception as exc:
        report.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    args = parser.parse_args()
    with keep_awake():
        print(json.dumps(convert(args.source, args.output, args.expected_source_sha256), indent=2), flush=True)


if __name__ == '__main__':
    main()
