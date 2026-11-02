import hashlib
import json

import numpy as np
import pytest
from ctranslate2.specs import model_spec

from scripts.compact_malayalam_asr import convert


class FixtureSpec(model_spec.ModelSpec):
    """Use the official serializer, independently of the candidate binary writer."""
    def __init__(self, variables):
        super().__init__()
        self._fixture_variables = variables

    @property
    def name(self):
        return 'WhisperSpec'

    @property
    def revision(self):
        return 3

    def variables(self, prefix='', ordered=False):
        values = {key: value if isinstance(value, str) else model_spec.NumpyVariable(value)
                  for key, value in self._fixture_variables.items()}
        return sorted(values.items()) if ordered else values


class OfficialWeights(model_spec.LayerSpec):
    def __init__(self, value):
        self.weight = model_spec.NumpyVariable(value.copy())
        self.weight_scale = model_spec.OPTIONAL


def make_source(folder):
    random = np.random.RandomState(45)
    convolution = random.normal(size=(7, 2, 3)).astype(np.float32)
    embedding = random.normal(size=(13, 4)).astype(np.float32)
    convolution[4] = 0
    embedding[5] = 0
    variables = {'encoder/num_heads': np.array(1, dtype=np.int16),
                 'decoder/num_heads': np.array(1, dtype=np.int16),
                 'encoder/conv1/weight': convolution,
                 'encoder/conv1/bias': random.normal(size=7).astype(np.float32),
                 'decoder/embeddings/weight': embedding,
                 'decoder/projection/weight': 'decoder/embeddings/weight'}
    folder.mkdir()
    FixtureSpec(variables)._serialize(str(folder/'model.bin'))
    (folder/'config.json').write_text('{"test": true}', encoding='utf-8')
    return variables, hashlib.sha256((folder/'model.bin').read_bytes()).hexdigest()


@pytest.mark.parametrize('keep_convolutions_fp32', [False, True])
def test_chunked_checkpoint_is_byte_identical_to_official_whole_tensor_quantization(tmp_path, keep_convolutions_fp32):
    source, output = tmp_path/'original', tmp_path/'compact'
    variables, digest = make_source(source)
    original_bytes = (source/'model.bin').read_bytes()
    expected = dict(variables)
    for key in ('encoder/conv1/weight', 'decoder/embeddings/weight'):
        if keep_convolutions_fp32 and key.startswith('encoder/conv'):
            continue
        layer = OfficialWeights(variables[key])
        layer._quantize('int8')
        expected[key] = layer.weight.numpy()
        expected[key+'_scale'] = layer.weight_scale.numpy()
    golden = tmp_path/'official.bin'
    FixtureSpec(expected)._serialize(str(golden))
    report = convert(source, output, digest, chunk_rows=3, keep_convolutions_fp32=keep_convolutions_fp32)
    assert (output/'model.bin').read_bytes() == golden.read_bytes()
    assert (source/'model.bin').read_bytes() == original_bytes
    assert (output/'config.json').read_bytes() == (source/'config.json').read_bytes()
    assert report['source_sha256'] == digest
    assert report['state'] == 'generated-unvalidated'
    assert report['quantized_tensors'] == (1 if keep_convolutions_fp32 else 2)


def test_wrong_source_hash_never_publishes_a_candidate_or_modifies_original(tmp_path):
    source, output = tmp_path/'original', tmp_path/'compact'
    _, digest = make_source(source)
    with pytest.raises(ValueError, match='SHA256'):
        convert(source, output, '0'*64, chunk_rows=3)
    assert not (output/'model.bin').exists()
    assert hashlib.sha256((source/'model.bin').read_bytes()).hexdigest() == digest
    assert json.loads((output/'conversion.json').read_text())['state'] == 'failed'


def test_existing_destination_is_never_overwritten(tmp_path):
    source, output = tmp_path/'original', tmp_path/'compact'
    _, digest = make_source(source)
    output.mkdir()
    protected = output/'model.bin'
    protected.write_bytes(b'preserve this checkpoint')
    with pytest.raises(FileExistsError):
        convert(source, output, digest)
    assert protected.read_bytes() == b'preserve this checkpoint'
