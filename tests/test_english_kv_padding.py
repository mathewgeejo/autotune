import numpy as np
import pytest

from scripts.english_kv_padding import pad_incremental_cache, compact_present


@pytest.mark.parametrize('window', [None, 2])
def test_masked_padding_preserves_attention_values_positions_and_compact_cache(window):
    random = np.random.RandomState(42)
    key = random.randn(1, 2, 3, 4).astype(np.float32)
    value = random.randn(1, 2, 3, 4).astype(np.float32)
    query, current_key, current_value = [random.randn(1, 2, 1, 4).astype(np.float32) for _ in range(3)]
    feeds = {'inputs_embeds': np.ones((1, 1, 8), dtype=np.float32),
             'position_ids': np.array([[3]], dtype=np.int64),
             'attention_mask': np.array([[1, 0, 1, 1]], dtype=np.int64),
             'past_key_values.0.key': key, 'past_key_values.0.value': value}
    padded, restore = pad_incremental_cache(feeds, 8)
    original_keys = np.concatenate((key, current_key), axis=2)
    original_values = np.concatenate((value, current_value), axis=2)
    padded_keys = np.concatenate((padded['past_key_values.0.key'], current_key), axis=2)
    padded_values = np.concatenate((padded['past_key_values.0.value'], current_value), axis=2)
    def attention(keys, values, mask):
        scores = query@keys.transpose(0, 1, 3, 2)
        allowed = mask[:, None, None, :] != 0
        if window:
            allowed = allowed & (np.arange(keys.shape[2]) >= keys.shape[2]-window)
        scores = np.where(allowed, scores, -np.inf)
        probabilities = np.exp(scores-np.max(scores, axis=-1, keepdims=True))
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        return probabilities@values
    assert np.allclose(attention(original_keys, original_values, feeds['attention_mask']),
                       attention(padded_keys, padded_values, padded['attention_mask']), atol=1e-6)
    logits = np.array([[1, 2]], dtype=np.float32)
    compact = compact_present([logits, padded_keys, padded_values], restore)
    assert np.array_equal(compact[1], original_keys)
    assert np.array_equal(compact[2], original_values)
    assert padded['position_ids'] is feeds['position_ids']
    assert np.array_equal(feeds['attention_mask'], [[1, 0, 1, 1]])
    assert key.shape[2] == value.shape[2] == 3


def test_prefill_and_capacity_limit_keep_original_feeds():
    feeds = {'inputs_embeds': np.ones((1, 4, 8))}
    assert pad_incremental_cache(feeds, 8) == (feeds, None)
    feeds = {'inputs_embeds': np.ones((1, 1, 8)), 'past_key_values.0.key': np.ones((1, 2, 8, 4))}
    assert pad_incremental_cache(feeds, 8) == (feeds, None)


def test_incompatible_attention_mask_is_rejected_before_inference():
    feeds = {'inputs_embeds': np.ones((1, 1, 8)), 'past_key_values.0.key': np.ones((1, 2, 3, 4)),
             'attention_mask': np.ones((1, 3))}
    with pytest.raises(ValueError, match='attention mask'):
        pad_incremental_cache(feeds, 8)
