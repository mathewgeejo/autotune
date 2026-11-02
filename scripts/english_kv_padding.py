"""Experimental fixed-capacity KV feeds for single-token English decoding."""
import numpy as np


def pad_incremental_cache(feeds, capacity):
    """Use leading masked slots so real history stays adjacent to the current token."""
    if not capacity or feeds['inputs_embeds'].shape[1] != 1:
        return feeds, None
    cache = [key for key in feeds if 'past_key_values' in key]
    if not cache:
        raise ValueError('No KV cache inputs')
    history = feeds[cache[0]].shape[2]
    if not history or history >= capacity:
        return feeds, None
    if feeds['attention_mask'].shape[1] != history+1:
        raise ValueError('The attention mask must include the history and current token')
    result = dict(feeds)
    for key in cache:
        value = feeds[key]
        if value.ndim != 4 or value.shape[2] != history:
            raise ValueError('KV history lengths disagree')
        shape = (*value.shape[:2], capacity, value.shape[3])
        padded = np.zeros(shape, dtype=value.dtype)
        padded[:, :, -history:] = value
        result[key] = padded
    mask = feeds['attention_mask']
    result['attention_mask'] = np.concatenate((
        np.zeros((mask.shape[0], capacity-history), dtype=mask.dtype), mask), axis=1)
    return result, (history, capacity)


def compact_present(values, restore):
    if restore is None:
        return values
    history, capacity = restore
    result = [values[0]]
    for value in values[1:]:
        if value.ndim != 4 or value.shape[2] < history+1:
            raise ValueError(f'Unexpected padded present-cache shape {value.shape}; need {history+1} positions')
        # A sliding cache may trim leading padding. The real history/current
        # token must remain at the tail, with their original relative positions.
        result.append(value[:, :, -(history+1):].copy())
    return result
