"""Reproducible acoustic diagnostics, with no synthesis-model quality claims.

Pitch uses normalized square-difference autocorrelation (McLeod-style peak
picking). Same-text comparisons use constrained dynamic time warping of MFCCs.
Different words are compared statistically: they cannot be aligned as errors.
"""
import csv
import json
from pathlib import Path
import unicodedata

import numpy as np
from scipy.fft import dct
from scipy.spatial.distance import cdist

from studio.audio import AudioError, decode

ANALYSIS_RATE = 16000
MAX_SECONDS = 60


def db(value):
    return float(20 * np.log10(max(float(value), 1e-10)))


def mel_filters(rate, nfft, count=40):
    frequencies = np.fft.rfftfreq(nfft, 1 / rate)
    mel = lambda hz: 2595 * np.log10(1 + np.asarray(hz) / 700)
    hz = lambda m: 700 * (10 ** (np.asarray(m) / 2595) - 1)
    bands = hz(np.linspace(mel(60), mel(min(7600, rate / 2)), count + 2))
    filters = []
    for left, center, right in zip(bands, bands[1:], bands[2:]):
        filters.append(np.maximum(0, np.minimum((frequencies-left)/(center-left),
                                                (right-frequencies)/(right-center))))
    return np.asarray(filters, dtype=np.float32)


def pitch_track(frames, rate, rms):
    """Return F0 and periodicity; unvoiced frames have F0=0, never NaN."""
    size = frames.shape[1]
    lags = np.arange(int(rate / 600), int(rate / 60) + 1)
    f0 = np.zeros(len(frames), dtype=np.float32)
    clarity = np.zeros_like(f0)
    floor = max(0.002, float(np.percentile(rms, 85)) * 0.05)
    for begin in range(0, len(frames), 256):
        x = frames[begin:begin+256].astype(np.float64)
        x -= x.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(x, n=2*size, axis=1)
        ac = np.fft.irfft(np.abs(spectrum)**2, n=2*size, axis=1)[:, :size]
        energy = np.cumsum(x*x, axis=1)
        denominator = energy[:, -1:] - energy[:, lags-1] + energy[:, size-lags-1]
        nsdf = 2 * ac[:, lags] / np.maximum(denominator, 1e-12)
        peaks = np.zeros_like(nsdf, dtype=bool)
        peaks[:, 1:-1] = (nsdf[:, 1:-1] > nsdf[:, :-2]) & (nsdf[:, 1:-1] >= nsdf[:, 2:])
        best = np.max(np.where(peaks, nsdf, 0), axis=1)
        # Prefer the first strong period to its later integer multiples.
        eligible = peaks & (nsdf >= best[:, None]*0.90) & (nsdf > 0.65)
        chosen = np.argmax(eligible, axis=1)
        for row, index in enumerate(chosen):
            if not eligible[row, index] or rms[begin+row] < floor:
                continue
            left, center, right = nsdf[row, index-1:index+2]
            denominator = left - 2*center + right
            offset = 0.5*(left-right)/denominator if abs(denominator) > 1e-10 else 0
            f0[begin+row] = rate / (lags[index] + np.clip(offset, -0.5, 0.5))
            clarity[begin+row] = center
    return f0, clarity


def acoustic_profile(samples, rate=ANALYSIS_RATE):
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise AudioError('Analysis needs nonempty audio containing finite samples.')
    if len(samples) / rate > MAX_SECONDS + 0.02:
        raise AudioError('Select at most 60 seconds for each analysis clip.')
    window, hop = int(rate * 0.05), int(rate * 0.01)
    padded = np.pad(samples, (window//2, window//2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, window)[::hop]
    rms = np.sqrt(np.mean(frames*frames, axis=1))
    f0, clarity = pitch_track(frames, rate, rms)
    nfft = 1024
    power = np.abs(np.fft.rfft(frames*np.hanning(window), n=nfft, axis=1))**2
    frequencies = np.fft.rfftfreq(nfft, 1/rate)
    totals = np.maximum(power.sum(axis=1), 1e-12)
    centroids = (power @ frequencies) / totals
    mel = np.maximum(power @ mel_filters(rate, nfft).T, 1e-10)
    logmel = np.log(mel)
    mfcc = dct(logmel, type=2, axis=1, norm='ortho')[:, 1:14]
    voiced = f0[f0 > 0]
    peak = float(np.max(np.abs(samples)))
    sample_rms = float(np.sqrt(np.mean(samples*samples)))
    silence = rms < max(0.00316, float(np.max(rms))*0.01)
    lufs = None
    if len(samples) >= int(rate*0.4) and sample_rms > 1e-8:
        import pyloudnorm as pyln
        measured = float(pyln.Meter(rate).integrated_loudness(samples.astype(np.float64)))
        lufs = measured if np.isfinite(measured) else None
    metrics = {
        'duration_seconds': len(samples)/rate, 'sample_rate': rate,
        'peak_dbfs': db(peak), 'rms_dbfs': db(sample_rms), 'loudness_lufs': lufs,
        'crest_factor_db': db(peak)-db(sample_rms), 'dc_offset': float(samples.mean()),
        'clipped_percent': float(np.mean(np.abs(samples) >= 0.999)*100),
        'silence_percent': float(silence.mean()*100),
        'voiced_percent': float(np.mean(f0 > 0)*100),
        'median_pitch_hz': float(np.median(voiced)) if len(voiced) else None,
        'pitch_p10_hz': float(np.percentile(voiced, 10)) if len(voiced) else None,
        'pitch_p90_hz': float(np.percentile(voiced, 90)) if len(voiced) else None,
        'spectral_centroid_hz': float(np.median(centroids[~silence])) if np.any(~silence) else None,
        'rms_dynamic_range_db': db(np.percentile(rms, 90))-db(np.percentile(rms, 10)),
    }
    warnings = []
    if metrics['clipped_percent'] > 0.1:
        warnings.append('Clipping detected; use a quieter recording level.')
    if metrics['rms_dbfs'] < -35:
        warnings.append('Very quiet recording; noise can dominate the voice characteristics.')
    if metrics['silence_percent'] > 35:
        warnings.append('Long silent regions; select a section with continuous speech.')
    if abs(metrics['dc_offset']) > 0.03:
        warnings.append('DC offset detected in the recording.')
    if metrics['voiced_percent'] < 10:
        warnings.append('Little periodic voicing detected; speech may be noisy, whispered, or absent.')
    return {'metrics': metrics, 'warnings': warnings,
            'time': np.arange(len(frames))*hop/rate, 'rms_dbfs': 20*np.log10(np.maximum(rms, 1e-10)),
            'pitch_hz': f0, 'periodicity': clarity, 'mfcc': mfcc,
            'mean_spectrum_db': 10*np.log10(np.maximum(power.mean(axis=0), 1e-10)),
            'frequency_hz': frequencies,
            'wave_time': np.arange(0, len(samples), max(1, len(samples)//3000))/rate,
            'waveform': samples[::max(1, len(samples)//3000)]}


def align_same_words(reference, candidate):
    # At most 601 x 601 cells for minute-long clips. A band prevents implausible
    # stretching while retaining different speaking rates and natural pauses.
    a, b = reference['mfcc'][::10], candidate['mfcc'][::10]
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    distances = cdist(a, b, metric='euclidean')
    n, m = distances.shape
    cost = np.full((n+1, m+1), np.inf)
    cost[0, 0] = 0
    path_choice = np.zeros((n, m), dtype=np.uint8)
    for i in range(1, n+1):
        center = i*m/n
        band = max(20, int(m*0.25))
        for j in range(max(1, int(center-band)), min(m, int(center+band))+1):
            choices = (cost[i-1, j-1], cost[i-1, j], cost[i, j-1])
            move = int(np.argmin(choices))
            cost[i, j] = distances[i-1, j-1] + choices[move]
            path_choice[i-1, j-1] = move
    if not np.isfinite(cost[n, m]):
        return {'available': False, 'reason': 'Speaking rates could not be aligned.'}
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        move = path_choice[i-1, j-1]
        i -= move != 2
        j -= move != 1
    path.reverse()
    rows = []
    for i, j in path:
        ref_f0, out_f0 = reference['pitch_hz'][i*10], candidate['pitch_hz'][j*10]
        rows.append({'reference_seconds': float(reference['time'][i*10]),
                     'output_seconds': float(candidate['time'][j*10]),
                     'spectral_distance': float(distances[i, j]),
                     'pitch_difference_cents': float(1200*np.log2(out_f0/ref_f0))
                     if ref_f0 > 0 and out_f0 > 0 else None})
    cents = [abs(row['pitch_difference_cents']) for row in rows if row['pitch_difference_cents'] is not None]
    return {'available': True, 'method': 'MFCC dynamic time warping, same spoken words only',
            'mean_spectral_distance': float(np.mean([row['spectral_distance'] for row in rows])),
            'median_absolute_pitch_difference_cents': float(np.median(cents)) if cents else None,
            'alignment': rows,
            'largest_spectral_differences': sorted(rows, key=lambda row: row['spectral_distance'], reverse=True)[:15]}


def normalize_text(text):
    text = unicodedata.normalize('NFC', text).casefold()
    return ' '.join(''.join(ch if unicodedata.category(ch)[0] in 'LMN' else ' '
                            for ch in text).split())


def edit_distance(expected, actual):
    previous = list(range(len(actual)+1))
    for i, item in enumerate(expected, 1):
        current = [i]
        for j, other in enumerate(actual, 1):
            current.append(min(previous[j]+1, current[-1]+1, previous[j-1]+(item != other)))
        previous = current
    return previous[-1]


def transcript_errors(expected, actual):
    import difflib
    import regex
    expected, actual = normalize_text(expected), normalize_text(actual)
    words_a, words_b = expected.split(), actual.split()
    chars_a, chars_b = regex.findall(r'\X', expected), regex.findall(r'\X', actual)
    if not words_a:
        return {'available': False, 'reason': 'Enter the expected words to measure content errors.'}
    edits = []
    for action, i, j, k, l in difflib.SequenceMatcher(None, words_a, words_b, autojunk=False).get_opcodes():
        if action != 'equal':
            edits.append({'type': action, 'expected': ' '.join(words_a[i:j]),
                          'recognized': ' '.join(words_b[k:l]), 'recognized_word_index': k})
    return {'available': True, 'word_error_rate': edit_distance(words_a, words_b)/len(words_a),
            'grapheme_error_rate': edit_distance(chars_a, chars_b)/max(1, len(chars_a)),
            'expected_normalized': expected, 'recognized_normalized': actual, 'differences': edits,
            'note': 'ASR is fallible. Numbers, abbreviations, accents, and code-switching can produce recognition errors.'}


def compare_profiles(reference, candidate, same_words=False):
    a, b = reference['metrics'], candidate['metrics']
    def difference(key):
        return b[key]-a[key] if a[key] is not None and b[key] is not None else None
    result = {'reference': a, 'output': b,
              'differences': {'loudness_lu': difference('loudness_lufs'),
                              'median_pitch_hz': difference('median_pitch_hz'),
                              'silence_percentage_points': difference('silence_percent'),
                              'spectral_centroid_hz': difference('spectral_centroid_hz')},
              'reference_warnings': reference['warnings'], 'output_warnings': candidate['warnings'],
              'same_spoken_words': bool(same_words),
              'interpretation': 'Acoustic differences are not a probability of speaker identity. '
                  'Different words naturally have different pitch, timing, and spectra.'}
    result['aligned_comparison'] = align_same_words(reference, candidate) if same_words else {
        'available': False, 'reason': 'Time alignment is disabled because the clips contain different words.'}
    return result


def analyze_files(reference_path, output_path, folder, same_words=False):
    import soundfile as sf
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    profiles = []
    for name, source in [('reference', reference_path), ('output', output_path)]:
        path = decode(source, folder/f'analysis-{name}.wav', duration=MAX_SECONDS+0.1, rate=ANALYSIS_RATE)
        samples, rate = sf.read(path, dtype='float32')
        profile = acoustic_profile(samples, rate)
        profiles.append(profile)
    result = compare_profiles(*profiles, same_words=same_words)
    result['schema_version'] = 1
    target = folder/'analysis.json'
    target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    csv_path = folder/'analysis-frames.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        fields = ['clip', 'time_seconds', 'pitch_hz', 'rms_dbfs', 'periodicity']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, profile in zip(('reference', 'output'), profiles):
            for t, pitch, level, clarity in zip(profile['time'], profile['pitch_hz'], profile['rms_dbfs'], profile['periodicity']):
                writer.writerow(dict(zip(fields, (name, float(t), float(pitch), float(level), float(clarity)))))
    return result, profiles, str(target), str(csv_path)


def comparison_figure(profiles):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    figure = make_subplots(rows=3, cols=1, subplot_titles=(
        'Pitch and voiced regions', 'Energy through time', 'Average spectrum'))
    for name, color, profile in zip(('Reference', 'Generated'), ('#a78bfa', '#34d399'), profiles):
        pitch = np.where(profile['pitch_hz'] > 0, profile['pitch_hz'], np.nan)
        for row, x, y in ((1, profile['time'], pitch),
                          (2, profile['time'], profile['rms_dbfs']),
                          (3, profile['frequency_hz'], profile['mean_spectrum_db'])):
            figure.add_trace(go.Scatter(x=x, y=y, name=name, legendgroup=name,
                showlegend=row == 1, line={'color': color, 'width': 1.5}, connectgaps=False), row=row, col=1)
    figure.update_layout(height=690, template='plotly_dark', margin={'l': 45, 'r': 20, 't': 45, 'b': 25},
                         paper_bgcolor='#151824', plot_bgcolor='#0f121d', hovermode='x unified')
    figure.update_yaxes(title_text='Hz', row=1, col=1)
    figure.update_yaxes(title_text='dBFS', row=2, col=1)
    figure.update_yaxes(title_text='dB power', row=3, col=1)
    figure.update_xaxes(title_text='Seconds', row=2, col=1)
    figure.update_xaxes(title_text='Frequency (Hz)', row=3, col=1)
    return figure
