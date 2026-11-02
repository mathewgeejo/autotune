"""Prepare blind same-words audio pairs; automatic checks never create listener ratings."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.audio import decode
from studio.recognition import evaluation_level
import soundfile as sf


PAGE = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Voice listening comparison</title>
<style>
body{font:17px/1.55 system-ui,sans-serif;max-width:820px;margin:0 auto;padding:24px;background:#f5f5f3;color:#17201e}
h1{font-size:1.8rem;line-height:1.2}article{background:white;border:1px solid #ccd4ce;border-radius:12px;padding:22px;margin:24px 0}
audio{display:block;width:100%;margin:8px 0 20px}label{display:block;margin:16px 0 4px}select,button{font:inherit;max-width:100%;padding:9px;border:1px solid #64736b;border-radius:5px}
button{background:#174b38;color:white;cursor:pointer}blockquote{margin:16px 0;padding:12px;background:#eff3ef}.muted{color:#4d5c55}
</style><main><h1>Voice listening comparison</h1>
<p>Each pair contains a recording and a synthesis of the same words. Source labels stay hidden on this page.</p>
<p>Use headphones and listen to both clips. Playback levels are matched; pitch and timing are preserved. Rate how closely the voices match and how natural each clip sounds.</p>
<form id="ratings">__CASES__<button type="submit">Download ratings</button></form>
<p class="muted">One listening response provides feedback. It does not establish perceptual indistinguishability across voices or languages.</p>
</main><script>
document.querySelector('#ratings').addEventListener('submit', function(event) {
 event.preventDefault();
 const ratings=Object.fromEntries(new FormData(event.currentTarget));
 const result={pack_id:__PACK_ID__,recorded_at:new Date().toISOString(),ratings:ratings,
   interpretation:'Listener-entered feedback; no automatic perceptual pass is inferred.'};
 const address=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));
 const link=document.createElement('a');link.href=address;link.download='voice-listening-review.json';link.click();
 setTimeout(()=>URL.revokeObjectURL(address),1000);
});
</script></html>'''


def rating(case, name, title, choices):
    options = '<option value="">Choose a rating</option>'+''.join(
        f'<option value="{html.escape(value)}">{html.escape(label)}</option>' for value, label in choices)
    identifier = f'{case}-{name}'
    return f'<label for="{identifier}">{html.escape(title)}</label><select required id="{identifier}" name="{identifier}">{options}</select>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pack_id = secrets.token_hex(8)
    manifest = {'pack_id': pack_id, 'cases': [], 'human_listening_results': [],
                'status': 'prepared-unreviewed', 'purpose': 'Blind feedback; neither automated checks nor pack creation count as human listening.'}
    answer_key, sections = {}, []
    for index, path in enumerate(args.reports, 1):
        report = json.loads(path.read_text(encoding='utf-8'))
        acoustic = report.get('same_words_acoustics', {})
        if report['status'] != 'measured' or not acoustic.get('same_spoken_words'):
            raise ValueError(f'{path}: a measured same-words trial is required')
        case = f'case-{index}'
        folder = args.output/case
        folder.mkdir()
        clips = [('recording', report['result']['reference_preview']), ('synthesis', report['result']['audio'])]
        secrets.SystemRandom().shuffle(clips)
        sources = {}
        for label, (kind, source) in zip(('A', 'B'), clips):
            decoded = decode(source, folder/f'{label}-decoded.wav', rate=24000, floating=True)
            waveform, rate = sf.read(decoded, dtype='float32')
            adjusted, level = evaluation_level(waveform)
            audio = folder/f'{label}.wav'
            sf.write(audio, adjusted, rate, subtype='PCM_16')
            # Keep the original-source mapping outside the blind player.
            sources[label] = {'kind': kind, 'original': str(source), 'playback_level': level,
                              'original_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                              'audio_sha256': hashlib.sha256(audio.read_bytes()).hexdigest()}
            # The intermediate filename does not reveal the source, but removing
            # it also avoids duplicate playback files in the finished pack.
            decoded.unlink()
        answer_key[case] = sources
        language = {'en': 'English', 'ml': 'Malayalam'}.get(report['language'], report['language'])
        heading = f'Pair {index}: {language}'
        section = f'<article><h2>{html.escape(heading)}</h2><blockquote>{html.escape(report["expected"])}</blockquote>'
        for label in ('A', 'B'):
            section += f'<label for="{case}-{label}">Clip {label}</label><audio id="{case}-{label}" controls preload="metadata" src="{case}/{label}.wav"></audio>'
        scale = [(str(value), str(value)) for value in range(1, 6)]
        section += rating(case, 'voice-match', 'Voice match: 1 = different voices, 5 = same voice', scale)
        section += rating(case, 'naturalness-A', 'Clip A naturalness: 1 = artificial, 5 = natural', scale)
        section += rating(case, 'naturalness-B', 'Clip B naturalness: 1 = artificial, 5 = natural', scale)
        section += rating(case, 'synthetic-guess', 'Which clip do you think is synthetic?', [('A', 'Clip A'), ('B', 'Clip B'), ('unsure', 'Unsure')])
        sections.append(section+'</article>')
        manifest['cases'].append({'id': case, 'language': report['language'], 'source_report': str(path),
                                  'text': report['expected'], 'level_matching': 'RMS 0.08, peak cap 0.95; no pitch or timing changes'})
    page = PAGE.replace('__CASES__', ''.join(sections)).replace('__PACK_ID__', json.dumps(pack_id))
    (args.output/'index.html').write_text(page, encoding='utf-8')
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output/'answer-key.json').write_text(json.dumps(answer_key, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Prepared unreviewed listening pack:', args.output/'index.html', flush=True)


if __name__ == '__main__':
    main()
