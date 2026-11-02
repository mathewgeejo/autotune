"""Real synthesis trial with fixed-reference baseline comparisons and human controls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.jobs import JobManager
from studio.recognition import full_audit, speaker_similarity
from studio.runtime import keep_awake
from scripts.smoke import wait
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--reference-text', type=Path)
    words = parser.add_mutually_exclusive_group(required=True)
    words.add_argument('--text')
    words.add_argument('--text-file', type=Path)
    parser.add_argument('--language', choices=['en', 'ml'], required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--baseline-reference', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.text_file:
        args.text = args.text_file.read_text(encoding='utf-8')
    reference_text = args.reference_text.read_text(encoding='utf-8') if args.reference_text else ''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'language': args.language, 'expected': args.text,
              'source_reference': str(args.reference.resolve()),
              'reference_text': reference_text,
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'status': 'running',
              'interpretation': 'Short automated comparisons are not proof of human indistinguishability.'}
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        temporary.replace(args.output)
    save()
    manager = JobManager()
    with keep_awake():
        try:
            job_id = manager.submit('clone', {'source': str(args.reference.resolve()), 'text': args.text,
                'reference_text': reference_text,
                'language': args.language, 'quality': 'Detailed'})
            report['job_id'] = job_id
            save()
            job = wait(manager, job_id, timeout=1800)
            result = job['result']
            report.update(status='generated', result=result, elapsed_seconds=job['elapsed'], peak_worker_mb=job['peak_mb'],
                selection=json.loads((job['folder']/'reference-section.json').read_text(encoding='utf-8')))
            save()
        except Exception as exc:
            report.update(status='failed', error=str(exc))
            save()
            raise
        finally:
            manager.close()
        folder = args.output.parent/args.output.stem
        folder.mkdir(exist_ok=True)
        progress = lambda message: print(message, flush=True)
        clean = result['reference_preview']
        if args.baseline:
            # Both outputs are compared to the identical selected natural clip.
            # The raw-reference view separately exposes evaluator sensitivity.
            report['baseline_against_selected_reference'] = speaker_similarity(clean, args.baseline, folder, progress)
            raw = args.baseline_reference or args.reference
            report['baseline_against_previous_reference'] = speaker_similarity(raw, args.baseline, folder, progress)
            report['candidate_against_previous_reference'] = speaker_similarity(raw, result['audio'], folder, progress)
            report['baseline_audio'] = str(args.baseline.resolve())
            save()
        audio, rate = sf.read(clean, dtype='float32')
        sf.write(folder/'human-first.wav', audio[:len(audio)//2], rate)
        sf.write(folder/'human-second.wav', audio[len(audio)//2:], rate)
        report['same_human_different_words_control'] = speaker_similarity(folder/'human-first.wav',
            folder/'human-second.wav', folder, progress)
        report['candidate_audit'] = full_audit(clean, result['audio'], args.text, folder, args.language, progress)
        from studio.analysis import normalize_text, analyze_files
        if reference_text and normalize_text(reference_text) == normalize_text(args.text):
            report['same_words_acoustics'], _, _, _ = analyze_files(clean, result['audio'], folder/'same-words', same_words=True)
        report['status'] = 'measured'
        save()
        summary = {key: report[key] for key in ('job_id', 'language', 'elapsed_seconds', 'peak_worker_mb', 'selection')}
        summary['candidate_speaker_cosine'] = report['candidate_audit']['speaker_verification'].get('cosine_similarity')
        summary['candidate_content_errors'] = report['candidate_audit']['content_comparison']
        summary['baseline_same_reference_cosine'] = report.get('baseline_against_selected_reference', {}).get('cosine_similarity')
        summary['human_control_cosine'] = report['same_human_different_words_control'].get('cosine_similarity')
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
