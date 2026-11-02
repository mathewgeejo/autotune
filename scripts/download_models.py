"""Cache the actual voice-cloning models before starting the app."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio.paths import configure
configure()
from studio.models import download_models, download_indic_models, download_asr, download_speaker

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Cache all speech engines, quality modes, and audit models (about 6.5 GB).')
    parser.add_argument('--language', choices=['en', 'hi', 'ml'], default='en')
    parser.add_argument('--quality', choices=['Balanced', 'Detailed'], default='Detailed')
    parser.add_argument('--audit', action='store_true', help='Also cache independent speaker and recognition models.')
    args = parser.parse_args()
    progress = lambda message: print(message, flush=True)
    if args.all or args.language == 'en':
        download_models(progress)
    if args.all:
        for quality in ('Balanced', 'Detailed'):
            download_indic_models(quality, progress)
    elif args.language in ('hi', 'ml'):
        download_indic_models(args.quality, progress)
    if args.all or args.audit or args.language in ('hi', 'ml'):
        download_asr(progress, 'ml' if args.language == 'ml' else None)
    if args.all:
        download_asr(progress, 'ml')
    if args.all or args.audit:
        download_speaker(progress)
    print('Selected models cached. Set HF_HUB_OFFLINE=1 to use cached files without network lookups.')
