"""Cache the actual voice-cloning models before starting the app."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio.paths import configure
configure()
from studio.models import download_models

if __name__ == '__main__':
    download_models(lambda message: print(message, flush=True))
    print('Voice-cloning models cached. File-based generation now works offline.')
