"""Check the bundled audio decoder. No singing or instrument assets are needed."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio.paths import configure
configure()
from studio.audio import ffmpeg

if __name__ == '__main__':
    if not Path(ffmpeg()).is_file():
        raise SystemExit('The bundled FFmpeg decoder is missing. Run setup.cmd again.')
    print('Audio decoder ready. Voice-cloning models download on first generation.')
