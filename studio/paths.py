import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
JOBS = DATA / 'jobs'
REFERENCES = DATA / 'references'
MODELS = DATA / 'models'
SEED = ROOT / 'vendor/seed-vc'


def configure():
    for path in (JOBS, REFERENCES, MODELS, DATA / 'tmp'):
        path.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(MODELS / 'huggingface')
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(MODELS / 'huggingface/hub')
    os.environ['TORCH_HOME'] = str(MODELS / 'torch')
    os.environ['GRADIO_TEMP_DIR'] = str(DATA / 'tmp')
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    os.environ['DO_NOT_TRACK'] = '1'
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
