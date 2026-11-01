import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('VOICE_STUDIO_DATA_DIR', ROOT / 'data')).expanduser().resolve()
JOBS = DATA / 'jobs'
REFERENCES = DATA / 'references'
MODELS = DATA / 'models'


def configure():
    threads = str(max(1, min(16, int(os.environ.get('VOICE_STUDIO_THREADS', '2')))))
    for path in (JOBS, REFERENCES, MODELS, DATA / 'tmp'):
        path.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(MODELS / 'huggingface')
    os.environ['HF_HUB_CACHE'] = str(MODELS / 'huggingface/hub')
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(MODELS / 'huggingface/hub')
    os.environ['TORCH_HOME'] = str(MODELS / 'torch')
    os.environ['GRADIO_TEMP_DIR'] = str(DATA / 'tmp')
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    os.environ['DO_NOT_TRACK'] = '1'
    os.environ['OMP_NUM_THREADS'] = threads
    os.environ['MKL_NUM_THREADS'] = threads
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY'] = '1'
    os.environ['HF_XET_NUM_CONCURRENT_RANGE_GETS'] = '4'
    os.environ['MPLCONFIGDIR'] = str(MODELS / 'matplotlib')
    os.environ['MPLBACKEND'] = 'Agg'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    # Gradio checks its own local URL at startup. System HTTP proxies must not
    # intercept requests between the browser interface and the local server.
    bypass = os.environ.get('NO_PROXY', os.environ.get('no_proxy', ''))
    hosts = [item for item in bypass.split(',') if item]
    for host in ('localhost', '127.0.0.1', '::1'):
        if host not in hosts:
            hosts.append(host)
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = ','.join(hosts)
