"""Launch Voice Studio locally or inside a Linux container."""
from studio.paths import configure, JOBS
configure()

import argparse
import os
from studio.ui import create_app, login_credentials, manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--host', default=os.environ.get('VOICE_STUDIO_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '7860')))
    args = parser.parse_args()
    create_app().queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=args.host, server_port=args.port, inbrowser=not args.no_browser,
        auth=login_credentials(), share=False, allowed_paths=[str(JOBS)], max_file_size='150mb')
