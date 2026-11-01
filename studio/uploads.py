"""Work around Gradio 5.49.1's Windows duplicate-upload copy race."""
from pathlib import Path
import shutil


def move_uploaded_files_to_cache(files, destinations):
    for source, destination in zip(files, destinations):
        source, destination = Path(source), Path(destination)
        # Gradio's destination directory is a hash of the complete upload.
        # A repeated upload therefore already has identical bytes here. On
        # Windows rename cannot replace it; upstream falls back to shutil.move,
        # which truncates the served file while the browser is fetching it.
        if destination.is_file():
            source.unlink(missing_ok=True)
        else:
            shutil.move(str(source), str(destination))


def install_upload_fix():
    import gradio
    import gradio.routes
    if gradio.__version__ == '5.49.1':
        gradio.routes.move_uploaded_files_to_cache = move_uploaded_files_to_cache
