"""Platform-specific subprocess behavior kept out of the audio engines."""
import os
import signal
import subprocess

HIDDEN_WINDOW = 0x08000000 if os.name == 'nt' else 0


def worker_options():
    return ({'creationflags': HIDDEN_WINDOW} if os.name == 'nt'
            else {'start_new_session': True})


def stop_tree(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       capture_output=True, check=False, creationflags=HIDDEN_WINDOW)
    else:
        # Every worker starts a new session. FFmpeg and other children inherit
        # its group, so cancellation releases the whole inference tree.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
    if os.name != 'nt':
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=3)
