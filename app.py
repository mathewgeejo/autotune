"""Local sample-to-speech voice cloning interface."""
from studio.paths import configure, DATA
configure()

import argparse
import gradio as gr

from studio.jobs import JobManager, BusyError
from studio.speech import split_text

manager = JobManager()

CSS = """
:root { --body-background-fill: #0c0e16; }
.gradio-container { max-width: 1180px !important; margin: auto; }
.studio-header { padding: 32px 0 18px; }
.brand { font: 700 15px ui-sans-serif, sans-serif; letter-spacing: 3px; color: #c6b9ff; }
.studio-header h1 { font-size: 46px; font-weight: 650; letter-spacing: -2px; margin: 18px 0 10px; }
.studio-header p { color: #a2a8ba; font-size: 16px; max-width: 690px; line-height: 1.7; }
.badge { display: inline-block; border: 1px solid #39344e; border-radius: 30px; padding: 6px 12px; font-size: 12px; color: #beb5da; margin: 8px 8px 0 0; }
.step-label { font-size: 12px; text-transform: uppercase; letter-spacing: 2px; color: #b5a1ff; margin: 8px 0; }
.footer { color: #858b9f; text-align: center; padding: 22px 0; font-size: 12px; }
#status textarea { font-family: ui-monospace, monospace; font-size: 13px; }
.primary { background: linear-gradient(110deg, #7756d8, #5d62d8) !important; border: none !important; }
button { transition: filter .15s; }
button:hover { filter: brightness(1.1); }
"""


def generate(source, text, start, end):
    try:
        split_text(text)
        job_id = manager.submit('clone', {'source': source, 'text': text,
                                         'start': start, 'end': end})
        return job_id, 'Starting voice cloning...', None, None, None, ''
    except (ValueError, OSError) as exc:
        raise gr.Error(str(exc)) from exc


def poll(job_id, delivered):
    with manager.lock:
        skip = gr.skip
        if not job_id:
            return (skip(),) * 7
        job = manager.snapshot(job_id)
        if not job or not job['folder'].exists():
            return 'Audio cleared. Ready for a new sample.', '', None, None, None, None, ''
        if delivered == job_id:
            return (skip(),) * 7
        elapsed = int(job['elapsed'])
        state = job['state']
        stage = job['status']['stage']
        if state == 'cancelled':
            stage = 'Cancelled. You can edit the text and generate again.'
        message = (f"{state.upper()} | {elapsed//60:02d}:{elapsed%60:02d} elapsed | "
                   f"{job['peak_mb']:.0f} MB peak worker memory\n{stage}")
        log_path = job['folder'] / 'worker.log'
        with log_path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-14000))
            log = stream.read().decode('utf-8', errors='replace')
        if state == 'running':
            preview = job['status'].get('preview')
            if preview and not delivered:
                return message, log, None, preview, skip(), skip(), job_id + '-preview'
            return message, log, None, skip(), skip(), skip(), skip()
        if state != 'complete':
            return message, log, str(log_path), None, None, skip(), job_id
        result = job['result']
        message += (f"\n{result['duration']:.1f}s of speech | {result['passages']} passage(s) | "
                    + ('cached voice reused' if result['cached_voice'] else 'voice sample learned'))
        return (message, log, str(log_path), result['audio'], result['audio'],
                result['reference_preview'], job_id)


def clear_audio():
    try:
        return manager.clear(), None, '', None, None, None, '', '', '', None
    except BusyError as exc:
        raise gr.Error(str(exc)) from exc


def clear_voices():
    try:
        return manager.clear(references=True)
    except BusyError as exc:
        raise gr.Error(str(exc)) from exc


def release_model():
    try:
        return manager.release()
    except BusyError as exc:
        raise gr.Error(str(exc)) from exc


def create_app():
    from studio.uploads import install_upload_fix
    install_upload_fix()
    theme = gr.themes.Base(primary_hue='violet', secondary_hue='indigo', neutral_hue='slate',
                           font=['system-ui', 'Segoe UI', 'sans-serif'],
                           font_mono=['Consolas', 'monospace']).set(
        body_background_fill='#0c0e16', body_background_fill_dark='#0c0e16',
        block_background_fill='#151824', block_background_fill_dark='#151824',
        block_border_color='#292d3f', block_border_color_dark='#292d3f',
        input_background_fill='#0f121d', input_background_fill_dark='#0f121d',
        body_text_color='#e5e6ef', body_text_color_dark='#e5e6ef',
        block_label_text_color='#adb3c8', block_label_text_color_dark='#adb3c8',
        block_label_background_fill='#151824', block_label_background_fill_dark='#151824',
        button_secondary_background_fill='#262b3e', button_secondary_background_fill_dark='#262b3e',
        button_secondary_text_color='#e5e6ef', button_secondary_text_color_dark='#e5e6ef')
    with gr.Blocks(title='Voice Studio | Voice Cloning', theme=theme, css=CSS,
                   analytics_enabled=False, delete_cache=(3600, 86400), fill_width=True) as demo:
        gr.HTML('''<div class="studio-header"><div class="brand">VOICE STUDIO</div>
        <h1>Give your words a familiar voice.</h1>
        <p>Upload a voice sample, type what you want to say, and hear it spoken in that voice.</p>
        <span class="badge">Runs on your PC</span><span class="badge">Voice cloning</span>
        <span class="badge">English speech</span></div>''')
        job_id = gr.State('')
        delivered = gr.State('')
        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                gr.HTML('<div class="step-label">01 / VOICE SAMPLE</div>')
                source = gr.Audio(sources=['microphone', 'upload'], type='filepath',
                                  label='Record or upload a voice sample', format=None, elem_id='voice-sample')
                gr.Markdown('Use **5-15 seconds of clear speech** from one person. Match the tone you want: '
                            'a calm sample gives calmer speech. Avoid music, echo, and other speakers.')
                with gr.Accordion('Choose a section from a longer sample', open=False):
                    with gr.Row():
                        start = gr.Number(value=0, minimum=0, label='Sample start (seconds)')
                        end = gr.Number(value=0, minimum=0, label='Sample end (0 = automatic)')
                    gr.Markdown('Up to 60 seconds per selection. The clearest continuous 10 seconds are used.')
                gr.HTML('<div class="step-label">02 / YOUR WORDS</div>')
                text = gr.Textbox(label='Text to speak (English)', lines=7, max_lines=14,
                                  placeholder='Type the words you want this voice to say...',
                                  max_length=5000, elem_id='speech-text')
                gr.Markdown('Punctuation shapes the pauses. Up to 5,000 characters per generation.')
                with gr.Row():
                    synthesize = gr.Button('Generate speech', variant='primary', scale=4)
                    cancel = gr.Button('Cancel', scale=1)
            with gr.Column(scale=4):
                gr.HTML('<div class="step-label">03 / LISTEN & DOWNLOAD</div>')
                output = gr.Audio(label='Generated speech', interactive=False, elem_id='speech-output')
                wav = gr.File(label='Download speech WAV', interactive=False, height=90)
                status = gr.Textbox(label='Studio activity',
                    value='Ready. Add a voice sample and type your text to begin.',
                    lines=5, interactive=False, elem_id='status')
                gr.Markdown('The first generation downloads the models. Later requests reuse the loaded engine '
                            'and your cached voice sample. Longer text plays a passage preview while it finishes.')
                with gr.Accordion('Voice sample used', open=False):
                    reference_preview = gr.Audio(label='Selected voice reference', interactive=False)
                with gr.Accordion('Processing details', open=False):
                    logs = gr.Textbox(label='Latest generation log', lines=10, interactive=False)
                    log_file = gr.File(label='Download full log', interactive=False)
                with gr.Accordion('Local storage & memory', open=False):
                    gr.Markdown('Samples, voice data, and speech stay in this project. '
                                'Deleting cached voice data keeps the downloaded models.')
                    release = gr.Button('Release model from memory', size='sm')
                    clear = gr.Button('Delete recordings & exports', size='sm')
                    clear_refs = gr.Button('Delete cached voice data', size='sm')
        gr.HTML('<div class="footer">Local voice cloning | Close the terminal to stop Voice Studio</div>')
        synthesize.click(generate, [source, text, start, end],
                         [job_id, status, output, wav, reference_preview, delivered],
                         concurrency_id='studio', api_name='generate_speech')
        cancel.click(manager.cancel, outputs=status, queue=False, api_name='cancel_generation')
        release.click(release_model, outputs=status, concurrency_id='studio')
        clear.click(clear_audio, outputs=[status, source, text, output, wav, reference_preview,
                                         job_id, delivered, logs, log_file], concurrency_id='studio')
        clear_refs.click(clear_voices, outputs=status, concurrency_id='studio')
        gr.Timer(1).tick(poll, [job_id, delivered], [status, logs, log_file, output, wav,
                                                   reference_preview, delivered], queue=False,
                         api_name=False)
    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    create_app().queue(default_concurrency_limit=1, max_size=8).launch(
        server_name='127.0.0.1', server_port=args.port, inbrowser=not args.no_browser,
        share=False, allowed_paths=[str(DATA)], max_file_size='150mb')
