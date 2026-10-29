"""Voice Studio: a private, local browser interface for audio transformation."""
from studio.paths import configure, DATA
configure()

import argparse
from pathlib import Path
import gradio as gr

from studio.jobs import JobManager, BusyError
from studio.audio import decode, AudioError

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


def start_job(mode, payload):
    try:
        job_id = manager.submit(mode, payload)
        return job_id, 'Starting…', None, None, None
    except (ValueError, OSError) as exc:
        raise gr.Error(str(exc)) from exc


def prepare_song(url, file):
    if not file and not (url or '').strip():
        raise gr.Error('Paste a song link or upload a reference song first.')
    return start_job('prepare', {'url': url or '', 'file': file})


def convert(mode, source, start, end, reference, ref_start, ref_duration, quality, instrument, octave, sensitivity):
    if not source:
        raise gr.Error('Record your voice or upload an audio file first.')
    if mode == 'Singer' and not reference:
        raise gr.Error('Prepare a reference song in the Singer tab first.')
    if end and end <= start:
        raise gr.Error('The end time must be later than the start time.')
    if end and end-start > 30.001:
        raise gr.Error('Choose a section of 30 seconds or less.')
    return start_job('singer' if mode == 'Singer' else 'instrument', {
        'source': source, 'start': start, 'end': end,
        'reference': reference, 'ref_start': ref_start, 'ref_duration': ref_duration,
        'steps': {'Preview': 10, 'Standard': 20, 'Detailed': 30}[quality],
        'instrument': instrument, 'octave': octave, 'sensitivity': sensitivity,
    })


def poll(job_id, delivered):
    skip = gr.skip
    # status, log, log-file, audio, wav, midi, reference-state, preview, ref-start,
    # ref-duration, delivered-state
    if not job_id:
        return (skip(),) * 11
    job = manager.snapshot(job_id)
    if not job:
        return ('Audio cleared. Ready for a new recording.', '', None, None, None, None,
                skip(), skip(), skip(), skip(), '')
    elapsed = job['elapsed']
    state = job['state']
    text = f"{state.upper()}  ·  {int(elapsed)//60:02d}:{int(elapsed)%60:02d} elapsed  ·  {job['peak_mb']:.0f} MB peak\n{job['status']['stage']}"
    log_path = job['folder'] / 'worker.log'
    log_text = log_path.read_text(encoding='utf-8', errors='replace')[-14000:]
    if state == 'running':
        return (text, log_text, None) + (skip(),)*8
    if delivered == job_id:
        return (text,) + (skip(),)*10
    if state != 'complete':
        return (text, log_text, str(log_path), skip(), skip(), skip(), skip(), skip(), skip(), skip(), job_id)
    result = job['result']
    if job['mode'] == 'prepare':
        text += '\nReference ready. Record your singing, then click Transform.'
        return (text, log_text, str(log_path), skip(), skip(), skip(), result['reference'],
                result['preview'], gr.update(value=result['start'], maximum=max(0, result['total']-2)),
                gr.update(value=result['duration']), job_id)
    text += '\nYour audio is ready to play and download.'
    return (text, log_text, str(log_path), result['audio'], result['audio'], result.get('midi'),
            skip(), skip(), skip(), skip(), job_id)


def preview_reference(reference, start, duration):
    if not reference or not Path(reference).is_file():
        raise gr.Error('Prepare a reference song first.')
    import uuid
    target = DATA / 'tmp' / f'preview-{uuid.uuid4().hex}.wav'
    try:
        decode(reference, target, start=float(start), duration=float(duration))
        return str(target)
    except AudioError as exc:
        raise gr.Error(str(exc)) from exc


def clear_audio():
    try:
        message = manager.clear()
        # Inputs and outputs must release paths before the next conversion.
        return message, None, None, None, None, None, '', '', '', None, None
    except BusyError as exc:
        raise gr.Error(str(exc)) from exc


def clear_references():
    try:
        return manager.clear(references=True), '', None
    except BusyError as exc:
        raise gr.Error(str(exc)) from exc


def create_app():
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
    with gr.Blocks(title='Voice Studio · Local Audio Lab', theme=theme, css=CSS,
                   analytics_enabled=False, delete_cache=(3600, 86400), fill_width=True) as demo:
        gr.HTML('''<div class="studio-header"><div class="brand">◉ VOICE STUDIO</div>
        <h1>Your voice. A new possibility.</h1>
        <p>Turn a melody into an instrument, or give your singing a different voice.
        Record a little idea and make it sound like something new.</p>
        <span class="badge">● Processing on your PC</span><span class="badge">No API keys</span>
        <span class="badge">Up to 30 seconds</span></div>''')
        job_id = gr.State('')
        delivered = gr.State('')
        reference = gr.State('')
        mode = gr.State('Instrument')
        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                gr.HTML('<div class="step-label">01 / YOUR PERFORMANCE</div>')
                source = gr.Audio(sources=['microphone', 'upload'], type='filepath',
                                  label='Record or upload your voice', format='wav')
                with gr.Row():
                    start = gr.Number(value=0, minimum=0, label='Start time (seconds)')
                    end = gr.Number(value=0, minimum=0, label='End time (0 = use recording)')
                gr.Markdown('Sing or hum one melody at a time. A quiet recording gives the clearest result.')
                gr.HTML('<div class="step-label">02 / CHOOSE YOUR SOUND</div>')
                with gr.Tabs():
                    with gr.Tab('Instrument') as instrument_tab:
                        instrument = gr.Radio(['Piano', 'Guitar', 'Violin', 'Synth'], value='Piano', label='Play my melody as')
                        with gr.Row():
                            octave = gr.Slider(-2, 2, value=0, step=1, label='Octave shift')
                            sensitivity = gr.Slider(0, 1, value=0.5, step=0.05, label='Note sensitivity')
                        gr.Markdown('Your humming becomes musical notes, played with the instrument you choose.')
                    with gr.Tab('Singer') as singer_tab:
                        url = gr.Textbox(label='Reference song link', placeholder='Paste a YouTube or supported audio link')
                        file = gr.File(label='Or upload the downloaded song', file_types=['audio'], type='filepath')
                        prepare = gr.Button('Prepare song reference', size='sm')
                        gr.Markdown('The app extracts vocals and picks a reference automatically. First use downloads the models. CPU processing can take several minutes.')
                        ref_audio = gr.Audio(label='Reference voice preview', interactive=False)
                        with gr.Accordion('Adjust reference section', open=False):
                            ref_start = gr.Slider(0, 600, value=0, step=0.1, label='Reference start (seconds)')
                            ref_duration = gr.Slider(2, 15, value=10, step=0.5, label='Reference length (seconds)')
                            preview = gr.Button('Preview selected section', size='sm')
                        quality = gr.Radio(['Preview', 'Standard', 'Detailed'], value='Standard', label='Conversion quality',
                                           info='Higher quality takes longer. Your notes and timing are preserved.')
                with gr.Row():
                    transform = gr.Button('Transform my voice  →', variant='primary', scale=4)
                    cancel = gr.Button('Cancel', scale=1)
            with gr.Column(scale=4):
                gr.HTML('<div class="step-label">03 / LISTEN & EXPORT</div>')
                output = gr.Audio(label='Your new sound', interactive=False)
                with gr.Row():
                    wav = gr.File(label='Download WAV', interactive=False, height=90, min_width=160)
                    midi = gr.File(label='Download MIDI', interactive=False, height=90, min_width=160)
                status = gr.Textbox(label='Studio activity', value='Ready when you are. Record your voice to begin.',
                                    lines=4, interactive=False, elem_id='status')
                with gr.Accordion('Processing details', open=False):
                    logs = gr.Textbox(label='Latest job log', lines=10, interactive=False)
                    log_file = gr.File(label='Download full log', interactive=False)
                with gr.Accordion('Local storage', open=False):
                    gr.Markdown('Recordings, exports, and prepared references stay in this project’s `data` folder. AI model files stay cached for offline use.')
                    clear = gr.Button('Delete recordings & exports', size='sm')
                    clear_refs = gr.Button('Delete cached song references', size='sm')
        gr.HTML('<div class="footer">Made for little melodies and big experiments · Local audio processing · Close the terminal to stop the studio</div>')
        instrument_tab.select(lambda: 'Instrument', outputs=mode, queue=False)
        singer_tab.select(lambda: 'Singer', outputs=mode, queue=False)
        kickoff_outputs = [job_id, status, output, wav, midi]
        prepare.click(prepare_song, [url, file], kickoff_outputs, concurrency_id='studio')
        transform.click(convert, [mode, source, start, end, reference, ref_start, ref_duration,
                                  quality, instrument, octave, sensitivity], kickoff_outputs, concurrency_id='studio')
        cancel.click(manager.cancel, outputs=status, queue=False)
        preview.click(preview_reference, [reference, ref_start, ref_duration], ref_audio, concurrency_id='studio')
        clear.click(clear_audio, outputs=[status, source, file, output, wav, midi, job_id, delivered, logs, log_file, ref_audio],
                    concurrency_id='studio')
        clear_refs.click(clear_references, outputs=[status, reference, ref_audio], concurrency_id='studio')
        gr.Timer(1).tick(poll, [job_id, delivered], [status, logs, log_file, output, wav, midi,
                                                  reference, ref_audio, ref_start, ref_duration, delivered], queue=False)
    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    create_app().queue(default_concurrency_limit=1, max_size=8).launch(
        server_name='127.0.0.1', server_port=args.port, inbrowser=not args.no_browser,
        share=False, allowed_paths=[str(DATA)], max_file_size='150mb')
