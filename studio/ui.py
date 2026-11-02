"""Responsive cloning workbench and reproducible audio-comparison views."""
import json
import os
import uuid

import gradio as gr

from studio.paths import DATA, JOBS
from studio.jobs import JobManager, BusyError
from studio.languages import LANGUAGES
from studio.speech import split_text, prepare_sample
from studio.style import CSS

manager = JobManager()
CSS += """
.gradio-container, .gradio-container > main {
 width: 100% !important; min-width: 0 !important; box-sizing: border-box;
}
@media (max-width: 640px) {
 .studio-header { padding: 16px 0 8px; }
 .studio-header h1 { font-size: 34px; line-height: 1.15; letter-spacing: -1px; }
 .studio-header p { font-size: 15px; }
 button, .tab-nav button { min-height: 44px; }
}
#analysis-summary { border-left: 3px solid #a78bfa; padding-left: 16px; }
#analysis-panel, #analysis-plot, .diagnostic-table { min-width: 0 !important; max-width: 100%; }
.diagnostic-table { overflow-x: auto; }
.diagnostic-table .table-wrap { max-width: 100%; overflow-x: auto; }
.footer { line-height: 1.6; }
"""


def generate(source, text, start, end, language='auto', quality='Detailed', reference_text='', reference_language='auto'):
    try:
        split_text(text)
        job_id = manager.submit('clone', {'source': source, 'text': text, 'start': start, 'end': end,
            'language': language, 'quality': quality, 'reference_text': reference_text or '',
            'reference_language': reference_language})
        return job_id, 'Starting voice cloning...', None, None, ''
    except (ValueError, OSError) as exc:
        raise gr.Error(str(exc)) from exc


def inspect_sample(source, start, end):
    if not source:
        return None, 'Add a speech recording up to one minute long.', ''
    folder = DATA/'tmp'/('selection-'+uuid.uuid4().hex)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        _, preview, _ = prepare_sample(source, folder, start, end)
        section = json.loads((folder/'reference-section.json').read_text(encoding='utf-8'))
        caption = (f"Selection: **{section['uploaded_selection_seconds']:.1f}s**. "
                   f"Conditioning uses **{section['used_duration_seconds']:.1f}s**, "
                   f"starting at **{section['used_start_seconds']:.1f}s**. "
                   'The model uses this focused window rather than the entire upload.')
        return str(preview), caption, ''
    except ValueError as exc:
        return None, str(exc), ''


def start_analysis(reference, candidate, expected, same_words, mode, language, last_reference, last_audio):
    try:
        job_id = manager.submit('analyze', {'source': reference or last_reference,
            'comparison': candidate or last_audio, 'text': expected or '', 'same_words': same_words,
            'full_audit': mode == 'Full quality audit', 'language': language})
        return job_id, 'Starting the audio comparison...', ''
    except (ValueError, OSError) as exc:
        raise gr.Error(str(exc)) from exc


def analysis_view(report):
    names = [('median_pitch_hz', 'Median voiced pitch', 'Hz'),
             ('pitch_p10_hz', 'Pitch: 10th percentile', 'Hz'),
             ('pitch_p90_hz', 'Pitch: 90th percentile', 'Hz'),
             ('loudness_lufs', 'Integrated loudness', 'LUFS'),
             ('rms_dbfs', 'RMS level', 'dBFS'), ('peak_dbfs', 'Peak level', 'dBFS'),
             ('silence_percent', 'Silence', '%'), ('voiced_percent', 'Periodic voicing', '%'),
             ('clipped_percent', 'Clipped samples', '%'),
             ('spectral_centroid_hz', 'Spectral centroid', 'Hz')]
    display = lambda value: 'Unavailable' if value is None else f'{value:.2f}'
    rows = []
    for key, label, unit in names:
        a, b = report['reference'][key], report['output'][key]
        rows.append([label, display(a), display(b), display(b-a) if a is not None and b is not None else '-', unit])
    summary = ['### What changed']
    for key, label in [('reference_warnings', 'Reference'), ('output_warnings', 'Generated')]:
        summary.extend(f'- **{label}:** {warning}' for warning in report[key])
    verifier = report.get('speaker_verification')
    if verifier and verifier.get('available'):
        summary.append(f"- **Independent speaker cosine:** {verifier['cosine_similarity']:.3f}. Higher means more similar; this is not a percentage of identity.")
    content = report.get('content_comparison')
    if content and content.get('available'):
        summary.append(f"- **Recognized word errors:** {content['word_error_rate']*100:.1f}%. **Grapheme errors:** {content['grapheme_error_rate']*100:.1f}%.")
        summary.append('- Recognition can also make mistakes. Inspect the word-level differences before deciding the voice model missed a word.')
    if not verifier:
        summary.append('- Choose **Full quality audit** to add independent speaker verification and multilingual word checking.')
    summary.append('- Different words naturally have different pitch, timing, and spectra. Alignment is available only for the same spoken words.')
    segments = [[row['reference_seconds'], row['output_seconds'], row['spectral_distance'], row['pitch_difference_cents']]
                for row in report['aligned_comparison'].get('largest_spectral_differences', [])]
    words = [[row['type'], row['expected'], row['recognized'], row.get('output_seconds')]
             for row in (content or {}).get('differences', [])]
    return '\n'.join(summary), rows, segments, words


def poll_updates(job_id, delivered):
    with manager.lock:
        if not job_id:
            return {}
        job = manager.snapshot(job_id)
        if not job or not job['folder'].exists():
            return {'status': 'Audio cleared. Ready for a new sample.', 'delivered': ''}
        if delivered == job_id:
            return {}
        elapsed, state = int(job['elapsed']), job['state']
        stage = job['status']['stage'] if state != 'cancelled' else 'Cancelled. You can edit the inputs and try again.'
        message = (f"{state.upper()} | {elapsed//60:02d}:{elapsed%60:02d} elapsed | "
                   f"{job['peak_mb']:.0f} MB peak worker memory\n{stage}")
        log_path = job['folder']/'worker.log'
        with log_path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-14000))
            log = stream.read().decode('utf-8', errors='replace')
        updates = {'status': message, 'logs': log}
        if state == 'running':
            if job['mode'] == 'clone' and job['status'].get('preview') and not delivered:
                updates.update(audio=job['status']['preview'], delivered=job_id+'-preview')
            return updates
        updates.update(log_file=str(log_path), delivered=job_id)
        if state != 'complete':
            return updates
        result = job['result']
        if job['mode'] == 'clone':
            language = LANGUAGES.get(result.get('language', 'en'), 'English')
            updates['status'] += (f"\n{language} | {result['duration']:.1f}s speech | {result['passages']} passage(s) | "
                                  + ('cached voice reused' if result['cached_voice'] else 'voice sample learned'))
            updates.update(audio=result['audio'], wav=result['audio'], selected=result['reference_preview'],
                           last_audio=result['audio'], last_reference=result['reference_preview'],
                           last_text=result.get('text', ''))
            if result.get('reference_text'):
                updates['reference_text'] = result['reference_text']
        else:
            from studio.analysis import load_tracks, comparison_figure
            summary, rows, segments, words = analysis_view(result['analysis'])
            updates.update(analysis_summary=summary, analysis_table=rows, analysis_segments=segments,
                analysis_words=words, analysis_plot=comparison_figure(load_tracks(result['analysis_tracks'])),
                analysis_json=result['analysis_json'], analysis_csv=result['analysis_csv'],
                analysis_archive=result['analysis_archive'], analysis_raw=result['analysis'])
        return updates


def safe_storage(action):
    try:
        return action()
    except BusyError as exc:
        raise gr.Error(str(exc)) from exc


def login_credentials():
    username, password = os.environ.get('VOICE_STUDIO_USERNAME', ''), os.environ.get('VOICE_STUDIO_PASSWORD', '')
    if bool(username) != bool(password):
        raise ValueError('Set both VOICE_STUDIO_USERNAME and VOICE_STUDIO_PASSWORD, or neither.')
    if os.environ.get('VOICE_STUDIO_REQUIRE_AUTH') == '1' and not username:
        raise ValueError('Public hosting requires VOICE_STUDIO_USERNAME and VOICE_STUDIO_PASSWORD.')
    return (username, password) if username else None


def create_app():
    from studio.uploads import install_upload_fix
    install_upload_fix()
    theme = gr.themes.Base(primary_hue='violet', secondary_hue='indigo', neutral_hue='slate',
        font=['system-ui', 'Nirmala UI', 'Segoe UI', 'sans-serif'], font_mono=['Consolas', 'monospace']).set(
        body_background_fill='#0c0e16', body_background_fill_dark='#0c0e16',
        block_background_fill='#151824', block_background_fill_dark='#151824',
        block_border_color='#292d3f', block_border_color_dark='#292d3f',
        input_background_fill='#0f121d', input_background_fill_dark='#0f121d',
        body_text_color='#e5e6ef', body_text_color_dark='#e5e6ef',
        block_label_text_color='#adb3c8', block_label_text_color_dark='#adb3c8',
        block_label_background_fill='#151824', block_label_background_fill_dark='#151824',
        table_text_color='#e5e6ef', table_text_color_dark='#e5e6ef',
        table_border_color='#363b50', table_border_color_dark='#363b50',
        table_even_background_fill='#151824', table_even_background_fill_dark='#151824',
        table_odd_background_fill='#1b2030', table_odd_background_fill_dark='#1b2030',
        button_secondary_background_fill='#262b3e', button_secondary_background_fill_dark='#262b3e',
        button_secondary_text_color='#e5e6ef', button_secondary_text_color_dark='#e5e6ef')
    language_options = [('Auto detect', 'auto')]+[(label, code) for code, label in LANGUAGES.items()]
    with gr.Blocks(title='Voice Studio | Clone, listen, compare', theme=theme, css=CSS,
                   analytics_enabled=False, delete_cache=(3600, 86400), fill_width=True) as demo:
        gr.HTML('''<div class="studio-header"><div class="brand">VOICE STUDIO</div>
        <h1>Give your words a familiar voice.</h1>
        <p>Clone a voice, hear it in another language, and inspect how closely the result matches.</p>
        <span class="badge">English / Hindi / Malayalam</span><span class="badge">One-minute samples</span>
        <span class="badge">Local audio analysis</span></div>''')
        job_id, delivered = gr.State(''), gr.State('')
        last_audio, last_reference, last_text = gr.State(''), gr.State(''), gr.State('')
        with gr.Tabs():
            with gr.Tab('Create speech'):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=6):
                        gr.HTML('<div class="step-label">01 / YOUR REFERENCE VOICE</div>')
                        source = gr.Audio(sources=['microphone', 'upload'], type='filepath', format=None,
                                          label='Record or upload a voice sample', elem_id='voice-sample')
                        gr.Markdown('Upload **up to 60 seconds** of one person speaking. Clear, natural speech '
                                    'without music gives the model the best reference for tone and accent.')
                        with gr.Accordion('Selected reference and transcript', open=False):
                            with gr.Row():
                                start = gr.Number(value=0, minimum=0, label='Sample start (seconds)')
                                end = gr.Number(value=0, minimum=0, label='Sample end (0 = automatic)')
                            selected = gr.Audio(label='Selected reference clip', interactive=False, format=None)
                            sample_info = gr.Markdown('Add a speech recording up to one minute long.')
                            reference_language = gr.Dropdown(language_options, value='auto', label='Reference language')
                            reference_text = gr.Textbox(label='Words in the selected reference (optional)', lines=3,
                                placeholder='Hindi and Malayalam use automatic transcription when this is blank.')
                            gr.Markdown('For stronger Indic pronunciation, review these words against the selected clip. '
                                        'The text below is what the output will say.')
                        gr.HTML('<div class="step-label">02 / WORDS & LANGUAGE</div>')
                        with gr.Row():
                            language = gr.Dropdown(language_options, value='auto', label='Speech language')
                            quality = gr.Radio(['Balanced', 'Detailed'], value='Detailed', label='Indic quality',
                                info='Detailed uses full precision and more refinement for Hindi/Malayalam. It takes longer.')
                        text = gr.Textbox(label='Text to speak', lines=6, max_lines=12, max_length=5000,
                                          placeholder='Type in English, Hindi, or Malayalam...', elem_id='speech-text')
                        gr.Markdown('Up to 5,000 characters. Longer text is generated in passages and joined into one WAV.')
                        with gr.Row():
                            synthesize = gr.Button('Generate speech', variant='primary', scale=4)
                            cancel = gr.Button('Cancel', scale=1)
                    with gr.Column(scale=4):
                        gr.HTML('<div class="step-label">03 / LISTEN & COMPARE</div>')
                        audio = gr.Audio(label='Generated speech', interactive=False, format=None, elem_id='speech-output')
                        wav = gr.File(label='Download speech WAV', interactive=False, height=90)
                        gr.Markdown('First use downloads the selected engine. Later generations reuse it and the '
                                    'reference conditioning. A passage preview appears during longer generations.')
                        use_result = gr.Button('Use this result in Voice analysis', size='sm')
            with gr.Tab('Voice analysis', elem_id='analysis-panel'):
                gr.Markdown('## Find the differences\nCompare pitch, loudness, spectra, speaker identity, and spoken words. '
                    'These diagnostics locate problems; no score guarantees human indistinguishability.')
                with gr.Row():
                    analysis_reference = gr.Audio(sources=['upload'], type='filepath', format=None,
                                                  label='Reference for comparison', elem_id='analysis-reference')
                    analysis_output = gr.Audio(sources=['upload'], type='filepath', format=None,
                                               label='Speech to evaluate', elem_id='analysis-candidate')
                expected = gr.Textbox(label='Expected spoken words', lines=3,
                    placeholder='Used only for scoring, never as a speech-recognition prompt.')
                with gr.Row():
                    audit_language = gr.Dropdown(language_options, value='auto', label='Spoken language for analysis')
                    audit_mode = gr.Radio(['Acoustic charts', 'Full quality audit'], value='Acoustic charts', label='Analysis depth')
                same_words = gr.Checkbox(label='Both clips contain the same spoken words', value=False)
                gr.Markdown('Compare up to one minute per clip. Full audit adds an independent speaker verifier and '
                    'multilingual recognizer, and frees the speech engine first to leave room in memory. '
                    'Enable same words only for a true reference reading of the generated passage.')
                compare = gr.Button('Analyze voices', variant='primary')
                analysis_summary = gr.Markdown('The comparison will appear here.', elem_id='analysis-summary')
                analysis_table = gr.Dataframe(headers=['Measurement', 'Reference', 'Generated', 'Difference', 'Unit'],
                    datatype=['str']*5, interactive=False, label='Acoustic measurements', min_width=0,
                    column_widths=[190, 105, 105, 105, 75], elem_classes=['diagnostic-table'])
                analysis_plot = gr.Plot(label='Pitch, energy, and spectrum', min_width=0, elem_id='analysis-plot')
                with gr.Accordion('Largest aligned differences (same words only)', open=False):
                    analysis_segments = gr.Dataframe(headers=['Reference seconds', 'Output seconds', 'Spectral distance', 'Pitch difference (cents)'],
                        interactive=False, label='Where the acoustic shape changes', min_width=0, elem_classes=['diagnostic-table'])
                with gr.Accordion('Recognized word differences', open=False):
                    analysis_words = gr.Dataframe(headers=['Type', 'Expected', 'Recognized', 'Output seconds'],
                        interactive=False, label='Content differences', min_width=0, elem_classes=['diagnostic-table'])
                with gr.Row():
                    analysis_json = gr.File(label='Measurements JSON', interactive=False)
                    analysis_csv = gr.File(label='Frame measurements CSV', interactive=False)
                    analysis_archive = gr.File(label='Complete diagnostics ZIP', interactive=False)
                with gr.Accordion('Raw analysis data', open=False):
                    analysis_raw = gr.JSON(label='Reproducible measurements and model provenance')
        status = gr.Textbox(label='Studio activity', value='Ready. Add a voice sample and your text.',
                            lines=4, interactive=False, elem_id='status')
        with gr.Accordion('Processing details', open=False):
            logs = gr.Textbox(label='Latest job log', lines=10, interactive=False)
            log_file = gr.File(label='Download full log', interactive=False)
        with gr.Accordion('Local storage & memory', open=False):
            gr.Markdown('Audio and diagnostics stay in this instance; model downloads are cached separately. '
                        'These controls affect all devices using this private instance.')
            with gr.Row():
                release = gr.Button('Release model from memory', size='sm')
                clear = gr.Button('Delete recordings & exports', size='sm')
                clear_refs = gr.Button('Delete cached voice data', size='sm')
        gr.HTML('<div class="footer">Voice Studio | Self-hostable voice cloning and audio diagnostics</div>')
        synthesize.click(generate, [source, text, start, end, language, quality, reference_text, reference_language],
                         [job_id, status, audio, wav, delivered], concurrency_id='studio', api_name='generate_speech')
        for component in (source, start, end):
            component.change(inspect_sample, [source, start, end], [selected, sample_info, reference_text], concurrency_id='selection')
        cancel.click(manager.cancel, outputs=status, queue=False, api_name='cancel_generation')
        use_result.click(lambda ref, out, words: (ref or None, out or None, words),
                         [last_reference, last_audio, last_text], [analysis_reference, analysis_output, expected], concurrency_id='studio')
        compare.click(start_analysis, [analysis_reference, analysis_output, expected, same_words, audit_mode,
            audit_language, last_reference, last_audio], [job_id, status, delivered], concurrency_id='studio')
        release.click(lambda: safe_storage(manager.release), outputs=status, concurrency_id='studio', api_name='release_model')
        clear_refs.click(lambda: safe_storage(lambda: manager.clear(references=True)), outputs=status, concurrency_id='studio')
        reset_outputs = [source, text, audio, wav, selected, reference_text, analysis_reference, analysis_output,
                         analysis_json, analysis_csv, analysis_archive, analysis_plot, last_audio, last_reference,
                         last_text, job_id, delivered, logs, log_file, analysis_table, analysis_segments,
                         analysis_words, analysis_raw, expected, analysis_summary, sample_info]
        def clear_all():
            message = safe_storage(manager.clear)
            def empty(item):
                if isinstance(item, (gr.Audio, gr.File, gr.Plot)):
                    return None
                if isinstance(item, gr.Dataframe):
                    return []
                if isinstance(item, gr.JSON):
                    return {}
                return ''
            return [message]+[empty(item) for item in reset_outputs]
        clear.click(clear_all, outputs=[status]+reset_outputs, concurrency_id='studio')
        output_names = ('status', 'logs', 'log_file', 'audio', 'wav', 'selected', 'delivered', 'last_audio',
            'last_reference', 'last_text', 'reference_text', 'analysis_summary', 'analysis_table', 'analysis_segments',
            'analysis_words', 'analysis_plot', 'analysis_json', 'analysis_csv', 'analysis_archive', 'analysis_raw')
        outputs = {name: value for name, value in locals().copy().items() if name in output_names}
        def poll_ui(job, previous):
            updates = poll_updates(job, previous)
            return {outputs[key]: value for key, value in updates.items()} if updates else (gr.skip(),)*len(outputs)
        gr.Timer(1).tick(poll_ui, [job_id, delivered], list(outputs.values()), queue=False, api_name=False)
    return demo
