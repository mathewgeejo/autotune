"""Calibrate automated judges on real human speech, before judging a clone."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import configure
configure()
from studio.recognition import recognize, speaker_similarity
from studio.analysis import transcript_errors
from studio.runtime import keep_awake
import soundfile as sf


def main():
    output = ROOT/'data/smoke/evaluator-controls'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    clips = [('ml', ROOT/'data/smoke/fleurs-ml_in/reference.wav', ROOT/'data/smoke/fleurs-ml_in/reference.txt'),
             ('hi', ROOT/'data/smoke/fleurs-hi_in/reference.wav', ROOT/'data/smoke/fleurs-hi_in/reference.txt'),
             ('en', ROOT/'data/smoke/dave.wav', ROOT/'data/smoke/dave.txt')]
    with keep_awake():
        for language, audio, caption in clips:
            folder = output/language
            folder.mkdir(exist_ok=True)
            samples, rate = sf.read(audio, dtype='float32')
            sf.write(folder/'first-half.wav', samples[:len(samples)//2], rate)
            sf.write(folder/'second-half.wav', samples[len(samples)//2:], rate)
            same = speaker_similarity(folder/'first-half.wav', folder/'second-half.wav', folder,
                                      lambda message: print(language, message, flush=True))
            recognized = recognize(audio, folder, language, lambda message: print(language, message, flush=True))
            errors = transcript_errors(caption.read_text(encoding='utf-8'), recognized['text'])
            record = {'language': language, 'real_human_audio': str(audio),
                      'same_human_different_words': same,
                      'human_reference_transcription': recognized, 'human_asr_errors': errors}
            records.append(record)
            (output/'controls.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({'language': language, 'same_speaker_cosine': same.get('cosine_similarity'),
                'human_word_error_rate': errors.get('word_error_rate'),
                'human_grapheme_error_rate': errors.get('grapheme_error_rate')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
