import torch
import librosa
import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence
import numpy as np
from mel_processing import spectrogram_torch
import gradio as gr
from indic_transliteration import sanscript


SCRIPT_DICT={
    'Devanagari':sanscript.DEVANAGARI,
    'IAST':sanscript.IAST,
    'SLP1':sanscript.SLP1,
    'HK':sanscript.HK
}

DEFAULT_TEXT='संस्कृतम् जगतः एकतमा अतिप्राचीना समृद्धा शास्त्रीया च भाषासु वर्तते । संस्कृतं भारतस्य जगत: वा भाषासु एकतमा‌ प्राचीनतमा ।'


def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def default_text(script):
    if script=='Devanagari':
        return DEFAULT_TEXT
    else:
        return sanscript.transliterate(DEFAULT_TEXT,sanscript.DEVANAGARI,SCRIPT_DICT[script])


def speech_synthesize(text,script, speaker_id, length_scale):
    text=text.replace('\n','')
    if script!='Devanagari':
        text=sanscript.transliterate(text,SCRIPT_DICT[script],sanscript.DEVANAGARI)
    print(text)
    stn_tst = get_text(text, hps_ms)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        sid = torch.LongTensor([speaker_id])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
    return (hps_ms.data.sampling_rate, audio)


def voice_convert(audio,origin_id,target_id):
    sampling_rate, audio = audio
    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != hps_ms.data.sampling_rate:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=hps_ms.data.sampling_rate)

    with torch.no_grad():
        y = torch.FloatTensor(audio).unsqueeze(0)
        spec = spectrogram_torch(y, hps_ms.data.filter_length,
            hps_ms.data.sampling_rate, hps_ms.data.hop_length, hps_ms.data.win_length,
            center=False)
        spec_lengths = torch.LongTensor([spec.size(-1)])
        sid_src = torch.LongTensor([origin_id])
        sid_tgt = torch.LongTensor([target_id])
        audio = net_g_ms.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][0,0].data.cpu().float().numpy()
    return (hps_ms.data.sampling_rate, audio)


if __name__=='__main__':
    hps_ms = utils.get_hparams_from_file('model/config.json')
    n_speakers = hps_ms.data.n_speakers
    n_symbols = len(hps_ms.symbols)
    speakers = hps_ms.speakers

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils.load_checkpoint('model/model.pth', net_g_ms)

    with gr.Blocks() as app:
        gr.Markdown('# Sanskrit Text to Speech\n')
        with gr.Tab('Text to Speech'):
            text_script=gr.Radio(['Devanagari','IAST','SLP1','HK'],label='Script',interactive=True,value='Devanagari')
            text_input = gr.TextArea(label='Text', placeholder='Type your text here',value=DEFAULT_TEXT)
            speaker_id=gr.Dropdown(speakers,label='Speaker',type='index',interactive=True,value=speakers[0])
            length_scale=gr.Slider(0.5,2,1,step=0.1,label='Speaking Speed',interactive=True)
            tts_button = gr.Button('Synthesize')
            audio_output = gr.Audio(label='Speech Synthesized')
            text_script.change(default_text,[text_script],[text_input])
            tts_button.click(speech_synthesize,[text_input,text_script,speaker_id,length_scale],[audio_output])
        with gr.Tab('Voice Conversion'):
            audio_input = gr.Audio(label='Audio',interactive=True)
            speaker_input = gr.Dropdown(speakers, label='Original Speaker',type='index',interactive=True, value=speakers[0])
            speaker_output = gr.Dropdown(speakers, label='Target Speaker',type='index',interactive=True, value=speakers[0])
            vc_button = gr.Button('Convert')
            audio_output_vc = gr.Audio(label='Voice Converted')
            vc_button.click(voice_convert,[audio_input,speaker_input,speaker_output],[audio_output_vc])
        gr.Markdown('## Based on\n'
        '- [VITS](https://github.com/jaywalnut310/vits)\n\n'
        '## Dataset\n'
        '- [Vāksañcayaḥ](https://www.cse.iitb.ac.in/~asr/)')

    app.launch()