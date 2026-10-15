"""CosyVoice3 web UI, v3-aware and vLLM-capable.

Upstream webui.py targets CosyVoice-300M: it hardcodes a `sft_spk` dropdown that
is empty for v3, and its "自然语言控制" mode calls `inference_instruct`, which
needs a preset speaker and therefore cannot work here. It also constructs the
model without vLLM. This replaces it with a UI that only offers modes v3 can
actually serve, and exposes `inference_instruct2` (natural-language control by
prompt audio) which the v3 examples use.

Every mode takes a prompt audio; the tts/prompt/instruct text is wrapped with
the `You are a helpful assistant.` system prompt if it is not already present,
because v3 does not add it for you and omitting it makes the LLM emit almost
nothing (the vocoder then fails with "Kernel size can't be greater than actual
input size", which looks like a code bug rather than a bad input).

    python docker/vllm/scripts/serve_webui.py --port 50001

Environment:
    COSYVOICE_MODEL_DIR      model directory (default pretrained_models/Fun-CosyVoice3-0.5B)
    COSYVOICE_LOAD_VLLM      1 to serve the LLM through vLLM (default 1)
    COSYVOICE_TEXT_FRONTEND  ttsfrd | wetext | none (default: degrade automatically)
"""
import argparse
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, 'third_party', 'Matcha-TTS'))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
from cosyvoice.utils.common import instruct_list, set_all_random_seed  # noqa: E402
from cosyvoice.utils.file_utils import logging  # noqa: E402

SYSTEM_PROMPT = 'You are a helpful assistant.'
END_OF_PROMPT = '<|endofprompt|>'

ZERO_SHOT = '零样本复刻'
CROSS_LINGUAL = '跨语种复刻'
INSTRUCT2 = '自然语言控制'
SFT = '预置音色'

MODE_HELP = {
    ZERO_SHOT: '1. 传一段 3~30s 的参考音频\n2. 填这段音频实际说的内容\n3. 填要合成的文本\n4. 生成',
    CROSS_LINGUAL: '1. 传参考音频\n2. 填要合成的文本（可与参考音频不同语言）\n3. 生成',
    INSTRUCT2: '1. 传参考音频\n2. 填指令（可从下方预设挑，如「请用四川话表达」）\n3. 填要合成的文本\n4. 生成',
    SFT: '1. 选择预置音色\n2. 填要合成的文本\n3. 生成',
}


def with_system_prompt(text, joiner=''):
    """Wrap text in v3's instruct system prompt unless it is already there."""
    text = (text or '').strip()
    if END_OF_PROMPT in text:
        return text
    return '{}{}{}{}'.format(SYSTEM_PROMPT, joiner, text, END_OF_PROMPT)


def build_ui(cosyvoice, spks, default_data, mode_list):
    with gr.Blocks(title='CosyVoice3') as demo:
        gr.Markdown('### CosyVoice3 本地推理（' + cosyvoice.model.__class__.__name__ + '）')

        with gr.Row():
            mode = gr.Radio(choices=mode_list, label='推理模式', value=mode_list[0])
            help_text = gr.Text(label='操作步骤', value=MODE_HELP[mode_list[0]], scale=0.5)

        with gr.Row():
            sft_dropdown = gr.Dropdown(choices=spks, label='预置音色（仅预置音色模式用）',
                                       value=spks[0] if spks else None, scale=1)
            stream = gr.Radio(choices=[('否', False), ('是', True)], label='流式', value=False, scale=0.5)
            speed = gr.Number(value=1.0, label='语速（仅非流式生效）', minimum=0.5, maximum=2.0, step=0.1, scale=0.5)
            seed = gr.Number(value=0, label='随机种子', scale=0.5)
            seed_button = gr.Button('\U0001F3B2', scale=0.1)

        with gr.Row():
            wav_upload = gr.Audio(sources=['upload'], type='filepath', label='参考音频（文件）')
            wav_record = gr.Audio(sources=['microphone'], type='filepath', label='参考音频（录制）')

        prompt_text = gr.Textbox(label='参考音频对应文本', lines=1,
                                 placeholder='必须与参考音频内容一致，不算标点差异')
        instruct_preset = gr.Dropdown(choices=instruct_list, label='指令预设（点了会填到下面）', value=None)
        instruct_text = gr.Textbox(label='指令文本', lines=1,
                                   placeholder='只写指令本身即可，如「请用四川话表达」，system prompt 会帮你补上')

        tts_text = gr.Textbox(label='合成文本', lines=2,
                              value='收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。')
        generate_button = gr.Button('生成音频', variant='primary')
        audio_output = gr.Audio(label='合成音频', autoplay=True, streaming=True)

        seed_button.click(lambda: random.randint(1, 100000000), inputs=[], outputs=seed, api_name=False)
        mode.change(lambda m: MODE_HELP[m], inputs=[mode], outputs=help_text, api_name=False)
        instruct_preset.change(lambda x: x or '', inputs=[instruct_preset], outputs=instruct_text, api_name=False)

        # api_name is explicit so the HTTP API path is stable; gradio would
        # otherwise derive it from the lambda's __name__. The fn has to be a
        # generator function itself -- returning a generator from a lambda is
        # not the same thing and gradio rejects it in postprocess.
        def handle(*args):
            yield from run_inference(cosyvoice, default_data, *args)

        generate_button.click(
            fn=handle,
            inputs=[tts_text, mode, sft_dropdown, prompt_text, wav_upload, wav_record, instruct_text,
                    seed, stream, speed],
            outputs=[audio_output],
            api_name='generate_audio')

    return demo


def run_inference(cosyvoice, default_data, tts_text, mode, sft_dropdown, prompt_text,
                  wav_upload, wav_record, instruct_text, seed, stream, speed):
    if not (tts_text or '').strip():
        gr.Warning('请填写合成文本')
        yield cosyvoice.sample_rate, default_data
        return

    prompt_wav = wav_upload or wav_record
    if mode != SFT and not prompt_wav:
        gr.Warning('请提供参考音频')
        yield cosyvoice.sample_rate, default_data
        return

    set_all_random_seed(int(seed))
    try:
        if mode == SFT:
            output = cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed)
        elif mode == ZERO_SHOT:
            output = cosyvoice.inference_zero_shot(tts_text, with_system_prompt(prompt_text),
                                                   prompt_wav, stream=stream, speed=speed)
        elif mode == CROSS_LINGUAL:
            output = cosyvoice.inference_cross_lingual(with_system_prompt(tts_text),
                                                       prompt_wav, stream=stream, speed=speed)
        else:
            output = cosyvoice.inference_instruct2(tts_text, with_system_prompt(instruct_text, ' '),
                                                   prompt_wav, stream=stream, speed=speed)
        for i in output:
            yield cosyvoice.sample_rate, i['tts_speech'].numpy().flatten()
    except Exception as e:
        logging.error('inference failed: %s', e)
        gr.Warning('推理失败：{}'.format(e))
        yield cosyvoice.sample_rate, default_data


def main():
    parser = argparse.ArgumentParser(description='CosyVoice3 web UI.')
    parser.add_argument('--model_dir', type=str,
                        default=os.environ.get('COSYVOICE_MODEL_DIR', 'pretrained_models/Fun-CosyVoice3-0.5B'))
    parser.add_argument('--port', type=int, default=50001)
    parser.add_argument('--load_vllm', type=int, default=int(os.environ.get('COSYVOICE_LOAD_VLLM', '1')))
    args = parser.parse_args()

    if args.load_vllm:
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
        ModelRegistry.register_model('CosyVoice2ForCausalLM', CosyVoice2ForCausalLM)

    logging.info('loading %s (vllm=%s)', args.model_dir, bool(args.load_vllm))
    cosyvoice = AutoModel(model_dir=args.model_dir, load_vllm=bool(args.load_vllm), fp16=False)
    logging.info('frontend in use: %r', cosyvoice.frontend.text_frontend)

    spks = cosyvoice.list_available_spks()
    if len(spks) == 0:
        logging.info('no preset speakers in this model, hiding the %s mode', SFT)
    mode_list = [ZERO_SHOT, CROSS_LINGUAL, INSTRUCT2] + ([SFT] if spks else [])

    build_ui(cosyvoice, spks, np.zeros(cosyvoice.sample_rate), mode_list).queue(max_size=4).launch(
        server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    main()
