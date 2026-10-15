"""Zero-shot synthesis with the CosyVoice LLM served through vLLM.

    python docker/vllm/scripts/vllm_infer.py
    python docker/vllm/scripts/vllm_infer.py --text "今天天气不错。" --iters 3

On first run CosyVoice exports the LLM to <model_dir>/vllm/ so the model
directory must be writable. The export is cached and reused afterwards.
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, 'third_party', 'Matcha-TTS'))

import torch  # noqa: E402
import torchaudio  # noqa: E402
from vllm import ModelRegistry  # noqa: E402
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM  # noqa: E402

ModelRegistry.register_model('CosyVoice2ForCausalLM', CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
from cosyvoice.utils.common import set_all_random_seed  # noqa: E402
from cosyvoice.utils.file_utils import logging  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='CosyVoice3 + vLLM zero-shot inference.')
    parser.add_argument('--model_dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B')
    parser.add_argument('--text', type=str,
                        default='收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，'
                                '笑容如花儿般绽放。')
    parser.add_argument('--prompt_text', type=str,
                        default='You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。')
    parser.add_argument('--prompt_wav', type=str, default='asset/zero_shot_prompt.wav')
    parser.add_argument('--iters', type=int, default=1, help='repeat the synthesis N times to average the RTF')
    parser.add_argument('--out', type=str, default='output/vllm_out_0.wav')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--no_vllm', action='store_true', help='load the plain PyTorch LLM instead, as a baseline')
    args = parser.parse_args()

    if torch.cuda.is_available() is False:
        raise SystemExit('no CUDA device visible in the container')

    logging.info('loading %s (vllm=%s)', args.model_dir, not args.no_vllm)
    t0 = time.time()
    cosyvoice = AutoModel(model_dir=args.model_dir, load_vllm=not args.no_vllm, fp16=args.fp16)
    logging.info('model loaded in %.1fs', time.time() - t0)
    logging.info('frontend in use: %r', cosyvoice.frontend.text_frontend)

    rtfs = []
    for i in range(args.iters):
        set_all_random_seed(i)
        t0 = time.time()
        chunks = list(cosyvoice.inference_zero_shot(args.text, args.prompt_text, args.prompt_wav, stream=False))
        if len(chunks) == 0:
            raise SystemExit('no audio produced')
        speech = torch.cat([c['tts_speech'] for c in chunks], dim=1)
        duration = speech.shape[1] / cosyvoice.sample_rate
        elapsed = time.time() - t0
        rtfs.append(elapsed / duration)
        logging.info('iter %d: %.2fs audio in %.2fs, rtf=%.2f', i, duration, elapsed, rtfs[-1])
        if i == 0:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            torchaudio.save(args.out, speech, cosyvoice.sample_rate)
            logging.info('saved %s', args.out)

    logging.info('mean rtf over %d run(s): %.2f', len(rtfs), sum(rtfs) / len(rtfs))
    logging.info('cuda peak memory: %.2f GB', torch.cuda.max_memory_allocated() / 1024 ** 3)


if __name__ == '__main__':
    main()
