"""Minimal end-to-end check for CosyVoice3 local setup.

Runs a single zero-shot synthesis and saves the output wav, so we can verify
that the environment (torch/CUDA, onnxruntime, Matcha-TTS, model weights) works
before running the full example.py.
"""
import sys
import time

sys.path.append('third_party/Matcha-TTS')

import torch  # noqa: E402
import torchaudio  # noqa: E402
from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

print('torch:', torch.__version__, '| cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))

t0 = time.time()
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
print('model loaded in {:.1f}s | sample_rate={}'.format(time.time() - t0, cosyvoice.sample_rate))

t0 = time.time()
n = 0
for i, j in enumerate(cosyvoice.inference_zero_shot(
        '八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。',
        'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
        './asset/zero_shot_prompt.wav', stream=False)):
    out = 'quickstart_out_{}.wav'.format(i)
    torchaudio.save(out, j['tts_speech'], cosyvoice.sample_rate)
    print('saved', out, '| shape', tuple(j['tts_speech'].shape))
    n += 1

print('OK: synthesized {} chunk(s) in {:.1f}s'.format(n, time.time() - t0))
