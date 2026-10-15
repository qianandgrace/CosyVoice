"""FastAPI wrapper around the upstream CosyVoice HTTP server, with vLLM support.

runtime/python/fastapi/server.py already defines the six inference endpoints, but
the model is only constructed under __main__ and there is no way to turn vLLM (or
the frontend choice) on. This loads the module up front, injects a model
configured from the environment, and serves it.

The three endpoints that take a prompt wav are also re-registered here. Upstream
loads the upload with `load_wav(prompt_wav.file, 16000)` and hands that 16 kHz
tensor to `inference_*`, but the v2/v3 frontend loads prompt_wav itself -- at
24000 Hz for the flow features and 16000 Hz for the speaker embedding -- so it
needs the original file, not an already-resampled tensor. Passing the tensor
fails with "TypeError: Invalid file: tensor([[...]])" inside torchaudio.

    python docker/vllm/scripts/serve_fastapi.py --port 50000

Environment:
    COSYVOICE_MODEL_DIR          model directory (default pretrained_models/Fun-CosyVoice3-0.5B)
    COSYVOICE_LOAD_VLLM          1 to serve the LLM through vLLM (default 1)
    COSYVOICE_TEXT_FRONTEND      ttsfrd | wetext | none (default: degrade automatically)
"""
import argparse
import importlib.util
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, 'third_party', 'Matcha-TTS'))

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import File, Form, UploadFile  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
from cosyvoice.utils.file_utils import logging  # noqa: E402

UPSTREAM_SERVER = os.path.join(REPO_ROOT, 'runtime', 'python', 'fastapi', 'server.py')

# Re-registered because upstream feeds the frontend a pre-resampled tensor.
PROMPT_WAV_PATHS = ('/inference_zero_shot', '/inference_cross_lingual', '/inference_instruct2')

if os.environ.get('COSYVOICE_LOAD_VLLM', '1') == '1':
    from vllm import ModelRegistry  # noqa: E402
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM  # noqa: E402
    ModelRegistry.register_model('CosyVoice2ForCausalLM', CosyVoice2ForCausalLM)


def load_upstream_app():
    spec = importlib.util.spec_from_file_location('cosyvoice_fastapi_server', UPSTREAM_SERVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def drop_routes(app, paths):
    app.router.routes = [r for r in app.router.routes if getattr(r, 'path', None) not in paths]
    app.openapi_schema = None


def spool_upload(upload):
    """Persist an UploadFile so load_wav can reopen it once per consumer."""
    suffix = os.path.splitext(upload.filename or '')[1] or '.wav'
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, 'wb') as f:
        shutil.copyfileobj(upload.file, f)
    return path


def stream_pcm(model_output, prompt_path):
    try:
        for i in model_output:
            yield (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
    finally:
        try:
            os.remove(prompt_path)
        except OSError:
            pass


def register_prompt_wav_routes(app, server):
    @app.post('/inference_zero_shot')
    async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
        path = spool_upload(prompt_wav)
        return StreamingResponse(stream_pcm(server.cosyvoice.inference_zero_shot(tts_text, prompt_text, path), path))

    @app.post('/inference_cross_lingual')
    async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
        path = spool_upload(prompt_wav)
        return StreamingResponse(stream_pcm(server.cosyvoice.inference_cross_lingual(tts_text, path), path))

    @app.post('/inference_instruct2')
    async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
        path = spool_upload(prompt_wav)
        return StreamingResponse(stream_pcm(server.cosyvoice.inference_instruct2(tts_text, instruct_text, path), path))


def main():
    parser = argparse.ArgumentParser(description='CosyVoice FastAPI server.')
    parser.add_argument('--model_dir', type=str,
                        default=os.environ.get('COSYVOICE_MODEL_DIR', 'pretrained_models/Fun-CosyVoice3-0.5B'))
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--load_vllm', type=int, default=int(os.environ.get('COSYVOICE_LOAD_VLLM', '1')))
    args = parser.parse_args()

    server = load_upstream_app()
    drop_routes(server.app, set(PROMPT_WAV_PATHS))
    register_prompt_wav_routes(server.app, server)

    logging.info('loading %s (vllm=%s)', args.model_dir, bool(args.load_vllm))
    server.cosyvoice = AutoModel(model_dir=args.model_dir, load_vllm=bool(args.load_vllm), fp16=False)
    logging.info('frontend in use: %r', server.cosyvoice.frontend.text_frontend)

    @server.app.get('/health')
    async def health():
        return {'status': 'ok', 'frontend': server.cosyvoice.frontend.text_frontend,
                'sample_rate': server.cosyvoice.sample_rate}

    uvicorn.run(server.app, host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    main()
