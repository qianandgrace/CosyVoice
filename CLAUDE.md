# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CosyVoice is Alibaba's LLM-based zero-shot text-to-speech (TTS) system, part of the FunAudioLLM family. There are three model generations, all supported by this single repo:

- **CosyVoice 1.0** (`CosyVoice-300M*`) — supervised semantic-token TTS with a 22050 Hz output.
- **CosyVoice 2.0** (`CosyVoice2-0.5B`) — 25 Hz token rate, streaming, built on Qwen2.
- **CosyVoice 3.0** (`Fun-CosyVoice3-0.5B-2512`) — DiT-based flow decoder + CausalHiFT vocoder, instruction following, 9 languages / 18+ Chinese dialects, pronunciation inpainting (pinyin / CMU phonemes). This is the current recommended model.

Output sample rate is 24 kHz for CosyVoice2/3 and comes from each model's own config for 1.0.

## Setup

```sh
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
git submodule update --init --recursive   # if the clone failed to pull submodules
```

`third_party/Matcha-TTS` is a **git submodule** and is required at import time. Any script that imports `cosyvoice` needs it on `sys.path` (repo scripts do `sys.path.append('third_party/Matcha-TTS')`); alternatively `export PYTHONPATH=third_party/Matcha-TTS`.

Python 3.10 + CUDA. `pip install -r requirements.txt` (pins `torch==2.3.1`, `transformers==4.51.3`, `diffusers==0.29.0`). `sox`/`libsox-dev` is needed on Linux for some deps.

Pretrained weights are downloaded into `pretrained_models/<model-id>/` via `modelscope.snapshot_download` or `huggingface_hub.snapshot_download` (see README). Each model dir contains a `cosyvoice*.yaml` config plus `llm.pt`, `flow.pt`, `hift.pt`, `campplus.onnx`, `speech_tokenizer_v*.onnx`, `spk2info.pt`.

Optional text-normalization frontend (`ttsfrd`) is installed from `pretrained_models/CosyVoice-ttsfrd/` (better than the default `wetext` fallback).

## Common commands

```sh
python example.py                      # runs cosyvoice3_example() by default; also has cosyvoice{,2,3}_example()
python3 webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M
python vllm_example.py                 # vLLM-accelerated CosyVoice2/3

# training (run from an examples/libritts/cosyvoice* dir; see its run.sh for stage numbers)
torchrun --nnodes=1 --nproc_per_node=$N --rdzv_id=$job_id --rdzv_backend=c10d --rdzv_endpoint=localhost:1234 \
  cosyvoice/bin/train.py \
  --config conf/cosyvoice3.yaml --model llm \
  --train_data data/train.data.list --cv_data data/dev.data.list \
  --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
  --onnx_path $pretrained_model_dir --checkpoint $pretrained_model_dir/llm.pt \
  --model_dir exp/cosyvoice3/llm/torch_ddp --tensorboard_dir tensorboard/cosyvoice3/llm/torch_ddp

# export for deployment / speedup
python cosyvoice/bin/export_jit.py --model_dir $model_dir
python cosyvoice/bin/export_onnx.py --model_dir $model_dir
python cosyvoice/bin/average_model.py --dst_model $model --src_path $exp_dir --num 5 --val_best
```

There is no test suite. Lint is enforced in CI (`.github/workflows/lint.yml`): `flake8` with `--max-line-length 180` and a specific ignore list, plus no-tabs and no-trailing-whitespace checks. Model weights, `.wav`, and generated files are gitignored.

## Architecture

### Three-stage synthesis pipeline

CosyVoice composes three trained modules in sequence, each a separate `torch.nn.Module` trained independently:

1. **LLM** (`cosyvoice/llm/llm.py`) — autoregressive Qwen2-based transformer that maps text tokens (+ speaker embedding + optional prompt speech tokens) to discrete **speech tokens** (semantic tokens, e.g. 6561 vocab for v3). `TransformerLM` (v1), `CosyVoice2LM`/`CosyVoice3LM` wrap a `Qwen2Encoder`/Qwen2 backbone.
2. **Flow matching** (`cosyvoice/flow/`) — converts speech tokens into an 80-dim mel spectrogram. v1/v2 use `MaskedDiffWithXvec` (Matcha-style); v3 uses `CausalMaskedDiffWithDiT` with a `DiT` decoder (`flow/DiT/dit.py`) and causal/streaming support.
3. **Vocoder** (`cosyvoice/hifigan/`) — HiFiGAN-style generator (`hift`) that converts mel to waveform. v3 uses `CausalHiFTGenerator` (causal ISTFT + F0 predictor); the `HiFiGan` module adds discriminator + GAN losses for training.

### Entry points and class hierarchy

`cosyvoice/cli/cosyvoice.py` is the public API. `AutoModel(model_dir=...)` inspects which yaml file is present (`cosyvoice.yaml` / `cosyvoice2.yaml` / `cosyvoice3.yaml`) and returns the matching `CosyVoice` / `CosyVoice2` / `CosyVoice3` class (v3 inherits v2 inherits v1). Each exposes `inference_sft`, `inference_zero_shot`, `inference_cross_lingual`, `inference_instruct{2}`, and `inference_vc`, all generators yielding `{'tts_speech': tensor}`.

- `cosyvoice/cli/frontend.py` (`CosyVoiceFrontEnd`) does all text/speech preprocessing: tokenization, text normalization (paragraph splitting via `split_paragraph`), speaker embedding (CAM++ ONNX), speech-token extraction (whisper log-mel → `speech_tokenizer_v*.onnx`), and mel feature extraction (Matcha `mel_spectrogram`).
- `cosyvoice/cli/model.py` (`CosyVoiceModel` / `CosyVoice2Model` / `CosyVoice3Model`) runs the actual synthesis loop: the LLM runs in a background thread filling a token buffer, and `token2wav` converts tokens → mel → waveform with overlap/caching for streaming. Per-request state is keyed by a `uuid`.

### Config-driven instantiation

The single most important architectural pattern: **everything is wired from one HyperPyYAML file** (`.yaml` in the model dir, plus `examples/libritts/cosyvoice*/conf/*.yaml` for training). Classes/functions are instantiated with `!new:` / `!name:` tags and cross-referenced with `!ref <key>`, so the full model graph (llm, flow, hift, tokenizer, data pipeline, training conf) is defined declaratively and parsed by `hyperpyyaml.load_hyperpyyaml`. To understand how any component is configured, read the yaml first, then the class it references.

### Training

`cosyvoice/bin/train.py` is the single training entry point. It uses `torchrun` (torch_ddp) or DeepSpeed; `--model` selects `llm`, `flow`, or `hifigan` (only one is trained per run — the other two are nulled out via yaml overrides). `--model hifigan` enables GAN training (`train_conf_gan`, discriminator). DPO is supported via `--dpo --ref_model`. Training data is parquet produced by `tools/make_parquet_list.py`; feature extraction is now **online** (whisper/embedding computed in the dataset pipeline via the `--onnx_path`), so pre-extraction (`tools/extract_embedding.py`, `tools/extract_speech_token.py`) is optional. Recipes live under `examples/libritts/cosyvoice{,2,3}/` as stage-based `run.sh` scripts (data download → prepare → extract → parquet → train → average → export).

RL fine-tuning (GRPO) for the CosyVoice2 LLM lives in `examples/grpo/cosyvoice2/`, using the external `veRL` framework.

### Inference accelerators / deployment

- **vLLM**: CosyVoice2/3 LLM only. `cosyvoice/vllm/cosyvoice2.py` is an inference-only Qwen2 model; `load_vllm` in `cli/model.py` exports weights then runs `LLMEngine`. Supports vLLM 0.9.0 (legacy) and 0.11.x+ (V1 engine only); 0.10.x is unsupported.
- **TensorRT**: `load_trt` swaps the flow decoder estimator for a `.plan` engine (`convert_onnx_to_trt` in `utils/file_utils.py`).
- **Deployment servers**: `runtime/python/fastapi/` and `runtime/python/grpc/` (gRPC protobufs are gitignored `*_pb2*.py` and generated with `grpcio-tools`). `runtime/triton_trtllm/` provides a TensorRT-LLM/Triton deployment for CosyVoice2/3.

## Conventions and gotchas

- **Special control tokens** are central to the API: `<|endofprompt|>` separates the instruct prefix from the text; `[breath]`, `[laughter]`, `<strong>`, `[cough]`, etc. control prosody; pinyin `[j][ǐ]` and CMU phonemes `[JH]` do pronunciation inpainting (see `cosyvoice/tokenizer/tokenizer.py` for the full token lists, `cosyvoice/utils/common.py` for instruct keywords).
- `CosyVoice3`/`CosyVoice2` prefix text with an instruct system prompt ("You are a helpful assistant.<|endofprompt|>") by default. Japanese input must be converted to katakana.
- Streaming tokens are rate-limited by hop/chunk-size constants in `cli/model.py` (`token_hop_len`, `chunk_size: 25`, `pre_lookahead_len`); these must match the model's `static_chunk_size` from training.
- The `ttsfrd` text frontend is preferred but optional; the code silently falls back to `wetext` → no frontend. Frontend choice is logged at init.
