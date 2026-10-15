# CosyVoice3 Docker 部署（vLLM 加速 + 三档文本前端对比）

把 Fun-CosyVoice3-0.5B 的完整推理链路打进一个 Linux 镜像，用 Docker Desktop 的
GPU 直通跑，并提供一个开关用来对比 **ttsfrd → wetext → 内置轻量规则** 三种文本
前端在复杂输入下的差异。

- 基础镜像：`nvidia/cuda:12.8.1-devel-ubuntu22.04` + Python 3.10（cudnn 由 torch 的
  pip 依赖自带，镜像里不装；`12.8.1` 没有 `cudnn-devel` 这个 tag）
- LLM 加速：vLLM 0.11.0（V1 engine）
- 权重：**挂载**宿主机 `pretrained_models/`，不烘进镜像
- 对外：CLI 脚本、三档对比工具、FastAPI HTTP 服务、Gradio WebUI

---

## 1. 前置条件

- Docker Desktop 已开启 **WSL2 后端** 和 **GPU 支持**（Settings → Resources → WSL integration）
- 宿主机有 NVIDIA 驱动，且 WSL2 下能跑 CUDA

验证 GPU 直通：

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

## 2. 准备资源

模型和文本前端资源都不在 git 里（`pretrained_models/*` 被 gitignore），需要单独拉：

```bash
bash docker/vllm/fetch_assets.sh
```

会下载（共约 6.5 GB）：

| 目录 | 内容 | 用途 |
|---|---|---|
| `pretrained_models/Fun-CosyVoice3-0.5B` | 模型权重 | 推理必需，挂载 |
| `pretrained_models/CosyVoice-ttsfrd` | ttsfrd wheel + `resource/`（483 MB） | ttsfrd 前端 |
| `pretrained_models/wetext` | wetext FST 语法文件（44 MB） | wetext 前端 |

> **为什么 wetext 要单独拉？** 原代码里 `wetext.Normalizer()` 会调
> `snapshot_download("pengzhendong/wetext")`，而 modelscope 现在**匿名请求返回 403**，
> 所以 wetext 前端一直静默降级到"无前端"。这里改为把 FST 文件下下来，通过
> `COSYVOICE_WETEXT_DIR` 直接指过去，`Normalizer(tagger_path=..., verbalizer_path=...)`
> 完全绕开 API。同理，ttsfrd 的 `resource.zip` 是 git-lfs 文件，需要额外 `git lfs pull`。

## 3. 构建

```bash
docker build -f docker/vllm/Dockerfile -t cosyvoice-vllm:0.11.0 .
```

首次构建约 20–40 分钟（取决于网络，要下 torch 2.8 + vLLM 的 CUDA 依赖，数 GB）。

**版本冲突说明**：CosyVoice 的 `requirements.txt` 锁 `torch==2.3.1`，而 vLLM 0.11.0
锁 `torch==2.8.0`（并自带匹配的 `torchaudio==2.8.0`，所以不会出现 torch/torchaudio
错配）。Dockerfile 里把 `torch`/`transformers` 的固定版本从
`requirements.docker.txt` 中剥离，先装 vLLM 再装其余依赖，保证只解析一次 torch。

如果新版 torch 导致 flow / vocoder 出问题，用无 vLLM 的精确上游栈重建：

```bash
docker build -f docker/vllm/Dockerfile -t cosyvoice-plain:2.3.1 \
  --build-arg INSTALL_VLLM=0 .
```

## 4. 运行

全部命令都用 `-f` 指定 compose 文件，或在 `docker/vllm/` 目录下执行。

```bash
# FastAPI，监听 :50000
docker compose -f docker/vllm/docker-compose.yml up -d

# WebUI，监听 :50001
docker compose -f docker/vllm/docker-compose.yml --profile ui up -d webui

# 单次 vLLM 推理 + RTF 统计
docker compose -f docker/vllm/docker-compose.yml --profile tools run --rm infer
```

> **一次只跑一个加载模型的服务。** 每个服务都会把整条链路放进显存（约 4–5 GB），
> 8 GB 卡上 `api` 和 `webui` 同时开会 OOM。所以 `webui` / 工具类服务都放在
> profile 里，默认不随 `up` 启动。

FastAPI 接口（复用上游 `runtime/python/fastapi/server.py`，六个端点）：

```bash
# 中文记得走文件，别直接写在 -F 里（见下方「编码」）
printf '今天天气不错，我们出去走走吧。\n' > /tmp/tts.txt
printf 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。\n' > /tmp/prompt.txt

curl -F "tts_text=</tmp/tts.txt" -F "prompt_text=</tmp/prompt.txt" \
     -F "prompt_wav=@asset/zero_shot_prompt.wav" \
     http://127.0.0.1:50000/inference_zero_shot -o out.raw     # 裸 PCM：int16 / 24 kHz 单声道
curl http://127.0.0.1:50000/health
```

端点与入参（v3 只支持下面这四种用法）：

| 端点 | 必填字段 |
|---|---|
| `/inference_zero_shot` | `tts_text`、`prompt_text`、`prompt_wav` |
| `/inference_cross_lingual` | `tts_text`、`prompt_wav` |
| `/inference_instruct2` | `tts_text`、`instruct_text`、`prompt_wav` |
| `/inference_sft`、`/inference_instruct` | 需要 `spk_id`，**v3 不可用**（见「已知限制」） |

> **`tts_text` / `instruct_text` 必须带 system prompt。** v3 不替你拼
> `You are a helpful assistant.<|endofprompt|>`：`inference_zero_shot` 把它放在 `prompt_text`
> 里，`inference_cross_lingual` 直接写在 `tts_text` 开头，`inference_instruct2` 写在
> `instruct_text` 里（如 `You are a helpful assistant. 请用广东话表达。<|endofprompt|>`）。
> 漏了这段，LLM 几乎不吐 token，会在声码器处炸成
> `Kernel size can't be greater than actual input size`（看着像代码 bug，其实是入参问题）。
> 另外 `text_normalize` 在文本里出现 `<|`/`|>` 时会**跳过**文本前端，这是刻意设计。

> **编码**：multipart 的表单字段是按字节透传的。Windows 的 git bash / cmd 会把中文按系统 ANSI
> 码页（GBK）编码，服务端拿到的就是乱码（日志里表现为 `Ï£ÍûÄãÒÔºó` 这种）。用
> `-F "field=<utf8.txt"` 从 UTF-8 文件读，或直接在 Linux/WSL 里发请求。


## 5. 三档前端对比

这是本套 Docker 化里跟部署无关、但最实用的一环：把三种文本前端在**同一批复杂输入**
上的输出并排打印，并可选地各自合成音频。

```bash
docker compose -f docker/vllm/docker-compose.yml --profile tools run --rm compare
```

对比脚本会**只加载一次模型**，然后通过改写 `frontend.text_frontend` 在三档之间切换
（ttsfrd 引擎只构造一次并缓存），所以不会重复吃显存。

覆盖的场景：金额/小数、日期时间、百分比、电话号码、序数、中英混排、区间与正负号、
多音字、时间格式、版本号。产物：

```
output/frontend_compare/text_report.tsv      # 归一化文本对比表
output/frontend_compare/<mode>/<场景>.wav    # 每档的合成音频（--synthesize）
```

常用变体：

```bash
# 只对比 ttsfrd 和内置规则，跳过音频
... run --rm compare python docker/vllm/scripts/compare_frontend.py --modes ttsfrd,builtin

# 只跑前 3 个场景 + 出音频，快速验证
... run --rm compare python docker/vllm/scripts/compare_frontend.py --synthesize --limit 3
```

### 三档到底是什么关系

`CosyVoiceFrontEnd.text_normalize()` 里，非 ttsfrd 分支的**内置清洗规则（去空白、
去角标、句末标点归一、分句）无论如何都会跑**，wetext 只是在它之上先叠一层 FST 文法：

| 档位 | 实际执行 |
|---|---|
| `ttsfrd` | ttsfrd 引擎独占，**不走**内置规则 |
| `wetext` | wetext FST 归一化 **+** 内置清洗规则 |
| `builtin` | 只有内置清洗规则（**不做数字/单位归一化**） |

所以 "builtin" 才是你什么都不装时实际拿到的效果。CosyVoice3 之所以在缺前端时仍能读对
数字，是模型自身在未归一化语料上训练出来的能力，不是前端在起作用——差别会体现在
金额、日期、单位这类格式复杂的输入上。

### 全局前端开关

除对比脚本外，各服务的前端由环境变量控制，未设置时保持"三级自动降级"：

```bash
COSYVOICE_TEXT_FRONTEND=ttsfrd docker compose -f docker/vllm/docker-compose.yml up -d
COSYVOICE_TEXT_FRONTEND=wetext docker compose ... up -d
COSYVOICE_TEXT_FRONTEND=none   docker compose ... up -d
```

## 6. 迁移到另一台机器

因为权重是挂载的，迁移 = 镜像 + 数据两部分：

```bash
# 源机器
docker save cosyvoice-vllm:0.11.0 -o cosyvoice-vllm.tar   # 约 15 GB
tar czf cosyvoice-assets.tar.gz pretrained_models         # 约 6.5 GB（可裁剪，见下）

# 目标机器
docker load -i cosyvoice-vllm.tar
tar xzf cosyvoice-assets.tar.gz
docker build ... # 不需要；直接 compose up
```

`pretrained_models/Fun-CosyVoice3-0.5B` 里可以安全删掉这些来省空间（约 4 GB）：

| 可删文件 | 省下 | 前提 |
|---|---|---|
| `llm.rl.pt` | 1.9 GB | 不用 RL 微调版 LLM |
| `flow.decoder.estimator.fp32.onnx` | 1.3 GB | 不用 TensorRT（本镜像默认 `load_trt=False`） |
| `speech_tokenizer_v3.batch.onnx` | 0.9 GB | 不做批量提取 |
| `.git` | ~10 GB | 不需要 git 历史 |

## 7. 已知限制

- **`load_trt` 未启用。** TensorRT 那套（`tensorrt-cu12*`，约 1.5 GB）已从镜像里剔除，
  `flow.decoder.estimator.fp32.onnx` 不会被用上。要开启就把依赖加回
  `requirements.docker.txt` 并给 `AutoModel` 传 `load_trt=True`——注意首次转换 `.plan`
  很慢。
- **onnxruntime 可能退回 CPU。** `onnxruntime-gpu==1.18.0` 是针对 cuDNN 8 构建的，
  而基础镜像是 CUDA 12.8 / cuDNN 9，CUDA provider 可能加载失败并静默回退到 CPU。
  只影响提示音频的特征提取，不影响正确性。想修就把 `onnxruntime-gpu` 换成更新的版本。
- **首次 vLLM 运行会写模型目录。** `load_vllm` 会把 LLM 导出到
  `<model_dir>/vllm/`（约 1 GB）并复用，所以 `pretrained_models` 是**读写挂载**。
  该目录被 gitignore 覆盖，不会污染 git 状态。
- **镜像体积约 15 GB**（CUDA devel 基础镜像 + torch 2.8 + vLLM + 依赖）。
- **`inference_sft` / `inference_instruct` 在 v3 上用不了。** 这两个端点要按 `spk_id` 取
  预置音色，但 `Fun-CosyVoice3-0.5B` 里**没有 `spk2info.pt`**（v3 是零样本/指令模型），
  调用会 `KeyError`。用 `zero_shot` / `cross_lingual` / `instruct2` 代替。

## 8. 文件清单

```
docker/vllm/
├── Dockerfile                    # 镜像定义
├── docker-compose.yml            # api / webui / infer / compare
├── requirements.docker.txt       # requirements.txt 的推理子集（见文件内注释）
├── fetch_assets.sh               # 拉取挂载用的模型与前端资源
├── README.md
└── scripts/
    ├── vllm_infer.py             # vLLM 零样本推理 + RTF
    ├── compare_frontend.py       # 三档文本前端对比
    └── serve_fastapi.py          # FastAPI（复用上游 server.py）
```

同时改动了仓库里两处代码，用于支撑上面的开关：

- `cosyvoice/cli/frontend.py`：新增 `COSYVOICE_TEXT_FRONTEND`（强制指定前端，不再降级）
  和 `COSYVOICE_WETEXT_DIR`（让 wetext 直接读本地 FST）。未设置这两个变量时行为与上游一致。
- `.dockerignore`、`.gitignore`：构建上下文与 `output/` 的忽略规则。

另外，**没有改动上游 `runtime/python/fastapi/server.py`**。它把上传的提示音频先
`load_wav(..., 16000)` 成 tensor 再往下传，而 v2/v3 前端要的是文件本身（自己会按
24000/16000 各取一次），传 tensor 会抛 `TypeError: Invalid file: tensor([[...]])`。
这个问题在 `docker/vllm/scripts/serve_fastapi.py` 里修：启动时摘掉三个带 prompt 音频的路由，
换成把上传落临时文件、传路径、流结束后删除的版本。
