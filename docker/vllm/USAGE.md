# CosyVoice3 Docker 使用手册

面向日常使用的速查文档。部署/构建/迁移见 [README.md](README.md)，这里只讲**怎么用**。

所有命令都在**仓库根目录**执行：

```bash
cd "C:/Users/qian gao/git_project/CosyVoice"
```

---

## 0. 前置确认

```bash
docker images | grep cosyvoice-vllm        # 应有 cosyvoice-vllm:0.11.0
ls pretrained_models/                      # Fun-CosyVoice3-0.5B / CosyVoice-ttsfrd / wetext
```

缺资源就跑 `bash docker/vllm/fetch_assets.sh`；缺镜像就看 README 的构建章节。

> **8GB 显存（RTX 4060 Laptop）同一时刻只能开一个加载模型的服务。** api、webui、infer、compare
> 都会把整条链路放进显存（约 4~5GB），同时开两个会 OOM。切换前先 `down`。

---

## 1. 拉起服务

### FastAPI（推荐，日常用这个）

```bash
docker compose -f docker/vllm/docker-compose.yml up -d
curl http://127.0.0.1:50000/health
# {"status":"ok","frontend":"ttsfrd","sample_rate":24000}
```

加载约 2~3 分钟；`docker ps` 看到 `(healthy)` 才算就绪。停止：

```bash
docker compose -f docker/vllm/docker-compose.yml down
```

### WebUI

```bash
docker compose -f docker/vllm/docker-compose.yml --profile ui up -d webui
# 浏览器打开 http://127.0.0.1:50001
```

停止：

```bash
docker compose -f docker/vllm/docker-compose.yml --profile ui down
```

> `webui` 在 `ui` group 里，所以不带 `--profile ui` 的 `up -d` 只会起 api。
> `infer` / `compare` 在 `tools` group 里，用 `run --rm` 一次性执行、不常驻。

### 三个前端开关（可选）

```bash
COSYVOICE_TEXT_FRONTEND=ttsfrd docker compose -f docker/vllm/docker-compose.yml up -d
COSYVOICE_TEXT_FRONTEND=wetext docker compose -f docker/vllm/docker-compose.yml up -d
COSYVOICE_TEXT_FRONTEND=none   docker compose -f docker/vllm/docker-compose.yml up -d
```

不设则按 `ttsfrd → wetext → 内置规则` 自动降级。三种前端的差异对比见第 5 节。

---

## 2. WebUI 怎么用

用的是 `docker/vllm/scripts/serve_webui.py`（不是上游那个只面向 300M 的 `webui.py`），
v3 能跑的模式直接列出来，跑不了的自动隐藏。

### 推理模式

| 模式 | 需要填 | 说明 |
|---|---|---|
| **零样本复刻** | 参考音频 + 参考音频文本 + 合成文本 | 最常用。音色完全由参考音频决定 |
| **跨语种复刻** | 参考音频 + 合成文本 | 合成文本可以和参考音频不同语言 |
| **自然语言控制** | 参考音频 + 指令 + 合成文本 | 用指令控风格/方言/语速，音色仍来自参考音频 |
| ~~预置音色~~ | — | v3 没有 `spk2info.pt`，列表为空时该模式**不显示** |

### 操作要点

1. **参考音频**：3~30 秒、采样率 ≥16kHz。上传或现场录制都行，同时给时优先用上传的。
2. **参考音频文本**：必须和音频**实际说的一样**，不用管标点差异。填错会明显影响效果。
3. **指令文本**（自然语言控制模式）：**只写指令本身就行**，比如 `请用四川话表达`，
   UI 会自动补成 `You are a helpful assistant. 请用四川话表达。<|endofprompt|>`。
   上面的「指令预设」下拉里内置了 26 条（各方言、快慢语速、开心/伤心/生气、
   小猪佩奇风格、机器人风格……），点一下会填到输入框，可以直接改。
4. **随机种子**：不满意就点 🎲 换种子重roll。同一个种子结果可复现。
5. **流式**：开流式会边生成边播；**语速调节只在非流式下生效**。

---

## 3. 单条推理（三种方式）

### 方式一：最快自检（一条命令，出 wav + RTF）

```bash
docker compose -f docker/vllm/docker-compose.yml --profile tools run --rm infer
# 产物：output/vllm_out_0.wav
# 日志：iter 0: 11.36s audio in 4.89s, rtf=0.43
```

要换文本/音色，改 `docker/vllm/scripts/vllm_infer.py` 里的 `--text` / `--prompt_wav`
默认值，或从命令行传（中文的坑见第 6 节）：

```bash
docker compose -f docker/vllm/docker-compose.yml --profile tools run --rm infer \
  python docker/vllm/scripts/vllm_infer.py \
  --text "要合成的文本" \
  --prompt_wav asset/zero_shot_prompt.wav \
  --iters 3 --out output/my.wav
```

### 方式二：HTTP 请求

```bash
printf '今天天气不错，我们出去走走吧。\n' > /tmp/tts.txt
printf 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。\n' > /tmp/prompt.txt

curl -F "tts_text=</tmp/tts.txt" -F "prompt_text=</tmp/prompt.txt" \
     -F "prompt_wav=@asset/zero_shot_prompt.wav" \
     http://127.0.0.1:50000/inference_zero_shot -o out.raw
```

返回的是**裸 PCM：int16 / 24kHz / 单声道**。转成能播的 wav：

```bash
ffmpeg -f s16le -ar 24000 -ac 1 -i out.raw out.wav
```

没装 ffmpeg 就用镜像里的 python（把 `out.raw` 放到 `output/` 下）：

```bash
docker compose -f docker/vllm/docker-compose.yml run --rm --no-deps api python -c "
import numpy as np, soundfile as sf
x = np.fromfile('/workspace/CosyVoice/output/out.raw', dtype='<i2').astype('float32') / 32768
sf.write('/workspace/CosyVoice/output/out.wav', x, 24000)"
```

端点与必填字段：

| 端点 | 必填 |
|---|---|
| `/inference_zero_shot` | `tts_text`、`prompt_text`、`prompt_wav` |
| `/inference_cross_lingual` | `tts_text`、`prompt_wav`（`tts_text` 要自带 system prompt） |
| `/inference_instruct2` | `tts_text`、`instruct_text`、`prompt_wav` |
| `/inference_sft`、`/inference_instruct` | 需要 `spk_id`，**v3 不可用** |
| `/health` | — |

> 走 HTTP 时 **system prompt 要你自己拼**（WebUI 才会自动补）：
> - `prompt_text` 用 `You are a helpful assistant.<|endofprompt|>参考音频说的话`
> - `instruct_text` 用 `You are a helpful assistant. 请用四川话表达。<|endofprompt|>`
> - `cross_lingual` 的 `tts_text` 开头加 `You are a helpful assistant.<|endofprompt|>`
>
> 漏了这段，LLM 几乎不吐 token，会在声码器里炸
> `Kernel size can't be greater than actual input size` —— 看着像代码 bug，其实是入参问题。

### 方式三：Python 里直接调

```bash
docker compose -f docker/vllm/docker-compose.yml --profile tools run --rm infer python - <<'PY'
import sys
sys.path.append('/workspace/CosyVoice')
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model('CosyVoice2ForCausalLM', CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio, torch
m = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_vllm=True, fp16=False)
out = list(m.inference_zero_shot('要合成的文本',
                                 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                                 'asset/zero_shot_prompt.wav', stream=False))
torchaudio.save('/workspace/CosyVoice/output/py_out.wav',
                torch.cat([i['tts_speech'] for i in out], dim=1), m.sample_rate)
PY
```

---

## 4. 换音色

**v3 没有预置音色表**（`Fun-CosyVoice3-0.5B` 不带 `spk2info.pt`）。
在本地，**音色 = 你喂的参考音频**。两种做法：

### 直接用（换音频即换音色）

WebUI 里换「参考音频」；HTTP 里换 `prompt_wav=@我的音色.wav`（记得同时改 `prompt_text`）。

### 注册成命名音色，当"音色表"复用

```python
cosyvoice.add_zero_shot_spk(prompt_text, '我的音色.wav', 'longanyang')   # 注意参数顺序
cosyvoice.save_spkinfo()          # 落到 pretrained_models/<model>/spk2info.pt
cosyvoice.list_available_spks()   # ['longanyang']

# 之后按名字用，prompt_text / prompt_wav 留空即可
cosyvoice.inference_zero_shot(文本, '', '', zero_shot_spk_id='longanyang')
```

`pretrained_models` 是**读写挂载**，所以 `save_spkinfo()` 会写到宿主机，容器重启后还在。
注册之后，WebUI 的「预置音色」模式也会自动出现（因为它读的就是这份 `spk2info`）。

### 想要云端的 `longanyang`（龙安洋）那种成品音色

那是**阿里云百炼 Model Studio 的 CosyVoice 系统音色**，不在本仓库里，
本仓库 grep 不到 `longanyang` / 龙安洋。要用得走云端 API：
`SpeechSynthesizer(model="cosyvoice-v2", voice="longanyang")`，
音色清单见[官方音色列表](https://help.aliyun.com/zh/model-studio/cosyvoice-voice-list)。

---

## 5. 三档文本前端对比

跟部署无关但最实用的一环：把 `ttsfrd` / `wetext` / 内置规则在同一批复杂输入上并排打印，可选各自合成音频。

```bash
docker compose -f docker/vllm/docker-compose.yml --profile tools run --rm compare
# 产物：output/frontend_compare/text_report.tsv
#       output/frontend_compare/<mode>/<场景>.wav
```

只比文本、不出音频（快很多）：

```bash
docker compose -f docker/vllm/docker-compose.yml --profile tools run --rm compare \
  python docker/vllm/scripts/compare_frontend.py --modes ttsfrd,builtin
```

实测差异（完整表见 `text_report.tsv`）：

| 输入 | ttsfrd | wetext | builtin |
|---|---|---|---|
| `￥1,299.50` | 一千二百九十九点五零元 | 一元,两百九十九点五零 ❌ | ￥1,299。50 ❌ |
| `5.2%` | 百分之五点二 | 百分之五点二 | 5。2% ❌ |
| `-10℃` | 零下十摄氏度 | 负十摄氏度 | -10℃ ❌ |
| `09:15` | 九点十五分 | 九点十五分 | 09:15 ❌ |

**结论**：ttsfrd 最准，wetext 次之（金额会误读），内置规则**完全不做数字归一化**，还会把小数点替换成句号。

---

## 6. 已知的坑

| 坑 | 现象 | 怎么办 |
|---|---|---|
| **Windows 命令行中文编码** | git bash 按 GBK 编码你敲的中文，服务端收到乱码（日志里是 `Ï£ÍûÄãÒÔºó`） | curl 用 `-F "字段=<utf8文件"`；或把文本写进脚本的 `default` |
| **system prompt 必须带** | 声码器报 `Kernel size can't be greater than actual input size` | 见第 3 节末尾那段。WebUI 已自动补，HTTP 要自己拼 |
| **首次走 vLLM 会写模型目录** | `pretrained_models/Fun-CosyVoice3-0.5B/vllm/` 多出约 1GB 导出 | 正常，只导出一次，后续复用；已被 gitignore 覆盖 |
| **onnxruntime 回退 CPU** | 启动时警告 `Failed to load library libonnxruntime_providers_cuda.so` | 预期行为，只影响提示音频特征提取，不影响正确性 |
| **生成长度离谱** | 13 个字合成出 11 秒音频 | 随机种子不好，换 🎲 或改「随机种子」重roll |
| **两个服务一起开** | OOM / CUDA out of memory | 8GB 卡一次只开一个，先 `down` 再起 |

---

## 7. 速查表

```bash
# 起 API（:50000）
docker compose -f docker/vllm/docker-compose.yml up -d

# 起 WebUI（:50001）
docker compose -f docker/vllm/docker-compose.yml --profile ui up -d webui

# 全部停掉
docker compose -f docker/vllm/docker-compose.yml --profile ui --profile tools down

# 单条推理 → output/vllm_out_0.wav
docker compose -f docker/vllm/docker-compose.yml --profile tools run --rm infer

# 三档前端对比 → output/frontend_compare/
docker compose -f docker/vllm/docker-compose.yml --profile tools run --rm compare

# 健康检查
curl http://127.0.0.1:50000/health

# 看日志
docker logs -f cosyvoice-api        # 或 cosyvoice-webui
```
