# CosyVoice 本地环境搭建踩坑记录

> 环境：Windows 11 · RTX 4060 Laptop 8GB · 驱动 592.00 (CUDA 13.1)
> 目标：新建 conda 虚拟环境，跑通 `example.py`（Fun-CosyVoice3-0.5B）
> 搭建日期：2026-10-14

本文记录从零搭建过程中遇到的所有问题、原因和解决办法。最终可用配置见文末。

---

## 坑 1：git 全局代理没开，submodule 拉不下来

**现象**

```text
$ git submodule update --init --recursive
fatal: unable to access 'https://github.com/shivammehta25/Matcha-TTS.git/':
Failed to connect to 127.0.0.1 port 7890 after 2105 ms: Could not connect to server
```

**原因**

git 全局配置了代理 `http://127.0.0.1:7890`（Clash/V2Ray 之类），但代理软件当时没运行。

```bash
$ git config --global --get http.proxy
http://127.0.0.1:7890
```

**解决**

两种方式二选一：

```bash
# 方式 A：本次命令临时绕过代理（推荐，直连 GitHub/modelscope 是通的）
git -c http.proxy= -c https.proxy= submodule update --init --recursive

# 方式 B：把代理软件打开，保持 7890 端口可用
```

> 同理，后面所有 `git clone`（拉模型）都需要加 `-c http.proxy= -c https.proxy=`，
> 否则会走那个不存在的代理。

---

## 坑 2：`pip install -r requirements.txt` 失败 —— `pkg_resources` 缺失

**现象**

```text
Collecting openai-whisper==20231117 (from -r requirements.txt)
  Getting requirements to build wheel ... error
  ModuleNotFoundError: No module named 'pkg_resources'
ERROR: Failed to build 'openai-whisper' when getting requirements to build wheel
```

**原因**

`openai-whisper==20231117` 只有源码包（sdist），构建时其 `setup.py` 里 `import pkg_resources`。
而 pip 构建隔离环境里用的 setuptools ≥ 81 **已经移除了 `pkg_resources`**，所以构建失败。

**解决**

先降级 setuptools，再关掉构建隔离单独装 whisper，最后装其余依赖：

```bash
pip install "setuptools<81" wheel -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
pip install openai-whisper==20231117 --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
pip install -r requirements_clean.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

**附带**：`requirements.txt` 头两行是国外的 `--extra-index-url`（pytorch.org / azure），
国内直连很慢或不通，安装前先删掉：

```bash
grep -v -E '^--(extra-)?index-url' requirements.txt > requirements_clean.txt
```

---

## 坑 3：装到的 torch 是 **CPU 版**，`torch.cuda.is_available()` 为 False

**现象**

```text
>>> import torch
>>> torch.__version__
'2.3.1+cpu'
>>> torch.cuda.is_available()
False
```

**原因**

Windows 上 PyPI（以及阿里云 PyPI 镜像）提供的 `torch` 是 **CPU-only** 构建。
CUDA 版必须在 `download.pytorch.org` 的 `cu121` 源里，而 `requirements.txt` 里那行
`--extra-index-url https://download.pytorch.org/whl/cu121` 正是为它准备的 —— 但被我们在坑 2 里删掉了
（而且国外源在国内也慢）。

**解决**

从阿里云的 pytorch-wheels 镜像装 CUDA 版（用 `--no-deps` 避免动其它已装依赖）：

```bash
pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 \
  -f https://mirrors.aliyun.com/pytorch-wheels/cu121/ --no-deps
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.3.1+cu121 True
```

> 注意：py3.12 不被 torch 2.3.1 支持，必须用 Python 3.10。

---

## 坑 4：镜像**单连接限速 ~0.5MB/s**，2.4GB 的 torch 要下 80 分钟

**现象**

`pip install` 下载 `torch-2.3.1+cu121-win_amd64.whl`（2,423,503,526 字节）时速度只有
~0.5MB/s，估计要 80 分钟。

**原因**

阿里云 pytorch-wheels 镜像**按单连接限速**。实测：

| 方式 | 速度 |
|---|---|
| 单连接 curl | ~0.5 MB/s |
| 4 连接并行 range | ~2.1 MB/s |
| 16 连接并行 range | **~9.6 MB/s** |

（其它镜像实测：SJTU 不通、tuna/nju/ustc 不托管 pytorch-wheels、华为云 32KB/s，aliyun 最快但需并行。）

**解决**

用并行 range 下载把 wheel 拉到本地，再本地安装：

```bash
U="https://mirrors.aliyun.com/pytorch-wheels/cu121/torch-2.3.1%2Bcu121-cp310-cp310-win_amd64.whl"
DL=/tmp/torchdl; mkdir -p $DL
SIZE=2423503526; N=16; CHUNK=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  s=$((i*CHUNK)); e=$((s+CHUNK-1)); [ $e -ge $SIZE ] && e=$((SIZE-1))
  curl -s --retry 8 -r $s-$e -o "$DL/part_$(printf '%02d' $i)" "$U" &
done; wait
cat $DL/part_* > "$DL/torch-2.3.1+cu121-cp310-cp310-win_amd64.whl"   # 注意按 part_00..part_15 顺序
pip install "$DL/torch-2.3.1+cu121-cp310-cp310-win_amd64.whl" --no-deps
```

> 顺序很重要：分片文件名要零填充（`part_00` 而非 `part_0`），否则 `cat` 的字典序会拼错。

---

## 坑 5：文本前端 wetext 报 403，日志提示 `no frontend is avaliable`

**现象**

```text
https://www.modelscope.cn:443 "GET /api/v1/models/pengzhendong/wetext/revisions" 403
modelscope - ERROR - Authentication token does not exist
INFO no frontend is avaliable
```

**原因**

CosyVoice 的文本前端按 `ttsfrd` → `wetext` → 无 的顺序降级。
`wetext` 内部用 `snapshot_download("pengzhendong/wetext")` 从 **modelscope API** 拉模型，
而 modelscope 现在**公开模型也要求登录 token**，匿名请求返回 403。
（`ttsfrd` 的 wheel 是 Linux-only，Windows 用不了。）

**影响**

- **CosyVoice3 无影响**：v3 自带数字/符号/格式归一化，不依赖传统前端（见 README 的
  *Text Normalization ... without a traditional frontend module*）。
- CosyVoice2 影响较大，建议修。

**解决（可选）**

登录 modelscope 拿到 token 后重跑，wetext 会自动下载模型：

```bash
modelscope login --token <你的 token>
# 或设置环境变量 MODELSCOPE_API_TOKEN=<你的 token>
```

> 注：`snapshot_download` 没有离线开关（`local_files_only` 非公开），
> 非离线模式一定会先调 API，所以**只靠预下载缓存绕不过 403**。

---

## 坑 6：onnxruntime 没有 CUDA provider，语音 tokenizer 跑在 CPU 上

**现象**

```text
UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.
Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
```

**原因**

`requirements.txt` 在 Windows 上装的是 **CPU 版** `onnxruntime`（`onnxruntime-gpu` 是 Linux 专用标记）。

**影响**

不影响正确性，只是 CAM++ 说话人嵌入 / speech tokenizer 走 CPU，稍慢。

**解决（可选）**

```bash
pip install onnxruntime-gpu==1.18.0 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

（需匹配 CUDA 12 运行库；torch cu121 自带相关库，一般可直接用。）

---

## 坑 7：模型仓库 `.git` 巨大（约 10GB）

**现象**

`pretrained_models/Fun-CosyVoice3-0.5B` 用 `git clone` 下来总共 19GB，其中约 10GB 是
`.git` 里的 git-lfs 完整历史。

**原因**

git-lfs 会把所有版本对象保留在 `.git/lfs` 下。推理只需要工作区文件。

**解决（可选，删了不影响推理）**

```bash
rm -rf pretrained_models/Fun-CosyVoice3-0.5B/.git
```

---

## 坑 8：无害告警（可忽略）

以下告警不影响结果，属正常现象：

```text
Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.
UserWarning: torch.nn.utils.weight_norm is deprecated ...
pkg_resources is deprecated as an API (from lightning) ... pin to Setuptools<81.
```

- 第 1 条：Qwen2 的 sliding window 在 sdpa 后端未实现，实际推理正常。
- 第 3 条：正是坑 2 里我们主动降到 setuptools<81 造成的，属预期。

---

## 最终可用配置

| 组件 | 版本 / 位置 |
|---|---|
| conda 环境 | `cosyvoice`（Python 3.10.22），路径 `C:\project\envs\cosyvoice` |
| torch | `2.3.1+cu121`（`cuda.is_available() == True`） |
| torchaudio | `2.3.1+cu121` |
| transformers | `4.51.3` |
| numpy | `1.26.4` |
| onnxruntime | `1.18.0`（CPU 版） |
| Matcha-TTS | `third_party/Matcha-TTS`（submodule） |
| 模型 | `pretrained_models/Fun-CosyVoice3-0.5B` |
| GPU | NVIDIA GeForce RTX 4060 Laptop，8GB |

### 全新机器上的完整搭建步骤

```bash
# 0. 创建环境
conda create -n cosyvoice python=3.10 -y
conda activate cosyvoice

# 1. 拉 submodule（绕过坏代理）
cd CosyVoice
git -c http.proxy= -c https.proxy= submodule update --init --recursive

# 2. 装依赖（先删国外 index，降 setuptools，单独装 whisper）
M="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com"
grep -v -E '^--(extra-)?index-url' requirements.txt > requirements_clean.txt
pip install "setuptools<81" wheel $M
pip install openai-whisper==20231117 --no-build-isolation $M
pip install -r requirements_clean.txt $M

# 3. 换 CUDA 版 torch（见坑 3、坑 4，大文件建议并行下载）
pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 \
  -f https://mirrors.aliyun.com/pytorch-wheels/cu121/ --no-deps

# 4. 下模型（绕过坏代理）
mkdir -p pretrained_models
git -c http.proxy= -c https.proxy= clone \
  https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512.git \
  pretrained_models/Fun-CosyVoice3-0.5B

# 5. 跑通
python example.py
```

### 跑通结果（RTX 4060 Laptop）

`example.py` 的 6 条示例全部成功，实测 RTF ≈ **0.6–0.75**（<1，比实时快）：

| 示例 | 音频时长 | RTF |
|---|---|---|
| zero_shot | 8.4s | 0.68 |
| fine-grained (`[breath]`) | 9.8s | 0.57 |
| instruct（粤语） | 4.6s | 0.65 |
| instruct（快速） | 7.9s | 0.63 |
| hotfix（拼音 `[j][ǐ]`） | 5.8s | 0.68 |
| 日语（片假名） | 7.2s | 0.74 |
