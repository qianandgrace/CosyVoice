#!/usr/bin/env bash
# Fetches everything docker-compose bind-mounts from pretrained_models/.
# That directory is gitignored, so a fresh clone has none of it.
#
#   bash docker/vllm/fetch_assets.sh
#
# Total download is roughly 5.5 GB for the model plus ~1 GB of frontend
# resources. Safe to re-run: each step is skipped if its target already exists.
#
# The proxy-clearing flags below assume direct access to modelscope.cn works.
# Drop them if you genuinely need an http proxy for outbound git.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PM="${REPO_ROOT}/pretrained_models"
GIT=(git -c http.proxy= -c https.proxy=)

mkdir -p "${PM}"

# 1. Model weights (llm.pt, flow.pt, hift.pt, the ONNX models, ...)
if [ -d "${PM}/Fun-CosyVoice3-0.5B" ]; then
    echo "[skip] ${PM}/Fun-CosyVoice3-0.5B already present"
else
    echo "[get ] Fun-CosyVoice3-0.5B"
    "${GIT[@]}" clone --depth 1 \
        https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512.git \
        "${PM}/Fun-CosyVoice3-0.5B"
fi

# 2. ttsfrd: the Python wheels (baked into the image at build time) and the
#    ~480 MB grammar resource (bind-mounted at run time).
if [ -d "${PM}/CosyVoice-ttsfrd/resource" ]; then
    echo "[skip] ${PM}/CosyVoice-ttsfrd/resource already present"
else
    echo "[get ] CosyVoice-ttsfrd"
    if [ ! -d "${PM}/CosyVoice-ttsfrd" ]; then
        "${GIT[@]}" clone --depth 1 \
            https://www.modelscope.cn/models/iic/CosyVoice-ttsfrd.git \
            "${PM}/CosyVoice-ttsfrd"
    fi
    echo "[lfs ] pulling resource archive (~660 MB)"
    (cd "${PM}/CosyVoice-ttsfrd" && "${GIT[@]}" lfs pull)
    echo "[unz ] resource.zip -> resource/"
    unzip -q -o "${PM}/CosyVoice-ttsfrd/resource.zip" -d "${PM}/CosyVoice-ttsfrd"
fi

# 3. wetext FST grammars. CosyVoiceTextFrontEnd is pointed at these via
#    COSYVOICE_WETEXT_DIR so it never calls snapshot_download().
if [ -d "${PM}/wetext/zh/tn" ]; then
    echo "[skip] ${PM}/wetext already present"
else
    echo "[get ] wetext FST grammars"
    "${GIT[@]}" clone --depth 1 \
        https://www.modelscope.cn/models/pengzhendong/wetext.git \
        "${PM}/wetext"
    rm -rf "${PM}/wetext/.git"
fi

echo
echo "done. pretrained_models/ now holds:"
du -sh "${PM}"/* 2>/dev/null || true
