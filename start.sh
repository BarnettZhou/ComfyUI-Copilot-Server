#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFY_ROOT="${COMFY_ROOT:-$HOME/comfyui}"
PYTHON="${PYTHON:-$COMFY_ROOT/venv/bin/python3}"
CONFIG="${CONFIG:-$SCRIPT_DIR/config.yaml}"
if [[ ! -x "$PYTHON" ]]; then echo "找不到 ComfyUI venv Python: $PYTHON" >&2; exit 1; fi
if [[ ! -f "$CONFIG" ]]; then echo "找不到配置文件: $CONFIG，请复制 example.config.yaml 为 config.yaml" >&2; exit 1; fi
# MPS 默认允许分配约 1.7 倍推荐显存（>物理内存），超限会靠 swap 硬撑导致整机卡死；
# 压到 0.9 倍让 torch 提前抛 OOM，由接口正常返回错误
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.9}"
exec "$PYTHON" "$SCRIPT_DIR/server.py" --config "$CONFIG" "$@"
