#!/bin/bash
# Sequentially run every model, then delete only that model's HF cache (downloaded by this script) to save disk.
cd "$(dirname "$0")"
run() {  # repo tag
  echo "=== $2 $(date)"; .venv/bin/python run_vlm.py "$1" "$2" all > "results/$2.log" 2>&1; echo "exit $? $(date)"
  d="$HOME/.cache/huggingface/hub/models--${1//\//--}"; [ -d "$d" ] && rm -rf "$d" && echo "removed cache $d"
}
run mlx-community/SmolVLM2-2.2B-Instruct-mlx smolvlm2-2.2b
run mlx-community/Qwen2.5-VL-3B-Instruct-bf16 qwen2.5vl-3b
run mlx-community/gemma-3-4b-it-8bit gemma3-4b-8bit
run mlx-community/Qwen3-VL-4B-Instruct-8bit qwen3vl-4b-8bit
