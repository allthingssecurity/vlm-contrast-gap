# Below the Clinical Threshold: artifacts

Code (`code/`) and raw outputs (`results/`) for every number in the paper.

* `stimuli.py`: calibrated stimuli (Weber contrast in linear sRGB, IEC 61966-2-1; achieved contrast measured after 8-bit quantisation).
* `run_vlm.py <hf_repo> <tag> [e1|e2|all|e3]`: runs one MLX VLM (greedy decoding) and writes `results/<tag>_<exp>.jsonl`.
* `run_all.sh` (E1+E2) and `run_e3.sh` (E3): the four models used:
  `mlx-community/SmolVLM2-2.2B-Instruct-mlx`, `mlx-community/Qwen2.5-VL-3B-Instruct-bf16`,
  `mlx-community/gemma-3-4b-it-8bit`, `mlx-community/Qwen3-VL-4B-Instruct-8bit`.
* `analyze.py [outdir]`: recomputes `results/summary.json`, both figures and Table I from the JSONL files.

Environment: Python 3.12, mlx-vlm 0.7.12, pillow, numpy, scikit-image 0.26, matplotlib, Apple-silicon GPU.
Images: scikit-image sample data (public domain / CC0 / no known copyright restrictions). Payloads are benign.
Responses: E1 5,120; E2 1,232; E3 2,408 (8,760 total).
