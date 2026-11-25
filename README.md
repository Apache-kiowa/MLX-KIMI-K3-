# kimi-k31-mlx — Kimi K3.1 Pro on Apple Silicon with mlx-lm

Tested 2026-11-19: mlx 0.32.4, mlx-lm main c4c9c76, tiny-random/kimi-k3.1-pro
(Linux CPU and Mac Studio). The full 1.56 TB model and the 4-Mac cluster are not tested yet.

kimi_k31.py   copy to <mlx-lm>/mlx_lm/models/ — Kimi K3 model + K3.1 DSA indexer.
              K31_INDEXER_FP8=0 (default): bf16/fp32 scoring, as NVIDIA NeMo AutoModel.
              K31_INDEXER_FP8=1: FP8 scoring, as vLLM serving kernels.

run/          full model, in order (edit run/config.sh first):
              Shared paths under /Users/Shared/k31 so every Mac has the same paths
              whatever the user name. Each Mac stores only its tensor-parallel part
              (scripts/presplit.py), loaded with K31_PRESPLIT_DIR; bit-identical to
              loading the full model and sharding (checked on tiny, 4 ranks).
  00_setup_env.sh      every Mac: Python 3.12 venv, MLX, mlx-lm, kimi_k31.py
  01_download.sh       rank 0: 1.56 TB from Hugging Face (resumable, run in tmux)
  02_convert.sh        rank 0: repack to MLX + verify_full.py
  03_split_copy.sh     rank 0, one Mac at a time: cut that Mac's part (~400 GB) and copy it
  04_cluster.sh        rank 0: hostfile for jaccl (Thunderbolt 5 + RDMA), version check
  05_check_cluster.sh  rank 0: tensor-parallel check of the tiny model on the cluster
  06_serve.sh          rank 0: OpenAI-compatible server on all 4 Macs
  07_test_request.sh   one chat request

scripts/      checks
  inspect_ckpt.py      tensor names and config from safetensors headers
  verify.py            tiny model: expert bytes, MXFP4 decode, indexer, logits
  verify_full.py       full model: names/shapes (strict), indexer, sampled expert bytes
  presplit.py          cut the full MLX model into per-rank parts (tp<N>-r<rank>-*.safetensors)
  tp_check.py          1 rank vs N ranks (mlx.launch); --topk forces sparse DSA
  diag_gpu.py          Metal vs CPU checks
  compare_hf.py        MLX vs official PyTorch code (Linux; dequant_src.py first)
  compare_dsa.py       MLX DSA vs PyTorch DSA reference (dsa_torch_ref.py)
  fla_torch_shim.py    official code without Triton
  dequant_src.py, mxfp4_ref.py, build_synthetic.py, convert_weights_only.py  helpers
