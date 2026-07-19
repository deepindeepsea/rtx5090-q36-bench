# NOTICE — Third-party attribution

## q36

This repository benchmarks and analyzes **q36**, an independent open-source project:

- **Project:** `ambud/q36` — https://github.com/ambud/q36
- **Author:** Ambud Sharma
- **License:** GNU AGPL-3.0
- **Author's description (quoted):** *"A hyper-optimized, zero-dependency C/CUDA inference engine
  for Qwen 3.6 35B on RTX 5090 / Blackwell."*
- **Author's status claim (quoted):** *"faster than llama.cpp at every measured point."*

**q36 source is NOT included, vendored, or modified in this repository.** The scripts here clone it
from its upstream repository at build time and run its `q36_bench` / `q36` / `q36_server` binaries.
The reports quote short factual descriptions and reproduce published performance numbers for
analysis and comparison, with attribution. If you redistribute a modified q36, or run a modified
version as a network service, the AGPL-3.0 terms apply to q36 — consult its LICENSE.

## llama.cpp

Benchmarked via the official image `ghcr.io/ggml-org/llama.cpp:full-cuda`
(https://github.com/ggml-org/llama.cpp, MIT License).

## Model

Qwen3.6-35B-A3B weights (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, Apache-2.0) are downloaded from
Hugging Face and are **not** included in this repository.

## R9700 field series

The Radeon AI PRO R9700 port plan builds on the "R9700 + vLLM" field series by Pradeep Nallimelli —
https://deepindeepsea.github.io/r9700-vllm-blogs/
