# RTX 5090 LLM Inference Benchmarks — Qwen3.6-35B-A3B (llama.cpp vs q36) + Radeon AI PRO port plan

Benchmarks of prefill and decode throughput for **Qwen3.6-35B-A3B** on an **NVIDIA RTX 5090**
(Blackwell, `sm_120`, 32 GB), comparing stock/tuned **llama.cpp** against the purpose-built
**q36** CUDA engine — plus a concrete plan for porting q36's ideas to **AMD ROCm / RDNA4**
(Radeon AI PRO R9700).

> **Credit — q36.** The custom engine benchmarked here is **[`ambud/q36`](https://github.com/ambud/q36)**
> by **Ambud Sharma**, described by its author as *"a hyper-optimized, zero-dependency C/CUDA
> inference engine for Qwen 3.6 35B on RTX 5090 / Blackwell"* with the status *"faster than
> llama.cpp at every measured point."* q36 is licensed **AGPL-3.0**. This repository does **not**
> vendor or modify q36 source; it reproduces and analyzes its published benchmarks and documents a
> separate ROCm port plan. See [`NOTICE.md`](NOTICE.md).

## Reports

| Report | What's in it |
|---|---|
| [`reports/benchmark_report_rtx5090.html`](reports/benchmark_report_rtx5090.html) | Full RTX 5090 story for an entry-level engineer: what prefill/decode mean, all runs, the micro-batch win, MXFP4 vs Q4_K_M, the decode roofline, and the LM Studio tuning postscript. |
| [`reports/radeon_ai_pro_next_steps.html`](reports/radeon_ai_pro_next_steps.html) | **Next steps on Radeon AI PRO R9700 (gfx1201):** optimization-by-optimization port map of q36 → RDNA4/ROCm, what reuses your existing kernels, realistic targets, and a phased build order. |
| [`reports/baseline_results.md`](reports/baseline_results.md) | Raw llama.cpp result tables (default → tuned → MXFP4). |

## Headline results (RTX 5090, Qwen3.6-35B-A3B)

| Engine | Prefill @ ~2k ctx | Peak prefill | Decode |
|---|---|---|---|
| llama.cpp, default (`-ub 512`) | ~7,080 t/s | ~7,529 t/s | 244 t/s |
| llama.cpp, tuned (`-ub 8192`, MXFP4) | ~11,772 t/s | ~11,878 t/s | ~219 t/s |
| **q36** (MXFP4, CUDA 13.1, `sm_120a`) | **13,020 t/s** | 13,020 t/s | **259.6 t/s** |
| q36 @ 90k context | 7,879 t/s | — | 178.8 t/s @ d90112 |

Two takeaways: (1) raising llama.cpp's **micro-batch** (`-ub`) from 512 → 8192 nearly doubled
prefill for free; (2) q36 adds ~11% prefill / ~19% decode on top via Blackwell-specific kernels.

## Reproduce on the RTX 5090 box

Requires Docker with NVIDIA GPU passthrough and the model GGUFs locally.

```bash
# llama.cpp baseline + tuned sweep (CUDA 13.3 container)
scripts/docker_llamabench.bat          # default
scripts/docker_llamabench_ub.bat       # micro-batch sweep
scripts/docker_llamabench_mxfp4.bat    # MXFP4, -ub 8192

# Build + benchmark q36 (CUDA 13.1 devel container, sm_120a)
scripts/run_q36_build.bat              # one-shot build + bench
scripts/run_q36dev.bat                 # persistent container 'q36dev' + 90k test
```

`q36_bench` and `llama-bench` share the same `-p / -n / -d / -r` flags, so results are directly
comparable. The `scripts/*` are Windows launchers; `build_q36.sh` / `setup_q36dev.sh` are the
Linux/container steps they invoke.

## Radeon AI PRO R9700 — the port

The ROCm/RDNA4 plan builds on the **[R9700 + vLLM field series](https://deepindeepsea.github.io/r9700-vllm-blogs/)**.
Short version: q36's five wins split into **two Blackwell-only hardware tricks** (native MXFP4
tensor-core MMA, 1-instruction fp4→fp16) that don't port, and **three architecture/systems ideas**
(zero-sync whole-token hipGraph decode, hybrid attention-KV + SSM-checkpoint cache, batch-bucketed
slot scheduling) that port cleanly and that no current R9700 runtime does. See the report for the
full map, kernel-reuse table, and phased build order.

## Environment captured

- **GPU:** RTX 5090, 32 GB, driver 610.47, CUDA 13.3 runtime, compute capability 12.0.
- **Model:** Qwen3.6-35B-A3B — hybrid MoE (10 attention + 30 gated-DeltaNet SSM layers), ~3B active/token.
- **Quants:** `Q4_K_M` (19.7 GB) and `MXFP4` (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, 20.6 GB).
- **Runtimes:** `ghcr.io/ggml-org/llama.cpp:full-cuda` (build 10066); q36 built in `nvidia/cuda:13.1.2-devel-ubuntu24.04`.

## License

Reports, scripts, and analysis in this repository: choose your license (e.g. MIT/Apache-2.0).
**q36 itself is AGPL-3.0 and is not included here** — see [`NOTICE.md`](NOTICE.md).
