# llama.cpp Baseline — RTX 5090 (Qwen3.6-35B-A3B)

**Date:** 2026-07-18
**Environment:** Docker `ghcr.io/ggml-org/llama.cpp:full-cuda` (build 86a9c79f8 / 10066), CUDA 13.3, WSL2 backend
**GPU:** RTX 5090, 32 GB, driver 610.47, compute capability 12.0 (Blackwell / sm_120)
**Model:** Qwen3.6-35B-A3B Q4_K_M (19.70 GiB, 34.66 B params), MoE hybrid
**Flags:** `-ngl 99 -fa 1 -r 3` (default batch 2048 / ubatch 512)

## Results

| Test    | t/s              |
| ------- | ---------------- |
| pp512   | 1485.42 ± 224.03 |
| pp1024  | 3848.86 ± 869.47 |
| pp2048  | 7080.16 ± 138.54 |
| pp4096  | 7528.68 ± 43.47  |
| pp8192  | 7447.51 ± 12.94  |
| pp16384 | 7284.05 ± 29.78  |
| tg128   | 243.94 ± 3.67    |

## Read

- **Peak prefill ≈ 7.5k tok/s** (pp4096); ~7.1k at 2k context.
- **Decode ≈ 244 tok/s.**
- Prefill flattens ~7.3–7.5k from 4k tokens up — model fits fully in VRAM, no CPU spill.

## Vs. Ambud's targets (q36 README)

| Metric              | This baseline (Q4_K_M) | Ambud llama.cpp (MXFP4) | q36 (MXFP4) |
| ------------------- | ---------------------- | ----------------------- | ----------- |
| Prefill @ ctx 2048  | ~7.1k                  | ~10k (claimed)          | 13.4k       |
| Decode              | 244                    | —                       | 270+        |

## UPDATE — ubatch retune (Q4_K_M, `-b 4096`)

Raising the micro-batch from the default 512 was the decisive lever:

| ubatch | pp2048 | pp4096  | pp8192  | tg128 |
| ------ | ------ | ------- | ------- | ----- |
| 512    | 7080   | 7529    | 7448    | 244   |
| 2048   | ~4705* | 10624   | 11090   | 243   |
| 4096   | 11125  | **11383** | 11303 | 247   |

*ub=2048 pp2048 was the first (warmup) config — high variance, ignore.

**Result: ~7.5k → ~11.4k tok/s prefill on Q4_K_M — already past Ambud's ~10k llama.cpp baseline and closing on q36's 13.4k, before switching to MXFP4.** Decode steady ~247 t/s. Best config so far: `-ngl 99 -fa 1 -b 4096 -ub 4096`.

Next: MXFP4 weights (downloading) for the format-matched number and to feed q36.

## UPDATE 2 — MXFP4 (format-matched to Ambud / q36)

Model: `Qwen3.6-35B-A3B-MXFP4_MOE.gguf` (20.65 GiB, 35.51 B). `-fa 1 -b 8192`.

| ubatch | pp2048 | pp4096  | pp8192 | pp16384 | tg128 |
| ------ | ------ | ------- | ------ | ------- | ----- |
| 4096   | ~2663* | ~8212*  | 11570  | 11631   | 218   |
| 8192   | 11772  | **11878** | 11192 | 11239 | 219   |

*ub=4096 pp2048/pp4096 were warmup configs (high variance) — ignore.

### Findings
- **Best llama.cpp prefill ≈ 11.9k tok/s** (MXFP4, ub=8192) — **exceeds Ambud's cited ~10k llama.cpp baseline**, mostly thanks to aggressive ubatch tuning.
- **MXFP4 ≈ Q4_K_M on prefill** here (11.9k vs 11.4k), and **MXFP4 decode is *lower*** (218 vs 247 t/s). llama.cpp's MXFP4 path isn't better than Q4_K on this GPU — consistent with q36's thesis that generic runtimes leave MXFP4/Blackwell performance on the table.

### Scoreboard (RTX 5090, Qwen3.6-35B-A3B)

| Engine                         | Prefill @≈2k | Peak prefill | Decode |
| ------------------------------ | ------------ | ------------ | ------ |
| llama.cpp default (ub=512, Q4) | ~7.1k        | ~7.5k        | 244    |
| **llama.cpp tuned (ub=8192, MXFP4)** | **~11.8k** | **~11.9k** | 218    |
| Ambud llama.cpp (cited)        | ~10k         | —            | —      |
| **q36 (target)**               | **13.4k**    | 13.4k        | **270+** |

So tuned llama.cpp already beats the ~10k baseline; q36's remaining edge is ~13% on prefill and ~+24% on decode (270 vs 218) — the decode gap is where its CUDA-graph / W4A8-MMA / zero-host-sync design should pay off. Best llama.cpp config: `-ngl 99 -fa 1 -b 8192 -ub 8192`.

## Gap analysis / levers to reach ~10k llama.cpp baseline

1. **Quantization: use MXFP4** (Ambud's format). Q4_K_M needs heavier dequant; MXFP4 maps to Blackwell native FP4 tensor-core paths → likely the biggest single lever. Requires downloading `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (~20 GB).
2. **Larger ubatch:** default `-ub 512`. Try `-ub 2048 -b 2048` — prefill throughput on big GPUs is very sensitive to ubatch.
3. **Native Linux (dual-boot):** removes WSL2/WDDM GPU-scheduling overhead (~5–15%) vs bare metal, which is what Ambud measured on.
4. **Stage model on fast storage:** copy GGUF into a Docker volume / WSL fs to avoid slow Windows bind-mount reads (iteration speed, not throughput).
