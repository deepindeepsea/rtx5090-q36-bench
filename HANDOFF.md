# HANDOFF — continuing the q36 → ROCm port on the R9700 lab machine

A runbook for picking this up on the **Radeon AI PRO R9700 (gfx1201, ROCm 7.2.1)** lab box, e.g.
driving it with Claude Enterprise / Cowork. Read
[`reports/radeon_ai_pro_next_steps.html`](reports/radeon_ai_pro_next_steps.html) first — this file
is the checklist version.

## What's already true (context)

- **RTX 5090 baseline is done and reproducible** (see `reports/` + `scripts/`). q36 = 13,020 t/s
  prefill, 259 t/s decode, 7,879 t/s @ 90k. Tuned llama.cpp = ~11.9k / ~219.
- **q36 is CUDA/Blackwell-only** — it will NOT build or run on the R9700. The port is a *new*
  engine that reuses q36's *ideas*, not its code (q36 is AGPL — keep it separate; see `NOTICE.md`).
- The R9700 already runs vLLM, llama.cpp (HIP + Vulkan), and SGLang on gfx1201 (per the field series).

## The split (why this is worth doing)

| Ports to RDNA4? | q36 idea |
|---|---|
| ❌ Blackwell-only | Native MXFP4 block-scaled tensor-core MMA; 1-instruction fp4→fp16 convert; Programmatic Dependent Launch |
| ✅ Ports cleanly | **Zero-sync whole-token hipGraph decode loop**; **hybrid attention-KV + SSM-checkpoint cache**; **batch-bucketed slot scheduling** |

The three portable ones are the whole point — none of your current R9700 runtimes do them.

## Build order (validate each step before the next)

- [ ] **0. Correctness harness.** CPU reference for the MoE FFN + gated-DeltaNet forward; bit-exact
      compare every kernel against it. (Mirror q36's `test_dequant` / `test_mma_bs` idea.)
- [ ] **1. Prefill path.** Expert GEMM via **INT4 (reuse `fused_moe_wvSplitK_int4_gemm`)** + flash
      attention (`-fa` mandatory on RDNA4) + a new **gated-DeltaNet chunked-scan** HIP kernel.
      Target: beat/near your llama.cpp HIP 3,491 t/s (pp512).
- [ ] **2. Zero-sync decode loop.** `hipGraph` capture of the full 40-block token step + on-device
      sampling; AWQ GEMV experts. This is the biggest portable win.
- [ ] **3. Hybrid cache.** Paged attn KV + the ~63 MB SSM checkpoint blob, DRAM/disk LRU, keyed on
      **prompt tokens** (not generated text). Turns the agentic re-prefill cost into a ~ms restore.
- [ ] **4. Batch / slots.** Bucketed graphs {8,16,32,48,64} + slot-move compaction; GEMV→tiled
      switch at batch ≥16. Unlocks the ~24-concurrent aggregate throughput.
- [ ] **5. (Optional) FP8 W8A8** experts via RDNA4 FP8 WMMA for quality/prefill headroom.

## Kernel reuse checklist (already gfx1201-native from your stack)

- [ ] `fused_moe_wvSplitK_int4_gemm` → batched/prefill expert GEMM
- [ ] `awq_gemv_hip` / `awq_gemv_moe_hip` → decode narrow-shape expert GEMV (does the INT4 unpack)
- [ ] `sgl_kernel`: `topk_softmax` (router), `moe_align_block_size`, `silu_and_mul`,
      `rotary_embedding`, `attention` (FA), `sampling`, `kvcacheio`
- [ ] ROCm 7.2.1 + PyTorch rocm7.2 toolchain, gfx1201 build recipes

## New kernels/systems to write (the "q36 layer")

- [ ] Zero-sync per-token **hipGraph decode driver** (bucketed batch graphs)
- [ ] **Gated-DeltaNet chunked-scan** HIP kernel (the 30 recurrent layers) — the one genuinely new kernel
- [ ] **Hybrid state-checkpoint cache** (paged KV + SSM blob, prompt-keyed, DRAM/disk LRU)
- [ ] MXFP4→INT4/FP8 **load-time repack + WMMA pre-swizzle**
- [ ] On-device end-to-end **sampling** wired into the graph

## Open questions to resolve first

- [ ] Does gfx1201 **hipGraph** capture the full forward reliably under ROCm 7.2.1? (Prototype this before anything else — it gates step 2.)
- [ ] Is there a **gated-DeltaNet** reference already compiling in your SGLang gfx1201 tree (Qwen3-Next linear-attn path) to validate a new HIP scan against bit-for-bit?
- [ ] MXFP4→INT4 **quality**: measure perplexity delta of AWQ-INT4 experts vs the MXFP4 original before committing to the INT4 path.
- [ ] Confirm the R9700's exact memory bandwidth (`rocm-smi --showbw` / spec) to firm up the decode roofline (plan assumes ~640 GB/s → ~234 t/s absolute ceiling).

## Realistic targets (Qwen3.6-35B-A3B on R9700)

| Metric | 5090 / q36 | R9700 target | Bounded by |
|---|---|---|---|
| Prefill pp2048 | 13,020 | ~4,000–6,000 | WMMA throughput; no zero-dequant MMA |
| Decode (fresh) | 259 | ~120–150 | 640 GB/s bandwidth (hard cap ~234) |
| Aggregate @ ~24 slots | ~1,650 | ~500–700 | compute-bound (scales with WMMA) |
| Agent re-prefill (78k) | 13.6 s → 0.4 s | similar ratio | cache is systems code — ports 1:1 |

## Reference

- q36 architecture: https://github.com/ambud/q36/blob/main/docs/ARCHITECTURE.md (AGPL-3.0, Ambud Sharma)
- R9700 field series: https://deepindeepsea.github.io/r9700-vllm-blogs/
