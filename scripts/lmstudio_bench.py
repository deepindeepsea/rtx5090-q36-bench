#!/usr/bin/env python3
"""
LM Studio prefill benchmark.

Measures prefill (time-to-first-token) and decode throughput against LM Studio's
OpenAI-compatible local server, across a sweep of prompt lengths.

No third-party dependencies -- uses only the Python standard library, so you can
run it with the plain `python` that ships on your machine:

    python lmstudio_bench.py

Prereqs on the Windows machine:
  1. Open LM Studio -> Developer tab -> Start Server (default http://localhost:1234).
  2. Make sure your Qwen model is loaded (or JIT loading is enabled).
  3. Run this script. Results print to the console and save to a timestamped CSV.

Key metrics (lower is better for the ms ones):
  - prefill_ms_per_tok : time to process each input token during prefill
                         (measured as time-to-first-token / prompt_tokens)
  - decode_ms_per_tok  : time to generate each output token
  - prefill_tok_per_s  : prompt tokens processed per second
  - decode_tok_per_s   : output tokens generated per second
"""

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

# ── Prompt lengths to sweep (approx. target input tokens) ──────────────
DEFAULT_PREFILL_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def http_json(url, method="GET", payload=None, timeout=600):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def detect_model(base_url):
    """Return the id of the first loaded model, or None."""
    try:
        out = http_json(f"{base_url}/models")
        models = out.get("data", [])
        if models:
            return models[0].get("id")
    except Exception as e:
        print(f"  ! could not list models: {e}")
    return None


def make_prompt(target_tokens):
    """
    Build a prompt of roughly `target_tokens` tokens.

    We can't tokenize exactly without the model's tokenizer, so we approximate
    ~0.75 tokens per word and pad generously. The per-token math later uses the
    server-reported prompt_tokens, so approximation here only affects the label,
    not the accuracy of the measurement.
    """
    words_needed = int(target_tokens / 0.75) + 8
    filler = ("The quick brown fox jumps over the lazy dog near the riverbank "
              "while thirteen weary travelers count seventy scattered stones. ").split()
    words = [filler[i % len(filler)] for i in range(words_needed)]
    return " ".join(words)


def stream_completion(base_url, model, prompt, max_tokens, timeout):
    """
    Stream a chat completion. Returns:
        (ttft_s, total_s, prompt_tokens, completion_tokens)
    ttft_s   = seconds from request send to first content token (prefill proxy)
    total_s  = seconds from request send to final token
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/chat/completions", data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    t0 = time.perf_counter()
    ttft = None
    t_end = None
    prompt_tokens = None
    completion_tokens = None
    seen_content = 0

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            chunk = line[len("data:"):].strip()
            if chunk == "[DONE]":
                t_end = time.perf_counter()
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    seen_content += len(content)
            usage = obj.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)

    if t_end is None:
        t_end = time.perf_counter()
    total_s = t_end - t0
    if ttft is None:
        ttft = total_s
    return ttft, total_s, prompt_tokens, completion_tokens


def bench_one(base_url, model, target_len, max_tokens, reps, timeout):
    """Run `reps` measurements for one prompt length; return the best (min TTFT) sample."""
    prompt = make_prompt(target_len)
    samples = []
    for r in range(reps):
        try:
            ttft, total, ptok, ctok = stream_completion(
                base_url, model, prompt, max_tokens, timeout
            )
        except urllib.error.URLError as e:
            print(f"    rep {r+1}: request failed: {e}")
            continue
        samples.append((ttft, total, ptok, ctok))
    if not samples:
        return None

    # Pick the rep with the lowest TTFT (least noise / contention).
    ttft, total, ptok, ctok = min(samples, key=lambda s: s[0])
    ptok = ptok or 0
    ctok = ctok or 0
    decode_s = max(total - ttft, 1e-9)

    return {
        "target_len": target_len,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "ttft_s": ttft,
        "total_s": total,
        "prefill_ms_per_tok": (ttft * 1000 / ptok) if ptok else None,
        "decode_ms_per_tok": (decode_s * 1000 / ctok) if ctok else None,
        "prefill_tok_per_s": (ptok / ttft) if ttft else None,
        "decode_tok_per_s": (ctok / decode_s) if ctok else None,
        "reps_ok": len(samples),
    }


def fmt(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


def main():
    ap = argparse.ArgumentParser(description="LM Studio prefill benchmark")
    ap.add_argument("--host", default="localhost", help="server host (default: localhost)")
    ap.add_argument("--port", type=int, default=1234, help="server port (default: 1234)")
    ap.add_argument("--model", default=None, help="model id (default: auto-detect first loaded)")
    ap.add_argument("--max-tokens", type=int, default=64, help="tokens to generate per run (default: 64)")
    ap.add_argument("--reps", type=int, default=3, help="repetitions per length, best is kept (default: 3)")
    ap.add_argument("--timeout", type=int, default=600, help="per-request timeout seconds (default: 600)")
    ap.add_argument("--lens", default=None,
                    help="comma-separated prompt lengths, e.g. 128,512,2048 (default: full sweep)")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1"
    prefill_lens = ([int(x) for x in args.lens.split(",")]
                    if args.lens else DEFAULT_PREFILL_LENS)

    print("=" * 62)
    print("  LM Studio Prefill Benchmark")
    print(f"  Server: {base_url}")
    print(f"  Date:   {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 62)

    # Connectivity + model check
    model = args.model or detect_model(base_url)
    if not model:
        print("\nERROR: no model detected. Is the LM Studio server running and a model loaded?")
        print("  LM Studio -> Developer tab -> Start Server, then load your Qwen model.")
        sys.exit(1)
    print(f"  Model:  {model}\n")

    # Warmup (JIT-load / spin up the model; not recorded)
    print("  Warming up model ...", end=" ", flush=True)
    try:
        stream_completion(base_url, model, make_prompt(64), 8, args.timeout)
        print("done\n")
    except Exception as e:
        print(f"\nERROR during warmup: {e}")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"lmstudio_bench_{ts}.csv"
    fields = ["target_len", "prompt_tokens", "completion_tokens", "ttft_s", "total_s",
              "prefill_ms_per_tok", "decode_ms_per_tok", "prefill_tok_per_s",
              "decode_tok_per_s", "reps_ok"]

    rows = []
    print("--- Running ---")
    for L in prefill_lens:
        print(f"  prompt~{L:<6} tok ...", end=" ", flush=True)
        row = bench_one(base_url, model, L, args.max_tokens, args.reps, args.timeout)
        if row is None:
            print("FAILED")
            continue
        rows.append(row)
        print(f"OK  (in={row['prompt_tokens']}, "
              f"prefill={fmt(row['prefill_ms_per_tok'])} ms/tok, "
              f"decode={fmt(row['decode_ms_per_tok'])} ms/tok)")

    with open(results_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n" + "=" * 62)
    print("  Summary")
    print("=" * 62)
    hdr = f"{'in_tok':>8} {'prefill ms/tok':>15} {'prefill tok/s':>14} {'decode ms/tok':>14} {'decode tok/s':>13}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['prompt_tokens']:>8} "
              f"{fmt(r['prefill_ms_per_tok']):>15} "
              f"{fmt(r['prefill_tok_per_s'],1):>14} "
              f"{fmt(r['decode_ms_per_tok']):>14} "
              f"{fmt(r['decode_tok_per_s'],1):>13}")
    print(f"\nSaved: {results_file}")


if __name__ == "__main__":
    main()
