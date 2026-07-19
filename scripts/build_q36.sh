#!/usr/bin/env bash
set -e
export DEBIAN_FRONTEND=noninteractive

MODEL=/models/unsloth/Qwen3.6-35B-A3B-MXFP4_MOE.gguf

echo "=== apt: install toolchain (git, build-essential) ==="
apt-get update -qq
apt-get install -y -qq git build-essential ca-certificates >/dev/null
echo "=== nvcc version ==="
nvcc --version | tail -2

echo "=== clone q36 ==="
cd /root
rm -rf q36
git clone --depth 1 https://github.com/ambud/q36.git
cd q36

echo "=== build: CPU tools (q36_info) ==="
make q36_info || echo "(q36_info build failed - continuing)"

echo "=== build: q36_bench (CUDA sm_120a) ==="
make q36_bench
echo "=== BUILD_DONE ==="

echo "=== q36_info: model sanity ==="
./q36_info "$MODEL" || echo "(q36_info run failed - continuing)"

echo "=== q36_bench: prefill 2048-16384, decode at depth 0/2048/16384 ==="
./q36_bench -m "$MODEL" -p 2048,4096,8192,16384 -n 128 -d 0,2048,16384 -r 3
echo "=== ALL_DONE ==="
