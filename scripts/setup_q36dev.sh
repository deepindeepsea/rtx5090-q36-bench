#!/usr/bin/env bash
set -e
export DEBIAN_FRONTEND=noninteractive
MODEL=/models/unsloth/Qwen3.6-35B-A3B-MXFP4_MOE.gguf

echo "=== toolchain (git, build-essential, tmux, vim) ==="
apt-get update -qq
apt-get install -y -qq git build-essential ca-certificates tmux vim >/dev/null

echo "=== clone + build q36 (cli, bench, server) ==="
cd /root
rm -rf q36
git clone --depth 1 https://github.com/ambud/q36.git
cd q36
make q36 q36_bench q36_server
echo "=== SETUP_DONE ==="

echo "=== 90k experiment: prefill 2048 & 90112, decode at depth 0 & 90112 ==="
./q36_bench -m "$MODEL" -p 2048,90112 -n 128 -d 0,90112 -r 3
echo "=== 90K_DONE ==="
