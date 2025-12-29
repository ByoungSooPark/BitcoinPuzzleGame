#!/bin/bash
# Quick test to capture pointDouble debug output

timeout 5 ./build/keyhunt -m bsgs -f tests/63.pub --gpu -g 0 --gpu-threads 1024 --gpu-blocks 0 --gpu-batch 128 -t 1 -b 63 -k 8 -s 5 -q 2>&1 | grep -A 15 "CUDA\]\[TEST\]" | head -30
