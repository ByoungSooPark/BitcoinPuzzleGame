#!/bin/bash
# GPU BSGS 검증 스크립트

echo "=== GPU BSGS 검증 테스트 ==="
echo ""

# 작은 테스트 범위 (Puzzle 66 근처)
TEST_RANGE_START="2832ed7000000000"
TEST_RANGE_END="2832ed8000000000"

cd /home/park/keyhuntM1CPU

echo "1. CPU 모드로 10초 실행..."
timeout 10 ./build/keyhunt -m bsgs -f tests/66.txt -r ${TEST_RANGE_START}:${TEST_RANGE_END} -t 1 -s 5 2>&1 | tee /tmp/cpu_test.log &
CPU_PID=$!
wait $CPU_PID

CPU_KEYS=$(grep -oE '[0-9]+ keys' /tmp/cpu_test.log | tail -1 | awk '{print $1}')
echo "CPU 처리: $CPU_KEYS keys"
echo ""

sleep 2

echo "2. GPU 모드로 10초 실행..."
timeout 10 ./build/keyhunt -m bsgs -f tests/66.txt -r ${TEST_RANGE_START}:${TEST_RANGE_END} --gpu -g 0 --gpu-threads 256 --gpu-blocks 256 -t 1 -s 5 2>&1 | tee /tmp/gpu_test.log &
GPU_PID=$!
wait $GPU_PID

GPU_KEYS=$(grep -oE '[0-9]+ keys' /tmp/gpu_test.log | tail -1 | awk '{print $1}')
echo "GPU 처리: $GPU_KEYS keys"
echo ""

echo "=== 결과 비교 ==="
echo "CPU: $CPU_KEYS keys"
echo "GPU: $GPU_KEYS keys"

if [ -n "$CPU_KEYS" ] && [ -n "$GPU_KEYS" ]; then
    RATIO=$(echo "scale=2; $GPU_KEYS / $CPU_KEYS" | bc)
    echo "GPU/CPU 비율: ${RATIO}x"
    
    if (( $(echo "$RATIO > 50" | bc -l) )); then
        echo "✅ GPU 가속 정상 동작 (50배 이상)"
    elif (( $(echo "$RATIO > 10" | bc -l) )); then
        echo "⚠️  GPU 가속 동작하지만 성능 낮음 (10-50배)"
    else
        echo "❌ GPU 가속 미동작 또는 문제 있음 (10배 미만)"
    fi
else
    echo "❌ 테스트 실패 - 로그 확인 필요"
fi

echo ""
echo "상세 로그:"
echo "  CPU: /tmp/cpu_test.log"
echo "  GPU: /tmp/gpu_test.log"
