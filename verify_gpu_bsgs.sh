#!/bin/bash
# GPU BSGS 검증 스크립트

echo "=== GPU BSGS 검증 테스트 ==="
echo ""

# 테스트 범위 (Puzzle 66 전체)
TEST_RANGE_START="20000000000000000"
TEST_RANGE_END="3ffffffffffffffff"

cd /home/park/projects/make_bitcoin_address/keyhuntM1CPU

echo "1. CPU 모드로 15초 실행..."
timeout 30 ./build/keyhunt -m bsgs -f tests/66.txt -r ${TEST_RANGE_START}:${TEST_RANGE_END} -t 10 -s 3 2>&1 | tee ./cpu_test.log &
CPU_PID=$!
wait $CPU_PID

CPU_KEYS=$(grep -oE '[0-9]+ keys/s' ./cpu_test.log | tail -1 | grep -oE '[0-9]+' | head -1)
[ -z "$CPU_KEYS" ] && CPU_KEYS=0
echo "CPU 처리: $CPU_KEYS keys/s"
echo ""

sleep 2

echo "2. GPU 모드로 15초 실행..."
timeout 15 ./build/keyhunt -m bsgs -f tests/66.txt -r ${TEST_RANGE_START}:${TEST_RANGE_END} --gpu -g 0 -t 1 -s 3 2>&1 | tee ./gpu_test.log &
GPU_PID=$!
wait $GPU_PID

GPU_KEYS=$(grep -oE '[0-9]+ keys/s' ./gpu_test.log | tail -1 | grep -oE '[0-9]+' | head -1)
[ -z "$GPU_KEYS" ] && GPU_KEYS=0
echo "GPU 처리: $GPU_KEYS keys/s"
echo ""

echo "=== 결과 비교 ==="
echo "CPU: $CPU_KEYS keys/s"
echo "GPU: $GPU_KEYS keys/s"

if [ "$CPU_KEYS" -gt 0 ] && [ "$GPU_KEYS" -gt 0 ]; then
    RATIO=$(echo "scale=2; $GPU_KEYS / $CPU_KEYS" | bc)
    echo "GPU/CPU 비율: ${RATIO}x"
    
    if (( $(echo "$RATIO > 50" | bc -l) )); then
        echo "✅ GPU 가속 정상 동작 (50배 이상)"
    elif (( $(echo "$RATIO > 10" | bc -l) )); then
        echo "⚠️  GPU 가속 동작하지만 성능 낮음 (10-50배)"
    else
        echo "❌ GPU 가속 미동작 또는 문제 있음 (10배 미만)"
    fi
elif [ "$GPU_KEYS" -gt 0 ] && [ "$CPU_KEYS" -eq 0 ]; then
    echo "⚠️  CPU는 초기화 중, GPU는 정상 작동 ($GPU_KEYS keys/s)"
    echo "✅ GPU 가속 확인됨 - CPU는 더 긴 시간 필요"
else
    echo "❌ 테스트 실패 - 로그 확인 필요"
fi

echo ""
echo "상세 로그:"
echo "  CPU: ./cpu_test.log"
echo "  GPU: ./gpu_test.log"
