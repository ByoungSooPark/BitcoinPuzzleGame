#!/bin/bash
# GPU BSGS 동작 확인 스크립트

echo "=== GPU BSGS 동작 검증 ==="
echo ""

LOG_FILE="/home/park/keyhunt_puzzle71.log"

# 1. CUDA 초기화 확인
echo "1. CUDA 초기화 확인:"
if grep -q "CUDA.*Enabled" "$LOG_FILE"; then
    echo "   ✅ CUDA 활성화됨"
    grep "CUDA.*Enabled" "$LOG_FILE" | tail -1
else
    echo "   ❌ CUDA 초기화 실패"
fi
echo ""

# 2. Bloom filter GPU 업로드 확인
echo "2. Bloom filter GPU 업로드:"
if grep -q "CUDA.*Bloom filter uploaded" "$LOG_FILE"; then
    echo "   ✅ Bloom filter GPU에 업로드됨"
    grep "CUDA.*Bloom filter uploaded" "$LOG_FILE" | tail -1
else
    echo "   ❌ Bloom filter 업로드 안됨"
fi
echo ""

# 3. 처리 속도 확인
echo "3. 처리 속도 확인:"
SPEED=$(grep -oE '[0-9]+\.?[0-9]* [PTGMK]?keys/s' "$LOG_FILE" | tail -1)
if [ -n "$SPEED" ]; then
    echo "   ✅ 처리 속도: $SPEED"
    
    # Tkeys/s 단위로 변환해서 확인
    if echo "$SPEED" | grep -q "Tkeys/s"; then
        TKEYS=$(echo "$SPEED" | grep -oE '[0-9]+\.?[0-9]*')
        if (( $(echo "$TKEYS > 1.0" | bc -l) )); then
            echo "   ✅ GPU 가속 정상 (1+ Tkeys/s)"
        else
            echo "   ⚠️  속도가 낮음 (< 1 Tkeys/s)"
        fi
    elif echo "$SPEED" | grep -q "Gkeys/s"; then
        echo "   ⚠️  Gkeys/s 단위 - GPU 미사용 가능성"
    else
        echo "   ❌ 속도가 너무 낮음 - GPU 미동작"
    fi
else
    echo "   ⏳ 아직 통계 출력 안됨 (60초 대기)"
fi
echo ""

# 4. GPU 사용률 확인
echo "4. GPU 사용률 확인:"
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
if [ -n "$GPU_UTIL" ]; then
    echo "   GPU 사용률: ${GPU_UTIL}%"
    if [ "$GPU_UTIL" -gt 80 ]; then
        echo "   ✅ GPU 활발히 사용 중"
    elif [ "$GPU_UTIL" -gt 30 ]; then
        echo "   ⚠️  GPU 사용 중이지만 낮음"
    else
        echo "   ❌ GPU 거의 사용 안함"
    fi
else
    echo "   ❌ nvidia-smi 실행 실패"
fi
echo ""

# 5. 프로세스 확인
echo "5. keyhunt 프로세스 확인:"
KEYHUNT_PROCS=$(ps aux | grep "keyhunt.*bsgs.*puzzle71" | grep -v grep | wc -l)
if [ "$KEYHUNT_PROCS" -gt 0 ]; then
    echo "   ✅ keyhunt 실행 중 ($KEYHUNT_PROCS 프로세스)"
    ps aux | grep "keyhunt.*bsgs.*puzzle71" | grep -v grep | awk '{printf "   PID: %s, CPU: %s%%, MEM: %s%%\n", $2, $3, $4}'
else
    echo "   ❌ keyhunt 실행 안됨"
fi
echo ""

# 6. 체크포인트 확인
echo "6. 체크포인트 저장 확인:"
if [ -f "/home/park/keyhunt_checkpoint.txt" ]; then
    CHECKPOINT=$(cat /home/park/keyhunt_checkpoint.txt)
    echo "   ✅ 체크포인트 저장됨: $CHECKPOINT"
else
    echo "   ⏳ 아직 체크포인트 없음 (5분 후 저장)"
fi
echo ""

# 7. 종합 판정
echo "=== 종합 판정 ==="
CUDA_OK=$(grep -q "CUDA.*Enabled" "$LOG_FILE" && echo "1" || echo "0")
BLOOM_OK=$(grep -q "CUDA.*Bloom filter uploaded" "$LOG_FILE" && echo "1" || echo "0")
SPEED_OK=$(grep -q "Tkeys/s" "$LOG_FILE" && echo "1" || echo "0")
GPU_OK=$([ "$GPU_UTIL" -gt 50 ] 2>/dev/null && echo "1" || echo "0")

SCORE=$((CUDA_OK + BLOOM_OK + SPEED_OK + GPU_OK))

if [ "$SCORE" -ge 3 ]; then
    echo "✅ GPU BSGS 정상 동작 중 ($SCORE/4 통과)"
elif [ "$SCORE" -ge 2 ]; then
    echo "⚠️  부분적으로 동작 ($SCORE/4 통과) - 로그 확인 필요"
else
    echo "❌ GPU BSGS 문제 있음 ($SCORE/4 통과)"
fi
echo ""
echo "상세 로그: tail -f $LOG_FILE"
