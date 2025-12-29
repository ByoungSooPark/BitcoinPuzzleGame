#!/bin/bash
# ============================================================================
# Keyhunt Vast.ai Deployment Script - GPU BSGS Edition
# Puzzle #71 Hunter with Discord Notifications & Checkpointing
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

DISCORD_WEBHOOK="https://discord.com/api/webhooks/1357451751908839576/qDswrcM9eK9zE02SWFQqIOA7068OTZWgdbsJ7_7END4cLgH57En7mj5TTIuQToBaJWCJ"

# Puzzle 71 공개키 (BSGS 모드 사용 가능!)
PUZZLE71_PUBKEY="0296b538e853519c726a2c91e61ec11600ae1390813a627c66fb8be7947be63c52"
TARGET_ADDRESS="1J6PYEzr4CUoGbnXrELyHszoTSz3wCsCaj"

BIT_RANGE=71

# GPU 설정 (RTX 5090 기준)
GPU_DEVICE=0           # 단일 GPU: 0, 멀티: "0,1,2,3..."
GPU_THREADS=256        # CUDA 블록당 스레드 수
GPU_BLOCKS=2048        # RTX 5090: 170 SM × 12 = 2040
CPU_THREADS=1          # GPU 모드에서는 1 권장

STATS_INTERVAL=60

# 파일 경로
WORK_DIR="/home/park"
LOG_FILE="${WORK_DIR}/keyhunt_puzzle71.log"
RESULT_FILE="${WORK_DIR}/keyhunt_FOUND.txt"
CHECKPOINT_FILE="${WORK_DIR}/keyhunt_checkpoint.txt"
RANGES_SEARCHED_FILE="${WORK_DIR}/keyhunt_ranges_searched.txt"

REPO_URL="https://github.com/consigcody94/keyhuntM1CPU.git"

# ============================================================================
# DISCORD NOTIFICATION
# ============================================================================

send_discord() {
    local title="$1"
    local message="$2"
    local color="$3"

    if [ -z "$DISCORD_WEBHOOK" ]; then
        echo "[WARN] Discord webhook not configured"
        return
    fi

    local gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
    local hostname=$(hostname)
    local searched_ranges=$(wc -l < "$RANGES_SEARCHED_FILE" 2>/dev/null || echo "0")

    curl -s -H "Content-Type: application/json" \
        -d "{
            \"embeds\": [{
                \"title\": \"🔑 $title\",
                \"description\": \"$message\",
                \"color\": $color,
                \"fields\": [
                    {\"name\": \"Machine\", \"value\": \"$hostname\", \"inline\": true},
                    {\"name\": \"GPU\", \"value\": \"$gpu_info\", \"inline\": true},
                    {\"name\": \"Ranges\", \"value\": \"$searched_ranges\", \"inline\": true},
                    {\"name\": \"Target\", \"value\": \"\`$TARGET_ADDRESS\`\", \"inline\": false}
                ],
                \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
            }]
        }" \
        "$DISCORD_WEBHOOK" || true
}

# ============================================================================
# CHECKPOINT
# ============================================================================

save_checkpoint() {
    echo "[CHECKPOINT] Saving progress..."
    local last_thread=$(tail -100 "$LOG_FILE" 2>/dev/null | grep -oE 'Thread 0x[0-9a-fA-F]+' | tail -1 || echo "")
    local current_pos=$(echo "$last_thread" | grep -oE '0x[0-9a-fA-F]+' || echo "")

    if [ -n "$current_pos" ]; then
        echo "$current_pos" > "$CHECKPOINT_FILE"
        echo "$(date -Iseconds) $current_pos" >> "$RANGES_SEARCHED_FILE"
        echo "[CHECKPOINT] Saved: $current_pos"
    fi
}

load_checkpoint() {
    if [ -f "$CHECKPOINT_FILE" ]; then
        cat "$CHECKPOINT_FILE" | tr -d '\n\r'
    else
        echo ""
    fi
}

# ============================================================================
# CLEANUP
# ============================================================================

cleanup() {
    echo "[SHUTDOWN] Saving progress..."
    save_checkpoint

    pkill -f "keyhunt.*bsgs" 2>/dev/null || true
    pkill -f "keyhunt_monitor" 2>/dev/null || true

    send_discord "Hunt Paused" "Progress saved" 16776960
    exit 0
}

trap cleanup SIGTERM SIGINT SIGHUP

# ============================================================================
# SETUP & BUILD
# ============================================================================

setup_environment() {
    echo "============================================"
    echo "Setting up environment..."
    echo "============================================"

    apt-get update
    apt-get install -y git cmake libssl-dev libgmp-dev libomp-dev curl

    nvcc --version || { echo "[ERROR] CUDA not found!"; exit 1; }
    nvidia-smi

    local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "[INFO] GPU: $gpu_name"

    if [[ "$gpu_name" == *"5090"* ]] || [[ "$gpu_name" == *"5080"* ]]; then
        export CUDA_ARCH="120"
        echo "[INFO] sm_120 (Blackwell)"
    elif [[ "$gpu_name" == *"4090"* ]] || [[ "$gpu_name" == *"4080"* ]]; then
        export CUDA_ARCH="89"
        echo "[INFO] sm_89 (Ada)"
    elif [[ "$gpu_name" == *"3090"* ]] || [[ "$gpu_name" == *"3080"* ]] || [[ "$gpu_name" == *"3060"* ]]; then
        export CUDA_ARCH="86"
        echo "[INFO] sm_86 (Ampere)"
    else
        export CUDA_ARCH="75;80;86;89"
        echo "[INFO] Multi-arch build"
    fi
}

build_keyhunt() {
    echo "============================================"
    echo "Building Keyhunt with CUDA..."
    echo "============================================"

    cd "$WORK_DIR"

    if [ ! -d "keyhuntM1CPU" ]; then
        git clone "$REPO_URL" keyhuntM1CPU || {
            echo "[ERROR] Clone failed. Authenticate with: gh auth login"
            exit 1
        }
    else
        cd keyhuntM1CPU && git pull && cd ..
    fi

    cd keyhuntM1CPU
    rm -rf build

    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DKEYHUNT_USE_CUDA=ON \
        -DKEYHUNT_APPLE_SILICON_ONLY=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

    cmake --build build -j$(nproc)

    if [ ! -f "build/keyhunt" ]; then
        echo "[ERROR] Build failed!"
        send_discord "Build Failed" "Binary not found" 16711680
        exit 1
    fi

    echo "[INFO] Build successful!"
    ./build/keyhunt -h | head -20 || true
}

# ============================================================================
# CREATE TARGET FILE
# ============================================================================

create_target_file() {
    # BSGS 모드는 공개키 필요
    echo "$PUZZLE71_PUBKEY" > "${WORK_DIR}/puzzle71_target.txt"
    echo "[INFO] Target file created (PUBLIC KEY for GPU BSGS)"
}

# ============================================================================
# RUN KEYHUNT
# ============================================================================

run_keyhunt() {
    echo "============================================"
    echo "Starting GPU BSGS Hunt - Puzzle #71"
    echo "============================================"

    cd "${WORK_DIR}/keyhuntM1CPU"

    touch "$RANGES_SEARCHED_FILE"

    local resume_pos=$(load_checkpoint)
    local range_flag=""

    if [ -n "$resume_pos" ]; then
        echo "[INFO] Resuming from: $resume_pos"
        # Bit 71: 0x400000000000000000 ~ 0x800000000000000000
        range_flag="-r ${resume_pos}:800000000000000000"
        send_discord "Hunt Resumed" "From checkpoint: $resume_pos" 3447003
    else
        echo "[INFO] Starting fresh"
        # 랜덤 시작 위치 생성 (Puzzle 71 범위 내)
        # 0x400000000000000000 ~ 0x800000000000000000
        local random_offset=$(printf "%016x" $((RANDOM * RANDOM * RANDOM)))
        local random_start="4${random_offset:1:17}"
        local random_end="800000000000000000"
        echo "[INFO] Random range: 0x${random_start} ~ 0x${random_end}"
        range_flag="-r ${random_start}:${random_end}"
        send_discord "Hunt Started" "GPU BSGS random range: 0x${random_start}" 65280
    fi

    # 모니터 스크립트 생성
    cat > "${WORK_DIR}/keyhunt_monitor.sh" << 'MONITOR_EOF'
#!/bin/bash
LOG_FILE="$1"
RESULT_FILE="$2"
DISCORD_WEBHOOK="$3"
CHECKPOINT_FILE="$4"
RANGES_FILE="$5"

send_found() {
    local key="$1"
    curl -s -H "Content-Type: application/json" \
        -d "{
            \"content\": \"@everyone\",
            \"embeds\": [{
                \"title\": \"🎉 PUZZLE #71 SOLVED! 🎉\",
                \"description\": \"**PRIVATE KEY FOUND!**\",
                \"color\": 65280,
                \"fields\": [
                    {\"name\": \"Private Key\", \"value\": \"\`$key\`\", \"inline\": false},
                    {\"name\": \"Address\", \"value\": \"1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU\", \"inline\": false},
                    {\"name\": \"⚠️ ACTION\", \"value\": \"TRANSFER BTC IMMEDIATELY!\", \"inline\": false}
                ]
            }]
        }" "$DISCORD_WEBHOOK"
}

save_checkpoint() {
    local pos=$(tail -100 "$LOG_FILE" 2>/dev/null | grep -oE 'Thread 0x[0-9a-fA-F]+' | tail -1 | grep -oE '0x[0-9a-fA-F]+' || echo "")
    if [ -n "$pos" ]; then
        echo "$pos" > "$CHECKPOINT_FILE"
        echo "$(date -Iseconds) $pos" >> "$RANGES_FILE"
    fi
}

last_progress=0
last_checkpoint=0

while true; do
    now=$(date +%s)

    # BSGS 모드 키 발견 패턴
    if grep -q "Key found privkey" "$LOG_FILE" 2>/dev/null; then
        echo "[MONITOR] KEY FOUND!"
        key=$(grep "Key found privkey" "$LOG_FILE" | grep -oE 'privkey [0-9a-fA-F]+' | tail -1 | awk '{print $2}')
        
        if [ -n "$key" ]; then
            echo "FOUND: $(date)" > "$RESULT_FILE"
            echo "Private Key: $key" >> "$RESULT_FILE"
            send_found "$key"
            pkill -f "keyhunt.*bsgs" || true
            exit 0
        fi
    fi

    # 5분마다 체크포인트
    if [ $((now - last_checkpoint)) -ge 300 ]; then
        save_checkpoint
        last_checkpoint=$now
    fi

    # 30분마다 진행상황
    if [ $((now - last_progress)) -ge 1800 ]; then
        speed=$(tail -20 "$LOG_FILE" | grep -oE '[0-9]+\.?[0-9]* [PTGMK]?keys/s' | tail -1 || echo "calculating...")
        ranges=$(wc -l < "$RANGES_FILE" 2>/dev/null || echo "0")
        
        curl -s -H "Content-Type: application/json" \
            -d "{\"embeds\": [{\"title\": \"📊 Progress\", \"description\": \"Speed: $speed\\nRanges: $ranges\", \"color\": 3447003}]}" \
            "$DISCORD_WEBHOOK" || true
        
        last_progress=$now
    fi

    sleep 30
done
MONITOR_EOF

    chmod +x "${WORK_DIR}/keyhunt_monitor.sh"

    # GPU BSGS 명령어 (핵심 수정!)
    local cmd="./build/keyhunt -m bsgs -f ${WORK_DIR}/puzzle71_target.txt $range_flag --gpu -g $GPU_DEVICE --gpu-threads $GPU_THREADS --gpu-blocks $GPU_BLOCKS -t $CPU_THREADS -s $STATS_INTERVAL"
    
    echo "[CMD] $cmd"
    
    nohup $cmd >> "$LOG_FILE" 2>&1 &
    echo $! > "${WORK_DIR}/keyhunt.pid"
    echo "[INFO] Keyhunt PID: $!"

    nohup "${WORK_DIR}/keyhunt_monitor.sh" "$LOG_FILE" "$RESULT_FILE" "$DISCORD_WEBHOOK" "$CHECKPOINT_FILE" "$RANGES_SEARCHED_FILE" >> "${WORK_DIR}/monitor.log" 2>&1 &
    echo $! > "${WORK_DIR}/monitor.pid"
    echo "[INFO] Monitor PID: $!"

    echo ""
    echo "============================================"
    echo "✅ GPU BSGS RUNNING"
    echo "============================================"
    echo ""
    echo "GPU: Device $GPU_DEVICE, $GPU_THREADS threads × $GPU_BLOCKS blocks"
    echo "Mode: BSGS with GPU acceleration"
    echo ""
    echo "Commands:"
    echo "  tail -f $LOG_FILE"
    echo "  ./vastai_deploy_fixed.sh status"
    echo "  ./vastai_deploy_fixed.sh stop"
    echo ""
}

# ============================================================================
# STATUS
# ============================================================================

status() {
    echo "============================================"
    echo "Keyhunt Status"
    echo "============================================"

    if [ -f "${WORK_DIR}/keyhunt.pid" ]; then
        pid=$(cat "${WORK_DIR}/keyhunt.pid")
        if ps -p $pid > /dev/null 2>&1; then
            echo "[✓] Running (PID: $pid)"
        else
            echo "[✗] Not running (stale PID)"
        fi
    else
        echo "[✗] Not running"
    fi

    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "Last position: $(cat $CHECKPOINT_FILE)"
    fi

    if [ -f "$RANGES_SEARCHED_FILE" ]; then
        echo "Ranges searched: $(wc -l < $RANGES_SEARCHED_FILE)"
    fi

    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "Last 10 lines:"
        tail -10 "$LOG_FILE"
    fi

    nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv 2>/dev/null || true
}

# ============================================================================
# STOP
# ============================================================================

stop() {
    echo "Stopping gracefully..."
    save_checkpoint

    pkill -f "keyhunt.*bsgs" 2>/dev/null || true
    pkill -f "keyhunt_monitor" 2>/dev/null || true
    
    rm -f "${WORK_DIR}/keyhunt.pid" "${WORK_DIR}/monitor.pid"

    echo "✅ Stopped. Resume with: ./vastai_deploy.sh run"
    send_discord "Hunt Stopped" "Progress saved" 16776960
}

# ============================================================================
# MAIN
# ============================================================================

case "${1:-run}" in
    setup)
        setup_environment
        build_keyhunt
        create_target_file
        echo "Setup complete! Run: ./vastai_deploy.sh run"
        ;;
    build)
        setup_environment
        build_keyhunt
        ;;
    run|resume)
        if [ ! -f "${WORK_DIR}/keyhuntM1CPU/build/keyhunt" ]; then
            echo "[INFO] Building first..."
            setup_environment
            build_keyhunt
        fi
        create_target_file
        run_keyhunt
        ;;
    status)
        status
        ;;
    stop)
        stop
        ;;
    *)
        echo "Usage: $0 {setup|build|run|resume|status|stop}"
        ;;
esac
