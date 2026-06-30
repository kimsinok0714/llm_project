#!/usr/bin/env bash
# GPU(VRAM)를 점유한 Jupyter 커널/파이썬 프로세스 정리 도구
#
# 사용법:
#   ./gpu_clean.sh            # 점유 프로세스 목록만 표시 (아무것도 죽이지 않음)
#   ./gpu_clean.sh --all      # GPU 점유 파이썬 프로세스 전부 종료
#   ./gpu_clean.sh 1234 5678  # 지정한 PID만 종료
#
# 노트북 사용 후 커널을 'Shutdown' 하지 않으면 프로세스가 VRAM을 계속
# 잡고 있어 CUDA out of memory 가 발생합니다. 그때 이 스크립트로 정리하세요.

set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "❌ nvidia-smi 를 찾을 수 없습니다 (NVIDIA 드라이버 미설치?)"
    exit 1
fi

show_apps() {
    echo "🖥️  GPU 점유 프로세스 (메모리 큰 순):"
    echo "------------------------------------------------------------"
    printf "%-10s %-12s %-10s %s\n" "PID" "VRAM(MiB)" "ELAPSED" "CMD"
    # pid, used_memory 를 가져와 ps 로 경과시간/명령을 보강
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
        | sort -t, -k2 -rn \
        | while IFS=',' read -r pid mem; do
            pid=$(echo "$pid" | tr -d ' ')
            mem=$(echo "$mem" | tr -d ' ')
            etime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ' || echo "?")
            cmd=$(ps -o comm= -p "$pid" 2>/dev/null || echo "?")
            printf "%-10s %-12s %-10s %s\n" "$pid" "$mem" "$etime" "$cmd"
        done
    echo "------------------------------------------------------------"
    echo "💡 오래된(ELAPSED 큰) 커널이 보통 좀비 커널입니다."
    echo "   현재 작업 중인 커널은 죽이지 마세요."
}

free_mem() {
    echo
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv
}

case "${1:-}" in
    "")
        show_apps
        free_mem
        echo
        echo "👉 종료하려면:  ./gpu_clean.sh <PID> [PID...]   또는   ./gpu_clean.sh --all"
        ;;
    --all)
        pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ')
        if [ -z "$pids" ]; then
            echo "✅ GPU 점유 프로세스가 없습니다."
            exit 0
        fi
        echo "⚠️  GPU 점유 파이썬 프로세스를 모두 종료합니다: $pids"
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
        echo "✅ 종료 완료."
        free_mem
        ;;
    *)
        echo "⚠️  종료할 PID: $*"
        kill -9 "$@" 2>/dev/null || true
        echo "✅ 종료 완료."
        free_mem
        ;;
esac