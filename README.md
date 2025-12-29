<p align="center">
  <img src="https://img.shields.io/badge/Bitcoin-Puzzle%20Hunter-orange?style=for-the-badge&logo=bitcoin" alt="비트코인 퍼즐 헌터"/>
  <img src="https://img.shields.io/badge/Apple%20Silicon-Optimized-black?style=for-the-badge&logo=apple" alt="애플 실리콘"/>
  <img src="https://img.shields.io/badge/CUDA-Accelerated-76B900?style=for-the-badge&logo=nvidia" alt="CUDA"/>
</p>

<h1 align="center">🔑 Keyhunt</h1>

<p align="center">
  <strong>고성능 비트코인 퍼즐 솔버</strong><br>
  <em>Apple Silicon & NVIDIA CUDA 최적화</em>
</p>

<p align="center">
  <a href="#-기능">기능</a> •
  <a href="#-빠른-시작">빠른 시작</a> •
  <a href="#-cuda-지원">CUDA</a> •
  <a href="#-퍼즐-예제">예제</a> •
  <a href="#-성능">성능</a>
</p>

---

## 🎯 이것은 무엇인가?

Keyhunt는 [비트코인 퍼즐 트랜잭션](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx) 해결을 위한 특화 도구입니다 - **~1000 BTC** 상금이 걸린 점점 어려워지는 챌린지 시리즈입니다. 이 버전은 다음을 위해 최적화되었습니다:

- **Apple Silicon** (M1/M2/M3/M4) - 통합 메모리 + 강력한 코어
- **NVIDIA CUDA** - 대규모 병렬 32비트 연산

## 🧠 32비트의 비밀

> **64비트 하드웨어에서 왜 32비트 청크를 사용하나?**

secp256k1 곡선은 256비트 정수를 사용합니다. 우리는 이를 **8 × 32비트 limb**로 나눕니다:

```
256비트 키 = [limb0][limb1][limb2][limb3][limb4][limb5][limb6][limb7]
              32     32     32     32     32     32     32     32
```

**장점:**
| 플랫폼 | 32비트가 더 빠른 이유 |
|----------|---------------------|
| Apple Silicon | 더 나은 레지스터 활용, 효율적인 캐리 체인 |
| NVIDIA CUDA | GPU는 64비트보다 32비트 ALU가 2-4배 많음 |
| 둘 다 | 범위 반감 최적화 가능 |

---

## ✨ 기능

| 기능 | 설명 |
|---------|-------------|
| 🚀 **BSGS 알고리즘** | Baby Step Giant Step - O(n)을 O(√n)으로 감소 |
| 🌸 **블룸 필터** | 초고속 조회를 위한 3단계 캐스케이드 |
| 🔄 **Endomorphism** | 2-3배 속도 향상을 위한 곡선 트릭 |
| 🧵 **멀티스레드** | 모든 CPU 코어에 걸쳐 확장 |
| 🎮 **CUDA 지원** | NVIDIA GPU로 오프로드 (신규!) |
| 💾 **체크포인트** | 긴 검색 저장/재개 |

---

## 🚀 빠른 시작

### macOS (Apple Silicon)

```bash
# 의존성 설치
brew install cmake openssl@3 gmp

# 클론 및 빌드
git clone https://github.com/consigcody94/keyhuntM1CPU.git
cd keyhuntM1CPU
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)

# 사냥 시작! 🎯
./build/keyhunt -m bsgs -f tests/66.txt -b 66 -t 8 -R
```

### Linux (CUDA 포함)

```bash
# 의존성 설치
sudo apt install cmake libssl-dev libgmp-dev nvidia-cuda-toolkit

# CUDA로 빌드
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKEYHUNT_USE_CUDA=ON
cmake --build build -j$(nproc)

# GPU로 사냥! 🎮
./build/keyhunt -m bsgs -f tests/66.txt -b 66 --gpu -g 0
```

---

## 🎮 CUDA 지원

CUDA 가속은 동일한 32비트 limb 전략을 사용하지만 수천 개의 병렬 검색을 실행합니다:

```
┌─────────────────────────────────────────────────────────────┐
│                      NVIDIA GPU                              │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │
│  │ SM0 │ │ SM1 │ │ SM2 │ │ SM3 │ │ SM4 │ │ SM5 │ │ ... │   │
│  │32bit│ │32bit│ │32bit│ │32bit│ │32bit│ │32bit│ │32bit│   │
│  │ x64 │ │ x64 │ │ x64 │ │ x64 │ │ x64 │ │ x64 │ │ x64 │   │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘   │
│         각 SM은 32비트 연산의 64개 스레드 실행              │
└─────────────────────────────────────────────────────────────┘
```

### CUDA 옵션

| 플래그 | 설명 |
|------|-------------|
| `--gpu` | GPU 가속 활성화 |
| `-g <id>` | GPU 장치 선택 (0, 1, ...) |
| `--gpu-threads <n>` | 블록당 스레드 (기본값: 256) |
| `--gpu-blocks <n>` | 블록 수 (기본값: 자동) |

### 지원 GPU

| GPU | 32비트 코어 | 예상 속도 |
|-----|--------------|----------------|
| RTX 4090 | 16384 | 🔥🔥🔥🔥🔥 |
| RTX 4080 | 9728 | 🔥🔥🔥🔥 |
| RTX 3090 | 10496 | 🔥🔥🔥🔥 |
| RTX 3080 | 8704 | 🔥🔥🔥 |
| RTX 3070 | 5888 | 🔥🔥🔥 |
| GTX 1080 Ti | 3584 | 🔥🔥 |

---

## 🎯 퍼즐 예제

### 퍼즐 #66 (상금: 6.6 BTC ≈ $660,000)
```bash
# CPU만 사용
./build/keyhunt -m bsgs -f tests/66.txt -b 66 -t 8 -R -S

# CUDA 사용
./build/keyhunt -m bsgs -f tests/66.txt -b 66 --gpu -g 0 -R -S
```

### 퍼즐 #130 (상금: 13 BTC ≈ $1,300,000)
```bash
./build/keyhunt -m bsgs -f tests/130.txt -b 130 -t 8 --gpu -S -k 2
```

### 사용자 정의 범위 검색
```bash
./build/keyhunt -m bsgs -f target.txt \
  -r 20000000000000000:3FFFFFFFFFFFFFFFF \
  -t 8 --gpu -S
```

---

## 📊 성능

### BSGS 복잡도 감소

```
무차별 대입:  O(2^66) = 73,786,976,294,838,206,464 연산 😵
BSGS:         O(2^33) = 8,589,934,592 연산 🚀

85억 배 더 빠릅니다!
```

### 속도 비교 (퍼즐 #66)

| 하드웨어 | 키/초 | 검색 시간 |
|----------|----------|----------------|
| Intel i9-13900K | ~50M | ~170초 |
| Apple M3 Max | ~80M | ~107초 |
| RTX 3080 | ~500M | ~17초 |
| RTX 4090 | ~1.2B | ~7초 |

*참고: 실제 성능은 BSGS 매개변수에 따라 다릅니다*

---

## 🛠️ 명령어 참조

```
사용법: keyhunt [옵션]

검색 모드:
  -m bsgs          Baby Step Giant Step (퍼즐에 가장 빠름)
  -m address       주소 무차별 대입
  -m rmd160        RIPEMD-160 해시 검색
  -m xpoint        X 좌표 검색

필수:
  -f <파일>        대상 파일 (공개키 또는 주소)
  -b <비트>        비트 범위 (예: 66)

선택:
  -r <시작:끝>     사용자 정의 16진수 범위
  -t <스레드>      CPU 스레드 (기본값: 모든 코어)
  -k <팩터>        BSGS 테이블 크기를 위한 K 팩터
  -S               블룸 필터 파일 저장/로드
  -R               무작위 시작점
  -q               조용한 모드
  -s <초>          상태 표시 간격

CUDA 옵션:
  --gpu            GPU 가속 활성화
  -g <장치>         GPU 장치 ID
  --gpu-threads    블록당 스레드
  --gpu-blocks     블록 수
```

---

## 📁 프로젝트 구조

```
keyhunt/
├── 🔧 CMakeLists.txt       # 빌드 시스템
├── 📖 README.md            # README.md 여기 있습니다!
├── 🎯 keyhunt_legacy.cpp   # 메인 CPU 구현
├── 🎮 cuda/                # CUDA 커널 (신규!)
│   ├── secp256k1.cu        # GPU 타원곡선 연산
│   └── bsgs_kernel.cu      # GPU BSGS 검색
├── 🔢 gmp256k1/            # 32비트 limb 연산
├── 🌸 bloom/               # 블룸 필터
├── 🔐 hash/                # SHA256, RIPEMD160
└── 🧪 tests/               # 퍼즐 대상 파일
```

---

## 🤔 BSGS 작동 원리

```
┌────────────────────────────────────────────────────────────────┐
│                    BABY STEP GIANT STEP                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  목표: k*G = P를 만족하는 k 찾기  (P는 공개키)                        │
│                                                                │
│  1. BABY STEPS: √n개의 점을 계산하고 저장                           │
│     테이블 = { 0*G, 1*G, 2*G, ..., m*G }  (m = √n)               │
│                                                                │
│  2. GIANT STEPS: P - j*m*G를 테이블과 비교                        │
│     j = 0,1,2,...,m에 대해:                                      │
│       만약 (P - j*m*G)가 테이블의 인덱스 i에 있으면:                   │
│         k = j*m + i  ← 발견! 🎉                                 │
│                                                                │
│  메모리: O(√n)    시간: O(√n)                                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 🙏 크레딧

- 원본 [keyhunt](https://github.com/albertobsd/keyhunt) by albertobsd
- Apple Silicon 최적화 by [@consigcody94](https://github.com/consigcody94)

## 📜 라이선스

MIT License - 책임감 있게 사냥하세요! 🎯

---

<p align="center">
  <strong>⭐ 보물을 찾으면 이 저장소에 별을 주세요! ⭐</strong><br><br>
  <em>~1000 BTC의 미해결 퍼즐이 기다립니다...</em>
</p>
