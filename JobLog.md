# JobLog

## 개요
`keyhuntM1CPU` 프로젝트에서 CUDA(GPU) 옵션/로직을 실제로 동작하는지 검증하고, GPU 경로의 병목을 계측한 뒤, **배치(여러 그룹을 1회 호출로 처리)** 방식으로 호출 오버헤드를 줄이는 개선을 적용했습니다.

---

## 1) Help/Usage에 CUDA 옵션 문구 추가
- **파일**: `keyhunt.cpp`
- **변경**: `menu()`의 `Example:` 섹션 바로 위에 CUDA 옵션 블록 추가
  - `--gpu` / `-g <device>` / `--gpu-threads` / `--gpu-blocks`

---

## 2) GPU 옵션/실제 GPU 로직 위치 확인
- **실제 파싱/실행 로직은 `keyhunt_legacy.cpp`에 존재**
  - `getopt_long` long option:
    - `--gpu` → `FLAGGPU=1`
    - `-g <device>` → `GPU_DEVICE`
    - `--gpu-threads <n>` → `GPU_THREADS`
    - `--gpu-blocks <n>` → `GPU_BLOCKS`
  - `FLAGGPU` 활성 시
    - `keyhunt_cudaGetDeviceCount()`
    - `cudaInit(GPU_DEVICE)`
    - bloom 업로드(`keyhunt_cudaSetBloom`)
    - BSGS 루프에서 GPU 체크(`keyhunt_cudaLegacyGroupCheck...`)

- **CUDA 구현 파일**: `cuda/bsgs_kernel.cu`
  - `cudaInit`, `keyhunt_cudaSetBloom`, `keyhunt_cudaBloomBatch*`, `keyhunt_cudaLegacyGroupCheck`

---

## 3) 빌드 설정(CUDA ON)
### Configure
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKEYHUNT_USE_CUDA=ON
```

### Build
```bash
cmake --build build -j$(nproc)
```

### RTX 3060(sm_86)용 아키텍처 지정 빌드
(초기에 `sm_52`로 빌드되는 것을 확인하여, `sm_86`로 재빌드)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKEYHUNT_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc)
```

---

## 4) 실행 커맨드 정리
### (주의) `-r` 사용법
- `-r`는 반드시 값이 필요: `-r start:end` (hex)
- 값 없이 `-r`만 쓰면 다음 옵션을 range로 오인하여
  - 예: `-r -b 66` → `Invalid hexstring : -b`
  - 예: `-r --gpu` → `Invalid hexstring : --gpu`

### CPU 실행 예시
```bash
./build/keyhunt -m bsgs -f tests/66.txt -t 1 -b 66 -s 5
```

### GPU 실행 예시
```bash
./build/keyhunt -m bsgs -f tests/66.txt --gpu -g 0 --gpu-threads 1024 --gpu-blocks 0 -t 1 -b 66 -s 5
```

### 출력 섞임 방지
- Thread 출력과 carriage return(`\r`) 때문에 라인이 섞일 수 있어 `-q` 권장
```bash
./build/keyhunt -m bsgs -f tests/66.txt --gpu -g 0 --gpu-threads 1024 --gpu-blocks 0 -t 1 -b 66 -s 5 -q
```

### GPU 상태 모니터링
```bash
nvidia-smi
nvidia-smi -l 1
nvidia-smi dmon -s pucm -d 1
```

---

## 5) GPU 실제 동작 검증(계측) 추가
### 목적
`Total ... keys/s`는 기존 코드에서 `steps * BSGS_N / seconds` 기반의 추정치라 실제 GPU 처리량/전력과 괴리가 있어, **GPU 커널이 실제 호출되는지/얼마나 호출되는지**를 확인하기 위해 계측을 추가.

### 변경 사항
- **파일**: `cuda/bsgs_kernel.cu`
  - `g_legacyGroupCheckCalls`, `g_legacyGroupCheckPoints` 카운터 추가
  - `keyhunt_cudaGetLegacyGroupCheckStats(...)`로 호출 수/처리 포인트 수 조회 가능
- **파일**: `keyhunt_legacy.cpp`
  - `-s` 주기마다 아래 형태로 출력
    - `[CUDA] legacyGroupCheck calls=... (+...) points=... (+...)`

### 검증 결과 예시
- `groupSize=1024` 기준에서 `points = calls * 1024` 형태로 증가 → GPU 경로가 실제로 실행됨을 확인.

---

## 6) 호출 시간(us/call) 계측 추가
### 목적
GPU 경로 병목 파악(커널/동기화/복사 오버헤드) 위해 `legacyGroupCheck` 1회 호출 시간 측정.

### 변경 사항
- **파일**: `cuda/bsgs_kernel.cu`
  - `std::chrono::steady_clock`로 `keyhunt_cudaLegacyGroupCheck...` 호출 구간 시간 측정
  - 누적 나노초 `g_legacyGroupCheckNanos` 추가
  - stats getter에 `nanos` 포함
- **파일**: `keyhunt_legacy.cpp`
  - 출력 추가:
    - `[CUDA] legacyGroupCheck avg_us_per_call total=... interval=...`

---

## 7) GPU 호출 오버헤드 완화
### 7.1) 기존 `keyhunt_cudaLegacyGroupCheck`의 반복 할당 제거
- **파일**: `cuda/bsgs_kernel.cu`
- **내용**: `cudaMalloc/cudaFree` 반복을 제거하기 위해 `thread_local` device buffer 재사용

### 7.2) 과도한 block 수 지정 방지
- **파일**: `cuda/bsgs_kernel.cu`
- **내용**: `numBlocks`가 `requiredBlocks`보다 크면 `requiredBlocks`로 clamp
  - `keyhunt_cudaLegacyGroupCheck`
  - `keyhunt_cudaBloomBatchRunConfig`

---

## 8) 배치 키우기(A안) 구현
### 목표
`1024개(groupSize)`씩 매우 자주 호출되던 구조에서, **한 번 호출에 `1024 * batch`를 처리**하도록 바꿔 호출 오버헤드를 분산.

### 변경 사항
- **파일**: `cuda/bsgs_kernel.cu`
  - `legacyGiantGroupBloomBatchKernel(...)` 추가
  - `extern "C" int keyhunt_cudaLegacyGroupCheckBatch(...)` 추가
    - start center를 batch로 받아 한번에 처리
    - hits는 `groupSize * batchCount` 크기
  - 컴파일 에러 해결을 위해 커널 forward declaration 추가

- **파일**: `keyhunt_legacy.cpp`
  - BSGS GPU 경로에서 batch start center를 만들고 `keyhunt_cudaLegacyGroupCheckBatch` 호출
  - hits 인덱스를 `(bb*CPU_GRP_SIZE + ii)`로 매핑
  - `startP`와 `j`를 batch만큼 전진

### 배치 크기 옵션 추가
- **파일**: `keyhunt_legacy.cpp`
- **옵션**: `--gpu-batch <n>`
  - `GPU_BATCH` 전역값
  - 시작 시 `[CUDA] Batch=<n>` 출력

---

## 9) 배치 적용 후 CPU 오버헤드 안정화
배치 경로에서 interval us/call이 튀는 현상이 있어, CPU측 할당/계산 반복 제거.

- **파일**: `keyhunt_legacy.cpp`
  - 루프 내 `std::vector` 생성 제거 → 함수 스코프에서 `startXB/startYB`를 유지하고 `resize` 재사용
  - `cuda_stepX/cuda_stepY` 변환을 루프 밖(스레드 초기화 시점)으로 이동

---

## 10) 테스트 커맨드 (배치/스레드 튜닝)
### 기본
```bash
./build/keyhunt -m bsgs -f tests/66.txt --gpu -g 0 --gpu-threads 1024 --gpu-blocks 0 --gpu-batch 16 -t 1 -b 66 -s 5 -q
```

### 멀티스레드 테스트 예시
```bash
./build/keyhunt -m bsgs -f tests/66.txt --gpu -g 0 --gpu-threads 1024 --gpu-blocks 0 --gpu-batch 16 -t 10 -b 66 -s 5 -q
./build/keyhunt -m bsgs -f tests/66.txt --gpu -g 0 --gpu-threads 1024 --gpu-blocks 0 --gpu-batch 32 -t 20 -b 66 -s 5 -q
./build/keyhunt -m bsgs -f tests/66.txt --gpu -g 0 --gpu-threads 1024 --gpu-blocks 0 --gpu-batch 64 -t 20 -b 66 -s 5 -q
```

---

## 참고: 현재 관찰된 특성
- `--gpu` 경로에서 GPU가 실제로 동작함은 `calls/points`로 확정.
- `-t`(스레드)를 크게 하면 `avg_us_per_call`이 커질 수 있음(호출 직렬화/컨텐션/동기화 영향).
- 배치가 클수록 `points` 증가량은 커지지만, `us/call`이 커질 수 있으므로 `--gpu-batch`와 `-t` 튜닝 필요.

---

## 11) tests/63.pub + -k 512 GPU 조기 종료(End) 디버깅 진행

### 11.1) 정상(레퍼런스) 케이스
사용자 제공 정상 케이스(구버전):
```bash
time ./keyhunt -m bsgs -t 8 -f tests/63.pub -k 512 -s 0 -S -b 63
```
특징:
- `.blm/.tbl` 파일을 읽어와서 실행
- `Thread Key found` 및 `All points were found`로 종료

### 11.2) 현재(legacy+GPU) 케이스에서 관측
```bash
./build/keyhunt -m bsgs -f tests/63.pub --gpu -g 0 --gpu-threads 1024 --gpu-blocks 0 --gpu-batch 128 -t 20 -b 63 -k 512 -s 5 -q
```
관측:
- bloom을 "Reading"이 아니라 직접 생성/체크섬 후 업로드
- `Sorting ... Done!` 직후 `End`로 즉시 종료되는 현상 발생

### 11.3) 조치/변경 사항(디버깅 및 옵션 추가)
- **`--gpu-batch <n>` 옵션 추가**
  - `GPU_BATCH` 전역값으로 배치 조절
  - 시작 시 `[CUDA] Batch=<n>` 출력

- **배치 크기 overshoot 방지(효과 배치 적용)**
  - `eff_batch = min(GPU_BATCH, cycles - j)` 형태로 마지막 구간에서 과도한 배치로 `j`가 건너뛰지 않도록 클램프

- **추가 진단 출력(원인 파악용)**
  - `[BSGS] ... expected_cycles=...` 출력 추가(메인 초기화 구간)
  - `[BSGS][T0] base_key=... range_end=...` 출력 추가(스레드 루프에서 base_key가 range_end에 도달하는지 확인)
  - GPU 경로에서(스레드0, 최대 3회) `hits` 카운트 출력:
    - `[BSGS][T0][GPU] cycles=... j=... eff_batch=... hits=...`
  - 실행 종료 직전 최종 CUDA 누적 카운터 출력:
    - `[CUDA] final legacyGroupCheck calls=... points=... avg_us_per_call=...`

### 11.4) 최근 관측 로그 요약
- `[BSGS][T0] base_key=... range_end=0x8000...`가 매우 많이 출력되며 `base_key`가 `range_end`를 넘어서는 지점에서 종료
- 즉, 스레드 루프에서 `base_key >= range_end` 조건으로 빠르게 break 되는 흐름이 확인됨
- 이 상태에서 GPU kernel 호출(`legacyGroupCheck`)이 실제로 수행되는지 여부를 확정하기 위해 final CUDA stats 및 hits 출력 계측을 추가함

---

## 12) GPU scalarMult/pointDouble 버그 디버깅 (2024-12-29)

### 12.1) 문제 상황
- GPU bloom check 결과 `hits=0`, `xmatch=0` (CPU와 GPU 결과 불일치)
- `fail_pow2=2` 출력 → `scalarMult(2, step)`에서 실패 (첫 번째 pointDouble에서 오류)

### 12.2) 디버깅용 추가 기능
- **scalarMult 디버그 커널**: `keyhunt_cudaLegacyDebugScalarMultX()` - k*step 계산 결과를 CPU와 비교
- **pointDouble 테스트 커널**: `keyhunt_cudaTestPointDouble()` - 알려진 G에서 2G 계산 후 알려진 값과 비교
- GPU 초기화 시 `[CUDA][TEST] pointDouble(G)->2G: x_match=? y_match=?` 출력

### 12.3) 발견한 버그들

#### (A) modMul reduction 버그 (수정 완료)
- **파일**: `cuda/secp256k1.cuh`
- **문제**: secp256k1 prime reduction에서 carry propagation 오류
- **수정**: modMul 완전 재작성
  - 즉시 carry 전파 방식
  - 2-pass reduction (limbs 8-15 → 0-7 fold)
  - 잔여 carry 처리 + while 루프로 r < p 보장

#### (B) add256/sub256 aliasing 버그 ⚠️ **핵심 버그** (수정 완료)

##### 버그 발생 위치
- **파일**: `cuda/secp256k1.cuh`
- **함수**: `add256()`, `sub256()`

##### 문제 상황
`pointDouble()` 함수에서 Y 좌표 계산 시 `8*Y^4`를 구하는 과정:
```cuda
// Y' = M*(S - X') - 8*Y^4
modSqr(&T, &Y2);            // T = Y^4
modAdd(&T, &T, &T);         // T = 2*Y^4  ← 여기서 r==a==b
modAdd(&T, &T, &T);         // T = 4*Y^4  ← 여기서 r==a==b
modAdd(&T, &T, &T);         // T = 8*Y^4  ← 여기서 r==a==b
modSub(&R->y, &R->y, &T);
```

`modAdd(&T, &T, &T)` 호출 시 내부적으로 `add256(&T, &T, &T)`가 호출되는데,
이때 **r, a, b가 모두 같은 메모리 주소**를 가리킴.

##### 버그 메커니즘 (상세)
```cuda
// 버그 있는 코드
__device__ uint32_t add256(uint256_t* r, const uint256_t* a, const uint256_t* b) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        // 문제: r == a == b일 때
        carry += (uint64_t)a->limbs[i] + (uint64_t)b->limbs[i];
        r->limbs[i] = (uint32_t)carry;  // ← 여기서 a->limbs[i]와 b->limbs[i]도 변경됨!
        carry >>= 32;
    }
    return (uint32_t)carry;
}
```

**구체적 예시** (T = 0x00000001_00000000_... 일 때 T+T 계산):
```
i=0: a->limbs[0]=0, b->limbs[0]=0 → carry=0, r->limbs[0]=0 (OK)
i=1: a->limbs[1]=1, b->limbs[1]=1 → carry=2, r->limbs[1]=2 (OK)
i=2: 여기서 a->limbs[2]를 읽으려 하는데, 이미 r->limbs[1]에 2를 썼음
     하지만 a==r이므로, a->limbs[1]도 2가 됨
     → 다음 iteration에서 잘못된 값 참조
```

실제로는 `i=0`에서 `r->limbs[0]`을 쓰는 순간 `a->limbs[0]`과 `b->limbs[0]`도 변경되므로,
`i=1`에서 이미 오염된 carry 값으로 계산하게 됨.

##### 증상
- `pointDouble(G) → 2G` 테스트 결과:
  - `x_match=1`: X 좌표는 정확 (X 계산에서는 aliasing 호출 없음)
  - `y_match=0`: Y 좌표 오류 (8*Y^4 계산에서 `modAdd(&T,&T,&T)` 3번 호출)
- `scalarMult(2, step)` 실패 → `fail_pow2=2`
- GPU bloom check 결과 `hits=0`, `xmatch=0`

##### 왜 X는 맞고 Y만 틀렸나?
X 계산 코드:
```cuda
modSqr(&R->x, &M);           // R->x = M^2
modSub(&R->x, &R->x, &S);    // R->x = M^2 - S
modSub(&R->x, &R->x, &S);    // R->x = M^2 - 2*S
```
여기서는 `modAdd(&X, &X, &X)` 같은 aliasing 패턴이 없음.

Y 계산에서만 `modAdd(&T, &T, &T)` (자기 자신을 두 배로) 패턴이 사용됨.

##### 수정 방법
입력값을 먼저 로컬 변수에 복사한 후 계산:
```cuda
// 수정된 코드
__device__ uint32_t add256(uint256_t* r, const uint256_t* a, const uint256_t* b) {
    // 1. 먼저 입력값을 로컬에 복사 (aliasing 방지)
    uint32_t av[8], bv[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        av[i] = a->limbs[i];
        bv[i] = b->limbs[i];
    }
    
    // 2. 복사된 값으로 계산
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        carry += (uint64_t)av[i] + (uint64_t)bv[i];
        r->limbs[i] = (uint32_t)carry;
        carry >>= 32;
    }
    return (uint32_t)carry;
}
```

##### 수정 적용
- `add256()`: 입력값 복사 후 계산하도록 수정
- `sub256()`: 동일하게 수정

##### 검증 방법
1. 빌드 후 실행
2. `[CUDA][TEST] pointDouble(G)->2G: x_match=1 y_match=1` 확인
3. `fail_pow2=0` 또는 출력 없음 확인
4. `hits > 0`, `xmatch=1` 확인

### 12.4) 추가 디버깅 진행 (2024-12-29 오후)

#### 단계 1: add256 aliasing 수정 후 테스트
- `add256`, `sub256`에 입력값 복사 로직 추가
- **결과**: `y_match=0` 여전히 발생

#### 단계 2: pointDouble aliasing 회피 수정
- `modAdd(&T, &T, &T)` 패턴을 임시 변수 사용으로 변경
- S, Z 계산에서도 aliasing 회피 적용
- **결과**: `y_match=0` 여전히 발생, computed 2G.y 값 동일

#### 단계 3: modAdd aliasing 테스트
- 테스트 커널에 간단한 테스트 추가: `x=1 → x+x=2 → x+x=4 → x+x=8`
- **결과**: `1->2=1, 2->4=1, 4->8=1, final=8` → 작은 값에서는 정상 작동!

#### 단계 4: Jacobian 좌표 출력 추가
- `toAffine` 전에 Jacobian Y, Z 값을 저장하고 출력
- **결과**:
  - `Jacobian Z = 9075b4ee4d4788cabb49f7f81c221151fa2f68914d0aa833388fa11ff621a970`
  - 알려진 2*Gy와 비교: **다름!**

#### 단계 5: Python으로 정확한 값 계산 및 비교
```python
p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B4480A6855419C47D08FFB10D4B8
2*Gy = 0x9075b4ee4d4788cabb49f7f81c221151fa2f689014d0aa83388fa11ff621a970
```

#### 단계 6: limb 단위 비교에서 버그 발견! ⚠️
```
GPU 2*Gy limbs (little-endian):
  limbs[0] = f621a970  ✓
  limbs[1] = 388fa11f  ✓
  limbs[2] = 4d0aa833  ✗ (expected: 14d0aa83)
  limbs[3] = fa2f6891  ✗ (expected: fa2f6890, +1)
  limbs[4] = 1c221151  ✓
  limbs[5] = bb49f7f8  ✓
  limbs[6] = 4d4788ca  ✓
  limbs[7] = 9075b4ee  ✓
```

**문제 분석**:
- limbs[0], [1]은 정확 → carry propagation 시작은 정상
- limbs[2]가 완전히 틀림 (0x14d0aa83 → 0x4d0aa833)
- limbs[3]에 1이 추가됨 (carry가 잘못 전파됨)
- limbs[4]~[7]은 다시 정확

**원인 추정**:
- `add256`에서 큰 값의 carry propagation에 문제
- 작은 값(1,2,4,8) 테스트는 통과하지만 큰 값에서 실패
- limbs[1]에서 limbs[2]로 carry 전파 시 값이 손상됨

### 12.5) 현재 상태 (작업 중단 시점)
- **진단 중**: `add256` carry propagation 버그 - 큰 값에서 limbs[2] 손상
- **테스트 추가**: Gy+Gy 직접 테스트로 `gy_doubled.limbs[2]` 값 확인 대기

### 12.6) 다음 작업 (재개 시)
1. **테스트 실행** (이미 빌드됨):
   ```bash
   ./build/keyhunt -m bsgs -f tests/63.pub --gpu -g 0 --gpu-threads 1024 --gpu-blocks 0 --gpu-batch 128 -t 1 -b 63 -k 8 -s 5 -q
   ```

2. **확인할 출력**:
   - `modAdd aliasing: 1->2=?, 2->4=?, 4->8=?, final=?`
   - `final` 값이 `0x14d0aa83`이면 정상, `0x4d0aa833`이면 버그 확인

3. **버그 확인 시 수정 방향**:
   - `add256` 구현 재검토
   - carry 변수 타입 및 shift 연산 확인
   - CUDA 컴파일러 최적화 문제 가능성 검토 (`#pragma unroll` 제거 시도)

---

## 참고: 수정된 파일 목록 (12번 작업)
- `cuda/secp256k1.cuh`: add256, sub256, modMul, pointDouble 수정
- `cuda/bsgs_kernel.cu`: testPointDoubleKernel 확장 (Jacobian Y/Z 출력, Gy+Gy 테스트)
- `keyhunt_legacy.cpp`: keyhunt_cudaTestPointDouble 선언 및 호출 업데이트



## 컴파일 방법
```bash
cmake --build build -j$(nproc)
./build/keyhunt -m bsgs -f tests/63.pub --gpu -g 0 --gpu-threads 1024 --gpu-blocks 0 --gpu-batch 128 -t 1 -b 63 -k 8 -s 5 -q
```

---

## 13) --gpu-batch 옵션 파싱 버그 수정 (2024-12-29)

### 문제
```
./build/keyhunt: invalid option -- '-'
[E] Unknow opcion -?
```
`--gpu-batch` 옵션이 인식되지 않음

### 원인
`keyhunt_legacy.cpp`의 `long_options` 배열에 `gpu-batch` 항목이 누락됨
- `case 1003` 처리 코드는 존재했으나 옵션 등록이 빠짐

### 수정
- **파일**: `keyhunt_legacy.cpp` (548-554행)
- `long_options` 배열에 `{"gpu-batch", required_argument, 0, 1003}` 추가

```c
static struct option long_options[] = {
    {"gpu", no_argument, 0, 1000},
    {"gpu-threads", required_argument, 0, 1001},
    {"gpu-blocks", required_argument, 0, 1002},
    {"gpu-batch", required_argument, 0, 1003},  // 추가됨
    {0, 0, 0, 0}
};
```
