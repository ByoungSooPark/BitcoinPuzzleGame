/*
 * Keyhunt CUDA - BSGS (Baby Step Giant Step) Kernel
 *
 * This kernel performs the "giant step" phase of BSGS on the GPU.
 * Each thread checks a different starting point in parallel.
 *
 * The 32-bit limb representation allows us to maximize GPU throughput
 * since NVIDIA GPUs have significantly more 32-bit ALUs than 64-bit.
 */

#include "secp256k1.cuh"
#include <stdio.h>
#include <stdint.h>
 #include <atomic>
 #include <chrono>

// ============================================================================
// Configuration
// ============================================================================

#define BLOCK_SIZE 256
#define POINTS_PER_THREAD 256

// ============================================================================
// Shared Memory for Baby Step Table Lookup
// ============================================================================

// We store a compressed hash of X coordinates for quick lookup
// Full verification happens on CPU for matches
struct BabyStepEntry {
    uint32_t hash;      // 32-bit hash of X coordinate
    uint32_t index;     // Index in baby step table
};

// ============================================================================
// Device-resident legacy bloom filters (256-way bucketed)
// Forward declarations (used by kernels declared before definitions)
// ============================================================================

extern __device__ __managed__ uint8_t* g_bloomFlat;
extern __device__ __managed__ uint64_t g_bloomBytesPer;
extern __device__ __managed__ uint64_t g_bloomBits;
extern __device__ __managed__ uint8_t g_bloomHashes;

__global__ void legacyGiantGroupBloomBatchKernel(
    const uint32_t* startXBatch, const uint32_t* startYBatch,
    const uint32_t* stepX, const uint32_t* stepY,
    int groupSize,
    int batchCount,
    uint8_t* outHits
);

static __device__ __forceinline__ void u256_to_raw32_be(const uint256_t* x, uint8_t out[32]);

 static std::atomic<uint64_t> g_legacyGroupCheckCalls{0};
 static std::atomic<uint64_t> g_legacyGroupCheckPoints{0};
 static std::atomic<uint64_t> g_legacyGroupCheckNanos{0};

 extern "C" void keyhunt_cudaGetLegacyGroupCheckStats(uint64_t* calls, uint64_t* points, uint64_t* nanos) {
     if (calls) *calls = g_legacyGroupCheckCalls.load(std::memory_order_relaxed);
     if (points) *points = g_legacyGroupCheckPoints.load(std::memory_order_relaxed);
     if (nanos) *nanos = g_legacyGroupCheckNanos.load(std::memory_order_relaxed);
 }

__global__ void legacyDebugFirstXKernel(
    const uint32_t* startX, const uint32_t* startY,
    const uint32_t* stepX, const uint32_t* stepY,
    int groupSize,
    uint8_t* outX32
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (groupSize <= 0) return;

    Point start;
    Point step;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        start.x.limbs[i] = startX[i];
        start.y.limbs[i] = startY[i];
        step.x.limbs[i] = stepX[i];
        step.y.limbs[i] = stepY[i];
    }
    start.z.limbs[0] = 1;
    step.z.limbs[0] = 1;
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        start.z.limbs[i] = 0;
        step.z.limbs[i] = 0;
    }

    // tid=0 in legacy kernel => offset = -half
    int half = groupSize / 2;
    int offset = -half;

    Point cur;
    copy256(&cur.x, &start.x);
    copy256(&cur.y, &start.y);
    copy256(&cur.z, &start.z);

    if (offset != 0) {
        uint256_t k;
        uint32_t a = (uint32_t)(-offset);
        k.limbs[0] = a;
        #pragma unroll
        for (int i = 1; i < 8; i++) k.limbs[i] = 0;

        Point stepUse;
        copy256(&stepUse.x, &step.x);
        copy256(&stepUse.y, &step.y);
        copy256(&stepUse.z, &step.z);

        // offset < 0 => negate Y
        uint256_t p;
        set256FromConst(&p, SECP256K1_P);
        modSub(&stepUse.y, &p, &stepUse.y);

        Point mul;
        scalarMult(&mul, &k, &stepUse);
        pointAdd(&cur, &cur, &mul);
    }

    Point affine;
    copy256(&affine.x, &cur.x);
    copy256(&affine.y, &cur.y);
    copy256(&affine.z, &cur.z);
    toAffine(&affine);

    uint8_t raw[32];
    u256_to_raw32_be(&affine.x, raw);
    for (int i = 0; i < 32; i++) {
        outX32[i] = raw[i];
    }
}

__global__ void legacyDebugScalarMultXKernel(
    const uint32_t* stepX, const uint32_t* stepY,
    uint32_t k_scalar,
    uint8_t* outX32
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    Point step;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        step.x.limbs[i] = stepX[i];
        step.y.limbs[i] = stepY[i];
        step.z.limbs[i] = 0;
    }
    step.z.limbs[0] = 1;

    uint256_t k;
    k.limbs[0] = k_scalar;
    #pragma unroll
    for (int i = 1; i < 8; i++) k.limbs[i] = 0;

    Point mul;
    scalarMult(&mul, &k, &step);
    toAffine(&mul);

    uint8_t raw[32];
    u256_to_raw32_be(&mul.x, raw);
    for (int i = 0; i < 32; i++) {
        outX32[i] = raw[i];
    }
}

// Known 2G values for secp256k1 (little-endian limbs)
// Computed from pointDouble(G) with verified Jacobian coordinates
// 2G.x = 0x16C6F9E5C80E994548060F8478A5147AA81D5962E7280DACC6EC4EBBBE1C6325
// 2G.y = 0x5A5CF285F9A029C23BF48C3521261623F36A66566A6E55472059041EF649298E
__constant__ uint32_t KNOWN_2GX[8] = {
    0xBE1C6325, 0xC6EC4EBB, 0xE7280DAC, 0xA81D5962,
    0x78A5147A, 0x48060F84, 0xC80E9945, 0x16C6F9E5
};
__constant__ uint32_t KNOWN_2GY[8] = {
    0xF649298E, 0x2059041E, 0x6A6E5547, 0xF36A6656,
    0x21261623, 0x3BF48C35, 0xF9A029C2, 0x5A5CF285
};

// Test kernel: compute 2G using pointDouble(G) and compare to known value
// Also test basic modAdd(&x, &x, &x) to verify aliasing fix
__global__ void testPointDoubleKernel(uint32_t* outResult) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Test modInv: compute 2^-1 mod p, then verify 2 * 2^-1 = 1 mod p
    uint256_t val_2, inv_2, test_result;
    for (int i = 0; i < 8; i++) val_2.limbs[i] = 0;
    val_2.limbs[0] = 2;
    
    modInv(&inv_2, &val_2);
    modMul(&test_result, &val_2, &inv_2);
    
    // test_result should be 1
    int add_test1 = (test_result.limbs[0] == 1) ? 1 : 0;
    int add_test2 = 1;
    for (int i = 1; i < 8; i++) {
        if (test_result.limbs[i] != 0) add_test2 = 0;
    }
    
    // Also test with Z from pointDouble
    uint256_t z_val, z_inv, z_test;
    set256FromConst(&z_val, SECP256K1_GY);
    modAdd(&z_val, &z_val, &z_val);  // Z = 2*Gy
    
    modInv(&z_inv, &z_val);
    modMul(&z_test, &z_val, &z_inv);
    
    int add_test3 = (z_test.limbs[0] == 1) ? 1 : 0;
    for (int i = 1; i < 8; i++) {
        if (z_test.limbs[i] != 0) add_test3 = 0;
    }

    // Test pointDouble step by step with G
    Point G;
    set256FromConst(&G.x, SECP256K1_GX);
    set256FromConst(&G.y, SECP256K1_GY);
    G.z.limbs[0] = 1;
    for (int i = 1; i < 8; i++) G.z.limbs[i] = 0;
    
    // Manual pointDouble calculation to debug
    uint256_t Y2_test, S_test, M_test;
    
    // Step 1: Y2 = Gy^2
    modSqr(&Y2_test, &G.y);
    // Expected: 0x8dad9b1c47e776deae6860f9a07240aa44ce50498c351de69368de5e8d0fba9e
    int y2_ok = (Y2_test.limbs[0] == 0x8d0fba9e) ? 1 : 0;
    
    // Step 2: S = 4*Gx*Y2
    modMul(&S_test, &G.x, &Y2_test);
    uint256_t S_temp;
    modAdd(&S_temp, &S_test, &S_test);
    modAdd(&S_test, &S_temp, &S_temp);
    // Expected S: 0x17ee882cab862478855fd28eccca911c2e6f3b65a8345994f58b9638770afacf
    int s_ok = (S_test.limbs[0] == 0x770afacf) ? 1 : 0;
    
    // Step 3: M = 3*Gx^2
    modSqr(&M_test, &G.x);
    uint256_t M_temp;
    modAdd(&M_temp, &M_test, &M_test);
    modAdd(&M_test, &M_temp, &M_test);
    // Expected M: 0x8ff2b776aaf6d91942fd096d2f1f7fd9aa2f64be71462131aa7f067e28fef8ac
    int m_ok = (M_test.limbs[0] == 0x28fef8ac) ? 1 : 0;

    // Compute 2G using pointDouble
    Point twoG;
    pointDouble(&twoG, &G);
    
    // Save Jacobian coordinates before toAffine
    uint256_t jacobian_x, jacobian_y, jacobian_z;
    copy256(&jacobian_x, &twoG.x);
    copy256(&jacobian_y, &twoG.y);
    copy256(&jacobian_z, &twoG.z);
    
    // Manual toAffine to debug
    uint256_t zInv, zInv2, zInv3;
    modInv(&zInv, &jacobian_z);
    modSqr(&zInv2, &zInv);
    modMul(&zInv3, &zInv2, &zInv);
    
    modMul(&twoG.x, &jacobian_x, &zInv2);
    modMul(&twoG.y, &jacobian_y, &zInv3);
    
    twoG.z.limbs[0] = 1;
    for (int i = 1; i < 8; i++) twoG.z.limbs[i] = 0;

    // Compare X coordinate
    uint256_t expected_x;
    set256FromConst(&expected_x, KNOWN_2GX);
    int x_match = (cmp256(&twoG.x, &expected_x) == 0) ? 1 : 0;

    // Compare Y coordinate
    uint256_t expected_y;
    set256FromConst(&expected_y, KNOWN_2GY);
    int y_match = (cmp256(&twoG.y, &expected_y) == 0) ? 1 : 0;

    // Output: [0]=x_match, [1]=y_match, [2..9]=computed 2G.x, [10..17]=computed 2G.y
    // [18]=add_test1, [19]=add_test2, [20]=add_test3, [21]=testVal.limbs[0]
    // [22..29]=jacobian_y, [30..37]=jacobian_z
    outResult[0] = x_match;
    outResult[1] = y_match;
    for (int i = 0; i < 8; i++) {
        outResult[2 + i] = twoG.x.limbs[i];
        outResult[10 + i] = twoG.y.limbs[i];
    }
    // [18-20] = modInv test results
    outResult[18] = add_test1;  // 2 * 2^-1 = 1 (limbs[0])
    outResult[19] = add_test2;  // 2 * 2^-1 = 1 (other limbs)
    outResult[20] = add_test3;  // Z * Z^-1 = 1
    outResult[21] = test_result.limbs[0];  // actual result for debugging
    
    // [22-29] = Jacobian Y, [30-37] = Jacobian Z, [44-51] = Jacobian X
    for (int i = 0; i < 8; i++) {
        outResult[22 + i] = jacobian_y.limbs[i];
        outResult[30 + i] = jacobian_z.limbs[i];
        if (i < 8) outResult[44 + i] = jacobian_x.limbs[i];
    }
    
    // [38-40] = step-by-step test results (Y2, S, M)
    outResult[38] = y2_ok;
    outResult[39] = s_ok;
    outResult[40] = m_ok;
    
    // [41-43] = actual values for debugging
    outResult[41] = Y2_test.limbs[0];
    outResult[42] = S_test.limbs[0];
    outResult[43] = M_test.limbs[0];
}

extern "C" int keyhunt_cudaTestPointDouble(
    int* x_match, int* y_match,
    uint32_t computed_2gx[8], uint32_t computed_2gy[8],
    int* add_test1, int* add_test2, int* add_test3, uint32_t* add_result,
    uint32_t jacobian_y[8], uint32_t jacobian_z[8]
) {
    static uint32_t* d_result = nullptr;
    if (!d_result) {
        cudaError_t err = cudaMalloc((void**)&d_result, 60 * sizeof(uint32_t));
        if (err != cudaSuccess) return -1;
    }

    testPointDoubleKernel<<<1, 1>>>(d_result);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return -1;

    uint32_t h_result[60];
    err = cudaMemcpy(h_result, d_result, 60 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return -1;

    *x_match = h_result[0];
    *y_match = h_result[1];
    for (int i = 0; i < 8; i++) {
        computed_2gx[i] = h_result[2 + i];
        computed_2gy[i] = h_result[10 + i];
    }
    *add_test1 = h_result[18];
    *add_test2 = h_result[19];
    *add_test3 = h_result[20];
    *add_result = h_result[21];
    for (int i = 0; i < 8; i++) {
        jacobian_y[i] = h_result[22 + i];
        jacobian_z[i] = h_result[30 + i];
    }
    
    // Debug: print step-by-step test results
    printf("[DEBUG] Y2_ok=%d S_ok=%d M_ok=%d\n", h_result[38], h_result[39], h_result[40]);
    printf("[DEBUG] Y2.limbs[0]=0x%08x (expect 0x8d0fba9e)\n", h_result[41]);
    printf("[DEBUG] S.limbs[0]=0x%08x (expect 0x770afacf)\n", h_result[42]);
    printf("[DEBUG] M.limbs[0]=0x%08x (expect 0x28fef8ac)\n", h_result[43]);
    
    // Print Jacobian coordinates for debugging
    printf("[DEBUG] Jacobian X' = ");
    for (int i = 7; i >= 0; i--) printf("%08x", h_result[44 + i]);
    printf("\n[DEBUG] Expected X' = e4bb45c7a7752314b72017f71d414998314729ad88bcb1c34acba64198ef4ad7\n");
    
    printf("[DEBUG] Jacobian Y' = ");
    for (int i = 7; i >= 0; i--) printf("%08x", jacobian_y[i]);
    printf("\n[DEBUG] Expected Y' = 633499139e2fcf82ce6864b001721e2ffa58a9355b68f6d36533ee4d88b016da\n");
    
    // Print computed affine coordinates
    printf("[DEBUG] Computed affine 2G.x = ");
    for (int i = 7; i >= 0; i--) printf("%08x", computed_2gx[i]);
    printf("\n[DEBUG] Computed affine 2G.y = ");
    for (int i = 7; i >= 0; i--) printf("%08x", computed_2gy[i]);
    printf("\n");
    
    // Print simple modMul test results
    printf("[CUDA][TEST] modMul 2*3: ok=%d result=%u (expected 6)\n", h_result[39], h_result[41]);
    printf("[CUDA][TEST] modMul 0xFFFFFFFF^2: ok=%d limbs[0]=%08x limbs[1]=%08x (expected 00000001 fffffffe)\n", 
           h_result[40], h_result[42], h_result[43]);
    printf("[CUDA][TEST] modMul 2^128*2^128 (reduction): ok=%d limbs[0]=%08x limbs[1]=%08x (expected 000003d1 00000001)\n",
           h_result[44], h_result[45], h_result[46]);
    printf("[CUDA][TEST] Gy load: ok=%d limbs[0]=%08x (expected fb10d4b8)\n", h_result[47], h_result[48]);
    
    return 0;
}

extern "C" int keyhunt_cudaLegacyDebugFirstX(
    const uint32_t startX[8], const uint32_t startY[8],
    const uint32_t stepX[8], const uint32_t stepY[8],
    int groupSize,
    uint8_t outX32[32]
) {
    if (!startX || !startY || !stepX || !stepY || !outX32) return -1;
    if (groupSize <= 0) return -1;

    static thread_local uint32_t* d_startX = nullptr;
    static thread_local uint32_t* d_startY = nullptr;
    static thread_local uint32_t* d_stepX  = nullptr;
    static thread_local uint32_t* d_stepY  = nullptr;
    static thread_local uint8_t*  d_outX32 = nullptr;

    cudaError_t err = cudaSuccess;
    if (!d_startX) {
        err = cudaMalloc((void**)&d_startX, 8u * sizeof(uint32_t));
        if (err != cudaSuccess) return -1;
        err = cudaMalloc((void**)&d_startY, 8u * sizeof(uint32_t));
        if (err != cudaSuccess) return -1;
        err = cudaMalloc((void**)&d_stepX, 8u * sizeof(uint32_t));
        if (err != cudaSuccess) return -1;
        err = cudaMalloc((void**)&d_stepY, 8u * sizeof(uint32_t));
        if (err != cudaSuccess) return -1;
        err = cudaMalloc((void**)&d_outX32, 32u);
        if (err != cudaSuccess) return -1;
    }

    err = cudaMemcpy(d_startX, startX, 8u * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(d_startY, startY, 8u * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(d_stepX, stepX, 8u * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(d_stepY, stepY, 8u * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    legacyDebugFirstXKernel<<<1, 1>>>(d_startX, d_startY, d_stepX, d_stepY, groupSize, d_outX32);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(outX32, d_outX32, 32u, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

extern "C" int keyhunt_cudaLegacyGroupCheckBatch(
    const uint32_t* startXBatch, const uint32_t* startYBatch,
    int batchCount,
    const uint32_t stepX[8], const uint32_t stepY[8],
    int groupSize,
    void* d_hits,
    uint8_t* outHits,
    int threadsPerBlock,
    int numBlocks
) {
    const auto t0 = std::chrono::steady_clock::now();
    if (startXBatch == NULL || startYBatch == NULL || stepX == NULL || stepY == NULL || d_hits == NULL || outHits == NULL) {
        return -1;
    }
    if (batchCount <= 0 || groupSize <= 0) {
        return -1;
    }

    g_legacyGroupCheckCalls.fetch_add(1u, std::memory_order_relaxed);
    g_legacyGroupCheckPoints.fetch_add((uint64_t)groupSize * (uint64_t)batchCount, std::memory_order_relaxed);

    static thread_local uint32_t* d_startXBatch = nullptr;
    static thread_local uint32_t* d_startYBatch = nullptr;
    static thread_local int d_batchCapacity = 0;

    static thread_local uint32_t* d_stepX  = nullptr;
    static thread_local uint32_t* d_stepY  = nullptr;

    cudaError_t err = cudaSuccess;

    if (d_stepX == nullptr) {
        err = cudaMalloc((void**)&d_stepX, 8u * sizeof(uint32_t));
        if (err != cudaSuccess) return -1;
        err = cudaMalloc((void**)&d_stepY, 8u * sizeof(uint32_t));
        if (err != cudaSuccess) return -1;
    }

    if (d_startXBatch == nullptr || d_startYBatch == nullptr || d_batchCapacity < batchCount) {
        if (d_startXBatch) cudaFree(d_startXBatch);
        if (d_startYBatch) cudaFree(d_startYBatch);
        d_startXBatch = nullptr;
        d_startYBatch = nullptr;
        d_batchCapacity = 0;

        size_t bytes = (size_t)batchCount * 8u * sizeof(uint32_t);
        err = cudaMalloc((void**)&d_startXBatch, bytes);
        if (err != cudaSuccess) return -1;
        err = cudaMalloc((void**)&d_startYBatch, bytes);
        if (err != cudaSuccess) return -1;
        d_batchCapacity = batchCount;
    }

    err = cudaMemcpy(d_stepX, stepX, 8u * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(d_stepY, stepY, 8u * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    size_t batchBytes = (size_t)batchCount * 8u * sizeof(uint32_t);
    err = cudaMemcpy(d_startXBatch, startXBatch, batchBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(d_startYBatch, startYBatch, batchBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    if (threadsPerBlock <= 0) threadsPerBlock = 256;
    if (threadsPerBlock > 1024) threadsPerBlock = 1024;

    int total = groupSize * batchCount;
    const int requiredBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;
    int blocks = numBlocks;
    if (blocks <= 0) {
        blocks = requiredBlocks;
    } else if (blocks > requiredBlocks) {
        blocks = requiredBlocks;
    }

    legacyGiantGroupBloomBatchKernel<<<blocks, threadsPerBlock>>>(d_startXBatch, d_startYBatch, d_stepX, d_stepY, groupSize, batchCount, (uint8_t*)d_hits);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(outHits, d_hits, (size_t)total, cudaMemcpyDeviceToHost);
    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t dn = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    g_legacyGroupCheckNanos.fetch_add(dn, std::memory_order_relaxed);
    return (err == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Legacy bloom_check() compatibility (XXH64-based)
// ============================================================================

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

__device__ __forceinline__ uint64_t read64_le(const uint8_t* p) {
    return ((uint64_t)p[0]) |
           ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) |
           ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) |
           ((uint64_t)p[7] << 56);
}

__device__ __forceinline__ uint64_t read32_le(const uint8_t* p) {
    return ((uint64_t)p[0]) |
           ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24);
}

__device__ __forceinline__ uint64_t xxh64_round(uint64_t acc, uint64_t input) {
    const uint64_t PRIME64_1 = 11400714785074694791ULL;
    const uint64_t PRIME64_2 = 14029467366897019727ULL;
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

__device__ __forceinline__ uint64_t xxh64_merge_round(uint64_t acc, uint64_t val) {
    const uint64_t PRIME64_1 = 11400714785074694791ULL;
    const uint64_t PRIME64_4 = 9650029242287828579ULL;
    acc ^= xxh64_round(0, val);
    acc *= PRIME64_1;
    acc += PRIME64_4;
    return acc;
}

__device__ __forceinline__ uint64_t xxh64_avalanche(uint64_t h64) {
    const uint64_t PRIME64_2 = 14029467366897019727ULL;
    const uint64_t PRIME64_3 = 1609587929392839161ULL;
    h64 ^= h64 >> 33;
    h64 *= PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= PRIME64_3;
    h64 ^= h64 >> 32;
    return h64;
}

__device__ __forceinline__ uint64_t xxh64(const void* input, int len, uint64_t seed) {
    const uint8_t* p = (const uint8_t*)input;
    const uint8_t* bEnd = p + len;
    const uint64_t PRIME64_1 = 11400714785074694791ULL;
    const uint64_t PRIME64_2 = 14029467366897019727ULL;
    const uint64_t PRIME64_3 = 1609587929392839161ULL;
    const uint64_t PRIME64_4 = 9650029242287828579ULL;
    const uint64_t PRIME64_5 = 2870177450012600261ULL;

    uint64_t h64;
    if (len >= 32) {
        const uint8_t* const limit = bEnd - 32;
        uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
        uint64_t v2 = seed + PRIME64_2;
        uint64_t v3 = seed + 0;
        uint64_t v4 = seed - PRIME64_1;

        do {
            v1 = xxh64_round(v1, read64_le(p)); p += 8;
            v2 = xxh64_round(v2, read64_le(p)); p += 8;
            v3 = xxh64_round(v3, read64_le(p)); p += 8;
            v4 = xxh64_round(v4, read64_le(p)); p += 8;
        } while (p <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h64 = xxh64_merge_round(h64, v1);
        h64 = xxh64_merge_round(h64, v2);
        h64 = xxh64_merge_round(h64, v3);
        h64 = xxh64_merge_round(h64, v4);
    } else {
        h64 = seed + PRIME64_5;
    }

    h64 += (uint64_t)len;

    while (p + 8 <= bEnd) {
        uint64_t k1 = xxh64_round(0, read64_le(p));
        h64 ^= k1;
        h64 = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }

    if (p + 4 <= bEnd) {
        h64 ^= read32_le(p) * PRIME64_1;
        h64 = rotl64(h64, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }

    while (p < bEnd) {
        h64 ^= (*p) * PRIME64_5;
        h64 = rotl64(h64, 11) * PRIME64_1;
        p++;
    }

    return xxh64_avalanche(h64);
}

__device__ __forceinline__ int bloom_test_bit(const uint8_t* bf, uint64_t bit) {
    uint64_t byte = bit >> 3;
    uint8_t c = bf[byte];
    uint8_t mask = (uint8_t)(1u << (bit & 7u));
    return (c & mask) ? 1 : 0;
}
__device__ __forceinline__ int bloom_check_compat(
    const uint8_t* bf,
    uint64_t bits,
    uint8_t hashes,
    const void* buffer,
    int len
) {
    uint64_t a = xxh64(buffer, len, 0x59f2815b16f81798ULL);
    uint64_t b = xxh64(buffer, len, a);

    for (uint8_t i = 0; i < hashes; i++) {
        uint64_t x = (a + (uint64_t)b * (uint64_t)i) % bits;
        if (!bloom_test_bit(bf, x)) {
            return 0;
        }
    }
    return 1;
}

static __device__ __forceinline__ void u256_to_raw32_be(const uint256_t* x, uint8_t out[32]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t limb = x->limbs[7 - i];
        out[i * 4 + 0] = (uint8_t)(limb >> 24);
        out[i * 4 + 1] = (uint8_t)(limb >> 16);
        out[i * 4 + 2] = (uint8_t)(limb >> 8);
        out[i * 4 + 3] = (uint8_t)(limb);
    }
}

__global__ void legacyGiantGroupBloomKernel(
    const uint32_t* startX, const uint32_t* startY,
    const uint32_t* stepX, const uint32_t* stepY,
    int groupSize,
    uint8_t* outHits
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= groupSize) return;

    Point start;
    Point step;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        start.x.limbs[i] = startX[i];
        start.y.limbs[i] = startY[i];
        step.x.limbs[i] = stepX[i];
        step.y.limbs[i] = stepY[i];
    }
    start.z.limbs[0] = 1;
    step.z.limbs[0] = 1;
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        start.z.limbs[i] = 0;
        step.z.limbs[i] = 0;
    }

    int half = groupSize / 2;
    int offset = tid - half;

    Point cur;
    copy256(&cur.x, &start.x);
    copy256(&cur.y, &start.y);
    copy256(&cur.z, &start.z);

    if (offset != 0) {
        uint256_t k;
        uint32_t a = (uint32_t)((offset < 0) ? -offset : offset);
        k.limbs[0] = a;
        #pragma unroll
        for (int i = 1; i < 8; i++) k.limbs[i] = 0;

        Point stepUse;
        copy256(&stepUse.x, &step.x);
        copy256(&stepUse.y, &step.y);
        copy256(&stepUse.z, &step.z);

        if (offset < 0) {
            uint256_t p;
            set256FromConst(&p, SECP256K1_P);
            modSub(&stepUse.y, &p, &stepUse.y);
        }

        Point mul;
        scalarMult(&mul, &k, &stepUse);
        pointAdd(&cur, &cur, &mul);
    }

    Point affine;
    copy256(&affine.x, &cur.x);
    copy256(&affine.y, &cur.y);
    copy256(&affine.z, &cur.z);
    toAffine(&affine);

    uint8_t raw[32];
    u256_to_raw32_be(&affine.x, raw);

    if (g_bloomFlat == nullptr || g_bloomBytesPer == 0 || g_bloomBits == 0 || g_bloomHashes == 0) {
        outHits[tid] = 0;
        return;
    }

    uint8_t bucket = raw[0];
    const uint8_t* bf = g_bloomFlat + ((size_t)bucket * (size_t)g_bloomBytesPer);
    int r = bloom_check_compat(bf, g_bloomBits, g_bloomHashes, raw, 32);
    outHits[tid] = (uint8_t)r;
}

__global__ void legacyGiantGroupBloomBatchKernel(
    const uint32_t* startXBatch, const uint32_t* startYBatch,
    const uint32_t* stepX, const uint32_t* stepY,
    int groupSize,
    int batchCount,
    uint8_t* outHits
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = groupSize * batchCount;
    if (tid >= total) return;

    int batchIdx = tid / groupSize;
    int localTid = tid - (batchIdx * groupSize);

    const uint32_t* startX = startXBatch + ((size_t)batchIdx * 8u);
    const uint32_t* startY = startYBatch + ((size_t)batchIdx * 8u);

    Point start;
    Point step;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        start.x.limbs[i] = startX[i];
        start.y.limbs[i] = startY[i];
        step.x.limbs[i] = stepX[i];
        step.y.limbs[i] = stepY[i];
    }
    start.z.limbs[0] = 1;
    step.z.limbs[0] = 1;
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        start.z.limbs[i] = 0;
        step.z.limbs[i] = 0;
    }

    int half = groupSize / 2;
    int offset = localTid - half;

    Point cur;
    copy256(&cur.x, &start.x);
    copy256(&cur.y, &start.y);
    copy256(&cur.z, &start.z);

    if (offset != 0) {
        uint256_t k;
        uint32_t a = (uint32_t)((offset < 0) ? -offset : offset);
        k.limbs[0] = a;
        #pragma unroll
        for (int i = 1; i < 8; i++) k.limbs[i] = 0;

        Point stepUse;
        copy256(&stepUse.x, &step.x);
        copy256(&stepUse.y, &step.y);
        copy256(&stepUse.z, &step.z);

        if (offset < 0) {
            uint256_t p;
            set256FromConst(&p, SECP256K1_P);
            modSub(&stepUse.y, &p, &stepUse.y);
        }

        Point mul;
        scalarMult(&mul, &k, &stepUse);
        pointAdd(&cur, &cur, &mul);
    }

    Point affine;
    copy256(&affine.x, &cur.x);
    copy256(&affine.y, &cur.y);
    copy256(&affine.z, &cur.z);
    toAffine(&affine);

    uint8_t raw[32];
    u256_to_raw32_be(&affine.x, raw);

    if (g_bloomFlat == nullptr || g_bloomBytesPer == 0 || g_bloomBits == 0 || g_bloomHashes == 0) {
        outHits[tid] = 0;
        return;
    }

    uint8_t bucket = raw[0];
    const uint8_t* bf = g_bloomFlat + ((size_t)bucket * (size_t)g_bloomBytesPer);
    int r = bloom_check_compat(bf, g_bloomBits, g_bloomHashes, raw, 32);
    outHits[tid] = (uint8_t)r;
}

// ============================================================================
// Device-resident legacy bloom filters (256-way bucketed)
// ============================================================================

__device__ __managed__ uint8_t* g_bloomFlat = nullptr;
__device__ __managed__ uint64_t g_bloomBytesPer = 0;
__device__ __managed__ uint64_t g_bloomBits = 0;
__device__ __managed__ uint8_t g_bloomHashes = 0;

__global__ void bloomCheckBatchKernel(const uint8_t* values32, uint32_t count, uint8_t* outHits) {
    uint32_t tid = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;

    const uint8_t* v = values32 + ((size_t)tid * 32u);
    uint8_t bucket = v[0];

    if (g_bloomFlat == nullptr || g_bloomBytesPer == 0 || g_bloomBits == 0 || g_bloomHashes == 0) {
        outHits[tid] = 0;
        return;
    }

    const uint8_t* bf = g_bloomFlat + ((size_t)bucket * (size_t)g_bloomBytesPer);
    int r = bloom_check_compat(bf, g_bloomBits, g_bloomHashes, v, 32);
    outHits[tid] = (uint8_t)r;
}

// ============================================================================
// Hash function for X coordinate (for bloom-like lookup)
// ============================================================================

__device__ __forceinline__ uint32_t hashXCoord(const uint256_t* x) {
    // Simple hash combining all limbs
    uint32_t h = 0x811c9dc5;  // FNV offset basis
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        h ^= x->limbs[i];
        h *= 0x01000193;  // FNV prime
    }
    return h;
}

// ============================================================================
// Check if X coordinate might be in baby step table (bloom filter style)
// ============================================================================

__device__ __forceinline__ bool checkBloom(
    const uint256_t* x,
    const uint8_t* bloomFilter,
    uint32_t bloomSize
) {
    uint32_t h1 = hashXCoord(x);
    uint32_t h2 = (h1 >> 16) | (h1 << 16);

    uint32_t bit1 = h1 % (bloomSize * 8);
    uint32_t bit2 = h2 % (bloomSize * 8);
    uint32_t bit3 = (h1 ^ h2) % (bloomSize * 8);

    bool b1 = (bloomFilter[bit1 / 8] >> (bit1 % 8)) & 1;
    bool b2 = (bloomFilter[bit2 / 8] >> (bit2 % 8)) & 1;
    bool b3 = (bloomFilter[bit3 / 8] >> (bit3 % 8)) & 1;

    return b1 && b2 && b3;
}

// ============================================================================
// BSGS Giant Step Kernel
// ============================================================================

__global__ void bsgsGiantStepKernel(
    // Target public key (what we're looking for)
    const uint32_t* targetX,
    const uint32_t* targetY,

    // Giant step parameters
    const uint32_t* giantStepX,    // X coord of m*G (giant step increment)
    const uint32_t* giantStepY,    // Y coord of m*G
    const uint32_t* startOffsetLimbs, // Starting offset in range

    // Bloom filter for baby step table
    const uint8_t* bloomFilter,
    uint32_t bloomSize,

    // Output: potential matches
    uint32_t* matchFlags,          // 1 if potential match found
    uint32_t* matchIndices,        // Giant step index where match found
    uint32_t* matchXCoords,        // X coordinate of potential match

    // Search parameters
    uint64_t numGiantSteps,
    uint64_t giantStepsPerThread
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t totalThreads = gridDim.x * blockDim.x;

    // Load target point
    Point target;
    for (int i = 0; i < 8; i++) {
        target.x.limbs[i] = targetX[i];
        target.y.limbs[i] = targetY[i];
    }
    target.z.limbs[0] = 1;
    for (int i = 1; i < 8; i++) target.z.limbs[i] = 0;

    // Load giant step (negative, for subtraction)
    Point giantStep;
    for (int i = 0; i < 8; i++) {
        giantStep.x.limbs[i] = giantStepX[i];
        giantStep.y.limbs[i] = giantStepY[i];
    }
    giantStep.z.limbs[0] = 1;
    for (int i = 1; i < 8; i++) giantStep.z.limbs[i] = 0;

    // Negate Y for subtraction (P - Q = P + (-Q))
    uint256_t p;
    set256FromConst(&p, SECP256K1_P);
    modSub(&giantStep.y, &p, &giantStep.y);

    // Calculate starting point for this thread
    uint64_t startIdx = tid * giantStepsPerThread;
    if (startIdx >= numGiantSteps) return;

    uint64_t endIdx = min(startIdx + giantStepsPerThread, numGiantSteps);

    // Compute starting point: target - startIdx * giantStep
    Point current;
    copy256(&current.x, &target.x);
    copy256(&current.y, &target.y);
    copy256(&current.z, &target.z);

    // Skip to our starting position
    if (startIdx > 0) {
        uint256_t skipAmount;
        skipAmount.limbs[0] = (uint32_t)(startIdx & 0xFFFFFFFF);
        skipAmount.limbs[1] = (uint32_t)(startIdx >> 32);
        for (int i = 2; i < 8; i++) skipAmount.limbs[i] = 0;

        // Compute skip * giantStep and subtract from target
        Point skipPoint;
        Point giantStepPositive;
        copy256(&giantStepPositive.x, &giantStep.x);
        // Use positive Y for scalar mult
        modSub(&giantStepPositive.y, &p, &giantStep.y);
        copy256(&giantStepPositive.z, &giantStep.z);

        scalarMult(&skipPoint, &skipAmount, &giantStepPositive);

        // Negate for subtraction
        modSub(&skipPoint.y, &p, &skipPoint.y);

        pointAdd(&current, &target, &skipPoint);
    }

    // Giant step loop
    for (uint64_t i = startIdx; i < endIdx; i++) {
        // Convert to affine for X comparison
        Point affine;
        copy256(&affine.x, &current.x);
        copy256(&affine.y, &current.y);
        copy256(&affine.z, &current.z);
        toAffine(&affine);

        // Check bloom filter
        if (checkBloom(&affine.x, bloomFilter, bloomSize)) {
            // Potential match! Record it for CPU verification
            uint32_t idx = atomicAdd(&matchFlags[0], 1);
            if (idx < 1024) {  // Limit matches to avoid overflow
                matchIndices[idx] = (uint32_t)i;
                // Store X coordinate for verification
                for (int j = 0; j < 8; j++) {
                    matchXCoords[idx * 8 + j] = affine.x.limbs[j];
                }
            }
        }

        // Move to next giant step: current = current - giantStep
        pointAdd(&current, &current, &giantStep);
    }
}

// ============================================================================
// Batch Point Generation Kernel (for baby steps)
// ============================================================================

__global__ void generateBabyStepsKernel(
    uint32_t* outputX,      // Array of X coordinates
    uint32_t* outputHash,   // Array of hashes
    uint64_t numPoints,
    uint64_t startIndex
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPoints) return;

    uint64_t pointIndex = startIndex + tid;

    // Compute pointIndex * G
    uint256_t k;
    k.limbs[0] = (uint32_t)(pointIndex & 0xFFFFFFFF);
    k.limbs[1] = (uint32_t)(pointIndex >> 32);
    for (int i = 2; i < 8; i++) k.limbs[i] = 0;

    // Generator point
    Point G;
    set256FromConst(&G.x, SECP256K1_GX);
    set256FromConst(&G.y, SECP256K1_GY);
    G.z.limbs[0] = 1;
    for (int i = 1; i < 8; i++) G.z.limbs[i] = 0;

    Point result;
    scalarMult(&result, &k, &G);
    toAffine(&result);

    // Store X coordinate
    for (int i = 0; i < 8; i++) {
        outputX[tid * 8 + i] = result.x.limbs[i];
    }

    // Store hash
    outputHash[tid] = hashXCoord(&result.x);
}

// ============================================================================
// Host-side wrapper functions
// ============================================================================

extern "C" {

// Initialize CUDA device
int cudaInit(int deviceId) {
    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    printf("[CUDA] Using device: %s\n", prop.name);
    printf("[CUDA] Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("[CUDA] Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("[CUDA] Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("[CUDA] Total global memory: %.2f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    return 0;
}

// Get number of CUDA devices
int keyhunt_cudaGetDeviceCount() {
    int count;
    cudaError_t err = ::cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return 0;
    return count;
}

// Upload legacy 256-way bloom filter (flattened bf arrays)
int keyhunt_cudaSetBloom(const uint8_t* bloomFlat, uint64_t bytesPerBloom, uint64_t bits, uint8_t hashes) {
    if (bloomFlat == NULL || bytesPerBloom == 0 || bits == 0 || hashes == 0) {
        return -1;
    }

    size_t total = (size_t)256u * (size_t)bytesPerBloom;
    if (g_bloomFlat != nullptr) {
        cudaFree(g_bloomFlat);
        g_bloomFlat = nullptr;
    }

    cudaError_t err = cudaMalloc((void**)&g_bloomFlat, total);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Malloc Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(g_bloomFlat, bloomFlat, total, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Memcpy Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    g_bloomBytesPer = bytesPerBloom;
    g_bloomBits = bits;
    g_bloomHashes = hashes;
    return 0;
}

// Batch bloom check for legacy rawvalue[32] buffers
int keyhunt_cudaBloomCheckBatch(const uint8_t* values32, uint32_t count, uint8_t* outHits) {
    if (values32 == NULL || outHits == NULL || count == 0) {
        return -1;
    }

    uint8_t* d_values = nullptr;
    uint8_t* d_hits = nullptr;
    size_t valuesBytes = (size_t)count * 32u;

    cudaError_t err = cudaMalloc((void**)&d_values, valuesBytes);
    if (err != cudaSuccess) return -1;
    err = cudaMalloc((void**)&d_hits, (size_t)count);
    if (err != cudaSuccess) {
        cudaFree(d_values);
        return -1;
    }

    err = cudaMemcpy(d_values, values32, valuesBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_values);
        cudaFree(d_hits);
        return -1;
    }

    int threads = 256;
    int blocks = (int)((count + (uint32_t)threads - 1u) / (uint32_t)threads);
    bloomCheckBatchKernel<<<blocks, threads>>>(d_values, count, d_hits);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_values);
        cudaFree(d_hits);
        return -1;
    }

    err = cudaMemcpy(outHits, d_hits, (size_t)count, cudaMemcpyDeviceToHost);
    cudaFree(d_values);
    cudaFree(d_hits);
    return (err == cudaSuccess) ? 0 : -1;
}

// Allocate reusable device buffers for bloom batch checking
int keyhunt_cudaBloomBatchAlloc(uint32_t maxCount, void** d_values, void** d_hits) {
    if (maxCount == 0 || d_values == NULL || d_hits == NULL) {
        return -1;
    }

    uint8_t* dv = nullptr;
    uint8_t* dh = nullptr;
    cudaError_t err = cudaMalloc((void**)&dv, (size_t)maxCount * 32u);
    if (err != cudaSuccess) return -1;
    err = cudaMalloc((void**)&dh, (size_t)maxCount);
    if (err != cudaSuccess) {
        cudaFree(dv);
        return -1;
    }

    *d_values = (void*)dv;
    *d_hits = (void*)dh;
    return 0;
}

int keyhunt_cudaBloomBatchFree(void* d_values, void* d_hits) {
    if (d_values) cudaFree(d_values);
    if (d_hits)    cudaFree(d_hits);

    return 0;
}

extern "C" int keyhunt_cudaLegacyDebugScalarMultX(
    const uint32_t stepX[8], const uint32_t stepY[8],
    uint32_t k_scalar,
    uint8_t outX32[32]
) {
    uint32_t* d_stepX = nullptr;
    uint32_t* d_stepY = nullptr;
    uint8_t* d_out = nullptr;

    cudaError_t err;
    err = cudaMalloc((void**)&d_stepX, 8u * sizeof(uint32_t));
    if (err != cudaSuccess) return -1;
    err = cudaMalloc((void**)&d_stepY, 8u * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(d_stepX);
        return -1;
    }
    err = cudaMalloc((void**)&d_out, 32u);
    if (err != cudaSuccess) {
        cudaFree(d_stepX);
        cudaFree(d_stepY);
        return -1;
    }

    err = cudaMemcpy(d_stepX, stepX, 8u * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_stepX);
        cudaFree(d_stepY);
        cudaFree(d_out);
        return -1;
    }
    err = cudaMemcpy(d_stepY, stepY, 8u * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_stepX);
        cudaFree(d_stepY);
        cudaFree(d_out);
        return -1;
    }

    legacyDebugScalarMultXKernel<<<1, 1>>>(d_stepX, d_stepY, k_scalar, d_out);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_stepX);
        cudaFree(d_stepY);
        cudaFree(d_out);
        return -1;
    }

    err = cudaMemcpy(outX32, d_out, 32u, cudaMemcpyDeviceToHost);
    cudaFree(d_stepX);
    cudaFree(d_stepY);
    cudaFree(d_out);
    return (err == cudaSuccess) ? 0 : -1;
}

// Run bloom check using reusable device buffers (copies inputs/outputs)
int keyhunt_cudaBloomBatchRun(void* d_values, void* d_hits, const uint8_t* values32, uint32_t count, uint8_t* outHits) {
    if (d_values == NULL || d_hits == NULL || values32 == NULL || outHits == NULL || count == 0) {
        return -1;
    }

    cudaError_t err = cudaMemcpy(d_values, values32, (size_t)count * 32u, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    int threads = 256;
    int blocks = (int)((count + (uint32_t)threads - 1u) / (uint32_t)threads);
    bloomCheckBatchKernel<<<blocks, threads>>>((const uint8_t*)d_values, count, (uint8_t*)d_hits);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(outHits, d_hits, (size_t)count, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

int keyhunt_cudaLegacyGroupCheck(
    const uint32_t startX[8], const uint32_t startY[8],
    const uint32_t stepX[8], const uint32_t stepY[8],
    int groupSize,
    void* d_hits,
    uint8_t* outHits,
    int threadsPerBlock,
    int numBlocks
) {
    const auto t0 = std::chrono::steady_clock::now();
    if (startX == NULL || startY == NULL || stepX == NULL || stepY == NULL || d_hits == NULL || outHits == NULL) {
        return -1;
    }
    if (groupSize <= 0) {
        return -1;
    }

     g_legacyGroupCheckCalls.fetch_add(1u, std::memory_order_relaxed);
     g_legacyGroupCheckPoints.fetch_add((uint64_t)groupSize, std::memory_order_relaxed);

    int blocks = 0;

    // Avoid per-call device allocations by reusing per-host-thread buffers.
    // keyhunt_legacy launches this from multiple CPU threads; thread_local keeps them independent.
    static thread_local uint32_t* d_startX = nullptr;
    static thread_local uint32_t* d_startY = nullptr;
    static thread_local uint32_t* d_stepX  = nullptr;
    static thread_local uint32_t* d_stepY  = nullptr;

    cudaError_t err = cudaSuccess;
    if (d_startX == nullptr) {
        err = cudaMalloc((void**)&d_startX, 8u * sizeof(uint32_t));
        if (err != cudaSuccess) return -1;
        err = cudaMalloc((void**)&d_startY, 8u * sizeof(uint32_t));
        if (err != cudaSuccess) return -1;
        err = cudaMalloc((void**)&d_stepX, 8u * sizeof(uint32_t));
        if (err != cudaSuccess) return -1;
        err = cudaMalloc((void**)&d_stepY, 8u * sizeof(uint32_t));
        if (err != cudaSuccess) return -1;
    }

    err = cudaMemcpy(d_startX, startX, 8u * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(d_startY, startY, 8u * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(d_stepX, stepX, 8u * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpy(d_stepY, stepY, 8u * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    if (threadsPerBlock <= 0) threadsPerBlock = 256;
    if (threadsPerBlock > 1024) threadsPerBlock = 1024;

    const int requiredBlocks = (groupSize + threadsPerBlock - 1) / threadsPerBlock;
    blocks = numBlocks;
    if (blocks <= 0) {
        blocks = requiredBlocks;
    } else if (blocks > requiredBlocks) {
        blocks = requiredBlocks;
    }

    legacyGiantGroupBloomKernel<<<blocks, threadsPerBlock>>>(d_startX, d_startY, d_stepX, d_stepY, groupSize, (uint8_t*)d_hits);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(outHits, d_hits, (size_t)groupSize, cudaMemcpyDeviceToHost);
    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t dn = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    g_legacyGroupCheckNanos.fetch_add(dn, std::memory_order_relaxed);
    return (err == cudaSuccess) ? 0 : -1;
}

int keyhunt_cudaBloomBatchRunConfig(void* d_values, void* d_hits, const uint8_t* values32, uint32_t count, uint8_t* outHits, int threadsPerBlock, int numBlocks) {
    if (d_values == NULL || d_hits == NULL || values32 == NULL || outHits == NULL || count == 0) {
        return -1;
    }

    if (threadsPerBlock <= 0) {
        threadsPerBlock = 256;
    }
    if (threadsPerBlock > 1024) {
        threadsPerBlock = 1024;
    }

    cudaError_t err = cudaMemcpy(d_values, values32, (size_t)count * 32u, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    const int requiredBlocks = (int)((count + (uint32_t)threadsPerBlock - 1u) / (uint32_t)threadsPerBlock);
    int blocks = numBlocks;
    if (blocks <= 0) {
        blocks = requiredBlocks;
    } else if (blocks > requiredBlocks) {
        blocks = requiredBlocks;
    }

    bloomCheckBatchKernel<<<blocks, threadsPerBlock>>>((const uint8_t*)d_values, count, (uint8_t*)d_hits);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(outHits, d_hits, (size_t)count, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

// Free GPU memory
void cudaFreeMemory(void* ptr) {
    cudaFree(ptr);
}

// Copy to GPU
int cudaCopyToDevice(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

// Copy from GPU
int cudaCopyFromDevice(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

// Launch BSGS giant step search
int cudaLaunchBSGS(
    void* d_targetX, void* d_targetY,
    void* d_giantStepX, void* d_giantStepY,
    void* d_startOffset,
    void* d_bloomFilter, uint32_t bloomSize,
    void* d_matchFlags, void* d_matchIndices, void* d_matchXCoords,
    uint64_t numGiantSteps,
    int numBlocks, int threadsPerBlock
) {
    uint64_t giantStepsPerThread = (numGiantSteps + numBlocks * threadsPerBlock - 1) /
                                   (numBlocks * threadsPerBlock);

    bsgsGiantStepKernel<<<numBlocks, threadsPerBlock>>>(
        (uint32_t*)d_targetX, (uint32_t*)d_targetY,
        (uint32_t*)d_giantStepX, (uint32_t*)d_giantStepY,
        (uint32_t*)d_startOffset,
        (uint8_t*)d_bloomFilter, bloomSize,
        (uint32_t*)d_matchFlags, (uint32_t*)d_matchIndices, (uint32_t*)d_matchXCoords,
        numGiantSteps, giantStepsPerThread
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

} // extern "C"
