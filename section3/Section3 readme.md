# 📕 Day 3 — Effective Memory Use in CUDA

> **Course:** Fundamentals of Accelerated Computing with CUDA Python
> **Provider:** NVIDIA Deep Learning Institute (DLI)
> **Section:** Section 3 of 3
> **Focus:** Memory coalescing, shared memory bank conflicts, tiling strategies, and performance profiling

---

## 🎯 Day 3 Objective

Day 3 is about making GPU code *fast*, not just correct. Writing a kernel that produces correct results is only the first step — the difference between a naive kernel and an optimized one can be **5–20× in performance**, often without changing a single line of computation logic. The key is understanding **how memory is physically accessed** on the GPU.

This section covers the two most impactful memory optimization techniques in CUDA: **memory coalescing** and **shared memory bank conflicts**.

---

## 🧠 Concepts Covered

### 1. The GPU Memory Hierarchy

Before optimizing, you need to understand where data lives:

```
┌──────────────────────────────────────────────────────┐
│                      GPU Chip                        │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │              Streaming Multiprocessor (SM)   │    │
│  │                                             │    │
│  │  ┌──────────┐  ┌──────────────────────────┐ │    │
│  │  │ Registers│  │     Shared Memory        │ │    │
│  │  │ (~1 cy)  │  │    (~5 cycles / 48 KB)   │ │    │
│  │  └──────────┘  └──────────────────────────┘ │    │
│  └─────────────────────────────────────────────┘    │
│                          │                           │
│  ┌───────────────────────┴─────────────────────┐    │
│  │              L2 Cache (~200 cycles)           │    │
│  └───────────────────────┬─────────────────────┘    │
│                          │                           │
│  ┌───────────────────────┴─────────────────────┐    │
│  │           Global Memory / DRAM (~500 cycles) │    │
│  └─────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

| Memory Type | Latency | Size | Scope |
|---|---|---|---|
| **Registers** | ~1 cycle | 32–255 per thread | Per thread |
| **Shared Memory** | ~5 cycles | 48 KB per block | Per block |
| **L1/L2 Cache** | ~30–200 cycles | 32–4096 KB | Per SM / chip |
| **Global Memory** | ~400–800 cycles | GB range | Entire GPU |

**Optimization goal:** Keep data in registers and shared memory as much as possible; access global memory in the most efficient pattern when you must.

---

### 2. Memory Coalescing

**Coalescing** refers to combining multiple thread memory accesses into a single, wide memory transaction.

When 32 threads in a warp access memory, the GPU can service all 32 accesses in **one transaction** if they access **consecutive memory addresses**. If the accesses are scattered or strided, the GPU must issue multiple transactions — dramatically reducing effective bandwidth.

#### ✅ Coalesced Access (fast)

```python
@cuda.jit
def coalesced_kernel(data, result):
    i = cuda.grid(1)
    if i < data.size:
        result[i] = data[i] * 2.0   # Thread i reads element i — consecutive!
```

```
Warp threads:  T0  T1  T2  T3  T4  T5  T6  T7 ...
Memory:        [0] [1] [2] [3] [4] [5] [6] [7] ...
               ──────────────────────────────────
               Single 256-byte transaction ✅
```

#### ❌ Strided Access (slow)

```python
@cuda.jit
def strided_kernel(data, result):
    i = cuda.grid(1)
    if i < data.size:
        result[i] = data[i * 2]    # Thread i reads element 2i — stride of 2!
```

```
Warp threads:  T0  T1  T2  T3  T4  T5  T6  T7 ...
Memory:        [0] [2] [4] [6] [8][10][12][14] ...
               ──────────────────────────────────
               Multiple transactions needed ❌
```

#### ❌ Random Access (worst)

```python
@cuda.jit
def random_kernel(data, indices, result):
    i = cuda.grid(1)
    if i < data.size:
        result[i] = data[indices[i]]   # Completely random — one transaction per thread ❌
```

**Rule of thumb:** For coalesced access in a 2D kernel (e.g., matrix), have threads in the same warp vary along the **column (x) dimension**, not the row:

```python
@cuda.jit
def matrix_kernel_fast(matrix, result):
    row, col = cuda.grid(2)
    # col = threadIdx.x varies within warp → consecutive memory → coalesced ✅
    if row < matrix.shape[0] and col < matrix.shape[1]:
        result[row, col] = matrix[row, col] * 2.0
```

---

### 3. Coalescing with Tiling — Matrix Transpose

Matrix transpose is the classic example where naïve access is uncoalesced in either the read or the write. The solution is **tiling with shared memory**:

1. Read a tile from global memory — **coalesced read** into shared memory
2. Transpose within shared memory (fast, on-chip)
3. Write the tile to the output — **coalesced write** from shared memory

```python
from numba import cuda, float32
import numpy as np

TILE = 32

@cuda.jit
def transpose_tiled(src, dst):
    # Allocate shared tile (+1 column to avoid bank conflicts)
    tile = cuda.shared.array(shape=(TILE, TILE + 1), dtype=float32)

    x = cuda.blockIdx.x * TILE + cuda.threadIdx.x
    y = cuda.blockIdx.y * TILE + cuda.threadIdx.y

    # Coalesced read from global memory into shared (row-major)
    if x < src.shape[1] and y < src.shape[0]:
        tile[cuda.threadIdx.y][cuda.threadIdx.x] = src[y, x]

    cuda.syncthreads()

    # Compute transposed output coordinates
    x_out = cuda.blockIdx.y * TILE + cuda.threadIdx.x
    y_out = cuda.blockIdx.x * TILE + cuda.threadIdx.y

    # Coalesced write from shared to global memory (column becomes row)
    if x_out < dst.shape[1] and y_out < dst.shape[0]:
        dst[y_out, x_out] = tile[cuda.threadIdx.x][cuda.threadIdx.y]
```

---

### 4. Shared Memory Bank Conflicts

Shared memory is divided into **32 memory banks** (for modern GPUs). Multiple threads accessing the **same bank simultaneously** causes a **bank conflict** — accesses are serialized, hurting performance.

#### Understanding Banks

Shared memory addresses map to banks in round-robin fashion:

```
Address:  0   1   2   3  ...  31  32  33  34  ...
Bank:     0   1   2   3  ...  31   0   1   2  ...
```

#### ✅ No Bank Conflict — Threads access different banks

```python
# Each thread accesses a different bank — parallel ✅
shared[threadIdx.x]         # Thread 0 → bank 0, Thread 1 → bank 1, etc.
```

#### ❌ Bank Conflict — Multiple threads access the same bank

```python
# Stride of 2 — threads 0,16 both access bank 0; threads 1,17 access bank 1; etc.
shared[threadIdx.x * 2]     # 2-way bank conflict ❌

# Stride of 32 — ALL threads access bank 0 → 32-way bank conflict ❌
shared[threadIdx.x * 32]
```

#### Fixing Bank Conflicts — Padding

The standard fix is to add **one extra column** of padding to the shared array:

```python
# ❌ With bank conflicts — 32-column tile, stride-32 access pattern
tile = cuda.shared.array(shape=(32, 32), dtype=float32)

# ✅ Without bank conflicts — padding shifts addresses to avoid overlap
tile = cuda.shared.array(shape=(32, 33), dtype=float32)  # +1 padding column
```

The `+1` column shifts each row's starting address by one bank, so threads that previously conflicted now land on different banks.

---

### 5. Combined Optimization — Tiled Matrix Multiplication

Matrix multiplication benefits enormously from shared memory tiling because each element of the input matrices is reused multiple times.

```python
from numba import cuda, float32
import numpy as np

TILE_SIZE = 16

@cuda.jit
def matmul_tiled(A, B, C):
    # Shared memory tiles for A and B
    sA = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)
    sB = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)

    row = cuda.blockIdx.y * TILE_SIZE + cuda.threadIdx.y
    col = cuda.blockIdx.x * TILE_SIZE + cuda.threadIdx.x
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y

    acc = float32(0.0)
    n_tiles = (A.shape[1] + TILE_SIZE - 1) // TILE_SIZE

    for tile_idx in range(n_tiles):
        # Load tile from A and B into shared memory
        sA[ty, tx] = A[row, tile_idx * TILE_SIZE + tx] if (
            row < A.shape[0] and tile_idx * TILE_SIZE + tx < A.shape[1]
        ) else 0.0

        sB[ty, tx] = B[tile_idx * TILE_SIZE + ty, col] if (
            tile_idx * TILE_SIZE + ty < B.shape[0] and col < B.shape[1]
        ) else 0.0

        cuda.syncthreads()

        # Compute partial dot product from shared memory
        for k in range(TILE_SIZE):
            acc += sA[ty, k] * sB[k, tx]

        cuda.syncthreads()

    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = acc
```

**Why this is fast:**
- Each element of A and B is loaded from global memory only **once per tile** instead of once per output element
- All subsequent accesses use **shared memory** (~100× faster)
- Access patterns are **coalesced** — threads in a warp read consecutive columns

---

### 6. Performance Impact Summary

| Optimization | Typical Speedup |
|---|---|
| Baseline naive kernel | 1× |
| Coalesced global memory access | 2–5× |
| Shared memory tiling | 5–10× |
| Coalescing + tiling + no bank conflicts | 10–20× |

---

### 7. Day 3 Final Assessment

The `assessment/` folder contains the final graded notebook for the course. You are given a CPU-based linear algebra algorithm and must:

1. Profile the CPU baseline
2. Implement a GPU kernel with correct results
3. Apply memory coalescing
4. Apply shared memory tiling
5. Eliminate bank conflicts
6. Benchmark and report speedup

Successful completion earns the **NVIDIA DLI Certificate of Competency**.

---

## 📁 Files in This Section

```
section3/
├── assessment/                      ← Final course assessment (graded)
│   └── assessment.ipynb
├── solutions/                       ← Reference solutions
│   └── *.ipynb
├── images/                          ← Memory diagrams and architecture visuals
├── coalescing-v3.pptx               ← Deep dive: memory coalescing patterns
├── bank_conflicts.pptx              ← Deep dive: shared memory bank conflicts
└── shared_coalescing.pptx           ← Combined: shared memory + coalescing
```

| File / Folder | Description |
|---|---|
| `assessment/` | Final graded notebook — end-to-end GPU optimization challenge |
| `solutions/` | Reference solutions for all Day 3 exercises |
| `coalescing-v3.pptx` | Detailed lecture on memory coalescing theory and patterns |
| `bank_conflicts.pptx` | Shared memory bank layout, conflict visualization, and fixes |
| `shared_coalescing.pptx` | Combined tiling strategy for coalesced + shared memory access |
| `images/` | GPU memory hierarchy diagrams, bandwidth charts |

---

## ✅ Exercises Completed

- [x] Identify coalesced vs uncoalesced access patterns in existing kernels
- [x] Rewrite a strided-access kernel to use coalesced access
- [x] Implement a tiled matrix transpose with coalesced read and write
- [x] Analyze and resolve shared memory bank conflicts using padding
- [x] Implement tiled matrix multiplication with shared memory
- [x] Complete final assessment — optimize a full CPU algorithm for GPU with measurable speedup

---

## 💡 Key Takeaways

- **Memory access pattern**, not just computation, is the dominant factor in GPU kernel performance.
- **Coalesced access** means threads in a warp read/write consecutive addresses — this maximizes memory bus utilization.
- **Shared memory tiling** reduces global memory traffic by loading reused data once and accessing it many times on-chip.
- **Bank conflicts** serialize shared memory accesses — add a single padding column (`+1`) to eliminate most conflicts.
- The **load → sync → compute → sync → write** pattern is the universal template for tiled, high-performance kernels.
- NVIDIA's GPU hardware is designed to reward these patterns — following them can yield **10–20× speedup** over naive implementations.

---

## 🏆 Course Completion

Upon passing the Day 3 assessment, participants receive the official **NVIDIA DLI Certificate of Competency** in Fundamentals of Accelerated Computing with CUDA Python (Course ID: `DLI C-AC-02`).

This certificate recognizes demonstrated ability to:
- Write GPU-accelerated Python applications with Numba
- Design and launch custom CUDA kernels
- Apply memory optimization techniques (coalescing, shared memory, bank conflict avoidance)

---

[← Day 2 — Custom CUDA Kernels](../section2/README.md) &nbsp;&nbsp;|&nbsp;&nbsp; [↑ Back to Root](../README.md)