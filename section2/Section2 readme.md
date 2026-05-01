# 📗 Day 2 — Custom CUDA Kernels in Python with Numba

> **Course:** Fundamentals of Accelerated Computing with CUDA Python
> **Provider:** NVIDIA Deep Learning Institute (DLI)
> **Section:** Section 2 of 3
> **Focus:** Custom CUDA kernels, 2D thread grids, shared memory, thread synchronization, and debugging

---

## 🎯 Day 2 Objective

Day 2 moves beyond basic ufuncs and introduces the full power of custom CUDA kernel programming. You learn to write multi-dimensional kernels, leverage **shared memory** for fast block-level data sharing, and synchronize threads to avoid race conditions. The section also includes hands-on **debugging exercises** where you identify and fix broken CUDA kernels.

---

## 🧠 Concepts Covered

### 1. Recap — Why Custom Kernels?

`@vectorize` ufuncs are excellent for simple element-wise operations, but many algorithms require:
- Threads to **communicate** with each other
- **Reusing data** across multiple operations (not just one element at a time)
- **Multi-dimensional** indexing for 2D problems (images, matrices)
- Fine-grained control over **memory access patterns**

Custom `@cuda.jit` kernels give you all of this.

---

### 2. 2D Thread Grids

For 2D data (images, matrices), CUDA supports 2D grids of 2D blocks.

```python
from numba import cuda
import numpy as np

@cuda.jit
def matrix_add_kernel(A, B, C):
    row, col = cuda.grid(2)   # 2D grid index
    if row < A.shape[0] and col < A.shape[1]:
        C[row, col] = A[row, col] + B[row, col]

# 2D launch configuration
rows, cols = 1024, 1024
A = np.random.rand(rows, cols).astype(np.float32)
B = np.random.rand(rows, cols).astype(np.float32)
C = np.zeros_like(A)

threads_per_block = (16, 16)       # 16x16 = 256 threads per block
blocks_per_grid_x = (rows + 15) // 16
blocks_per_grid_y = (cols + 15) // 16
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

matrix_add_kernel[blocks_per_grid, threads_per_block](A, B, C)
```

**2D index breakdown:**

| Variable | Value | Meaning |
|---|---|---|
| `cuda.threadIdx.x/y` | 0–15 | Thread position within block |
| `cuda.blockIdx.x/y` | 0–63 | Block position within grid |
| `cuda.blockDim.x/y` | 16 | Block dimension |
| `cuda.grid(2)` | `(row, col)` | Global 2D index |

---

### 3. Shared Memory — The Fast Lane

Global GPU memory (DRAM) has high latency. **Shared memory** is on-chip memory local to each thread block — it is **~100× faster** than global memory but limited in size (typically 48 KB per block).

The pattern for using shared memory:
1. Allocate shared memory array in the kernel
2. Load data from **global memory → shared memory** (each thread loads its element)
3. **Synchronize threads** to ensure all loads are complete
4. Perform computation using shared memory
5. Write results back to global memory

```python
from numba import cuda, float32
import numpy as np

@cuda.jit
def shared_memory_kernel(data, result):
    # Allocate shared memory: shape and dtype must be known at compile time
    shared = cuda.shared.array(shape=256, dtype=float32)

    tx = cuda.threadIdx.x
    i  = cuda.grid(1)

    # Step 1: Load global → shared
    if i < data.size:
        shared[tx] = data[i]
    else:
        shared[tx] = 0.0

    # Step 2: Synchronize — wait for ALL threads in block to finish loading
    cuda.syncthreads()

    # Step 3: Use shared memory (example: each element + its neighbor)
    if tx > 0 and i < result.size:
        result[i] = shared[tx] + shared[tx - 1]
```

> ⚠️ **`cuda.syncthreads()` is critical.** Without it, some threads may read shared memory before other threads have finished writing — a race condition that causes wrong results.

---

### 4. Thread Synchronization Deep Dive

`cuda.syncthreads()` acts as a **barrier** — no thread in a block may pass this line until every thread in the block has reached it.

```
Thread 0:  Load → ──────────── BARRIER ──── Compute
Thread 1:  Load → ───── BARRIER ─────────── Compute
Thread 2:  Load → BARRIER ──────────────── Compute
Thread 3:  Load → ────────────────── BARRIER Compute
                              ↑
                     All threads must arrive here
                     before any can proceed
```

**Common mistake — divergent synchronization:**
```python
# ❌ WRONG — syncthreads inside a conditional
@cuda.jit
def bad_kernel(data):
    i = cuda.grid(1)
    if i % 2 == 0:
        cuda.syncthreads()   # Only even threads reach this — DEADLOCK

# ✅ CORRECT — syncthreads at consistent execution point
@cuda.jit
def good_kernel(data):
    i = cuda.grid(1)
    shared = cuda.shared.array(shape=256, dtype=float32)
    shared[cuda.threadIdx.x] = data[i] if i < data.size else 0.0
    cuda.syncthreads()          # All threads reach this
    # Now use shared safely
```

---

### 5. A Complete Shared Memory Example — Stencil Operation

A stencil (sliding window) is a classic use case for shared memory. Each output element depends on several neighboring input elements — shared memory avoids re-loading those neighbors from slow global memory.

```python
from numba import cuda, float32
import numpy as np

BLOCK_SIZE = 256
RADIUS = 3  # window of ±3 neighbors

@cuda.jit
def stencil_kernel(src, dst):
    # Shared memory with halo (padding for boundary elements)
    shared = cuda.shared.array(shape=BLOCK_SIZE + 2 * RADIUS, dtype=float32)

    tx = cuda.threadIdx.x
    i  = cuda.grid(1)

    # Load central elements
    shared[tx + RADIUS] = src[i] if i < src.size else 0.0

    # Load halo (boundary) elements
    if tx < RADIUS:
        shared[tx] = src[i - RADIUS] if i >= RADIUS else 0.0
        shared[tx + BLOCK_SIZE + RADIUS] = (
            src[i + BLOCK_SIZE] if i + BLOCK_SIZE < src.size else 0.0
        )

    cuda.syncthreads()

    # Apply stencil
    if i < dst.size:
        total = 0.0
        for offset in range(-RADIUS, RADIUS + 1):
            total += shared[tx + RADIUS + offset]
        dst[i] = total / (2 * RADIUS + 1)
```

---

### 6. Debugging CUDA Kernels

The `debug/` folder contains intentionally broken kernels for practice. Common bugs include:

**Bug 1 — Missing boundary check:**
```python
# ❌ Causes out-of-bounds memory access
@cuda.jit
def broken_kernel(data):
    i = cuda.grid(1)
    data[i] *= 2.0   # crashes when i >= data.size

# ✅ Fixed
@cuda.jit
def fixed_kernel(data):
    i = cuda.grid(1)
    if i < data.size:
        data[i] *= 2.0
```

**Bug 2 — Wrong index calculation:**
```python
# ❌ All threads in different blocks get the same index
@cuda.jit
def broken_kernel(data):
    i = cuda.threadIdx.x   # Missing blockIdx contribution!
    data[i] *= 2.0

# ✅ Fixed
@cuda.jit
def fixed_kernel(data):
    i = cuda.grid(1)   # = threadIdx.x + blockIdx.x * blockDim.x
    if i < data.size:
        data[i] *= 2.0
```

**Bug 3 — Race condition without syncthreads:**
```python
# ❌ Thread reads shared memory before another thread has written to it
@cuda.jit
def broken_kernel(data, result):
    shared = cuda.shared.array(shape=256, dtype=float32)
    tx = cuda.threadIdx.x
    i  = cuda.grid(1)
    shared[tx] = data[i]
    # Missing cuda.syncthreads() here!
    result[i] = shared[255 - tx]   # reads potentially uninitialized values

# ✅ Fixed
@cuda.jit
def fixed_kernel(data, result):
    shared = cuda.shared.array(shape=256, dtype=float32)
    tx = cuda.threadIdx.x
    i  = cuda.grid(1)
    shared[tx] = data[i]
    cuda.syncthreads()             # Wait for all loads
    result[i] = shared[255 - tx]
```

---

### 7. Day 2 Assessment

The `assessment/` folder contains a graded notebook where you apply Day 2 skills to a real problem — typically accelerating a linear algebra or signal processing operation using shared memory and custom kernels.

---

## 📁 Files in This Section

```
section2/
├── Custom CUDA Kernels in Python with Numba.ipynb  ← Main lab notebook
├── AC_CUDA_Python_2.pptx                           ← Day 2 lecture slides
├── assessment/                                      ← Graded assessment notebook
│   └── assessment.ipynb
├── debug/                                           ← Debugging exercises
│   └── debug_kernels.ipynb
├── solutions/                                       ← Reference solutions
│   └── *.ipynb
├── images/                                          ← Diagrams used in lab
└── img/                                             ← Additional visual assets
```

| File / Folder | Description |
|---|---|
| `Custom CUDA Kernels in Python with Numba.ipynb` | Complete hands-on lab notebook |
| `AC_CUDA_Python_2.pptx` | Lecture slides — shared memory, thread sync, 2D kernels |
| `assessment/` | Graded notebook — applies all Day 2 techniques |
| `debug/` | Practice debugging broken CUDA kernels |
| `solutions/` | Reference solutions for self-assessment |
| `images/` & `img/` | Architecture diagrams, memory diagrams |

---

## ✅ Exercises Completed

- [x] Write and launch a 2D CUDA kernel for matrix operations
- [x] Allocate and use on-chip shared memory with `cuda.shared.array()`
- [x] Synchronize threads correctly with `cuda.syncthreads()`
- [x] Implement a stencil kernel using shared memory with halo elements
- [x] Identify and fix 3 types of common CUDA kernel bugs
- [x] Complete graded assessment — GPU-accelerated algorithm using shared memory

---

## 💡 Key Takeaways

- **2D thread grids** map naturally to matrix and image problems — use `cuda.grid(2)`.
- **Shared memory** is ~100× faster than global memory and is the primary optimization tool for compute-bound kernels.
- **`cuda.syncthreads()`** must be called at a point all threads reach unconditionally — never inside a conditional branch.
- Shared memory allocation uses a fixed-size compile-time shape — dynamic sizes require `numba.cuda.shared.array` with care.
- **Debugging GPU code** requires systematic reasoning — race conditions and out-of-bounds errors don't always crash immediately.
- The **load → sync → compute → write** pattern is the standard template for shared memory kernels.

---

## ➡️ Up Next: Day 3

Day 3 covers **effective memory use** — memory coalescing, bank conflicts, and tiling strategies to maximize memory bandwidth on real GPU hardware.

[← Day 1 — Intro to CUDA Python](../Section1/README.md) &nbsp;&nbsp;|&nbsp;&nbsp; [→ Day 3 — Effective Memory Use](../section3/README.md)