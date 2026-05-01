# 📘 Day 1 — Introduction to CUDA Python with Numba

> **Course:** Fundamentals of Accelerated Computing with CUDA Python
> **Provider:** NVIDIA Deep Learning Institute (DLI)
> **Section:** Section 1 of 3
> **Focus:** GPU programming fundamentals, Numba JIT compilation, CUDA thread hierarchy, and memory management

---

## 🎯 Day 1 Objective

Day 1 builds the complete foundation required for GPU programming in Python. By the end of this session, you will understand how a GPU executes code in parallel, how to write and launch CUDA kernels using the **Numba** JIT compiler, and how to efficiently move data between the CPU (host) and GPU (device).

No prior CUDA experience is needed — just Python and NumPy.

---

## 🧠 Concepts Covered

### 1. Why GPUs? — The Parallel Computing Advantage

Modern CPUs have a small number of powerful cores (4–64) optimized for sequential, low-latency tasks. GPUs, by contrast, contain **thousands of smaller cores** designed to execute many operations simultaneously.

```
CPU:  ████ ████ ████ ████         (few powerful cores)
GPU:  ■■■■■■■■■■■■■■■■■■■■■■■■   (thousands of smaller cores)
      ■■■■■■■■■■■■■■■■■■■■■■■■
      ■■■■■■■■■■■■■■■■■■■■■■■■
```

This makes GPUs ideal for data-parallel problems like:
- Array and matrix operations
- Image and signal processing
- Physics simulations
- Deep learning forward/backward passes

---

### 2. What is Numba?

**Numba** is an open-source JIT (Just-In-Time) compiler that translates Python functions into optimized machine code at runtime using LLVM. With its CUDA backend, Numba lets you write GPU kernels in pure Python — no C, no C++.

Key Numba decorators used in this section:

| Decorator | Purpose |
|---|---|
| `@vectorize(target='cuda')` | Creates a GPU-accelerated NumPy ufunc |
| `@cuda.jit` | Compiles a low-level CUDA kernel in Python |

---

### 3. GPU-Accelerated ufuncs with `@vectorize`

The simplest way to move computation to the GPU is with Numba's `@vectorize` decorator. It works identically to NumPy ufuncs but runs element-wise across GPU threads.

```python
from numba import vectorize
import numpy as np

@vectorize(['float32(float32, float32)'], target='cuda')
def gpu_add(a, b):
    return a + b

a = np.ones(10_000_000, dtype=np.float32)
b = np.ones(10_000_000, dtype=np.float32)

result = gpu_add(a, b)  # Runs on GPU automatically
```

**What happens under the hood:**
- Numba compiles the function for the GPU
- Input arrays are automatically transferred to device memory
- Each GPU thread handles one element of the array
- Results are returned to host memory

---

### 4. The CUDA Thread Hierarchy

CUDA organizes parallel execution into a three-level hierarchy:

```
┌─────────────────── GRID ───────────────────┐
│                                             │
│   ┌─── BLOCK 0 ───┐   ┌─── BLOCK 1 ───┐   │
│   │ T0 T1 T2 T3   │   │ T0 T1 T2 T3   │   │
│   │ T4 T5 T6 T7   │   │ T4 T5 T6 T7   │   │
│   └───────────────┘   └───────────────┘   │
│                                             │
└─────────────────────────────────────────────┘
```

| Level | Variable | Description |
|---|---|---|
| **Thread** | `cuda.threadIdx.x` | Smallest unit; one thread per data element |
| **Block** | `cuda.blockIdx.x` | Group of threads; can share memory |
| **Grid** | All blocks together | The entire kernel launch |

**Calculating a unique global thread index:**
```python
@cuda.jit
def my_kernel(data):
    # Unique index for each thread across all blocks
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i < data.size:
        data[i] *= 2.0
```

Numba provides a shorthand: `cuda.grid(1)` for 1D, `cuda.grid(2)` for 2D.

```python
@cuda.jit
def my_kernel(data):
    i = cuda.grid(1)   # equivalent to threadIdx.x + blockIdx.x * blockDim.x
    if i < data.size:
        data[i] *= 2.0
```

---

### 5. Launching a CUDA Kernel

Kernels are launched with a special `[blocks, threads]` syntax:

```python
import numpy as np
from numba import cuda

@cuda.jit
def double_elements(data):
    i = cuda.grid(1)
    if i < data.size:
        data[i] *= 2.0

# Setup
N = 1_000_000
data = np.ones(N, dtype=np.float32)

# Launch configuration
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

# Launch kernel
double_elements[blocks_per_grid, threads_per_block](data)
cuda.synchronize()  # Wait for GPU to finish
```

**Choosing thread block size:**
- Always use multiples of **32** (one GPU warp = 32 threads)
- Common choices: 128, 256, 512
- `threads_per_block = 256` is a safe default

---

### 6. Host ↔ Device Memory Management

By default, Numba handles memory transfers automatically. For performance-critical applications, **explicit memory management** avoids redundant transfers:

```python
import numpy as np
from numba import cuda

a = np.random.rand(1_000_000).astype(np.float32)
b = np.random.rand(1_000_000).astype(np.float32)

# Transfer arrays to GPU once
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.device_array(a.shape, dtype=np.float32)  # allocate output on GPU

@cuda.jit
def add_kernel(a, b, c):
    i = cuda.grid(1)
    if i < a.size:
        c[i] = a[i] + b[i]

threads = 256
blocks = (a.size + threads - 1) // threads
add_kernel[blocks, threads](d_a, d_b, d_c)

# Transfer result back to CPU only when needed
result = d_c.copy_to_host()
```

**Memory transfer cost is real** — minimizing host↔device transfers is one of the most impactful GPU optimizations.

---

### 7. Benchmarking CPU vs GPU

```python
import time
import numpy as np
from numba import cuda, vectorize

@vectorize(['float32(float32)'], target='cpu')
def cpu_sqrt(x):
    return x ** 0.5

@vectorize(['float32(float32)'], target='cuda')
def gpu_sqrt(x):
    return x ** 0.5

data = np.random.rand(10_000_000).astype(np.float32)

# CPU timing
start = time.time()
cpu_result = cpu_sqrt(data)
print(f"CPU: {(time.time() - start)*1000:.2f} ms")

# GPU timing (warm-up run first)
gpu_sqrt(data)
cuda.synchronize()
start = time.time()
gpu_result = gpu_sqrt(data)
cuda.synchronize()
print(f"GPU: {(time.time() - start)*1000:.2f} ms")
```

> ⚠️ Always run a **warm-up** GPU call before benchmarking — the first call includes JIT compilation time.

---

## 📁 Files in This Section

```
Section1/
├── Introduction to CUDA Python with Numba.ipynb   ← Main lab notebook
├── AC_CUDA_Python_1.pptx                          ← Day 1 lecture slides
└── images/                                         ← Diagrams used in the notebook
```

| File | Description |
|---|---|
| `Introduction to CUDA Python with Numba.ipynb` | Complete hands-on lab — GPU ufuncs, thread indexing, memory management, benchmarking |
| `AC_CUDA_Python_1.pptx` | Lecture slides covering GPU architecture, CUDA model, and Numba overview |
| `images/` | Diagrams of thread hierarchy, memory model, and GPU architecture |

---

## ✅ Exercises Completed

- [x] Accelerate a NumPy ufunc to run on the GPU with `@vectorize`
- [x] Write and launch a custom CUDA kernel using `@cuda.jit`
- [x] Calculate correct 1D thread indices for arbitrary array sizes
- [x] Handle arrays whose length is not a perfect multiple of block size (boundary check)
- [x] Explicitly manage device memory with `cuda.to_device()` and `copy_to_host()`
- [x] Benchmark and compare CPU vs GPU execution times

---

## 💡 Key Takeaways

- GPUs excel at **data-parallel** problems where the same operation applies to millions of independent elements.
- Numba's `@cuda.jit` brings CUDA kernel programming to Python with near-zero overhead.
- **Thread index calculation** is the core skill of CUDA programming — every kernel depends on it.
- **Always include a boundary check** (`if i < array.size`) to prevent out-of-bounds memory access.
- Explicit memory management (`cuda.to_device`) is essential for real-world performance.
- GPU benchmarking requires `cuda.synchronize()` — GPU execution is asynchronous by default.

---

## ➡️ Up Next: Day 2

Day 2 goes deeper with **custom CUDA kernels** — writing multi-dimensional kernels, using **shared memory** for block-level communication, and synchronizing threads.

[→ Go to Day 2 — Custom CUDA Kernels](../section2/README.md)