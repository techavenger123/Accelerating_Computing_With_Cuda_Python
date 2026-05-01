# ⚡ Fundamentals of Accelerated Computing with CUDA Python

![NVIDIA DLI](https://img.shields.io/badge/NVIDIA-Deep%20Learning%20Institute-76b900?style=for-the-badge&logo=nvidia&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Numba](https://img.shields.io/badge/Numba-CUDA-00A3E0?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

> A 3-day hands-on workshop by **NVIDIA Deep Learning Institute (DLI)** covering GPU-accelerated computing using CUDA Python with Numba — from writing your first kernel to optimizing shared memory access patterns.

---

## 📖 About This Course

This repository contains all lab notebooks, slide decks, solutions, and reference material from the **NVIDIA DLI "Fundamentals of Accelerated Computing with CUDA Python"** course.

The course teaches how to harness the massive parallel computing power of NVIDIA GPUs using Python — without writing a single line of C/C++. Using **Numba**, a Python JIT compiler, participants write real CUDA kernels, manage GPU memory, and optimize performance through advanced memory access techniques.

---

## 🗂️ Repository Structure

```
📦 Accelerating_Computing_With_Cuda_Python
│
├── 📁 Section1/                        # Day 1 — Introduction to CUDA Python with Numba
│   ├── Introduction to CUDA Python with Numba.ipynb
│   ├── AC_CUDA_Python_1.pptx
│   └── images/
│
├── 📁 section2/                        # Day 2 — Custom CUDA Kernels
│   ├── Custom CUDA Kernels in Python with Numba.ipynb
│   ├── AC_CUDA_Python_2.pptx
│   ├── assessment/
│   ├── debug/
│   ├── solutions/
│   └── images/
│
├── 📁 section3/                        # Day 3 — Memory Optimization & Advanced Patterns
│   ├── coalescing-v3.pptx
│   ├── bank_conflicts.pptx
│   ├── shared_coalescing.pptx
│   ├── assessment/
│   ├── solutions/
│   └── images/
│
├── 📁 solutions/                       # Consolidated solutions across all sections
├── 📁 images/                          # Global assets and diagrams
└── 📄 README.md
```

---

## 📅 3-Day Learning Journey

### 🟢 Day 1 — Introduction to CUDA Python with Numba
> **Folder:** `Section1/`

The first day establishes the theoretical and practical foundation of GPU computing. You learn how the CUDA execution model maps work across thousands of threads and write your first GPU-accelerated functions in Python.

**Topics Covered:**
- Why GPU acceleration? CPU vs GPU architecture
- The CUDA thread hierarchy: Grid → Block → Thread
- Writing GPU kernels with Numba's `@cuda.jit` decorator
- Calculating unique thread indices with `cuda.grid()`
- Host ↔ Device memory management (`cuda.to_device`, `copy_to_host`)
- High-level GPU ufuncs with `@vectorize(target='cuda')`
- Benchmarking CPU vs GPU execution times

**Key Files:**
| File | Description |
|---|---|
| `Introduction to CUDA Python with Numba.ipynb` | Main lab notebook |
| `AC_CUDA_Python_1.pptx` | Lecture slides — GPU architecture & CUDA model |

---

### 🟡 Day 2 — Custom CUDA Kernels in Python
> **Folder:** `section2/`

Day 2 goes deeper into writing custom CUDA kernels, thread synchronization, and leveraging **shared memory** for intra-block communication. You also tackle hands-on debugging of common GPU programming mistakes.

**Topics Covered:**
- Writing multi-dimensional kernels (1D, 2D grids)
- Thread synchronization with `cuda.syncthreads()`
- Shared memory allocation using `cuda.shared.array()`
- Common kernel bugs and how to debug them
- Hands-on assessment: accelerating a real-world problem

**Key Files:**
| File | Description |
|---|---|
| `Custom CUDA Kernels in Python with Numba.ipynb` | Main lab notebook |
| `AC_CUDA_Python_2.pptx` | Lecture slides — shared memory & synchronization |
| `debug/` | Debugging exercises with broken kernels to fix |
| `assessment/` | Day 2 graded assessment notebook |
| `solutions/` | Reference solutions for Day 2 exercises |

---

### 🔴 Day 3 — Memory Optimization & Advanced Patterns
> **Folder:** `section3/`

The final day focuses on squeezing maximum performance out of GPU code through advanced memory optimization techniques — the difference between a slow kernel and a production-ready one.

**Topics Covered:**
- **Memory coalescing** — aligning memory access patterns to maximize bandwidth
- **Shared memory tiling** — reducing global memory round-trips
- **Bank conflicts** — understanding and eliminating shared memory bottlenecks
- Performance profiling strategies
- Final assessment: optimizing an end-to-end GPU pipeline

**Key Files:**
| File | Description |
|---|---|
| `coalescing-v3.pptx` | Deep dive into memory coalescing patterns |
| `bank_conflicts.pptx` | Shared memory bank conflict analysis |
| `shared_coalescing.pptx` | Combined shared memory + coalescing optimization |
| `assessment/` | Day 3 final assessment notebook |
| `solutions/` | Reference solutions for Day 3 exercises |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.8+** | Primary programming language |
| **Numba** | JIT compiler — Python → CUDA kernels |
| **NumPy** | Array operations and CPU baseline |
| **CUDA Toolkit** | NVIDIA's parallel computing platform |
| **Jupyter Notebook** | Interactive lab environment |

### Installation

```bash
# Using conda (recommended)
conda install numba cudatoolkit numpy jupyter -c conda-forge

# Or using pip
pip install numba numpy jupyter
# Note: CUDA Toolkit must be installed separately from https://developer.nvidia.com/cuda-toolkit
```

> **Requirements:** An NVIDIA CUDA-capable GPU with compatible drivers. The DLI course labs were originally run on NVIDIA's cloud GPU infrastructure.

---

## 🎯 Learning Outcomes

By completing this course, I was able to:

- ✅ Explain the GPU thread hierarchy and map algorithms to it correctly
- ✅ Write, launch, and debug custom CUDA kernels in pure Python using Numba
- ✅ Manage GPU memory explicitly for performance-sensitive applications
- ✅ Apply shared memory to reduce global memory access overhead
- ✅ Identify and fix memory coalescing issues and bank conflicts
- ✅ Profile and benchmark GPU code to quantify speedup over CPU baselines

---

## 📊 CPU vs GPU — What We're Solving

```
A simple element-wise operation on 10 million floats:

  CPU (single-core):   ~120 ms
  GPU (CUDA kernel):   ~1.8 ms
  ─────────────────────────────
  Speedup:             ~66x faster
```

The magnitude of speedup grows significantly for more complex workloads like matrix operations, convolutions, and simulations.

---

## 🔗 References & Further Reading

- [NVIDIA DLI Official Course](https://www.nvidia.com/en-us/training/instructor-led-workshops/fundamentals-of-accelerated-computing-with-cuda-python/)
- [Numba CUDA Documentation](https://numba.readthedocs.io/en/stable/cuda/index.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [An Even Easier Introduction to CUDA (NVIDIA Blog)](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

---

## 📜 License & Acknowledgements

The course content, lab notebooks, and slide decks are the intellectual property of **NVIDIA Corporation** and were accessed through the NVIDIA Deep Learning Institute. This repository is maintained purely for **personal learning and documentation purposes**.

---

<div align="center">
  <sub>Built with 💚 during the NVIDIA DLI Fundamentals of Accelerated Computing with CUDA Python workshop</sub>
</div>