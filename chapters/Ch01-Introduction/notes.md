# 1. Introduction

[TOC]

---

## 1.1 Heterogeneous Parallel Computing
CPUs performance blocked by energy consumption and heat dissipation, ledding vendors to switch to models with multiple processor units. Two main trajectories:

- **multi-core** seeking to maintain the execution **latency of sequential programs**.
  - Sophisticated control logic to allow [SIMD](https://en.wikipedia.org/wiki/Instruction-level_parallelism) or [Out-of-order](https://en.wikipedia.org/wiki/Out-of-order_execution).
  - Large cache memories to reduce the instruction and data access latency.
  - Low-latency arithmetic units

- **many-thread** focusing more on the execution **throughput of parallel programs**.
  - Large memory bandwidths to move extremely large amounts of data in and out of its [DRAM](https://en.wikipedia.org/wiki/Dynamic_random-access_memory).
  - Small cache memories to reduce the number of accesses to DRAM.
  - Maximized chip area and power budget dedicated to floating-point calculations.

The **computationally intensive** parts of a program are the main focus of parallel programming

---

## 1.2 Factor of Speeding Up
- How much is the parallelizable portion of the application.
[**Amdahl's Law**](https://en.wikipedia.org/wiki/Amdahl%27s_law): the overall performance improvement gained by optimizing a single part of a system is limited by the fraction of time that the improved part is actually used.
end2end theoretical $speedup=\frac{1}{(1-p)}$, where p is the portion of the program can be speedup

- How fast data can be accessed from and written to the memory.

---

## 1.3 Challenges in parallel programming
- Design parallel algorithms with the same level of algorithmic (computational) complexity as that of sequential algorithms.
- The execution speed of many applications is limited by memory access latency and/or throughput.
- The execution speed of parallel programs is often more sensitive to the input data characteristics than is the case for their sequential counterparts.
- Some applications can be parallelized while requiring little collaboration across different threads.

## 1.4 Thought Processes of Parallized Programming
1. Identifying the part of application programs to be parallelized
2. Isolating the data to be used by the parallelized code, using an API function to allocate memory on the parallel computing device
3. Using an API function to transfer data to the parallel computing device
4. Developing the parallel part into a kernel function that will be executed by parallel threads
5. Launching a kernel function for execution by parallel threads
6. Eventually transferring the data back to the host processor with an API function call.

---

## 1.5 Related Parallel Programming Interfaces
- OpenMP (Open, 2005) for shared memory multiprocessor systems -> CUDA gives programmers explicit control of these parallel programming details
- Message Passing Interface (MPI) (MPI, 2009) for scalable cluster computing -> NVIDIA Collective Communications Library (NCCL)

---
