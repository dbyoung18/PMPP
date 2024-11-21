*1. Assume that each atomic operation in a DRAM system has a total latency of 100 ns. What is the maximum throughput that we can get for atomic operations on the same global memory variable?*

**Correct answer:** 0.01G atomic operations per second
```
No other atomic operation can touch the same variable for the entire duration of 100ns.
The maximal thoughput = 1s / 100ns = 0.01G atomic operations per second
```

---

*2. For a processor that supports atomic operations in L2 cache, assume that each atomic operation takes 4 ns to complete in L2 cache and 100 ns to complete in DRAM. Assume that 90% of the atomic operations hit in L2 cache. What is the approximate throughput for atomic operations on the same global memory variable?*

**Correct answer:** 0.0735G atomic operations per second
```
The average latency = 4ns * 90% + 100ns * 10% = 13.6ns.
The average throughput = 1s / 13.6ns = 0.0735G atomic operations per second
```

---

*3. In Exercise 1, assume that a kernel performs five floating-point operations per atomic operation. What is the maximum floating-point throughput of the kernel execution as limited by the throughput of the atomic operations?*

**Correct answer:** 0.05GFLOPS
```
The maximal floating-point thoughput = 1s / 100ns * 5 = 0.05GFLOPS
```

---

*4. In Exercise 1, assume that we privatize the global memory variable into shared memory variables in the kernel and that the shared memory access latency is 1 ns. All original global memory atomic operations are converted into shared memory atomic operation. For simplicity, assume that the additional global memory atomic operations for accumulating privatized variable into the global variable adds 10% to the total execution time. Assume that a kernel performs five floating-point operations per atomic operation. What is the maximum floating-point throughput of the kernel execution as limited by the throughput of the atomic operations?*

**Correct answer:** 4.545 GFLOPS
```
throughput = 5 * 1 / (1 * 110%) = 4.545 GFLOPS
```

---

*5. To perform an atomic add operation to add the value of an integer variable Partial to a global memory integer variable Total, which one of the following statements should be used?
a. atomicAdd(Total, 1);
b. atomicAdd(&Total, &Partial);
c. atomicAdd(Total, &Partial);
d. atomicAdd(&Total, Partial);*

**Correct answer:** d

---

*6. Consider a histogram kernel that processes an input with 524,288 elements to produce a histogram with 128 bins. The kernel is configured with 1024 threads per block.
a. What is the total number of atomic operations that are performed on global memory by the kernel in Fig. 9.6 where no privatization, shared memory, and thread coarsening are used?
b. What is the maximum number of atomic operations that may be performed on global memory by the kernel in Fig. 9.10 where privatization and shared memory are used but not thread coarsening?
c. What is the maximum number of atomic operations that may be performed on global memory by the kernel in Fig. 9.14 where privatization, shared memory, and thread coarsening are used with a coarsening factor of 4?*

**Correct answer:**
a. 524288
```
• n = 524288
• blockDim = 1024
• NUM_BINS = 128
• Every input element is directly updated into the global memory histogram using an atomic operation.
• Global Atomic Operations are equal to n.
```
b. 65536
```
• gridDim = n / blockDim = 524288 / 1024 = 512
• At the end of the kernel, each block performs NUM_BINS atomic operations (one per bin in its shared histogram) to update the global histogram.
• Global Atomic Operations = gridDim * NUM_BINS = 512 * 128 = 65536
```
c. 16384
```
• gridDim = n / coarsen_factor / blockDim = 524,288 / 4 / 1024 = 128
• Global Atomic Operations = gridDim * NUM_BINS = 128 * 128 = 16384
```