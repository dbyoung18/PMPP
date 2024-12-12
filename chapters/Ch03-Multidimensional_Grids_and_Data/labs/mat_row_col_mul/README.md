## Matrix-Vector Multiplication

*Code for [Exercise 1](../exercises.md)*

---

a. Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design.  
**Correct answer:**  See [mat_row_mul.cu](./mat_row_mul.cu)

---

b. Write a kernel that has each thread produce one output matrix column. Fill in the execution configuration parameters for the design.  
**Correct answer:** See [mat_col_mul.cu](./mat_col_mul.cu)

---

c. Analyze the pros and cons of each of the two kernel designs.
**Correct answer:**
Kernel where each thread produces one output matrix row:
Pros:
- Memory access pattern: This approach typically results in better memory coalescing when accessing matrix A. Since each thread processes one entire row of the result matrix, it accesses A in a sequential manner, which can improve memory access efficiency.
- Simplicity: This design is straightforward to implement and understand. It is easy to match the number of blocks to the number of rows in the result matrix.
Cons:
- Column access in matrix B: Each thread must access elements of matrix B across different rows (non-sequential), which can cause non-coalesced memory accesses and reduce performance when reading matrix B.
- Thread utilization: If the number of rows (M) is not a multiple of the block size, some threads might be idle in the last block, resulting in lower efficiency.

Kernel where each thread produces one output matrix column:
Pros:
- Memory access pattern for matrix B: This design can provide better memory access patterns when reading from matrix B. Each thread accesses one entire column, which can be more sequential and more coalesced, especially if the column-major storage order is used.
- Parallelism: This design can achieve good parallelism if N (the number of columns) is large, as each thread computes a full column.
Cons:
- Row access in matrix A: Each thread needs to access A across different columns (non-sequential access), which can result in non-coalesced memory accesses and reduced performance when reading from A.
- Thread utilization: Similar to the row-based approach, if N (number of columns) is not a multiple of the block size, there might be some idle threads in the last block.

Overall Comparison:
- The row-based kernel is generally better when the number of rows M is large, as it provides more sequential access to matrix A and simplifies the memory access pattern for matrix A.
- The column-based kernel can be more efficient if the number of columns N is large, as it offers better memory access patterns for matrix B.
- Both kernels have their strengths and weaknesses, and their performance will depend on the specific matrix dimensions and memory access patterns in your application.

Final Thoughts:
- For typical matrix-multiplication problems, the row-based kernel (a) is often a better choice in terms of memory coalescing and thread utilization, especially when the number of rows M is large.
- The column-based kernel (b) might perform better when working with certain data layouts or matrix sizes but might struggle with inefficient memory access to matrix A.