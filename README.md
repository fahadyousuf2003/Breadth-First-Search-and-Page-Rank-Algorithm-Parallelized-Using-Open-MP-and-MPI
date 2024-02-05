# Parallel BFS and PAGE RANK Algorithm Implementation Using OPEN MP and MPI

## Overview
This project focuses on the implementation and optimization of two fundamental graph algorithms, Breadth-First Search (BFS) and Page Rank, utilizing parallel computing techniques. The parallelization is achieved through Message Passing Interface (MPI) for distributed-memory parallelism and OpenMP for shared-memory parallelism. The algorithms play a crucial role in network analysis, social network analysis, and web page ranking.

## Methodology

### 1. Graph Representation and Data Input:
- Graphs are represented using adjacency matrices for both BFS and Page Rank algorithms.
- Graph data is read from a file for the MPI implementation, while random graphs are generated for OpenMP Page Rank.

### 2. Parallel BFS using OpenMP:
- BFS algorithm is parallelized using OpenMP for shared-memory parallelism.
- OpenMP directives are used to parallelize the loop that explores neighboring nodes.
- Time is measured for BFS execution, and relevant information is printed.

### 3. Parallel BFS using MPI:
- MPI is employed for distributed-memory parallelism in BFS.
- Process 0 reads the graph, specifies the start node, and broadcasts necessary information to all processes.
- BFS is executed in parallel across processes, and memory is freed after completion.

### 4. Page Rank using OpenMP:
- Page Rank algorithm is parallelized with OpenMP to harness shared-memory parallelism.
- Random graphs are generated, and an iterative approach is taken for Page Rank calculation.
- OpenMP directives parallelize the loop that calculates Page Rank scores.
- Convergence is checked, and results, including iteration-wise runtime, are printed.

### 5. Page Rank using MPI:
- MPI is utilized for distributed-memory parallelism in the Page Rank algorithm.
- Memory is allocated for a global vector, and matrix-vector multiplication is performed.
- Information exchange is carried out between processes, and convergence is checked globally.
- Iteration-wise information is printed, and Page Rank scores are written to an output file.

## Performance Comparison: OpenMP vs. MPI
Results of the performance comparison for different datasets are presented in an Excel sheet, showcasing computation time, communication time, and total time for each algorithm and parallelization strategy.

## Conclusion
- Page Rank OpenMP vs. Page Rank MPI: OpenMP generally outperforms MPI despite varying performance.
- BFS MPI vs. BFS OpenMP: OpenMP consistently outperforms MPI in terms of total time across different datasets.
- Recommendations: OpenMP is recommended for better BFS performance in the given workload.

## Future Work
Further analysis and testing on a broader range of graphs are suggested for a comprehensive understanding of parallelization efficiency.

## How to Run
1. Clone the repository.
2. Compile the OpenMP and MPI versions of the code.
3. Follow instructions in each algorithm's directory for specific execution steps.
