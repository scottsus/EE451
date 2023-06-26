# Achieving Parallelization using MPI

## Usage

Execute makefile
```
make all
```

This compiles the .c files using `mpicc` and send a batch job using `sbatch` to a multicore platform, for instance USC CARC.


### Problem 1: Message Passing in a Ring
Use 4 processes to cycle an integer, incrementing it each time. `p1.c`

### Problem 2: Adding 64 Numbers
This is done using 3 methods

- Equal Split: Every process reads the whole array. P_0 computes A[i] to A[15], P_1 computes A[16] to A[31], P_2 computes A[32] to A[47], P_3 computes A[48] to A[63]. `p2_1.c`
- Broadcast: P_0 reads the array and broadcasts the entire array to P_1, P_2, P_3, each of which computes their portions as per the above method. `p2_2.c`
- Scatter and Gather: P_0 reads the array and scatters it. Each process receives random portions of the original array. Gather is used at the end. `p2_3.c`
