# Transpose Benchmark — H200 GPUs, 100 iterations

## 4 GPUs — With Accumulation (B += A^T)

### IPC Direct (per-phase) vs GPU-aware MPI vs Staged MPI
Matrix Size | Mode           | GB/s
------------|----------------|------
1024²       | IPC (direct)   | 280.0
1024²       | GPU-aware MPI  | 200.0
1024²       | Staged MPI     |  71.7
2048²       | IPC (direct)   | 554.0
2048²       | GPU-aware MPI  | 540.0
2048²       | Staged MPI     | 108.0
4096²       | IPC (direct)   | 693.0
4096²       | GPU-aware MPI  | 850.0
4096²       | Staged MPI     | 105.8
8192²       | IPC (direct)   | 812.0
8192²       | GPU-aware MPI  | 1024.0
8192²       | Staged MPI     | 126.6
16384²      | IPC (direct)   | 866.0
16384²      | GPU-aware MPI  | 1077.0
16384²      | Staged MPI     | 114.9

### IPC Direct (single-kernel) vs Buffered IPC
Matrix Size | Mode                    | GB/s
------------|-------------------------|------
1024²       | IPC direct (single-K)   | 570.4
1024²       | IPC (buffered)          | 178.9
2048²       | IPC direct (single-K)   | 1162.2
2048²       | IPC (buffered)          | 509.4
4096²       | IPC direct (single-K)   | 881.8
4096²       | IPC (buffered)          | 845.2
8192²       | IPC direct (single-K)   | 889.2
8192²       | IPC (buffered)          | 1017.7
16384²      | IPC direct (single-K)   | 884.6
16384²      | IPC (buffered)          | 1062.0

### NVSHMEM (with accumulation)
Matrix Size | Mode              | GB/s
------------|-------------------|------
1024²       | NVSHMEM direct    | 147.3
1024²       | NVSHMEM single-K  | 358.3
1024²       | NVSHMEM buffered  | 139.2
2048²       | NVSHMEM direct    | 384.2
2048²       | NVSHMEM single-K  | 847.4
2048²       | NVSHMEM buffered  | FAILED (error 8.25e+11)
4096²       | NVSHMEM direct    | 601.6
4096²       | NVSHMEM single-K  | 826.8
4096²       | NVSHMEM buffered  | 788.8
8192²       | NVSHMEM direct    | 776.6
8192²       | NVSHMEM single-K  | 872.0
8192²       | NVSHMEM buffered  | 1009.3
16384²      | NVSHMEM direct    | 855.0
16384²      | NVSHMEM single-K  | 880.0
16384²      | NVSHMEM buffered  | 1083.1

## 4 GPUs — Without Accumulation (B = A^T)

### IPC (no accumulation)
Matrix Size | Mode                    | GB/s
------------|-------------------------|------
1024²       | IPC direct (per-phase)  | 333.2
1024²       | IPC direct (single-K)   | 701.0
1024²       | IPC buffered            | 182.1
1024²       | GPU-aware MPI           | 206.8
1024²       | Staged MPI              |  76.4
2048²       | IPC direct (per-phase)  | 757.9
2048²       | IPC direct (single-K)   | 1218.6
2048²       | IPC buffered            | 509.5
2048²       | GPU-aware MPI           | 556.6
2048²       | Staged MPI              | 116.0
4096²       | IPC direct (per-phase)  | 1076.7
4096²       | IPC direct (single-K)   | 1344.9
4096²       | IPC buffered            | 883.3
4096²       | GPU-aware MPI           | 900.7
4096²       | Staged MPI              | 123.6
8192²       | IPC direct (per-phase)  | 1231.8
8192²       | IPC direct (single-K)   | 1310.3
8192²       | IPC buffered            | 1097.9
8192²       | GPU-aware MPI           | 1101.7
8192²       | Staged MPI              | 124.7
16384²      | IPC direct (per-phase)  | 1282.0
16384²      | IPC direct (single-K)   | 1302.2
16384²      | IPC buffered            | 1152.2
16384²      | GPU-aware MPI           | 1170.1
16384²      | Staged MPI              | 117.0

### NVSHMEM (no accumulation)
Matrix Size | Mode              | GB/s
------------|-------------------|------
1024²       | NVSHMEM direct    | 162.4
1024²       | NVSHMEM single-K  | 400.5
1024²       | NVSHMEM buffered  | 152.3
2048²       | NVSHMEM direct    | 472.9
2048²       | NVSHMEM single-K  | 1066.7
2048²       | NVSHMEM buffered  | 428.6
4096²       | NVSHMEM direct    | 875.9
4096²       | NVSHMEM single-K  | 1217.5
4096²       | NVSHMEM buffered  | 808.5
8192²       | NVSHMEM direct    | 1152.5
8192²       | NVSHMEM single-K  | 1279.3
8192²       | NVSHMEM buffered  | 1085.1
16384²      | NVSHMEM direct    | 1259.8
16384²      | NVSHMEM single-K  | 1295.1
16384²      | NVSHMEM buffered  | 1178.3

## 2 GPUs — With Accumulation (Buffered IPC)
Matrix Size | Mode           | GB/s
------------|----------------|------
1024²       | IPC (buffered) | 389.3
1024²       | GPU-aware MPI  | 361.9
1024²       | Staged MPI     |  92.4
2048²       | IPC (buffered) | 612.2
2048²       | GPU-aware MPI  | 581.3
2048²       | Staged MPI     | 104.7
4096²       | IPC (buffered) | 727.0
4096²       | GPU-aware MPI  | 693.7
4096²       | Staged MPI     | 108.4
8192²       | IPC (buffered) | 769.1
8192²       | GPU-aware MPI  | 738.6
8192²       | Staged MPI     | 105.6
16384²      | IPC (buffered) | 772.4
16384²      | GPU-aware MPI  | 742.0
16384²      | Staged MPI     | 104.3
