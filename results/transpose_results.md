# Transpose Benchmark — H200 GPUs, 100 iterations

## 4 GPUs — With Accumulation (B += A^T)

### IPC / MPI — All Modes
Matrix Size | Mode                    | GB/s
------------|-------------------------|-------
1024²       | IPC direct (per-phase)  |  284.3
1024²       | IPC direct (single-K)   |  595.6
1024²       | IPC buffered            |  175.4
1024²       | GPU-aware MPI           |  200.1
1024²       | Staged MPI              |   81.6
2048²       | IPC direct (per-phase)  |  559.9
2048²       | IPC direct (single-K)   | 1171.7
2048²       | IPC buffered            |  499.8
2048²       | GPU-aware MPI           |  542.2
2048²       | Staged MPI              |  117.1
4096²       | IPC direct (per-phase)  |  696.3
4096²       | IPC direct (single-K)   |  882.3
4096²       | IPC buffered            |  839.3
4096²       | GPU-aware MPI           |  851.1
4096²       | Staged MPI              |  131.7
8192²       | IPC direct (per-phase)  |  812.1
8192²       | IPC direct (single-K)   |  889.6
8192²       | IPC buffered            | 1011.7
8192²       | GPU-aware MPI           | 1023.7
8192²       | Staged MPI              |  121.2
16384²      | IPC direct (per-phase)  |  865.7
16384²      | IPC direct (single-K)   |  884.8
16384²      | IPC buffered            | 1060.3
16384²      | GPU-aware MPI           | 1077.1
16384²      | Staged MPI              |  114.8

### NVSHMEM
Matrix Size | Mode              | GB/s
------------|-------------------|-------
1024²       | NVSHMEM direct    |  147.3
1024²       | NVSHMEM single-K  |  358.3
1024²       | NVSHMEM buffered  |  139.2
2048²       | NVSHMEM direct    |  384.2
2048²       | NVSHMEM single-K  |  847.4
2048²       | NVSHMEM buffered  | (rerun pending)
4096²       | NVSHMEM direct    |  601.6
4096²       | NVSHMEM single-K  |  826.8
4096²       | NVSHMEM buffered  |  788.8
8192²       | NVSHMEM direct    |  776.6
8192²       | NVSHMEM single-K  |  872.0
8192²       | NVSHMEM buffered  | 1009.3
16384²      | NVSHMEM direct    |  855.0
16384²      | NVSHMEM single-K  |  880.0
16384²      | NVSHMEM buffered  | 1083.1

## 4 GPUs — Without Accumulation (B = A^T)

### IPC / MPI — All Modes
Matrix Size | Mode                    | GB/s
------------|-------------------------|-------
1024²       | IPC direct (per-phase)  |  342.1
1024²       | IPC direct (single-K)   |  722.3
1024²       | IPC buffered            |  177.6
1024²       | GPU-aware MPI           |  207.0
1024²       | Staged MPI              |   74.9
2048²       | IPC direct (per-phase)  |  759.3
2048²       | IPC direct (single-K)   | 1240.9
2048²       | IPC buffered            |  501.9
2048²       | GPU-aware MPI           |  551.8
2048²       | Staged MPI              |  119.8
4096²       | IPC direct (per-phase)  | 1085.5
4096²       | IPC direct (single-K)   | 1349.4
4096²       | IPC buffered            |  875.5
4096²       | GPU-aware MPI           |  908.4
4096²       | Staged MPI              |  112.6
8192²       | IPC direct (per-phase)  | 1233.5
8192²       | IPC direct (single-K)   | 1310.6
8192²       | IPC buffered            | 1095.3
8192²       | GPU-aware MPI           | 1099.2
8192²       | Staged MPI              |  140.0
16384²      | IPC direct (per-phase)  | 1282.4
16384²      | IPC direct (single-K)   | 1302.8
16384²      | IPC buffered            | 1151.0
16384²      | GPU-aware MPI           | 1171.0
16384²      | Staged MPI              |  127.3

### NVSHMEM
Matrix Size | Mode              | GB/s
------------|-------------------|-------
1024²       | NVSHMEM direct    |  162.4
1024²       | NVSHMEM single-K  |  400.5
1024²       | NVSHMEM buffered  |  152.3
2048²       | NVSHMEM direct    |  472.9
2048²       | NVSHMEM single-K  | 1066.7
2048²       | NVSHMEM buffered  |  428.6
4096²       | NVSHMEM direct    |  875.9
4096²       | NVSHMEM single-K  | 1217.5
4096²       | NVSHMEM buffered  |  808.5
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
