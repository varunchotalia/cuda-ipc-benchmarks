# Device-initiated RMA: first tests (October 2026)

First cut of the device API from the MPI RMA working group proposal
(mpiwg-rma/mpi-standard PR #13), built on the WinIPC interposer:

- `MPIX_Win_get_device_handle` (`mpi-intercept/mpiwrap_ipc.cc`) copies the
  window's per-rank peer table into device memory (`mpi-intercept/mpiw_dev_win.h`).
- `MPIX_Put`, `MPIX_Get` and `MPIX_Win_flush_all` with a thread/warp/block
  scope (`mpi-intercept/mpiw_device.cuh`).
- Test: `mpi-intercept/test_device_rma.cu`, run by `scripts/run_device_rma.sbatch`.

## Intra-node test: job 957153, PASS

Node h200x4-01 (4x H200 NVL), commit `9ff9fc2`, np=2 and np=4:

```
RESULT np=2 PASS fails=0 skips=0
RESULT np=4 PASS fails=0 skips=0
```

What that covers. Each rank puts 1024 doubles into rank+1 from a kernel, once
per scope, and the receiver checks every value:

| Operation | Scopes |
|---|---|
| Put | thread, warp, block |
| Get (reads back what it put) | thread, block |

It also checks these error returns:

| Side | Case | Expected |
|---|---|---|
| host | no `device_id` in info | `MPI_ERR_INFO_NOKEY` |
| host | wrong or non-numeric `device_id` | `MPI_ERR_INFO_VALUE` |
| host | window over host memory | `MPI_ERR_WIN` |
| host | second call on the same window | same handle |
| device | target out of range | `MPI_ERR_RANK` |
| device | Open MPI's own `MPI_DOUBLE` | `MPI_ERR_TYPE` |
| device | origin and target datatypes differ | `MPI_ERR_TYPE` |
| device | origin and target counts differ | `MPI_ERR_COUNT` |
| device | displacement past the window end | `MPI_ERR_RMA_RANGE` |
| device | invalid scope | `MPI_ERR_ARG` |

## Overhead over raw pointer access: job 963126

`mpi-intercept/bench_device_rma.cu`, run by `scripts/run_device_rma_bench.sbatch`.
Node h200x4-04, one H200, np=1, so every operation targets the rank's own
window (same device). Commit `04a243c`.

Each case is timed three ways, all with the same copy loop:

- **raw**: copy through the pointer from host `MPI_Win_shared_query`, with no
  lookup and no checks.
- **ct**: `MPIX_Put` / `MPIX_Get`, with the scope as a compile-time constant.
- **rt**: the same calls, with the scope passed as a kernel argument.

So **ct-raw** is the cost of the handle lookup and the argument checks, and
**rt-ct** is the cost of not knowing the scope at compile time.

Values are cycles per operation: thread 0's `clock64()` over 1000
back-to-back operations, median of 7 launches. ns uses the 1785 MHz clock
attribute, which may not be the clock the GPU actually ran at.

**Put (no flush)**

| scope | bytes | raw | ct | rt | ct-raw | rt-ct |
|---|--:|--:|--:|--:|--:|--:|
| thread | 8 | 171 | 634 | 660 | 463 | 26 |
| thread | 64 | 552 | 1025 | 1144 | 473 | 119 |
| thread | 512 | 2960 | 3466 | 5011 | 506 | 1545 |
| thread | 4096 | 22254 | 23068 | 35952 | 814 | 12885 |
| warp | 8 | 194 | 688 | 806 | 494 | 118 |
| warp | 64 | 194 | 688 | 806 | 494 | 118 |
| warp | 512 | 250 | 744 | 862 | 494 | 118 |
| warp | 4096 | 1247 | 1741 | 1859 | 494 | 118 |
| block | 8 | 198 | 655 | 691 | 458 | 36 |
| block | 64 | 198 | 655 | 691 | 457 | 36 |
| block | 512 | 182 | 640 | 675 | 458 | 35 |
| block | 4096 | 261 | 718 | 751 | 457 | 34 |

**Put, then flush after every operation**

| scope | bytes | raw | ct | rt | ct-raw | rt-ct |
|---|--:|--:|--:|--:|--:|--:|
| thread | 8 | 1421 | 2719 | 2787 | 1298 | 68 |
| thread | 64 | 2076 | 3382 | 3561 | 1306 | 180 |
| thread | 512 | 8268 | 9618 | 11317 | 1349 | 1699 |
| thread | 4096 | 56270 | 57919 | 71888 | 1648 | 13969 |
| warp | 8 | 1445 | 2768 | 2920 | 1323 | 153 |
| warp | 64 | 1450 | 2772 | 2925 | 1322 | 153 |
| warp | 512 | 1768 | 3092 | 3247 | 1324 | 155 |
| warp | 4096 | 6377 | 7700 | 7882 | 1323 | 182 |
| block | 8 | 1863 | 3102 | 3246 | 1239 | 144 |
| block | 64 | 1863 | 3105 | 3245 | 1242 | 140 |
| block | 512 | 1864 | 3106 | 3247 | 1243 | 140 |
| block | 4096 | 2553 | 3811 | 3909 | 1258 | 98 |

**Get, then flush after every operation**

| scope | bytes | raw | ct | rt | ct-raw | rt-ct |
|---|--:|--:|--:|--:|--:|--:|
| thread | 8 | 1373 | 2667 | 2748 | 1294 | 80 |
| thread | 64 | 2009 | 3307 | 3483 | 1297 | 176 |
| thread | 512 | 7840 | 9140 | 10842 | 1300 | 1702 |
| thread | 4096 | 54460 | 55789 | 69681 | 1329 | 13892 |
| warp | 8 | 1411 | 2721 | 2876 | 1310 | 155 |
| warp | 64 | 1417 | 2729 | 2880 | 1312 | 151 |
| warp | 512 | 1717 | 3029 | 3180 | 1312 | 151 |
| warp | 4096 | 6168 | 7480 | 7631 | 1312 | 151 |
| block | 8 | 1864 | 3097 | 3234 | 1233 | 137 |
| block | 64 | 1864 | 3097 | 3234 | 1232 | 137 |
| block | 512 | 1865 | 3102 | 3240 | 1238 | 137 |
| block | 4096 | 2466 | 3766 | 3869 | 1300 | 103 |

What it shows:

1. **The lookup and checks cost about 460-500 cycles (about 260-280 ns) per
   Put or Get**, nearly independent of size and scope. That is about 3.7x the
   raw time for an 8-byte put, and about 4% for a 4 KiB thread-scope put.
2. **With a flush after every operation, the extra cost is about 1240-1350
   cycles (about 700-750 ns)**, again constant. Why it is larger than without
   flush is not known yet. One guess, not verified: the fence stops the next
   operation's handle loads from overlapping with earlier work.
3. **A runtime scope costs little at warp and block scope** (35-155 cycles),
   **but a lot at thread scope for larger messages** (+12885 cycles, 56%, at
   4 KiB). With a compile-time thread scope the compiler knows one thread does
   the whole copy and optimizes the loop. With a runtime scope it cannot. So
   the cost is the lost loop optimization, not the scope branches themselves.

Static instruction counts agree. Compiled for sm_90, the rt kernel is the
same for all three scopes (175 instructions, 21 branches). The ct warp and
block kernels have 155-165 instructions and 17 branches, so the compiler does
remove the scope branches when it knows the scope. The ct kernels issue 11
global loads where raw issues 2: the handle lookups.

Caveats: one run, one node, same device only, timed from one thread.

## Not tested yet

- **Flush.** The test reads the data after the kernel has finished, and a
  finished kernel's writes are visible anyway. So this test cannot show that
  `MPIX_Win_flush_all` does anything. That needs a kernel that puts, flushes
  and then sets a flag, with the target spinning on the flag.
- **Unreachable peers.** Job 957155 ran np=2 and np=8 on h200x8-04 and
  passed with 0 skips: every put landed, including across the GPU 0-3 / 4-7
  boundary. In September (jobs 92211, 92654; `island_cross_boundary.md`)
  no cross-island write on this node arrived. So the node's behaviour
  appears to have changed, and it no longer gives us an unreachable peer to
  test with. Job 959275 (raw CUDA IPC recheck plus `nvidia-smi topo -m`) is
  queued to confirm.
- Get at warp scope, many blocks at once, overhead between two GPUs,
  inter-node.

## GDAKI on nvwulf: not available

Job 957154 (`scripts/run_gdaki_probe.sbatch`, h200x4-01) ran NVSHMEM 3.2.5's
`shmem_put_bw` twice:

- Defaults (NVLink): works, 114.7 GB/s at 4 MiB.
- IBGDA forced, NVLink off: NVSHMEM fails to initialize with
  `neither nv_peer_mem, or nvidia_peermem detected. Skipping transport.`

The NIC itself is fine: mlx5_0, InfiniBand, active, 200 Gb/s (2X NDR). What is
missing is GPUDirect RDMA, which lets the NIC read and write GPU memory.

`nvidia_peermem` loaded, by node checked:

| Node | `nvidia_peermem` | Driver `RegistryDwords` |
|---|---|---|
| h200x4-01 | not loaded | empty |
| h200x4-03 | not loaded | empty |
| h200x8-03 | not loaded | empty |
| b40x4-05 | not loaded | empty |

Jobs 957154, 958822, 958823, 958824. Every node has an active NDR
InfiniBand NIC (400 Gb/s on h200x8-03, 200 Gb/s on the others).

The kernels on these nodes do export dma-buf symbols. Whether a dma-buf route
could stand in for `nvidia_peermem` here has not been checked.
