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

## Overhead and cost of acting as a group: job 2337920

`mpi-intercept/bench_device_rma.cu`, run by `scripts/run_device_rma_bench.sbatch`.
Node h200x4-01, one H200, np=1, so every operation targets the rank's own
window (same device). Commit `8614db9`.

Every thread copies the same number of bytes, and the message grows with the
scope: 32x for a warp, 256x for a block. Every thread does the same work, so
how much longer a warp or block takes than one thread is the cost of acting
as a group. Each case is timed two ways, with the same copy loop:

- **raw**: copy through the pointer from host `MPI_Win_shared_query`, with no
  lookup and no checks.
- **ct**: `MPIX_Put` / `MPIX_Get`, with the scope as a compile-time constant.

So **ct-raw** is the cost of the handle lookup and the argument checks.

Values are cycles per operation: thread 0's `clock64()` over 1000
back-to-back operations, after 5 untimed launches, median of 7 launches.
Scopes were measured in the order warp, thread, block. Slots are 1 MiB apart
so no two messages overlap.

**Put (no flush)**

| bytes per thread | scope | message bytes | raw | ct | ct-raw |
|--:|---|--:|--:|--:|--:|
| 8 | thread | 8 | 171 | 634 | 463 |
| 8 | warp | 256 | 178 | 672 | 494 |
| 8 | block | 2048 | 187 | 642 | 456 |
| 64 | thread | 64 | 552 | 1024 | 473 |
| 64 | warp | 2048 | 677 | 1171 | 494 |
| 64 | block | 16384 | 715 | 1175 | 459 |
| 512 | thread | 512 | 2960 | 3466 | 506 |
| 512 | warp | 16384 | 4667 | 5161 | 494 |
| 512 | block | 131072 | 6244 | 6543 | 299 |
| 4096 | thread | 4096 | 22252 | 23066 | 814 |
| 4096 | warp | 131072 | 37424 | 37944 | 521 |
| 4096 | block | 1048576 | 168326 | 168621 | 295 |

**Put, then flush after every operation**

| bytes per thread | scope | message bytes | raw | ct | ct-raw |
|--:|---|--:|--:|--:|--:|
| 8 | thread | 8 | 1577 | 2832 | 1256 |
| 8 | warp | 256 | 1587 | 2868 | 1281 |
| 8 | block | 2048 | 2526 | 3746 | 1220 |
| 64 | thread | 64 | 2220 | 3487 | 1266 |
| 64 | warp | 2048 | 3732 | 5011 | 1279 |
| 64 | block | 16384 | 4871 | 6102 | 1230 |
| 512 | thread | 512 | 8263 | 9552 | 1289 |
| 512 | warp | 16384 | 21774 | 23035 | 1261 |
| 512 | block | 131072 | 23547 | 24965 | 1418 |
| 4096 | thread | 4096 | 54746 | 56373 | 1627 |
| 4096 | warp | 131072 | 165348 | 166675 | 1327 |
| 4096 | block | 1048576 | 173563 | 176255 | 2692 |

**Get, then flush after every operation**

| bytes per thread | scope | message bytes | raw | ct | ct-raw |
|--:|---|--:|--:|--:|--:|
| 8 | thread | 8 | 1548 | 2802 | 1254 |
| 8 | warp | 256 | 1572 | 2842 | 1270 |
| 8 | block | 2048 | 2436 | 3621 | 1184 |
| 64 | thread | 64 | 2182 | 3436 | 1254 |
| 64 | warp | 2048 | 3804 | 5072 | 1269 |
| 64 | block | 16384 | 4842 | 6056 | 1214 |
| 512 | thread | 512 | 8072 | 9330 | 1258 |
| 512 | warp | 16384 | 21708 | 22978 | 1270 |
| 512 | block | 131072 | 23784 | 24979 | 1195 |
| 4096 | thread | 4096 | 54875 | 56154 | 1279 |
| 4096 | warp | 131072 | 165227 | 166489 | 1262 |
| 4096 | block | 1048576 | 373793 | 375053 | 1260 |

What it shows:

1. **With no flush and 8 bytes per thread, a warp or block takes about as long
   as one thread** (178 and 187 cycles vs 171, raw). Issuing an operation as
   a group costs almost nothing.
2. **As the bytes per thread grow, the group falls behind, without any
   barrier**: at 4 KiB per thread a warp takes 1.7x and a block 7.6x as long
   as one thread (raw put). There is no synchronization in a put without
   flush, so this is the group moving 32x or 256x more data, not the cost of
   coordinating. One guess, not verified: a single block's memory throughput
   is the limit.
3. **The barrier shows up with flush.** At 8 bytes per thread, a warp adds 10
   cycles over one thread and a block adds 949 (raw put+flush; get+flush:
   24 and 888). A warp barrier is close to free; a block barrier costs about
   900-950 cycles (about 500-530 ns), including waiting for the slowest
   thread's stores. The gap grows with size (64 bytes per thread: +1512 for
   a warp, +2651 for a block), because flush waits until all of the group's
   data has arrived.
4. **The MPIX overhead does not depend on the group size**: ct-raw is about
   455-495 cycles (about 255-280 ns at the 1785 MHz clock attribute) at 8
   bytes per thread for all three scopes, about 1180-1290 with a flush after
   every operation (up to 64 bytes per thread), and small next to the copy for the largest messages.

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
