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

## Overhead over raw pointer access: job 2337906

`mpi-intercept/bench_device_rma.cu`, run by `scripts/run_device_rma_bench.sbatch`.
Node h200x4-01, one H200, np=1, so every operation targets the rank's own
window (same device). Commit `c60b3f5`.

Each case is timed three ways, all with the same copy loop:

- **raw**: copy through the pointer from host `MPI_Win_shared_query`, with no
  lookup and no checks.
- **ct**: `MPIX_Put` / `MPIX_Get`, with the scope as a compile-time constant.
- **rt**: the same calls, with the scope passed as a kernel argument.

So **ct-raw** is the cost of the handle lookup and the argument checks, and
**rt-ct** is the cost of not knowing the scope at compile time.

Values are cycles per operation: thread 0's `clock64()` over 1000
back-to-back operations, after 5 untimed launches, median of 7 launches.
Scopes were measured in the order warp, thread, block. ns uses the 1785 MHz
clock attribute, which may not be the clock the GPU actually ran at.

**Put (no flush)**

| bytes | scope | raw | ct | rt | ct-raw | rt-ct |
|--:|---|--:|--:|--:|--:|--:|
| 8 | thread | 172 | 681 | 762 | 509 | 81 |
| 8 | warp | 196 | 762 | 912 | 566 | 150 |
| 8 | block | 202 | 705 | 798 | 503 | 93 |
| 64 | thread | 581 | 1072 | 1246 | 491 | 174 |
| 64 | warp | 196 | 762 | 912 | 566 | 150 |
| 64 | block | 202 | 705 | 798 | 504 | 93 |
| 512 | thread | 2980 | 3585 | 5113 | 605 | 1528 |
| 512 | warp | 252 | 817 | 968 | 565 | 151 |
| 512 | block | 185 | 688 | 782 | 503 | 94 |
| 4096 | thread | 22300 | 23689 | 36052 | 1389 | 12364 |
| 4096 | warp | 1249 | 1814 | 1965 | 565 | 151 |
| 4096 | block | 266 | 765 | 858 | 499 | 93 |

**Put, then flush after every operation**

| bytes | scope | raw | ct | rt | ct-raw | rt-ct |
|--:|---|--:|--:|--:|--:|--:|
| 8 | thread | 1576 | 2877 | 2988 | 1301 | 111 |
| 8 | warp | 1604 | 2966 | 3117 | 1362 | 152 |
| 8 | block | 2204 | 3495 | 3631 | 1291 | 136 |
| 64 | thread | 2250 | 3535 | 3736 | 1285 | 201 |
| 64 | warp | 1607 | 2966 | 3122 | 1359 | 156 |
| 64 | block | 2204 | 3493 | 3631 | 1290 | 138 |
| 512 | thread | 8274 | 9670 | 11237 | 1397 | 1567 |
| 512 | warp | 1917 | 3277 | 3438 | 1360 | 161 |
| 512 | block | 2210 | 3492 | 3637 | 1282 | 145 |
| 4096 | thread | 54805 | 56998 | 69396 | 2193 | 12399 |
| 4096 | warp | 6325 | 7686 | 7832 | 1361 | 146 |
| 4096 | block | 2867 | 4153 | 4268 | 1286 | 116 |

**Get, then flush after every operation**

| bytes | scope | raw | ct | rt | ct-raw | rt-ct |
|--:|---|--:|--:|--:|--:|--:|
| 8 | thread | 1548 | 2843 | 2954 | 1295 | 111 |
| 8 | warp | 1584 | 2932 | 3093 | 1349 | 161 |
| 8 | block | 2205 | 3468 | 3616 | 1263 | 147 |
| 64 | thread | 2218 | 3472 | 3689 | 1255 | 217 |
| 64 | warp | 1591 | 2941 | 3097 | 1350 | 157 |
| 64 | block | 2205 | 3468 | 3611 | 1263 | 143 |
| 512 | thread | 8055 | 9364 | 11045 | 1310 | 1681 |
| 512 | warp | 1891 | 3240 | 3391 | 1349 | 152 |
| 512 | block | 2202 | 3468 | 3616 | 1266 | 148 |
| 4096 | thread | 54703 | 56322 | 69848 | 1619 | 13526 |
| 4096 | warp | 6335 | 7694 | 7850 | 1359 | 156 |
| 4096 | block | 2824 | 4140 | 4248 | 1316 | 108 |

What it shows:

1. **The lookup and checks cost about 490-610 cycles (about 275-340 ns) per
   Put or Get** up to 512 bytes, nearly independent of scope. That is about
   4x the raw time for an 8-byte put, and 6% for a 4 KiB thread-scope put.
2. **With a flush after every operation, the extra cost is about 1250-1400
   cycles (about 700-780 ns)**, more for a 4 KiB thread-scope operation
   (1619-2193). Why it is larger than without flush is not
   known yet. One guess, not verified: the fence stops the next operation's
   handle loads from overlapping with earlier work.
3. **A runtime scope costs little at warp and block scope** (93-161 cycles),
   **but a lot at thread scope for larger messages** (+12364 cycles, 52%, for
   a 4 KiB put). With a compile-time thread scope the compiler knows one
   thread does the whole copy and optimizes the loop. With a runtime scope it
   cannot. So the cost is the lost loop optimization, not the scope branches.

Static instruction counts agree. Compiled for sm_90, the rt put kernel is the
same for all three scopes (191 instructions, 22 branches). The ct warp and
block kernels have 167-174 instructions and 18 branches, so the compiler does
remove the scope branches when it knows the scope. The ct kernels issue 11
global loads where raw issues 2: the handle lookups.

**The overhead moves with the surrounding code.** Earlier runs on the same
node with the previous benchmark loop (jobs 966563, 966580) measured
ct-raw about 70 cycles lower (494 vs 566 for an 8-byte warp put), with raw
within 1%. The device API did not change in between. Commit `c60b3f5` made
the loop read its slot spacing from a kernel argument instead of a constant,
which added 4 instructions to the raw kernels and 9-15 to the MPIX kernels.
So these overhead figures carry an uncertainty of roughly that size.

## Cost of acting as a group: job 2337906

Same run. Here every thread copies the same number of bytes, and the message
grows with the scope: 32x for a warp, 256x for a block. Every thread does
the same work, so how much longer a warp or block takes than one thread is
the cost of acting as a group. The thread rows match the first table's to
within 0.2%.

**Put (no flush)**

| bytes per thread | scope | message bytes | raw | ct | ct-raw |
|--:|---|--:|--:|--:|--:|
| 8 | thread | 8 | 172 | 681 | 509 |
| 8 | warp | 256 | 180 | 745 | 565 |
| 8 | block | 2048 | 188 | 690 | 502 |
| 64 | thread | 64 | 581 | 1072 | 491 |
| 64 | warp | 2048 | 679 | 1244 | 565 |
| 64 | block | 16384 | 731 | 1220 | 489 |
| 512 | thread | 512 | 2980 | 3585 | 605 |
| 512 | warp | 16384 | 4669 | 5234 | 565 |
| 512 | block | 131072 | 6318 | 6545 | 226 |
| 4096 | thread | 4096 | 22300 | 23689 | 1389 |
| 4096 | warp | 131072 | 38126 | 38723 | 597 |
| 4096 | block | 1048576 | 169437 | 168661 | -776 |

**Put, then flush after every operation**

| bytes per thread | scope | message bytes | raw | ct | ct-raw |
|--:|---|--:|--:|--:|--:|
| 8 | thread | 8 | 1577 | 2878 | 1301 |
| 8 | warp | 256 | 1588 | 2949 | 1361 |
| 8 | block | 2048 | 2527 | 3825 | 1298 |
| 64 | thread | 64 | 2248 | 3532 | 1284 |
| 64 | warp | 2048 | 3733 | 5092 | 1359 |
| 64 | block | 16384 | 4886 | 6166 | 1281 |
| 512 | thread | 512 | 8276 | 9664 | 1388 |
| 512 | warp | 16384 | 21759 | 23100 | 1341 |
| 512 | block | 131072 | 23739 | 24836 | 1097 |
| 4096 | thread | 4096 | 54765 | 56932 | 2167 |
| 4096 | warp | 131072 | 165103 | 166547 | 1444 |
| 4096 | block | 1048576 | 174904 | 174744 | -160 |

**Get, then flush after every operation**

| bytes per thread | scope | message bytes | raw | ct | ct-raw |
|--:|---|--:|--:|--:|--:|
| 8 | thread | 8 | 1546 | 2844 | 1298 |
| 8 | warp | 256 | 1572 | 2918 | 1347 |
| 8 | block | 2048 | 2453 | 3692 | 1239 |
| 64 | thread | 64 | 2216 | 3469 | 1253 |
| 64 | warp | 2048 | 3790 | 5137 | 1347 |
| 64 | block | 16384 | 4842 | 6122 | 1280 |
| 512 | thread | 512 | 8048 | 9357 | 1309 |
| 512 | warp | 16384 | 21662 | 23013 | 1352 |
| 512 | block | 131072 | 23956 | 25057 | 1101 |
| 4096 | thread | 4096 | 54662 | 56308 | 1646 |
| 4096 | warp | 131072 | 165024 | 166348 | 1324 |
| 4096 | block | 1048576 | 374022 | 374511 | 488 |

What it shows:

1. **With no flush and 8 bytes per thread, a warp or block takes about as long
   as one thread** (180 and 188 cycles vs 172, raw). Issuing an operation as
   a group costs almost nothing.
2. **As the bytes per thread grow, the group falls behind, without any
   barrier**: at 4 KiB per thread a warp takes 1.7x and a block 7.6x as long
   as one thread (raw put). There is no synchronization in a put without
   flush, so this is the group moving 32x or 256x more data, not the cost of
   coordinating. One guess, not verified: a single block's memory throughput
   is the limit.
3. **The barrier shows up with flush.** At 8 bytes per thread, a warp adds 11
   cycles over one thread and a block adds 950 (raw put+flush; get+flush:
   26 and 907). A warp barrier is close to free; a block barrier costs about
   950 cycles (about 530 ns), including waiting for the slowest thread's
   stores. The gap grows with size (64 bytes per thread: +1486 for a warp,
   +2638 for a block), because flush waits until all of the group's data
   has arrived.
4. **The MPIX overhead does not depend on the group size**: ct-raw is about
   500-570 cycles at 8 bytes per thread for all three scopes, and is lost in
   the noise for the largest messages.

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
