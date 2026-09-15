# Cross-island CUDA IPC on h200x8-04: the peer query lies

Raw logs are gitignored; this is the committed record. Jobs 92211, 92654.

h200x8-04 has eight GPUs in two NVLink islands, 0-3 and 4-7, with only PCIe
between them. h200x8-03 is NVSwitch all-to-all and has no such boundary, which
is why every number in the paper is unaffected: the stencil results use 4 GPUs
inside one island and the 8-rank LULESH ran on -03.

## 1. The platform behaviour, measured directly

Job 92211 ran `mpi-intercept/test_island_p2p.cu` -- raw CUDA IPC, no
interposer, no benchmark. Every rank publishes a handle, writes a sentinel into
every peer it can open, and after a barrier reports what actually arrived.

| pair type | pairs | OPEN | CAN | LAND |
|---|--:|--:|--:|--:|
| same island | 24 | 24 | 24 | **24** |
| cross island | 32 | 32 | 32 | **0** |

`OPEN` = `cudaIpcOpenMemHandle` succeeded. `CAN` = `cudaDeviceCanAccessPeer`
said yes. `LAND` = the sentinel was there afterwards.

**Every cross-island pair opens, reports itself peer-accessible, and delivers
nothing.** No error is returned at any point. Because this uses raw CUDA IPC
with no interposer, it is CUDA behaviour on this topology, not a WinIPC bug.

That kills `cudaDeviceCanAccessPeer` as a gate: it is true for all 32 broken
pairs. CUDA does P2P over PCIe, so "can these GPUs reach each other" is the
wrong question -- the right one is "does a store actually arrive".

## 2. The delivery probe works

Commit 63df376 replaced the reachability gate with `verify_peer_delivery()`,
which writes a sentinel at window construction and uses an Alltoall so each
rank learns which of ITS writes reached their target -- the direction
`MPI_Win_shared_query` has to gate on.

Job 92654 confirms it fires: every rank reports "4 of 7 peers opened but did
not deliver", and both boundary ranks print
`not IPC-reachable ... using MPI send/recv fallback` -- rank 3 for its right
neighbour, rank 4 for its left. Detection and the fallback decision are both
correct.

## 3. But np=8 is still wrong, and now we know exactly where

| job | gate | np=8 L2 |
|---|---|---|
| 90162 | reachability (95a4607) | 0.6600931148 |
| 92654 | delivery probe (63df376) | 0.6600931148 |

Byte-identical. Per-rank norms localise it completely:

| np | per-rank local L2 |
|---|---|
| 4 (correct) | 0, **3.5034**, **3.7679**, 0 |
| 8 (wrong) | 0, 0, 0, 0, **0.6601**, 0, 0, 0 |

At np=8 the heat source sits at global column 512, which is rank 4's left edge.
Heat should spread left into rank 3 immediately and take ~128 iterations to
cross rank 4 rightward, so rank 5 being zero is correct. **Rank 3 being zero is
the entire bug**: rank 4's left edge never reaches it. That one exchange is the
cross-island pair, and it is now supposed to be travelling over MPI.

## 4. What is not yet explained

The fallback looks correct on inspection. Both ranks take it. The tags pair
(rank 3 sends tag 0 rightward and receives tag 1; rank 4 receives tag 0 from
the left and sends tag 1), and `cudaDeviceSynchronize()` runs between the pack
kernels and the `MPI_Isend`, so the packed buffer is ready before it is sent.

The gap in the testing so far: every run either fitted inside one island
(np=2, np=4 -- no fallback needed) or spanned both (np=8 -- fallback needed and
the answer wrong). **The fallback path has never once been observed to work.**
It may be broken in general rather than at the boundary specifically, and
nothing run so far separates those.

Job 96753 forces the question with `CUDA_VISIBLE_DEVICES=3,4` and np=2: two
ranks, one per island, a single neighbour pair that must use the fallback, and
a same-island np=2 control alongside it.
