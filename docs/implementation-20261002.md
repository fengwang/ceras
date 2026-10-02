# Review remediation implementation and evidence

Implementation of [the approved plan](fix-plan-20261002.md) against reviewed `main` commit `48fe97c1ef30181cffa0446af8f6831140c9963d`. The remediation is prepared as one local commit following user authorization; no publication was requested. The final working-tree version passed the validation matrix below. All 32 accepted review findings have repair and regression evidence.

## Result and scope

W01–W13 are implemented: standard owning storage, checked dimensions, transactional loaders, one image implementation, safe parallel ranges, execution-owned graph records, corrected derivatives, optimizer-owned state, weak registration, model relocation, deterministic seeding, bounded CUDA dispatch and regression CI. Optional W14 was evaluated with an actual C++26 compiler/library probe; C++20 remains the supported baseline.

The objective is to close the 32 evidenced review findings and make their regressions enforceable. Passing these tests is not a proof that every historical operation or raw-pointer API is free of defects. Historical convolution/RNN/custom-layer combinations were not exhaustively re-audited by this implementation. Shared mutable graphs across threads and fast-math compilation remain outside the tested contract. Follow the [migration guide](migration-20261002.md) for the supported ownership and concurrency boundaries.

## Validation environment and results

Linux x86-64; AMD Ryzen 9 7900X3D, 24 logical CPUs; GCC 16.2.1 (20260810); Clang 22.1.8; CMake 4.4.3. CUDA 13.3.73/cuBLAS executed on an NVIDIA RTX 5090. Floating-point builds use `-fno-fast-math`. Release and sanitizer RelWithDebInfo builds define `NDEBUG`.

| Configuration | Executed gate | Result |
| --- | --- | --- |
| GCC Debug | 31 core tests | Passed |
| GCC Release | 31 core + seeded MNIST | Passed |
| Clang Debug and Release | 31 tests each | Passed |
| GCC ASan+UBSan Debug and NDEBUG | 31 tests each, leak checks enabled | Passed |
| GCC TSan | `contexts`, `ranges`, `parallel` | Passed |
| CBLAS Release | 31 tests including float/double transpose parity | Passed |
| CUDA Release, real GPU | 32 tests; independent and shared backend contexts | Passed |
| GCC C++26 Release | 31 tests; feature-report target | Passed |
| Isolated source export | Configure/build/31 tests without unrelated untracked files | Passed |
| Fault injection | Zero mean gradient, skip SGD update, permit extra serialized values | All three gates rejected the intended fault |
| Long parser smoke | 100,000 deterministic tensor/LZW inputs, seed 123, bounded output | Passed under ASan+UBSan; LSan disabled for this sandboxed long run |

The initial sanitizer run failed because LeakSanitizer cannot inspect a ptrace sandbox. Both full sanitizer configurations subsequently ran outside the sandbox with leak checking enabled. No sanitizer suppression or permanent expected-failure test was introduced. GPU access likewise required execution outside the sandbox. The local matrix was executed; the newly written GitHub Actions workflow has not been run remotely.

MNIST uses the first 512 training samples, batches of 32, 320 SGD steps, seed 42, and the first 256 validation samples. It requires loss below half its initial value and accuracy at least 65%. The measured result was loss **2.30258 → 0.132798**, accuracy **85.9375%**. This is a small convergence gate, not a full-dataset accuracy claim. Dataset-free synthetic training and numerical tests run independently.

The [source hash manifest](evidence/20261002/source-sha256.txt) identifies the build inputs; the [machine-readable results](evidence/20261002/validation-summary.json) record test totals. Runtime logs and baseline failures are retained in [evidence/20261002](evidence/20261002/). `tools/check_clean_build.py` copies tracked source files plus the explicit new-source manifest into a temporary directory. `tools/check_regression_mutations.py` makes separate temporary source mutations, requires successful compilation, and then requires the intended runtime assertion to fail. Neither script changes the working sources.

## Finding closure evidence

The tests below have unconditional failures in both Debug and Release. All rows refer to working-tree changes; the original finding severity, reviewer attribution and evidence remain in [the review](review-20261002.md).

| Findings | Implemented repair | Executable evidence |
| --- | --- | --- |
| F01, F07, F08 | Unsafe cache/buffer/vector owners replaced with standard allocation and storage; allocation-before-commit | `headers`, `allocation`, `storage`; over-alignment, throwing allocator, resize aliases, moves/self-moves |
| F02 | Checked extent/byte arithmetic and backend narrowing; standard-vector allocation limits | `storage`, `allocation`, `boundaries`; overflowing element/byte counts and int conversion |
| F03 | Bounded tensor input; staged session restore; bounded decompression without shared temp files | `parse`, `parsers`, `restore`; extra/missing/junk values, unknown IDs, malformed second record, round-trip |
| F04 | Shared IDX validation for both datasets, exact payloads and label range | `idx`; valid tiny fixture, trailing byte rejection, label 255 rejection |
| F05, F06 | Grayscale channel count and rank validation; actual float HDR samples; RAII decode buffer | `image`, `hdr`; grayscale/color resize, invalid ranks/channels, HDR decode values under sanitizers |
| F09 | Complete records owned by the executing session; invalidation after optimizer update | `graph`, `gradients`, `contexts`, `optimizer_state`; reused/weighted nodes and missing records |
| F10 | Full softmax VJP using independent gradient storage | `softmax`, `gradients`; constant-sum property, arbitrary upstream central differences |
| F11 | Mean derivative divides by total element count; empty mean rejects | `mean`, `gradients`, `normalization` |
| F12 | Matching mean reduction in CE; stable log-sum-exp; no optimizer batch divisor | `crossentropy`, `gradients`, `normalization`; batch duplication preserves loss and update |
| F13, F14 | ELU/GELU use forward input and upstream gradient | `elu`, `gelu`, `gradients`; zero upstream and central differences |
| F15 | Independent Adadelta gradient/update accumulators | `adadelta`, `recurrences`; five independently calculated steps |
| F16 | Optimizer-owned, independently copied moments; shape reset and explicit state reset | `optimizers`, `optimizer_state`, `recurrences`; switching, recurrence, reset, freeze, unrelated variables |
| F17 | Valid maximum identities; NaN/infinity propagation and empty-input rejection | `maximum`, `boundaries` |
| F18 | STB implementation in `src/image.cpp`; inline terminal-width helper | `multi_tu`, `headers`, `image`; two core TUs and two image-consumer TUs |
| F19 | Optimizer owns its loss handle; model relocation no longer retains a source-member reference | `api`; move, source destruction, subsequent training |
| F20 | Weak registries, expired-handle pruning and binding deduplication | `lifetime`; parameter state expires, 2,000 bindings remain bounded |
| F21 | Public run uses a session reference; tap explicitly mutates an lvalue | `api` |
| F22 | Seed actual engine; deliberate cross-TU/thread/context RNG ownership | `multi_tu`, `contexts` |
| F23 | Removed false unconditional noexcept along allocating tensor/graph paths | `allocation`; exceptions propagate from injected allocator failure |
| F24 | Offset-correct partitioning, bounded arithmetic, RAII joins and exception propagation | `parallel`, `ranges`, TSan; near-limit ranges, workers 0/1/2/8, uneven work, callback exception |
| F25 | Serial softmax inner loops, minimum work per worker, nested parallelism guard | `contexts` large-row parity; before/after benchmark CSVs below |
| F26 | Removed implicit calibration entirely; explicit dispatch is immediate and bounded by one requested GEMM | `gemm` with test timeouts; source contains no calibration search |
| F27 | CUDA context owns resources; shared-context mutex covers the complete transaction | `cuda_concurrency` and `gemm` on RTX 5090; varying shapes, growth, per-thread and shared contexts |
| F28 | Independent numerical oracles, sanitizer matrix, seeded integration, parser budgets and fault injection | Full CI manifest plus three detected mutations |
| F29 | Standalone CPU oracle; optional backend checks | `gemm` float/double, nonsquare, all transpose combinations |
| F30 | Actual CBLAS option, official header/library and configured backend | `cblas-release`; explicit CBLAS parity test |
| F31 | Explicit maintained test manifest and clean source export | `tools/check_clean_build.py`; `make` delegates maintained targets to CMake |
| F32 | Traversal uses current node accessors and shared trainability metadata | `api`; freeze after training, composition and later training |

Additional defects found by the new checks were repaired: addition/subtraction could modify shared forward operands; ELU/leaky-ReLU construction referenced a missing serializer helper; a stored concatenate factory captured temporary expressions by reference; empty-tensor deep copy changed its size; and a strided reduction constructed a pointer past the legal end. The concatenate and empty-copy regressions failed before their repairs. The vendor PNG transparency loop now states its three-channel bound explicitly, resolving a release compiler warning without disabling diagnostics.

## Performance evidence

Same machine and compiler flags (`g++ -std=c++20 -O3 -fno-fast-math -pthread`), six calls per case: one cold call and five warm observations. Shapes cover widths 7/8/9/24/32, row counts 1/8, and float/double. Columns report cold/min/median/max microseconds, actual spawned workers, ordinary `operator new` calls, and process peak RSS. Aligned allocation calls are not included in that allocation count; RSS is cumulative process high-water memory. This small benchmark does not establish convolution or end-to-end training speed.

| Float case | Before median | After median | Threads across six calls |
| --- | --- | --- | --- |
| 1 × 8 | 8.620 µs | 4.210 µs | 0 → 0 |
| 1 × 9 | 378.466 µs | 4.070 µs | 96 → 0 |
| 8 × 9 | 2,826.810 µs | 4.230 µs | 768 → 0 |
| 8 × 32 | 8,277.010 µs | 4.750 µs | 2,208 → 0 |

For float 8 × 32, ordinary allocation calls fell from 4,540 to 28 across six calls. All measured cases improved; no measured median regression exceeded the provisional 5% noise budget. The large differences track eliminated thread creation, but timings remain hardware/workload specific. No thread pool was introduced. The separate `contexts` test exercises sufficiently large rows that still use parallel execution.

Raw data: [before](evidence/20261002/benchmark-before.csv), [after](evidence/20261002/benchmark-after.csv). The benchmark measures the repaired implementation immediately before versus after the scheduling change, not every difference from the historical main branch.

## Deliberate plan adjustments and compatibility

- Removed calibration instead of adding a bounded automatic tuner. This eliminates the no-crossover loop completely; explicit backend selection and user-set thresholds remain. A fake timing backend is unnecessary because there is no internal search to simulate.
- The default execution context is per thread; explicit contexts serialize entry. Sharing mutable parameters across independent contexts concurrently remains unsupported. CPU TSan tests use independent graphs, while CUDA tests deliberately share the backend resource context.
- Compiled-model relocation became safe by retaining an owning loss handle inside the optimizer. Copying model graphs still shares weights; optimizer state copies are independent. A new clone API is outside this repair.
- Kept allocator names as documentation-deprecated aliases to standard allocators; removed the unsupported memory-cache API. Retired historical demo Make targets in favor of a small maintained test manifest.
- Optional modern-library adoption is a separate API task. The installed C++26 probe reported `__cplusplus=202400`, `span=202311`, `expected=202211`, `mdspan=202406`, `inplace_vector=202603`, and `contracts=202502`. These describe this compiler/library only. The repaired library does not require those C++26 facilities or rely on contracts for external-input checks.

The [migration guide](migration-20261002.md) lists the behavior changes users must handle, particularly learning-rate normalization, resize versus reshape, image linking, and supported context concurrency.
