# Implementation progress

## Start

Read approved plan, repository RTK instructions, TDD and verification skills. Implementation in progress; no findings closed yet. Existing source unchanged at start.

W01: added explicit CMake/preset test harness and maintained-target cleanup. Initial regression compilation failed because ELU references a missing serializer helper and passes it in the shape-calculator position. Repaired that prerequisite before recording behavior failures.

Baseline regression results: gemm passed; storage, parse, image, parallel, graph, softmax, mean, ELU, GELU, maximum and lifetime failed (red.log in /tmp/ceras-implementation). W02 replaces unsafe allocator algorithms with standard allocation and the raw vector owner with a standard-vector adapter; introduces checked extents and transactional structural resize.

W03/W04 targeted tests: parse, IDX malformed payload, grayscale resize, HDR and GEMM/storage all pass. W05–W07 focused run: 13/13 tests passed (lifetime intentionally still red). ELU prerequisite repair retained. Broad noexcept removal limited to allocation-heavy graph/tensor APIs; further boundary audit pending.

W08/W09: optimizer-switch regression aborted on shared slot access; Adadelta produced -2 instead of 1.999684. After optimizer-owned state and weak session registration, all 17 runtime cases passed. API freeze/composition compile regression added before accessor repair. Session restore now stages records and decompresses with an output budget; expanded restore/fuzz tests pending.

Multi-TU link exposed another header definition in utils/tqdm.hpp; marked function inline and initialized terminal width fallback. GPU query failed inside sandbox but approved external query found RTX 5090 (32 GB); hardware tests can proceed with sandbox escalation when needed.


W06–W10 expansion: finite differences for softmax, mean, ELU, GELU, sigmoid, square, stable cross-entropy and weighted reused-node graphs pass. Optimizer scalar recurrences, state copies/resets, freeze, mean-loss batch duplication, explicit contexts/RNG and destroyed-source model relocation now have gates. Empty deep-copy and retained concatenate-factory regressions failed before repair and then passed.

W11: RTX 5090 tests pass for varying GEMM shapes and both per-thread and shared backend contexts. Removed automatic calibration entirely, so initialization performs no timing search. Official CUDA/cuBLAS and CBLAS headers replace hand-maintained ABI declarations.

W12: GCC/Clang Debug/Release, both ASan+UBSan configurations with external leak checks, TSan concurrency, CBLAS, real CUDA, C++26 and a clean source export passed before the last two lifetime fixes; the final matrix is being refreshed. The three isolated injected faults were all detected for the intended reason. 100,000 seeded parser inputs passed ASan+UBSan with bounded output. Seeded MNIST reached 85.9375% accuracy against a 65% gate; loss fell from 2.30258 to 0.132798.

W13: minimum-work partitioning and row-level softmax avoid tiny thread creation. Benchmark CSVs and methodology are in docs/evidence/20261002 and docs/implementation-20261002.md. No pool added. W14 feature availability was probed under GCC C++26; the supported language baseline remains C++20.


Final verification complete: GCC Debug 31/31; GCC Release 32/32 including MNIST; Clang Debug and Release 31/31 each; ASan+UBSan Debug and NDEBUG 31/31 each with external leak detection; TSan 3/3 concurrency tests; CBLAS 31/31; CUDA RTX 5090 32/32; C++26 31/31; isolated clean source export 31/31. No sanitizer diagnostics were found in final runtime logs. All 32 finding checklist entries now link to evidence. Source hashes, raw benchmark data, baseline failures and test logs are retained under docs/evidence/20261002. No commit, push, deployment, or unrelated workspace-file edit was performed.

User authorized a local commit after implementation. Verified all tested source hashes and the explicit 80-file staging scope; unrelated untracked files are excluded.
