# Implementation findings

- All 32 baseline defects and reproductions are in docs/review-20261002.md.
- HEAD equals reviewed main; many unrelated untracked files must remain untouched.
- CMake 4.4.3, GCC 16, Clang 22 and a pkg-config BLAS library are present.
- Public headers currently emit STB implementations. Until W04, use single-TU regression executables.
- Forward/backward state and default-session ownership changes require coordinated optimizer/session changes; retain recorded inputs across optimizer execution.


## Implementation discoveries

- Same-shape broadcasting could alias the left operand of tensor addition/subtraction. Mutating the result corrupted saved graph inputs; weighted diamond finite differences caught the error.
- Header-only core multi-TU linking also required an inline terminal-width helper after moving STB definitions.
- ELU/leaky-ReLU serializer wiring failed to compile before a small prerequisite repair.
- Stored concatenate factories captured temporary graph expressions by reference; the new API regression segfaulted before value capture.
- Deep-copying the default empty tensor allocated a scalar. The new allocation regression failed before preserving the empty state.
- Strided reduction end-iterator construction could form an out-of-range pointer; the reduction now indexes valid elements directly.
- RTX 5090 execution was available outside the sandbox. The historical CUDA concurrent test failed; independent and shared RAII backend contexts pass hardware checks.
- LSan fails inside the ptrace sandbox. Authorized external full test runs passed with LSan enabled; the long 100,000-input sandbox fuzz run explicitly disabled LSan only.
- Before/after softmax CSVs show thread creation eliminated for tiny rows. Measurements are machine-specific and do not justify introducing a pool.
