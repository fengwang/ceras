# Migrating to the reviewed Ceras implementation

C++20 remains the supported minimum. The core stays header-only; image implementation symbols now come from one compiled translation unit. Correctness repairs do not require C++26.

## Build and link

```cmake
add_subdirectory(ceras)
target_link_libraries(my_application PRIVATE ceras::ceras)
# Add this only when calling image I/O or resize functions:
target_link_libraries(my_image_application PRIVATE ceras::image)
```

For direct compiler invocations, compile `src/image.cpp` exactly once and link its object into image consumers. Do not define STB implementation macros in application headers. Image shapes keep the existing **width, height, channels** order; a rank-two tensor represents one grayscale channel. HDR writing converts to actual float samples without byte normalization. Byte-valued HDR input is converted numerically (37 becomes 37.0), not scaled to [0,1]. Non-HDR `imwrite` still normalizes floating pixels to bytes; `direct_imwrite` accepts unsigned-byte pixels for those formats.

`make test` and `make ci` run the maintained CMake manifest. `make CBLAS=1 test`, `make CUDA=1 test`, `make gemm`, `make image`, `make mnist`, and `make benchmarks` are maintained entry points. Historical per-demo Make targets were retired; their source examples remain available. Builds go under `build/`, with strict floating-point semantics by default. CBLAS builds require both the provider's `cblas.h` and BLAS library. CUDA builds require official CUDA runtime/cuBLAS headers and libraries.

## Tensor ownership and errors

Tensor copies still share element storage. `deep_copy()` creates independent element storage and preserves an empty tensor. The destination-taking deep-copy overload commits independent storage only after successful allocation. Resizing to a different element count detaches the resized tensor so sibling aliases retain valid shape/storage pairs. Reshaping only changes dimensions, accepts a trailing `~0UL` inferred extent, and rejects a different element count; use `resize` when changing storage size. Addition/subtraction now return independent results even when broadcasting can return a view.

The custom allocator algorithms and cache collector were removed. `cached_allocator<T>` and `buffered_allocator<T, Bytes>` remain compatibility aliases for `std::allocator<T>`; `Bytes` no longer selects inline storage. These names are deprecated in documentation. Prefer `std::allocator` in new code. `get_memory_cache().gc()` has no replacement: normal owner destruction releases memory. Pinned CUDA allocations belong to backend operations rather than tensor shape storage.

Shape overflow, checked tensor indexing, invalid reshape/reduction arguments, invalid image dimensions, and invalid external data report exceptions in Debug and Release. Allocation failures propagate through the repaired tensor/graph APIs. Tensor stream extraction sets `failbit`, honors the stream exception mask, and preserves its destination on validation failure. File loaders throw contextual exceptions. Default parser limits are 64 dimensions, 16 million elements, and 256 MiB byte/line budgets; configure the thread-local `tensor_io_limits` deliberately for larger trusted inputs. Session restore validates all records before updating any parameter. Compressed restores have an output budget and no shared temporary file.

Global and axis min/max reject empty tensors, preserve infinities, and propagate NaNs. Mean rejects empty input. These choices require strict floating-point compilation; `-ffast-math` is outside the tested numerical contract.

## Gradients and training

Mean and cross-entropy gradients now differentiate the averaged forward scalar. **Optimizers no longer divide the learning rate by batch size.** The retained `batch_size` constructor argument is checked for positivity but does not rescale gradients. Revisit learning rates when migrating; there is no universal conversion factor because the old gradient and optimizer errors could partially cancel for some graphs.

Softmax uses its full vector-Jacobian product. ELU/GELU apply the incoming gradient correctly. Cross-entropy uses stable log-sum-exp, matching nonempty `[batch, classes]` tensors with at least two classes, and consistent label smoothing. Its target gradient differentiates the same smoothed-target expression. It no longer clips very small predicted probabilities before taking logarithms.

Each optimizer owns independent moment tensors keyed by parameter identity and visits only parameters reachable from its loss. Copying an optimizer copies its state independently while sharing the graph's parameter handles. `reset_state()` restarts its history; a changed parameter shape resets that parameter's state. Freezing pauses updates and preserves moments for later unfreezing. SGD/Nesterov, Adagrad, RMSprop, Adadelta and Adam have scalar recurrence tests. Adam now uses adaptive updates on its first step; AMSGrad maintains its own maximum variance. SGD decay is measured against the base rate rather than compounded into it.

Run the loss before each optimizer step. The step consumes and invalidates that forward execution. A backward call after invalidation fails until another forward pass. Backward within a valid execution accumulates leaf gradients; a new training forward resets them. Reused graph nodes retrieve complete execution records rather than relying on wrapper-local saved inputs. Borrowed upstream gradients are not used as writable scratch.

Compiled models retain graph ownership through the optimizer's loss handle, so relocation and destruction of the source are supported. Model copies share parameter handles; this is not a clone of learned weights. Freeze metadata is shared across those handles. Session registries hold weak handles and prune expired registrations.

## Contexts, phases and reproducibility

The convenience default session is **thread-local**. Sharing a mutable graph/parameter set between threads is unsupported without application-level synchronization. Independent graphs and explicit sessions are the tested CPU concurrency boundary. A shared explicit session serializes `run`, `backward`, and `generate`; combine a whole forward/backward/update sequence in `generate` if it must be one transaction. Do not mutate parameters, bindings, or raw tensor storage concurrently with execution. Legacy raw pointer views still require callers to keep storage alive and supply valid extents.

```cpp
ceras::ceras_private::session<ceras::tensor<double>> context;
context.seed_random(42);
auto data = context.generate([] { return ceras::random<double>({2, 3}); });
auto output = context.run(expression);
context.backward(expression, ceras::ones_like(output));
```

Explicit contexts own RNG, phase, graph scratch, and forward records. `context.phase_` selects training (1) or inference (0). `clear_forward_cache()` releases retained execution tensors. For default convenience APIs, call `ceras::seed_random(42)`; assigning `random_seed` alone does not reseed an already-created engine. The engine is shared deliberately across translation units within a thread. Reproducibility is tested within one standard-library implementation, not across different libraries' distribution algorithms. Prediction/evaluation restore the previous training phase even if execution throws.

## CPU and GPU execution

`gemm_cpu`, `cblas_gemm`, and `cuda_gemm` explicitly select a backend. Normal `gemm` uses CBLAS when enabled, otherwise the scalar CPU implementation. Automatic first-call CUDA calibration has been removed; there is no implicit timing or allocation search. Set the thread-local `cuda_gemm_threshold` after measuring a workload, or call CUDA directly. `update_cuda_gemm_threshold()` is retained as a compatibility reset to CPU-first dispatch, not a tuner.

CUDA resources are owned by `cuda_backend_context`: device, stream, handle, scratch and a transaction mutex. The default CUDA contexts are per thread/device; an explicit context can be shared and serializes reserve, transfers and GEMM together. Backend errors throw; allocation does not retry without a bound. Do not change an explicit context's resource fields during use. Independent mutable model training across streams is not implied by GEMM concurrency support.

`parallel_min_work` defaults to 4096. A worker processes at least that much work before another worker is added, nested calls stay serial, and softmax parallelizes row groups with serial inner loops. Set a higher threshold or compile with `NOPARALLEL` when an application or threaded BLAS already owns CPU parallelism. This is a tuning choice, not a universal fastest setting.

## Validation commands

Run the corresponding configure, build and test preset for `cpu-debug`, `cpu-release`, `asan-ubsan-debug`, `asan-ubsan-release`, `tsan`, `cblas-release` or `cuda-release`. Use `ctest --preset tsan -L concurrency --no-tests=error` for the supported concurrency surface. Leak detection must run outside ptrace-based sandboxes when the sanitizer runtime requires it.

`cmake --build build/cpu-release --target fuzz-smoke` runs deterministic bounded parser smoke. `CERAS_FUZZ_ROUNDS=100000 build/asan-ubsan-debug/ceras_verification parsers` extends it. `python3 tools/check_regression_mutations.py` checks that three representative faults fail the intended gates in a temporary source copy. `python3 tools/check_clean_build.py` builds a source export that cannot see unrelated untracked workspace files.

MNIST is optional and local: configure `-DCERAS_MNIST_DIR=/absolute/path/to/dataset/mnist` to register its seeded convergence test. Core tests never depend on downloading a dataset. CUDA CI is opt-in through `CERAS_CUDA_RUNNER=enabled` and a provisioned runner; a skipped workflow job is not hardware verification.

The C++26 feature-report target is `ceras_modern_features`, configured with `-DCMAKE_CXX_STANDARD=26`. It reports the actual language/library feature macros. It does not enable contracts, change the supported baseline, or substitute for runtime validation. Optional `expected`, `mdspan`, and fixed-capacity-container API migrations remain separate compatibility work.
