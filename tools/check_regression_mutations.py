#!/usr/bin/env python3
"""Prove three release gates detect representative faults in an isolated copy."""
import pathlib
import shutil
import subprocess
import tempfile

root = pathlib.Path(__file__).resolve().parents[1]
mutations = [
    ("include/operation.hpp", "ans /= static_cast<typename Tsor::value_type>(batch_size);", "ans *= 0;", "mean"),
    ("include/optimizer.hpp", "else data += moments;", "else (void)moments;", "optimizers"),
    ("include/tensor.tcc", "if (!values.eof()) return fail();", "// mutation: permit trailing values", "parse"),
]
with tempfile.TemporaryDirectory(prefix="ceras-mutations-") as directory:
    copy = pathlib.Path(directory)
    for name in ("include", "src", "test"):
        shutil.copytree(root / name, copy / name)
    shutil.copy2(root / "CMakeLists.txt", copy)
    build = copy / "build"
    subprocess.run(["cmake", "-S", str(copy), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"], check=True)
    for name, before, after, test in mutations:
        path = copy / name
        original = path.read_text()
        if original.count(before) != 1:
            raise RuntimeError(f"mutation anchor changed: {name}: {before}")
        path.write_text(original.replace(before, after))
        subprocess.run(["cmake", "--build", str(build), "--target", "ceras_regression", "--parallel", "2"], check=True)
        result = subprocess.run([str(build / "ceras_regression"), test], text=True, capture_output=True)
        if result.returncode != 1 or "FAIL:" not in result.stderr:
            raise RuntimeError(f"gate failed to detect {test}: {result}")
        print(f"Detected {test}: {result.stderr.strip()}", flush=True)
        path.write_text(original)
