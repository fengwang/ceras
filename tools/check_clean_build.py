#!/usr/bin/env python3
"""Build a source export without access to unrelated untracked workspace files."""
from pathlib import Path
import shutil
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
tracked=subprocess.check_output(["git","ls-files","-z"],cwd=root).decode().split("\0")
extras=(root/"tools/maintained_sources.txt").read_text().splitlines()
with tempfile.TemporaryDirectory(prefix="ceras-clean-") as directory:
    destination=Path(directory)
    for name in set(tracked+extras):
        if not name or not (name.startswith(("include/","test/","src/")) or name in ("CMakeLists.txt","CMakePresets.json","Makefile")):
            continue
        source=root/name
        if not source.is_file():raise RuntimeError(f"missing source: {name}")
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    build=destination/"build"
    subprocess.run(["cmake","-S",str(destination),"-B",str(build),"-DCMAKE_BUILD_TYPE=Release"],check=True)
    subprocess.run(["cmake","--build",str(build),"--parallel","2"],check=True)
    subprocess.run(["ctest","--test-dir",str(build),"--output-on-failure","--no-tests=error"],check=True)
