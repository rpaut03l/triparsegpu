"""Compile CUDA kernels from kernels/*.cu via CuPy RawModule (nvcc backend)."""
import functools, pathlib
import cupy as cp

KDIR = pathlib.Path(__file__).resolve().parent.parent / "kernels"
ARCH = "sm_120"   # RTX PRO 6000 Blackwell, compute capability 12.0


@functools.lru_cache(maxsize=None)
def module(name: str, defines: tuple = ()):
    src = (KDIR / f"{name}.cu").read_text()
    opts = ("-std=c++17", f"-arch={ARCH}", "-O3", f"-I{KDIR}") + defines
    return cp.RawModule(code=src, backend="nvcc", options=opts)


def kernel(name: str, fn: str, defines: tuple = ()):
    return module(name, defines).get_function(fn)
