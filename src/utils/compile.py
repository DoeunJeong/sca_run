# src/compile.py
from contextlib import contextmanager
from functools import wraps
import inspect
import os
import typing as tp
import torch
from torch import cuda

_compile_disabled: bool = False

@contextmanager
def no_compile():
    global _compile_disabled
    prev_disabled = _compile_disabled
    _compile_disabled = True
    try: yield
    finally: _compile_disabled = prev_disabled

def torch_compile_lazy(fun):
    if os.environ.get("NO_TORCH_COMPILE"): return fun
    fun_compiled = None
    @wraps(fun)
    def _wrapped(*args, **kwargs):
        nonlocal fun_compiled
        if _compile_disabled: return fun(*args, **kwargs)
        if fun_compiled is None:
            # Latency 최적화 모드 사용
            #fun_compiled = torch.compile(fun, mode="reduce-overhead")
            fun_compiled = torch.compile(fun, mode="default")
        return fun_compiled(*args, **kwargs)
    return _wrapped

# ... (나머지 Moshi 코드는 그대로 두거나 필요시 복사) ...
# 최소한 torch_compile_lazy는 필수입니다.