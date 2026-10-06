"""vLLM plugin: replace AWQ GEMM with the KernelAgent Triton kernel."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_KERNEL = None
_CALLS = 0
_FALLBACKS = 0


def _load_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    path = Path(__file__).resolve().parents[1] / "kernel.py"
    spec = importlib.util.spec_from_file_location("qwen_awq_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _KERNEL = mod.kernel_function
    return _KERNEL


def register() -> None:
    if os.environ.get("KERNELAGENT_AWQ") != "1":
        return

    import torch

    import vllm._custom_ops as ops
    import vllm.model_executor.layers.quantization.auto_awq as auto_awq
    import vllm.model_executor.layers.quantization.utils.marlin_utils as marlin_utils

    def _no_marlin(*_args, **_kwargs) -> bool:
        return False

    # Chosen at quant-method selection. Keep this off Marlin so apply()
    # calls ops.awq_gemm, which is the hook below.
    auto_awq.check_marlin_supported = _no_marlin
    marlin_utils.check_marlin_supported = _no_marlin

    kernel_function = _load_kernel()

    def awq_gemm(inp, qweight, scales, qzeros, split_k_iters):
        global _CALLS, _FALLBACKS
        _CALLS += 1
        try:
            x = inp if inp.dtype == torch.bfloat16 else inp.to(torch.bfloat16)
            sc = scales if scales.dtype == torch.bfloat16 else scales.to(torch.bfloat16)
            y = kernel_function(x, qweight, sc, qzeros)
            if y.dtype != inp.dtype:
                y = y.to(inp.dtype)
            if _CALLS == 1 or _CALLS % 200 == 0:
                print(
                    f"KERNELAGENT_AWQ calls={_CALLS} fallbacks={_FALLBACKS} "
                    f"shape={tuple(inp.shape)}",
                    flush=True,
                )
            return y
        except Exception as exc:
            _FALLBACKS += 1
            if _FALLBACKS <= 3:
                print(f"KERNELAGENT_AWQ fallback: {type(exc).__name__}: {exc}", flush=True)
            return ops_orig(inp, qweight, scales, qzeros, split_k_iters)

    ops_orig = ops.awq_gemm
    ops.awq_gemm = awq_gemm
    print("KERNELAGENT_AWQ plugin installed", flush=True)
