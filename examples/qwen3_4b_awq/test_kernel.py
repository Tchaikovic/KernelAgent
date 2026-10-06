"""Correctness test for the Qwen3-4B-AWQ W4A16 GEMM."""

import torch


def _pack_int4(unpacked: torch.Tensor) -> torch.Tensor:
    """Pack uint8 nibbles [..., N] into int32 [..., N // 8], low nibble first."""
    if unpacked.shape[-1] % 8 != 0:
        raise ValueError("N must be a multiple of 8")
    x = unpacked.to(torch.int32)
    x = x.reshape(*x.shape[:-1], x.shape[-1] // 8, 8)
    packed = torch.zeros(*x.shape[:-1], dtype=torch.int32, device=x.device)
    for i in range(8):
        packed |= x[..., i] << (4 * i)
    return packed


def _reference(x, qweight, scales, qzeros) -> torch.Tensor:
    """AWQ GEMM reference. Returns bf16 [M, N]."""
    m, k = x.shape
    n = scales.shape[1]
    group = 128
    device = x.device

    shifts = (4 * torch.arange(8, device=device, dtype=torch.int32)).view(1, 1, 8)
    w_packed = qweight.to(torch.int32).view(k, n // 8, 1)
    w_i4 = ((w_packed >> shifts) & 0xF).reshape(k, n).to(torch.float32)

    g = k // group
    z_packed = qzeros.to(torch.int32).view(g, n // 8, 1)
    z_i4 = ((z_packed >> shifts) & 0xF).reshape(g, n).to(torch.float32)
    z_i4 = z_i4.repeat_interleave(group, dim=0)

    scale = scales.to(torch.float32).repeat_interleave(group, dim=0)
    weight = (w_i4 - z_i4) * scale
    y = x.to(torch.float32) @ weight
    return y.to(torch.bfloat16)


def _make_case(m, k, n, device, seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(m, k, dtype=torch.bfloat16)
    q = torch.randint(0, 16, (k, n), dtype=torch.uint8, generator=g)
    z = torch.randint(0, 16, (k // 128, n), dtype=torch.uint8, generator=g)
    scales = (torch.rand(k // 128, n, generator=g) * 0.1 + 0.01).to(torch.bfloat16)
    return (
        x.to(device),
        _pack_int4(q).to(device),
        scales.to(device),
        _pack_int4(z).to(device),
    )


def test_kernel():
    """Test decode and a short prefill at real Qwen3-4B-AWQ layer shapes."""
    try:
        from kernel import kernel_function

        if not callable(kernel_function):
            print("kernel_function is not callable")
            return False
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        device = "cuda"
        # Tiny case catches packing bugs quickly. Then the two largest layers.
        cases = [
            (2, 256, 128, 0),
            (1, 2560, 9728, 1),  # gate_proj / up_proj, decode
            (8, 9728, 2560, 2),  # down_proj, short prefill
        ]
        for m, k, n, seed in cases:
            x, qweight, scales, qzeros = _make_case(m, k, n, device, seed)
            expected = _reference(x, qweight, scales, qzeros)
            result = kernel_function(x, qweight, scales, qzeros)
            if not isinstance(result, torch.Tensor):
                print(f"expected a tensor, got {type(result)}")
                return False
            if result.shape != expected.shape or result.dtype != torch.bfloat16:
                print(
                    f"bad result meta: got {result.shape} {result.dtype}, "
                    f"expected {expected.shape} {expected.dtype}"
                )
                return False
            if result.device != x.device:
                print(f"result device {result.device} != input {x.device}")
                return False
            if not torch.allclose(result, expected, rtol=2e-2, atol=2e-2):
                diff = (result.float() - expected.float()).abs()
                print(
                    f"MISMATCH M={m} K={k} N={n} max_abs={diff.max().item():.5f} "
                    f"got={result.flatten()[:4].tolist()} "
                    f"exp={expected.flatten()[:4].tolist()}"
                )
                return False
            print(f"ok M={m} K={k} N={n}")
        return True
    except Exception as e:
        print(f"Test failed: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    import sys

    sys.exit(0 if test_kernel() else 1)
