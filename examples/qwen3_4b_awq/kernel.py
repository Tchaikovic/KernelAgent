import torch
import triton
import triton.language as tl

# ============================================
# Triton Kernel: AWQ GEMM (W4A16) Dequant + GEMM
# ============================================
#
# Fuses:
#  1) Unpacking of 4-bit weight values from packed int32
#  2) Subtraction of per-128-row zero-points
#  3) Scaling by per-128-row per-column bf16→f32 scales
#  4) Matrix multiply-accumulate with input X (bf16→f32)
#  5) Cast accumulator to bf16 and store

@triton.jit
def _awq_w4a16_gemm(
    x_ptr,      # bf16 pointer [M, K]
    qw_ptr,     # int32 pointer [K, N//8] packed int4 weights
    sc_ptr,     # bf16 pointer [K//128, N] scales
    qz_ptr,     # int32 pointer [K//128, N//8] packed int4 zeros
    y_ptr,      # bf16 pointer [M, N]
    M, K, N,    # matrix dimensions
    stride_xm, stride_xk,
    stride_qwm, stride_qwn,
    stride_sm,  stride_sn,
    stride_qzm, stride_qzn,
    stride_ym,  stride_yn,
    BLOCK_M: tl.constexpr,  # e.g. 64
    BLOCK_N: tl.constexpr,  # e.g. 32  <- reduced to stay under SMEM limits
    BLOCK_K: tl.constexpr,  # 128
):
    # tile indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # compute the M- and N-indices for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # masks for boundary checks
    mask_m = offs_m < M
    mask_n = offs_n < N

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K in increments of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # load X block [BLOCK_M, BLOCK_K], bf16→f32
        x_ptrs = (
            x_ptr
            + offs_m[:, None] * stride_xm
            + offs_k[None, :] * stride_xk
        )
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # unpack 4-bit weights:
        #   each int32 packs 8 nibbles along N
        packed_n = offs_n[None, :] >> 3                # which word
        nib_shift = (offs_n[None, :] & 7) * 4          # shift amount

        # load packed weights [BLOCK_K, N//8]
        qw_ptrs = (
            qw_ptr
            + offs_k[:, None] * stride_qwm
            + packed_n * stride_qwn
        )
        valid_qw = mask_k[:, None] & (packed_n < ((N + 7) // 8))
        qw_p = tl.load(qw_ptrs, mask=valid_qw, other=0)
        w_i4 = (qw_p >> nib_shift) & 0xF               # [BLOCK_K, BLOCK_N]

        # unpack zero-points the same way
        grp = (offs_k // BLOCK_K)                       # which 128-row group
        qz_ptrs = (
            qz_ptr
            + grp[:, None] * stride_qzm
            + packed_n * stride_qzn
        )
        valid_qz = (grp[:, None] < (K + BLOCK_K - 1)//BLOCK_K) & (packed_n < ((N + 7)//8))
        qz_p = tl.load(qz_ptrs, mask=valid_qz, other=0)
        z_i4 = (qz_p >> nib_shift) & 0xF

        # load scales bf16→f32
        sc_ptrs = (
            sc_ptr
            + grp[:, None] * stride_sm
            + offs_n[None, :] * stride_sn
        )
        valid_s = (grp[:, None] < (K + BLOCK_K - 1)//BLOCK_K) & mask_n[None, :]
        s = tl.load(sc_ptrs, mask=valid_s, other=0.0).to(tl.float32)

        # dequantize & accumulate
        w_f32 = (w_i4.to(tl.float32) - z_i4.to(tl.float32)) * s   # [BLOCK_K, BLOCK_N]
        acc = tl.dot(x, w_f32, acc)

    # write back result as bf16
    out = acc.to(tl.bfloat16)
    y_ptrs = (
        y_ptr
        + offs_m[:, None] * stride_ym
        + offs_n[None, :] * stride_yn
    )
    tl.store(y_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


def kernel_function(x, qweight, scales, qzeros):
    """
    Wrapper for AWQ W4A16 GEMM
      y = x @ ((unpack4(qweight) - unpack4(qzeros_per128)) * scales_per128)
    x:       [M, K]    bf16
    qweight: [K, N//8] int32 (packed 4-bit)
    scales:  [K//128, N] bf16
    qzeros:  [K//128, N//8] int32
    returns  y: [M, N] bf16
    """
    # sanity checks
    assert x.device.type == 'cuda', "x must be on CUDA"
    assert x.dtype == torch.bfloat16, "x must be bf16"
    assert qweight.dtype == torch.int32 and qzeros.dtype == torch.int32, \
        "qweight/qzeros must be int32"
    assert scales.dtype == torch.bfloat16, "scales must be bf16"

    M, K = x.shape
    K_q, N8 = qweight.shape
    G, N_s = scales.shape
    Gz, N8_z = qzeros.shape
    assert K_q == K, "qweight K mismatch"
    assert G * 128 == K, "scales group size must be 128"
    assert Gz == G, "qzeros group count mismatch"
    assert N8 * 8 == N_s, "qweight N//8 does not match scales N"

    N = N_s

    # allocate output
    y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    # strides
    sxm, sxk = x.stride(0),            x.stride(1)
    sqwm, sqwn = qweight.stride(0),    qweight.stride(1)
    ssm, ssn   = scales.stride(0),     scales.stride(1)
    sqzm, sqzn = qzeros.stride(0),     qzeros.stride(1)
    sym, syn   = y.stride(0),          y.stride(1)

    # RTX 4070 shared memory is 99 KB. 64x32x128 fp32 tiles do not fit.
    BLOCK_M = 16
    BLOCK_N = 32
    BLOCK_K = 128

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _awq_w4a16_gemm[grid](
        x, qweight, scales, qzeros, y,
        M, K, N,
        sxm, sxk,
        sqwm, sqwn,
        ssm,  ssn,
        sqzm, sqzn,
        sym,  syn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_stages=1,
        num_warps=4,
    )
    return y