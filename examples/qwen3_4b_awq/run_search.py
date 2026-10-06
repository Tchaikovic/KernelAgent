"""Search for a Qwen3-4B-AWQ W4A16 GEMM kernel with KernelAgent."""

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from triton_kernel_agent import TritonKernelAgent

HERE = Path(__file__).resolve().parent


def main() -> int:
    problem = (HERE / "problem.txt").read_text()
    test_code = (HERE / "test_kernel.py").read_text()
    agent = TritonKernelAgent(
        num_workers=2,
        max_rounds=4,
        model_name="o4-mini",
        log_dir=str(HERE / "logs"),
        test_timeout_s=180,
        high_reasoning_effort=True,
    )
    result = agent.generate_kernel(
        problem,
        test_code=test_code,
        generate_default_test=False,
    )
    print(result)
    if result.get("success"):
        out = HERE / "kernel.py"
        out.write_text(result["kernel_code"])
        print(f"wrote {out}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
