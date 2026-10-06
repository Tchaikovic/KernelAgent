"""Time one vLLM generate of Qwen/Qwen3-4B-AWQ."""

import os
import time

from vllm import LLM, SamplingParams
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def main() -> None:
    eager = os.environ.get("EAGER", "1") == "1"
    llm = LLM(
        model="Qwen/Qwen3-4B-AWQ",
        dtype="float16",
        max_model_len=256,
        gpu_memory_utilization=0.85,
        enforce_eager=eager,
        trust_remote_code=True,
        attention_backend=AttentionBackendEnum.TRITON_ATTN,
    )
    prompts = ["Explain what RMSNorm does in two short sentences."]
    params = SamplingParams(temperature=0.0, max_tokens=32)
    llm.generate(prompts, params)
    torch_sync = __import__("torch").cuda.synchronize
    torch_sync()
    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    torch_sync()
    elapsed = time.perf_counter() - start
    text = outputs[0].outputs[0].text
    ntok = len(outputs[0].outputs[0].token_ids)
    print(
        f"RESULT eager={eager} tokens={ntok} sec={elapsed:.3f} "
        f"tok_s={ntok / elapsed:.2f}"
    )
    print("TEXT", text.replace("\n", " ")[:300])


if __name__ == "__main__":
    main()
