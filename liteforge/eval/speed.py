"""速度基准：单次前向吞吐 + 自回归生成吞吐。

诚实口径：
- forward 吞吐衡量"稠密算力"（剪枝后权重仍稠密存储，反映不出稀疏加速）；
- generate 吞吐是自回归解码口径（含 KV cache）；
- 部署引擎（vLLM/llama.cpp）的实测见 deploy/，那才是"压缩换速度"的最终证据。
"""

import time

import torch


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_forward(model, tokenizer, seqlen: int = 1024, batch_size: int = 1,
                      warmup: int = 3, iters: int = 10) -> dict:
    device = next(model.parameters()).device
    vocab = max(model.config.vocab_size, tokenizer.vocab_size + 1) if tokenizer else model.config.vocab_size
    ids = torch.randint(0, model.config.vocab_size - 1, (batch_size, seqlen), device=device)

    for _ in range(warmup):
        model(input_ids=ids)
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        model(input_ids=ids)
    _sync(device)
    dt = time.perf_counter() - t0

    tokens = batch_size * seqlen * iters
    return {
        "forward_tokens_per_s": round(tokens / dt, 1),
        "forward_ms_per_iter": round(dt / iters * 1000, 2),
        "seqlen": seqlen,
        "batch_size": batch_size,
    }


@torch.no_grad()
def benchmark_generate(model, tokenizer, prompt: str = "The meaning of life is",
                       max_new_tokens: int = 64, warmup: int = 1, iters: int = 3) -> dict:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    for _ in range(warmup):
        model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    t0 = time.perf_counter()
    out_len = 0
    for _ in range(iters):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
        out_len = out.shape[1] - inputs["input_ids"].shape[1]
    _sync(device)
    dt = time.perf_counter() - t0

    return {
        "generate_tokens_per_s": round(out_len * iters / dt, 1),
        "new_tokens": out_len,
        "iters": iters,
    }
