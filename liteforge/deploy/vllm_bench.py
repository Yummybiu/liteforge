"""vLLM 部署基准：对 OpenAI 兼容服务（vllm serve 默认端口 8000）压测。

用法（在部署了 vLLM 的 GPU 机器上）：
    # 1) 起服务
    vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 4096
    # 2) 压测
    python -m liteforge.deploy.vllm_bench --model Qwen/Qwen2.5-0.5B-Instruct \
        --num-requests 64 --concurrency 8 --max-tokens 128

口径：
- TTFT：发请求到收到首个流式 chunk 的时间（首 token 延迟）；
- decode 吞吐：所有请求新生成 token 总数 / 总墙钟时间；
- 输入长度：以词表近似构造指定长度 prompt（压测关注形状而非语义）。
仅依赖标准库（urllib），无需 openai 客户端。
"""

import argparse
import json
import threading
import time
import urllib.request

WORDS = ("the model of language neural network attention transformer gradient "
         "optimization inference quantization pruning deploy benchmark throughput "
         "latency token batch schedule memory compute kernel sparse dense serve").split()


def build_prompt(num_words: int) -> str:
    return " ".join(WORDS[i % len(WORDS)] for i in range(num_words)) or "hello"


def stream_completion(endpoint: str, model: str, prompt: str, max_tokens: int):
    """流式请求，返回 (ttft_s, n_tokens, latency_s)。"""
    url = endpoint.rstrip("/") + "/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    ttft, n_tokens, t0 = None, 0, time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if ttft is None:
                ttft = time.perf_counter() - t0
            choice = (chunk.get("choices") or [{}])[0]
            n_tokens += len(choice.get("text", ""))  # 近似：按字符计 token 略高估
    latency = time.perf_counter() - t0
    if ttft is None:
        ttft = latency
    return ttft, n_tokens, latency


def run_bench(endpoint: str, model: str, num_requests: int, concurrency: int,
              input_words: int, max_tokens: int) -> dict:
    prompts = [build_prompt(input_words) for _ in range(num_requests)]
    results = []
    lock = threading.Lock()
    idx = {"i": 0}
    t0 = time.perf_counter()

    def worker():
        while True:
            with lock:
                i = idx["i"]
                idx["i"] += 1
            if i >= len(prompts):
                return
            try:
                ttft, n_tok, lat = stream_completion(endpoint, model, prompts[i], max_tokens)
                results.append({"ttft_s": ttft, "tokens": n_tok, "latency_s": lat})
            except Exception as e:  # 单请求失败不拖垮整体
                results.append({"error": str(e)[:200]})

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if "error" not in r]
    total_tokens = sum(r["tokens"] for r in ok)
    summary = {
        "endpoint": endpoint,
        "model": model,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "input_words": input_words,
        "max_tokens": max_tokens,
        "success_rate": round(len(ok) / max(len(results), 1), 3),
        "wall_s": round(wall, 2),
        "output_tokens_total": total_tokens,
        "decode_tokens_per_s": round(total_tokens / max(wall, 1e-6), 1),
        "ttft_mean_s": round(sum(r["ttft_s"] for r in ok) / max(len(ok), 1), 3),
        "latency_mean_s": round(sum(r["latency_s"] for r in ok) / max(len(ok), 1), 3),
    }
    return summary


def main():
    p = argparse.ArgumentParser(description="LiteForge vLLM 部署基准")
    p.add_argument("--endpoint", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--input-words", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--out", default=None, help="结果 JSON 输出路径")
    args = p.parse_args()

    summary = run_bench(args.endpoint, args.model, args.num_requests,
                        args.concurrency, args.input_words, args.max_tokens)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        from ..utils import save_json
        save_json({"task": "vllm-bench", "model": args.model,
                   "method": "vllm-serve", "params": {
                       "concurrency": args.concurrency,
                       "num_requests": args.num_requests,
                       "max_tokens": args.max_tokens},
                   "metrics": {k: v for k, v in summary.items()
                               if k in ("decode_tokens_per_s", "ttft_mean_s",
                                        "latency_mean_s", "success_rate")}},
                  args.out)


if __name__ == "__main__":
    main()
