"""示例 06：投机解码 —— 小模型起草、大模型验证，质量零损失加速。

运行：
    python examples/06_speculative_decoding.py

你将学到：贪心验证为什么输出与 target 纯贪心逐 token 一致（精确性定理，
本仓库用单测证明）、接受率与每 target 前向产出 token 数的统计口径。
"""

import torch

from liteforge.models import load_model_and_tokenizer
from liteforge.speculative import greedy_generate_baseline, speculative_generate
from liteforge.utils import seed_everything, setup_logging


def main():
    setup_logging()
    seed_everything()
    target, tok = load_model_and_tokenizer("cache/models/Qwen2.5-1.5B")
    draft, _ = load_model_and_tokenizer("cache/models/Qwen2.5-0.5B")
    ids = tok("The meaning of life is", return_tensors="pt")["input_ids"]
    ids = ids.to(next(target.parameters()).device)

    ref = greedy_generate_baseline(target, ids, max_new_tokens=128,
                                   eos_id=tok.eos_token_id)
    out = speculative_generate(target, draft, ids, max_new_tokens=128, k=4,
                               eos_id=tok.eos_token_id)

    exact = torch.equal(out["ids"], ref["ids"])
    print("\n=== 投机解码（target=1.5B, draft=0.5B, k=4）===")
    print(f"输出与 target 纯贪心逐 token 一致: {exact}（必须为 True）")
    print(f"接受率 α: {out['stats']['acceptance_rate']:.3f}")
    print(f"每 target 前向产出: {out['stats']['tokens_per_target_forward']:.2f} tokens")
    print(f"墙钟加速: {ref['stats']['wall_s']/out['stats']['wall_s']:.2f}×")
    print(f"生成文本: {tok.decode(out['ids'][0, ids.shape[1]:], skip_special_tokens=True)[:120]}…")


if __name__ == "__main__":
    main()
