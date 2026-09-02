"""困惑度评测：非重叠滑窗（标准 GPT 系 PPL 协议），从零实现。

协议说明（面试常考点）：
- 文本拼接后切成互不重叠的 block（默认 2048），每个 block 内
  用前 n-1 个 token 预测后 n-1 个 token，交叉熵按全部预测 token 求均值；
- PPL = exp(mean NLL)。所有 block 的 token 权重相同；
- 与 HF 官方 perplexity 指南和 lm-eval-harness 的 wikitext PPL 口径一致
  （滑动重叠窗口版本是变体，见 TODO）。
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..data.text import BlockBatcher, load_eval_text


@dataclass
class PerplexityResult:
    ppl: float
    nll: float
    n_tokens: int
    n_blocks: int
    block_size: int

    def to_dict(self) -> dict:
        return {
            "ppl": round(self.ppl, 4),
            "nll": round(self.nll, 6),
            "n_tokens": self.n_tokens,
            "n_blocks": self.n_blocks,
            "block_size": self.block_size,
        }


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer=None,
    dataset: str | None = None,
    text: str | None = None,
    block_size: int = 2048,
    batch_size: int = 4,
    max_blocks: int | None = None,
    device=None,
) -> PerplexityResult:
    """dataset: 'wikitext2' 等（经 load_eval_text 解析）；或直接给 text。"""
    if text is None:
        assert dataset is not None, "需要 dataset 或 text 之一"
        text = load_eval_text(dataset)

    device = device or next(model.parameters()).device
    batcher = BlockBatcher(tokenizer, text, block_size=block_size,
                            batch_size=batch_size, max_blocks=max_blocks)
    if len(batcher.blocks) == 0:
        raise ValueError(f"文本太短，不足一个 {block_size}-token 块")

    total_nll, total_tokens = 0.0, 0
    for batch in batcher:
        input_ids = batch.to(device)
        logits = model(input_ids=input_ids).logits
        # shift：用前 n-1 个位置预测第 2..n 个 token
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1)).float()
        shift_labels = input_ids[:, 1:].reshape(-1)
        nll = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        total_nll += nll.item()
        total_tokens += shift_labels.numel()

    mean_nll = total_nll / total_tokens
    return PerplexityResult(
        ppl=math.exp(mean_nll),
        nll=mean_nll,
        n_tokens=total_tokens,
        n_blocks=len(batcher.blocks),
        block_size=block_size,
    )
