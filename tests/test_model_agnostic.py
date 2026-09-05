"""模型无关性测试：框架承诺支持任意 HF CausalLM——用三种架构族证明。

覆盖三种代表性结构：Llama 系（Qwen/LLaMA）、GPTNeoX 系（融合 QKV）、
Mistral 系（滑动窗口注意力）。find_linears 的排除表（lm_head/embed_out）
必须对全部架构正确工作。
"""

import pytest
import torch
from transformers import (GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig,
                          LlamaForCausalLM, MistralConfig, MistralForCausalLM)

VOCAB = 48


def _ids():
    return torch.randint(0, VOCAB - 1, (1, 16))


def _tiny(cls, cfg_cls, **kw):
    torch.manual_seed(0)
    cfg = cfg_cls(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=128, **kw)
    if cls is GPTNeoXForCausalLM:  # NeoX 无 num_key_value_heads
        del kw
        cfg = cfg_cls(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      max_position_embeddings=128)
    return cls(cfg).eval()


ARCHS = [
    ("llama", LlamaForCausalLM, LlamaConfig),
    ("gptneox", GPTNeoXForCausalLM, GPTNeoXConfig),
    ("mistral", MistralForCausalLM, MistralConfig),
]


@pytest.mark.parametrize("name,cls,cfg_cls", ARCHS)
def test_find_linears_across_architectures(name, cls, cfg_cls):
    from liteforge.utils import find_linears
    model = _tiny(cls, cfg_cls)
    linears = find_linears(model, exclude=("lm_head", "embed_out"))
    assert linears, f"{name}: 未找到任何 Linear"
    # 输出头必须被排除
    assert all("lm_head" not in n and "embed_out" not in n for n, _ in linears)
    model(input_ids=_ids())  # 前向通过


@pytest.mark.parametrize("name,cls,cfg_cls", ARCHS)
def test_wanda_across_architectures(name, cls, cfg_cls):
    from liteforge.data.text import BlockBatcher
    from liteforge.prune import WandaPruner
    from liteforge.prune.base import PruneConfig

    model = _tiny(cls, cfg_cls)
    tok = _CharTok()
    batcher = BlockBatcher(tok, "the quick brown fox jumps over the lazy dog. " * 40,
                           block_size=64, batch_size=2)
    result = WandaPruner(model, PruneConfig(sparsity=0.5)).run(
        calib_batches=batcher, max_batches=2)
    assert abs(result.overall_sparsity - 0.5) < 0.01
    assert model(input_ids=_ids()).logits.isfinite().all()


@pytest.mark.parametrize("name,cls,cfg_cls", ARCHS)
def test_rtn_across_architectures(name, cls, cfg_cls):
    from liteforge.quant import RTNConfig, RTNQuantizer

    model = _tiny(cls, cfg_cls)
    q = RTNQuantizer(model, RTNConfig(bits=4, group_size=16))
    report = q.quantize_()
    assert report["n_layers"] > 0
    assert model(input_ids=_ids()).logits.isfinite().all()
    q.restore()


class _CharTok:
    """字符级 tokenizer（与 conftest 一致的最小实现，避免循环导入 fixture）。"""

    LETTERS = "abcdefghijklmnopqrstuvwxyz ."
    eos_token = "\n"
    pad_token = "\n"
    eos_token_id = 29
    vocab_size = 30

    def __call__(self, text, add_special_tokens=False, **kw):
        stoi = {c: i for i, c in enumerate(self.LETTERS)}
        return {"input_ids": [stoi.get(c, 0) for c in text]}
