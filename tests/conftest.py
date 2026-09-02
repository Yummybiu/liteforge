"""测试夹具：微型 Llama + 字符级 mock tokenizer，全部离线可跑。"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VOCAB_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?-()=+\n"  # 47 chars


class TinyTokenizer:
    """字符级分词器，接口对齐 HF tokenizer 的最小子集。"""

    def __init__(self):
        self.itos = list(VOCAB_CHARS)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self._eos = len(self.itos)
        self._pad = self._eos

    @property
    def vocab_size(self):
        return len(self.itos) + 1

    @property
    def eos_token(self):
        return "\n"

    @property
    def pad_token(self):
        return "\n"

    @property
    def eos_token_id(self):
        return self._eos

    def encode(self, text):
        unk = 0
        return [self.stoi.get(c, unk) for c in text]

    def __call__(self, text, add_special_tokens=False, return_tensors=None, **kw):
        ids = self.encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


@pytest.fixture()  # function 级：剪枝测试会修改权重，必须每个测试用全新模型
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(42)
    cfg = LlamaConfig(
        vocab_size=48, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256,
    )
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture()
def tiny_tokenizer():
    return TinyTokenizer()


@pytest.fixture()
def sample_text():
    # ~5000+ 字符，保证能切出若干个 64-token 块
    unit = "the quick brown fox jumps over the lazy dog. neural nets are fun, "
    return unit * 80
