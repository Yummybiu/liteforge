"""models.py 回归：显式 dtype 加载路径（models.py 是唯一零测试的运行时路径，
torch_dtype/dtype 关键字兼容问题曾在此静默埋雷——审查 P0）。"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.fixture()
def tiny_saved(tmp_path):
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(
        vocab_size=48, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128))
    m.save_pretrained(tmp_path)
    return str(tmp_path)


def test_load_explicit_dtype(tiny_saved, monkeypatch):
    """显式 dtype 加载 + 关键字兼容（v4: torch_dtype / v5: dtype）。"""
    import transformers
    from tests.conftest import TinyTokenizer

    class _FakeAT:
        @staticmethod
        def from_pretrained(path, **kw):
            return TinyTokenizer()

    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeAT)

    from liteforge.models import load_model_and_tokenizer
    model, tok = load_model_and_tokenizer(tiny_saved, dtype="fp32")
    assert next(model.parameters()).dtype == torch.float32
    dev = next(model.parameters()).device
    out = model(input_ids=torch.randint(0, 47, (1, 8)).to(dev))
    assert out.logits.isfinite().all()


def test_load_auto_path(tiny_saved, monkeypatch):
    import transformers
    from tests.conftest import TinyTokenizer

    class _FakeAT:
        @staticmethod
        def from_pretrained(path, **kw):
            return TinyTokenizer()

    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeAT)

    from liteforge.models import load_model_and_tokenizer
    model, tok = load_model_and_tokenizer(tiny_saved, dtype="auto")
    dev = next(model.parameters()).device
    out = model(input_ids=torch.randint(0, 47, (1, 8)).to(dev))
    assert out.logits.isfinite().all()
