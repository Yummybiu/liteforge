"""困惑度与报告聚合测试。"""

import math

from liteforge.data.text import BlockBatcher
from liteforge.eval import compute_perplexity
from liteforge.report import collect_results, to_markdown
from liteforge.report.plots import effective_bits


def test_perplexity_finite_and_positive(tiny_model, tiny_tokenizer, sample_text):
    res = compute_perplexity(
        tiny_model, tiny_tokenizer, text=sample_text,
        block_size=64, batch_size=2, max_blocks=3,
    )
    assert math.isfinite(res.ppl) and res.ppl > 1.0
    assert res.n_tokens == 3 * 63  # 3 块，每块预测 63 个 token


def test_block_batcher_shapes(tiny_tokenizer, sample_text):
    b = BlockBatcher(tiny_tokenizer, sample_text, block_size=64, batch_size=2)
    batches = list(b)
    assert all(t.shape == (2, 64) for t in batches)
    assert len(b) * 2 <= len(b.blocks) + 1  # 不超过块数（向上取整）


def test_collect_and_markdown(tmp_path):
    import json
    r1 = {"task": "eval-ppl", "model": "Qwen/Qwen2.5-0.5B", "method": "dense",
          "params": {}, "metrics": {"ppl": 9.5}, "env": {}}
    r2 = {"task": "prune", "model": "Qwen/Qwen2.5-0.5B", "method": "wanda",
          "params": {"sparsity": 0.5}, "metrics": {"ppl": 24.3}, "env": {}}
    for rec, name in [(r1, "a.json"), (r2, "b.json")]:
        with open(tmp_path / name, "w", encoding="utf-8") as f:
            json.dump(rec, f)
    records = collect_results(str(tmp_path))
    assert len(records) == 2
    md = to_markdown(records)
    assert "wanda" in md and "9.5" in md and "24.3" in md


def test_effective_bits_semantics():
    assert effective_bits({"method": "dense", "params": {}}) == 16.0
    assert effective_bits({"method": "rtn", "params": {"bits": 4}}) == 4.0
    assert effective_bits({"method": "wanda", "params": {"sparsity": 0.5}}) == 8.0
    assert effective_bits({"method": "wanda", "params": {"sparsity": 0.75}}) == 4.0
