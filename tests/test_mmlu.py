"""MMLU 评测器测试：mock 模型下验证似然评分路径与分桶统计。"""

import pytest
import torch


class _To(dict):
    """支持 .to() 的 dict（对齐 HF batch 输出）。"""

    def to(self, device):
        return self


class _MockTok:
    """prompt → 递增 id；" A".." D" → 单 token 40-43。"""

    LETTERS = {" A": [40], " B": [41], " C": [42], " D": [43]}

    def __init__(self):
        self._n = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=False, **kw):
        if text in self.LETTERS:
            ids = list(self.LETTERS[text])
        else:
            ids = [min(self._n + i, 9) for i in range(8)]
            self._n += 1
        return _To({"input_ids": torch.tensor([ids])})


@pytest.fixture()
def mock_env(monkeypatch):
    """4 题固定 mock：前 3 题正确答案是 A，第 4 题是 C；模型 logit 偏置可控。"""
    items = [
        {"question": f"q{i}", "choices": ["x", "y", "z", "w"], "answer": a, "subject": s}
        for i, (a, s) in enumerate([(0, "college_mathematics"), (0, "philosophy"),
                                    (0, "high_school_biology"), (2, "jurisprudence")])
    ]
    import liteforge.eval.mmlu as m
    monkeypatch.setattr(m, "load_mmlu", lambda n=500, seed=42: items)

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    class _MockModel:
        """最后位置 logit：给 letter 40-43 加偏置 bias。"""

        def __init__(self, bias):
            self.bias = bias
            self._p = torch.nn.Parameter(torch.zeros(1))  # 满足 .parameters() 协议

        def parameters(self):
            return iter([self._p])

        def __call__(self, input_ids, **kw):
            logits = torch.zeros(1, input_ids.shape[1], 44)
            for i, ch in enumerate("ABCD"):
                logits[0, -1, 40 + i] = self.bias.get(ch, 0.0)
            return _Out(logits)

    return m, _MockModel


def test_mmlu_single_token_scoring_and_buckets(mock_env):
    m, Model = mock_env
    # 偏置让 A(40) 最强 → 前 3 题对，第 4 题(answer=C)错
    model = Model({"A": 2.0, "B": 0.0, "C": 1.0, "D": 0.0})

    class _Tok(_MockTok):
        pass

    stats = m.evaluate_mmlu(model, _Tok(), n=4)
    assert stats["n"] == 4
    assert abs(stats["accuracy"] - 0.75) < 1e-6
    # MMLU 官方分类：college_mathematics 和 high_school_biology 都是 STEM
    assert stats["by_bucket"]["stem"]["n"] == 2
    assert stats["by_bucket"]["humanities"]["n"] == 2
    assert "other" not in stats["by_bucket"]


def test_mmlu_scoring_respects_logits(mock_env):
    m, Model = mock_env
    # 反转偏置：D(43) 最强 → 全错
    model = Model({"A": 0.0, "B": 0.0, "C": 0.0, "D": 2.0})

    class _Tok(_MockTok):
        pass

    stats = m.evaluate_mmlu(model, _Tok(), n=4)
    assert stats["accuracy"] == 0.0


def test_format_prompt_structure():
    ex = {"question": "2+2=?", "choices": ["3", "4", "5", "6"], "answer": 1,
          "subject": "elementary_mathematics"}
    p = m_format(ex)
    assert "A. 3" in p and "D. 6" in p and "Answer:" in p
    assert "elementary mathematics" in p  # subject 下划线转空格


def m_format(ex):
    from liteforge.eval.mmlu import format_prompt
    return format_prompt(ex)
