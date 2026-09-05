"""投机解码测试：精确性定理（输出 == target 纯贪心）与统计一致性。"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from liteforge.speculative import greedy_generate_baseline, speculative_generate


def _tiny(seed=0, hidden=32):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=48, hidden_size=hidden, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256)
    return LlamaForCausalLM(cfg).eval()


def test_same_model_accepts_all_and_matches_greedy():
    """draft == target：接受率应为 1，输出逐 token 等于纯贪心。"""
    target = _tiny(0)
    ids = torch.randint(0, 47, (1, 8))
    ref = greedy_generate_baseline(target, ids, max_new_tokens=24)
    out = speculative_generate(target, target, ids, max_new_tokens=24, k=4)
    assert torch.equal(out["ids"], ref["ids"]), "同模型投机输出必须与贪心一致"
    assert out["stats"]["acceptance_rate"] == 1.0
    assert out["stats"]["tokens_generated"] == 24
    # 有效性：每 target 前应产出 k+1 个 token（全接受 + 红利）
    assert out["stats"]["tokens_per_target_forward"] > 4.0


def test_exactness_with_different_draft():
    """draft ≠ target：输出仍必须逐 token 等于 target 纯贪心（精确性定理）。"""
    for seed in range(3):
        target = _tiny(0, hidden=32)
        draft = _tiny(seed + 10, hidden=48)
        ids = torch.randint(0, 47, (1, 8))
        ref = greedy_generate_baseline(target, ids, max_new_tokens=20)
        out = speculative_generate(target, draft, ids, max_new_tokens=20, k=4)
        assert torch.equal(out["ids"], ref["ids"]), \
            f"seed={seed}: 投机输出偏离了 target 贪心——精确性被破坏！"
        assert 0.0 <= out["stats"]["acceptance_rate"] <= 1.0
        assert out["stats"]["target_forwards"] * 4 >= out["stats"]["tokens_generated"]


def test_k_sweep_consistency():
    """k=1/2/8 的输出都应与贪心一致（k 不影响正确性，只影响效率）。"""
    target = _tiny(0)
    draft = _tiny(3, hidden=40)
    ids = torch.randint(0, 47, (1, 8))
    ref = greedy_generate_baseline(target, ids, max_new_tokens=16)["ids"]
    for k in (1, 2, 8):
        out = speculative_generate(target, draft, ids, max_new_tokens=16, k=k)
        assert torch.equal(out["ids"], ref), f"k={k} 破坏精确性"
