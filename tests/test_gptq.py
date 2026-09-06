"""自研 GPTQ 测试：误差反馈应不劣于 RTN；restore 正确。"""

import torch

from liteforge.quant.gptq import _gptq_layer, GPTQQuantizer
from liteforge.quant.rtn import RTNConfig, RTNQuantizer, quantize_tensor
from liteforge.utils.hessian import damp_inverse


def _synthetic(seed=1, n_out=16, n_in=128, n_tokens=2048):
    g = torch.Generator().manual_seed(seed)
    # 让激活有相关性（更像真实校准），误差反馈才有发挥空间
    A = torch.randn(n_in, n_in, generator=g) / n_in
    X = torch.randn(n_tokens, n_in, generator=g) @ A.T
    W = torch.randn(n_out, n_in, generator=g) * 0.3
    return X, W


def test_gptq_output_error_not_worse_than_rtn():
    for seed in range(3):
        X, W = _synthetic(seed)
        H = X.T @ X
        U = torch.linalg.cholesky(damp_inverse(H), upper=True).float()
        Wg = _gptq_layer(W.clone(), U, bits=4, group_size=32, symmetric=False)
        Wr = quantize_tensor(W, 4, 32, False)
        gptq_err = (X @ Wg.T - X @ W.T).pow(2).mean().item()
        rtn_err = (X @ Wr.T - X @ W.T).pow(2).mean().item()
        assert gptq_err <= rtn_err * 1.05, \
            f"seed={seed}: GPTQ {gptq_err:.5f} 应不劣于 RTN {rtn_err:.5f}"


def test_gptq_values_on_grid():
    X, W = _synthetic()
    H = X.T @ X
    U = torch.linalg.cholesky(damp_inverse(H), upper=True).float()
    bits, gs = 4, 32
    Wg = _gptq_layer(W.clone(), U, bits=bits, group_size=gs, symmetric=False)
    # 量化值必然落在 2^bits=16 个格点上 → 每行每组的不同取值数 ≤ 16
    # （尺度由每组起点权重决定，不能用 min/max 反推，故验证取值基数）
    for g0 in range(0, W.shape[1], gs):
        sl = Wg[:, g0:g0 + gs]
        for row in range(sl.shape[0]):
            n_distinct = sl[row].unique().numel()
            assert n_distinct <= 2 ** bits, \
                f"组 {g0} 行 {row} 有 {n_distinct} 个不同取值，超过 2^{bits}"


def test_gptq_quantizer_end_to_end(tiny_model, tiny_tokenizer, sample_text):
    from liteforge.data.text import BlockBatcher
    from liteforge.utils import find_linears
    name, m = find_linears(tiny_model, exclude=("lm_head",))[0]
    orig = m.weight.data.clone()
    q = GPTQQuantizer(tiny_model, RTNConfig(bits=4, group_size=16))
    batcher = BlockBatcher(tiny_tokenizer, sample_text, block_size=64, batch_size=2)
    report = q.quantize_(batcher, max_batches=3)
    assert report["n_layers"] > 0
    assert not torch.equal(m.weight.data, orig)
    q.restore()
    assert torch.equal(m.weight.data, orig)


def test_gptq_act_order():
    """act-order（按激活对角降序处理列 + static groups）：
    含强离群激活列时应不劣于自然列序（文献：高激活列先量化）。"""
    from liteforge.quant.gptq import _gptq_layer
    from liteforge.utils.hessian import damp_inverse
    g = torch.Generator().manual_seed(5)
    n_in, n_out, T = 128, 32, 1024
    A = torch.randn(n_in, n_in, generator=g) / (n_in ** 0.5)
    X = torch.randn(T, n_in, generator=g) @ A.T
    X[:, [10, 60, 100]] *= 20.0                  # 少数高激活列
    W = torch.randn(n_out, n_in, generator=g) * 0.3
    H = X.T @ X
    U = torch.linalg.cholesky(damp_inverse(H), upper=True).float()
    bits, gs = 3, 32                             # 3bit 差距更明显
    Wo = _gptq_layer(W.clone(), U, bits, gs, False, act_order=True,
                     h_diag=H.diagonal().float())
    Wn = _gptq_layer(W.clone(), U, bits, gs, False)
    ref = X @ W.T
    e_o = (X @ Wo.T - ref).pow(2).mean().item()
    e_n = (X @ Wn.T - ref).pow(2).mean().item()
    assert e_o <= e_n * 1.05, f"act-order {e_o:.4f} 应不劣于自然序 {e_n:.4f}"
