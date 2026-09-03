"""OBC 剪枝测试：补偿有效性（合成数据）+ 端到端稀疏度。"""

import torch

from liteforge.prune.obc import OBCPruner, _sequential_zero, _sequential_zero_dynamic
from liteforge.prune.base import PruneConfig, per_row_topk_mask
from liteforge.utils.hessian import damp_inverse
from liteforge.utils import find_linears


def _synthetic(seed=0, n_out=8, n_in=64, n_tokens=512):
    """带相关性的合成激活：iid 随机 X 的 Hessian 近似对角，列间无相关性，
    误差补偿无从发力（补偿量≈0），检验不出 OBC 的价值。
    真实激活天然相关（GPTQ/SparseGPT 论文设定相同）。"""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n_in, n_in, generator=g) / (n_in ** 0.5)
    X = torch.randn(n_tokens, n_in, generator=g) @ A.T
    W = torch.randn(n_out, n_in, generator=g) * 0.5
    return X, W


def _recon_err(X, W, W_ref):
    """输出重构误差：||XWᵀ − XW_refᵀ||²（必须相对参考输出计算）。"""
    return (X @ W.T - X @ W_ref.T).pow(2).mean().item()


def test_obc_compensation_beats_naive_masking():
    """同一删除掩码下，OBC 补偿后的重构误差应严格小于朴素置零。"""
    for seed in range(3):
        X, W = _synthetic(seed)
        H = X.T @ X
        Hinv = damp_inverse(H)
        score = W.pow(2) / Hinv.diagonal().clamp(min=1e-12).unsqueeze(0)
        mask = per_row_topk_mask(score, 0.25)
        naive = (W * mask).float()
        U = torch.linalg.cholesky(Hinv, upper=True)
        obc = _sequential_zero(W.clone(), U.float(), mask)
        naive_err = _recon_err(X, naive, W)
        obc_err = _recon_err(X, obc, W)
        assert obc_err < naive_err, \
            f"seed={seed}: OBC {obc_err:.5f} 应优于朴素 {naive_err:.5f}"


def test_obc_end_to_end(tiny_model, tiny_tokenizer, sample_text):
    from liteforge.data.text import BlockBatcher
    pruner = OBCPruner(tiny_model, PruneConfig(sparsity=0.5))
    batcher = BlockBatcher(tiny_tokenizer, sample_text, block_size=64, batch_size=2)
    result = pruner.run(calib_batches=batcher, max_batches=4)
    assert abs(result.overall_sparsity - 0.5) < 0.02
    ids = torch.randint(0, 47, (1, 32))
    assert tiny_model(input_ids=ids).logits.isfinite().all()


def test_obc_dynamic_mask(tiny_model, tiny_tokenizer, sample_text):
    """SparseGPT 忠实版：块级动态重评分，稀疏度精确到目标。"""
    from liteforge.data.text import BlockBatcher
    pruner = OBCPruner(tiny_model, PruneConfig(sparsity=0.5), mask_mode="dynamic")
    batcher = BlockBatcher(tiny_tokenizer, sample_text, block_size=64, batch_size=2)
    result = pruner.run(calib_batches=batcher, max_batches=4)
    assert abs(result.overall_sparsity - 0.5) < 0.02
    ids = torch.randint(0, 47, (1, 32))
    assert tiny_model(input_ids=ids).logits.isfinite().all()


def test_obc_dynamic_beats_naive_zeroing_synthetic():
    """相关数据上，动态重评分 + 补偿应严格优于朴素置零（无补偿）。"""
    from liteforge.utils.hessian import damp_inverse
    g = torch.Generator().manual_seed(7)
    n_in, n_out, T = 64, 8, 512
    A = torch.randn(n_in, n_in, generator=g) / (n_in ** 0.5)
    X = torch.randn(T, n_in, generator=g) @ A.T
    W = torch.randn(n_out, n_in, generator=g) * 0.5
    Hinv = damp_inverse(X.T @ X)
    U = torch.linalg.cholesky(Hinv, upper=True).float()
    score = W.pow(2) / Hinv.diagonal().clamp(min=1e-12).unsqueeze(0)
    naive = (W * per_row_topk_mask(score, 0.25)).float()
    dyn = _sequential_zero_dynamic(W.clone(), U, 0.25)
    ref = X @ W.T
    e_dyn = (X @ dyn.T - ref).pow(2).mean().item()
    e_naive = (X @ naive.T - ref).pow(2).mean().item()
    assert e_dyn < e_naive, f"dynamic {e_dyn:.5f} 应优于朴素置零 {e_naive:.5f}"


def test_score_dry_ranks_layers(tiny_model, tiny_tokenizer, sample_text):
    """敏感度评分：应返回所有层且损失为非负数。"""
    from liteforge.data.text import BlockBatcher
    pruner = OBCPruner(tiny_model, PruneConfig(sparsity=0.5))
    batcher = BlockBatcher(tiny_tokenizer, sample_text, block_size=64, batch_size=2)
    sens = pruner.score_dry(batcher, max_batches=2)
    assert len(sens) == len(find_linears(tiny_model, exclude=("lm_head", "embed_out")))
    assert all(v >= 0 for v in sens.values())
