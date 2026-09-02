"""剪枝模块测试：稀疏度精确性、2:4 结构正确性、剪枝后前向可运行。"""

import torch

from liteforge.data.text import BlockBatcher
from liteforge.prune import MagnitudePruner, PruneConfig, WandaPruner, layer_sparsity
from liteforge.utils import find_linears


def _linears(model):
    return find_linears(model, exclude=("lm_head", "embed_out"))


def test_magnitude_prune_reaches_target_sparsity(tiny_model):
    cfg = PruneConfig(sparsity=0.5)
    pruner = MagnitudePruner(tiny_model, cfg)
    result = pruner.run()
    assert abs(result.overall_sparsity - 0.5) < 0.01
    for name, m in _linears(tiny_model):
        sp = layer_sparsity(m)
        assert abs(sp - 0.5) < 0.01, f"{name} 稀疏度 {sp} 偏离目标"
    # 前向仍可运行
    ids = torch.randint(0, 47, (1, 32))
    out = tiny_model(input_ids=ids)
    assert out.logits.shape[-1] == 48


def test_wanda_prune_with_calibration(tiny_model, tiny_tokenizer, sample_text):
    cfg = PruneConfig(sparsity=0.5)
    pruner = WandaPruner(tiny_model, cfg)
    batcher = BlockBatcher(tiny_tokenizer, sample_text, block_size=64, batch_size=2)
    result = pruner.run(calib_batches=batcher, max_batches=4)
    assert abs(result.overall_sparsity - 0.5) < 0.01
    assert len(result.layer_reports) == len(_linears(tiny_model))
    ids = torch.randint(0, 47, (1, 32))
    assert tiny_model(input_ids=ids).logits.isfinite().all()


def test_wanda_2to4_structure(tiny_model, tiny_tokenizer, sample_text):
    cfg = PruneConfig(sparsity=0.5, structure="2:4")
    pruner = WandaPruner(tiny_model, cfg)
    batcher = BlockBatcher(tiny_tokenizer, sample_text, block_size=64, batch_size=2)
    result = pruner.run(calib_batches=batcher, max_batches=2)
    assert result.overall_sparsity == 0.5  # 2:4 恒为 50%
    for name, m in _linears(tiny_model):
        w = m.weight.data
        n_out, n_in = w.shape
        zeros_per_group = (w == 0).float().view(n_out, n_in // 4, 4).sum(-1)
        assert torch.all(zeros_per_group == 2), f"{name} 不满足 2:4 结构"


def test_wanda_requires_calibration(tiny_model):
    pruner = WandaPruner(tiny_model, PruneConfig(sparsity=0.5))
    try:
        pruner.run()
        raise AssertionError("无校准数据时应报错")
    except RuntimeError:
        pass
