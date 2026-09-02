"""幅度剪枝（Magnitude Pruning）：最经典的基线方法。

逐输出行保留 |W| 最大的权重。作为激活感知方法（Wanda 等）的对照组，
幅度剪枝在不看任何校准数据的情况下通常质量损失明显更大——这正是
实验矩阵里必须保留它的原因。
"""

import torch

from .base import BasePruner


class MagnitudePruner(BasePruner):
    method_name = "magnitude"

    def compute_scores(self, layer_inputs) -> dict:
        scores = {}
        for name, m in self.linears:
            scores[name] = m.weight.data.detach().abs().to(torch.float32)
        return scores
