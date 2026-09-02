"""Wanda：用激活范数加权权重大小的免训练剪枝。

论文：*A Simple and Effective Pruning Approach for Large Language Models*
(Sun et al., NeurIPS 2023, arXiv:2306.11695)。

核心打分：score_ij = |W_ij| · ||X_j||
其中 ||X_j|| 是第 j 个输入特征在校准数据上的 RMS（对 token 求均值的平方和开根）。
直觉：权重再大，若它作用的输入通道几乎不活跃，删掉也无妨；反之小幅权重
乘上高活跃通道也可能致命。

本实现从零构建（不依赖官方代码）：
- 前向预钩子按层累计输入平方和（float64 累加防溢出）；
- act_norm_j = sqrt(x_sum_j / n_tokens)；
- 逐输出行 top-k 掩码，支持非结构化与 2:4 半结构化。
"""

import logging
import math

import torch

from .base import BasePruner, group_2to4_mask, per_row_topk_mask

logger = logging.getLogger(__name__)


class WandaPruner(BasePruner):
    method_name = "wanda"

    def compute_scores(self, layer_inputs) -> dict:
        if not layer_inputs:
            raise RuntimeError(
                "Wanda 需要校准数据：请传入 calib_batches（见 CLI --calib-size）"
            )
        scores = {}
        for name, m in self.linears:
            if name not in layer_inputs:
                raise KeyError(f"层 {name} 未采集到激活统计")
            x_sum, n = layer_inputs[name]
            if n == 0:
                raise RuntimeError(f"层 {name} 校准 token 数为 0")
            act_norm = (x_sum / n).sqrt().to(torch.float32)      # [n_in]
            w = m.weight.data.detach().to(torch.float32)          # [n_out, n_in]
            scores[name] = w.abs() * act_norm.unsqueeze(0)
            logger.debug(
                "%s: act_norm mean=%.4f max=%.4f", name,
                act_norm.mean().item(), act_norm.max().item(),
            )
        return scores

    def run(self, calib_batches=None, max_batches: int = 32):
        if self.config.structure == "2:4":
            # 2:4 只看组内相对大小，sparsity 固定 0.5
            self.config.sparsity = 0.5
        return super().run(calib_batches=calib_batches, max_batches=max_batches)
