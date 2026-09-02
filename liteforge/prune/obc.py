"""OBC 剪枝：Optimal Brain Compression 家族（二阶误差补偿剪枝）。

借鉴关系（诚实声明）：算法框架来自 Frantar & Alistarh 的 OBC/SparseGPT 论文
（NeurIPS 2022 / ICML 2023），本实现从零手写，与 WandaPruner 共享工程链路，
用于回答"免补偿的 Wanda 和带误差补偿的 OBS 族差多少"。

数学：线性层 y = Wx，校准输出的 L2 损失 Hessian 为 H = 2E[xxᵀ]。
固定删除集合 S（每行各自选择）后，剩余权重的最优补偿（多权重 OBS）：

    Δw_K = −(H⁻¹)_{K,S} ((H⁻¹)_{S,S})⁻¹ w_S

逐列顺序形式等价于上式（SparseGPT 定理），且只需 H⁻¹ 的 Cholesky 上三角 U：

    对第 j 列的已删行：err = w_j / U_jj，后续列 W[:, j+1:] −= err ⊗ U[j, j+1:]

删除集合的打分（选谁删）用 OBS 单权重损失：score_ij = w_ij² / (H⁻¹)_jj
——这正是 Wanda 分数 |w|·||x|| 的二阶严格版（‖x_j‖² ≈ H_jj 对角元）。
"""

import logging
import time

import torch

from .base import BasePruner, PruneConfig, PruneResult, layer_sparsity_target, per_row_topk_mask
from ..utils import find_linears
from ..utils.hessian import collect_xtx, damp_inverse

logger = logging.getLogger(__name__)


def _sequential_zero(W: torch.Tensor, U: torch.Tensor, keep_mask: torch.Tensor,
                     blocksize: int = 128) -> torch.Tensor:
    """顺序误差补偿：已删行把误差按 Cholesky 因子传播给该行后续列。

    W: [out, in] float32（会被修改，请传副本）；U: H⁻¹ 的 upper-Cholesky；
    keep_mask: 1=保留，0=删除。
    """
    n_in = W.shape[1]
    for b0 in range(0, n_in, blocksize):
        b1 = min(b0 + blocksize, n_in)
        for j in range(b0, b1):
            d = U[j, j]
            w = W[:, j]
            removed = keep_mask[:, j] == 0
            if not removed.any():
                continue
            err = torch.zeros_like(w)
            err[removed] = w[removed] / d
            W[removed, j] = 0
            if j + 1 < n_in:
                W[removed, j + 1:] -= err[removed].unsqueeze(1) * U[j, j + 1:].unsqueeze(0)
    return W


class OBCPruner(BasePruner):
    method_name = "obc"

    def __init__(self, model, config: PruneConfig, percdamp: float = 0.01,
                 blocksize: int = 128):
        super().__init__(model, config)
        self.percdamp = percdamp
        self.blocksize = blocksize

    def compute_scores(self, layer_inputs) -> dict:  # 不走基类的 Wanda 式打分
        raise NotImplementedError

    def _scores_from_hessian(self, H_dict):
        scores = {}
        for name, m in self.linears:
            Hinv = damp_inverse(H_dict[name], self.percdamp)
            Hdiag = Hinv.diagonal().clamp(min=1e-12)
            w = m.weight.data.detach().to(torch.float64)
            scores[name] = (w * w / Hdiag.unsqueeze(0)).to(torch.float32)
        return scores

    def run(self, calib_batches=None, max_batches: int = 32) -> PruneResult:
        assert calib_batches is not None, "OBC 需要校准数据"
        t0 = time.time()
        logger.info("[obc] 采集校准 Hessian (XᵀX) + 阻尼求逆 ...")
        scores = self.prepare(calib_batches, max_batches)

        layer_reports, pruned, total = [], 0, 0
        for name, m in self.linears:
            s = layer_sparsity_target(self.config.sparsity, name)
            W = m.weight.data.to(torch.float32)
            mask = per_row_topk_mask(scores[name], s)          # 1=keep
            Wc = _sequential_zero(W.clone(), self._U[name], mask, self.blocksize)
            m.weight.data.copy_(Wc.to(m.weight.data.dtype))
            sp = (m.weight.data == 0).float().mean().item()
            pruned += int((m.weight.data == 0).sum().item())
            total += m.weight.data.numel()
            layer_reports.append({"layer": name, "sparsity": round(sp, 4)})

        result = PruneResult(
            method=self.method_name,
            sparsity_target=self.config.sparsity if isinstance(self.config.sparsity, float) else -1,
            structure=self.config.structure,
            overall_sparsity=round(pruned / max(total, 1), 4),
            duration_s=time.time() - t0,
            layer_reports=layer_reports,
        )
        logger.info("[obc] 完成：整体稀疏度 %.2f%%，耗时 %.1fs",
                    result.overall_sparsity * 100, result.duration_s)
        return result

    def prepare(self, calib_batches, max_batches: int = 32):
        """分离出 Hessian 准备步骤（score_only 与 run 共用；SGMix 用）。"""
        H_dict = collect_xtx(self.model, self.linears, calib_batches, max_batches)
        scores = self._scores_from_hessian(H_dict)
        self._U = {}
        for name, _ in self.linears:
            Hinv = damp_inverse(H_dict[name], self.percdamp)
            self._U[name] = torch.linalg.cholesky(Hinv, upper=True).to(torch.float32)
        return scores

    def score_dry(self, calib_batches, max_batches: int = 32) -> dict:
        """不落刀的敏感度评分：每层 OBS 理论删除损失 = 被删项 score 之和。

        返回 {layer_name: estimated_loss}（SGMix 的敏感度输入）。
        """
        assert calib_batches is not None
        scores = self.prepare(calib_batches, max_batches)
        sens = {}
        for name, _ in self.linears:
            s = layer_sparsity_target(self.config.sparsity, name)
            mask = per_row_topk_mask(scores[name], s)
            sens[name] = float((scores[name] * (1 - mask)).sum().item())
        return sens
