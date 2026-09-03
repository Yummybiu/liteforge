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


def _sequential_zero_dynamic(W: torch.Tensor, U: torch.Tensor, sparsity: float,
                             blocksize: int = 128) -> torch.Tensor:
    """SparseGPT 忠实版：block 级动态重评分 + 顺序误差补偿。

    静态预选掩码的缺陷：w²/H⁻¹diag 打分会集中删除"被强保护的列簇"，
    但簇内保护者彼此删除后保护失效，高稀疏度下实际损失远超单列估计。
    官方解法：每个 block 处理时，用**当前已补偿**的权重在该 block 内
    重新评分、重新决定删留（跨块动态、块内定秩）。
    """
    n_out, n_in = W.shape
    n_keep_row = max(1, int(round(n_in * (1.0 - sparsity))))
    diag = U.diagonal().clamp(min=1e-12)
    for b0 in range(0, n_in, blocksize):
        b1 = min(b0 + blocksize, n_in)
        # 块内定秩：用当前权重评分，每行保留块内 top-n_keep_blk
        blk_len = b1 - b0
        n_keep_blk = max(1, int(round(blk_len * (1.0 - sparsity))))
        Wblk = W[:, b0:b1]
        score_blk = Wblk * Wblk / (diag[b0:b1] ** 2).unsqueeze(0)
        mask_blk = per_row_topk_mask(score_blk, 1.0 - n_keep_blk / blk_len)
        block_err = torch.zeros(blk_len, n_out, device=W.device)
        for j in range(b0, b1):
            d = U[j, j]
            w = W[:, j]
            removed = mask_blk[:, j - b0] == 0
            if not removed.any():
                continue
            err = torch.zeros_like(w)
            err[removed] = w[removed] / d
            W[removed, j] = 0
            if j + 1 < b1:
                W[removed, j + 1:b1] -= err[removed].unsqueeze(1) * U[j, j + 1:b1].unsqueeze(0)
            block_err[j - b0] = err
        if b1 < n_in:
            W[:, b1:] -= block_err.T @ U[b0:b1, b1:]
    return W


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
                 blocksize: int = 128, mask_mode: str = "dynamic"):
        super().__init__(model, config)
        self.percdamp = percdamp
        self.blocksize = blocksize
        self.mask_mode = mask_mode  # dynamic=SparseGPT 忠实版 | static=静态预选（消融用）

    def compute_scores(self, layer_inputs) -> dict:  # 不走基类的 Wanda 式打分
        raise NotImplementedError

    def _layer_U(self, H, percdamp=None):
        """单层 Hessian → float32 Cholesky 上三角（用完即弃，避免全模型驻留 U）。"""
        Hinv = damp_inverse(H, percdamp or self.percdamp)
        return Hinv, torch.linalg.cholesky(Hinv, upper=True).to(torch.float32)

    def prepare(self, calib_batches, max_batches: int = 32):
        """采集全模型 Hessian（运行期驻留的是 H 本身，U 逐层现算现放）。"""
        H_dict = collect_xtx(self.model, self.linears, calib_batches, max_batches)
        return H_dict

    def run(self, calib_batches=None, max_batches: int = 32) -> PruneResult:
        assert calib_batches is not None, "OBC 需要校准数据"
        t0 = time.time()
        logger.info("[obc] 采集校准 Hessian (XᵀX) + 逐层阻尼求逆 ...")
        H_dict = self.prepare(calib_batches, max_batches)

        layer_reports, pruned, total = [], 0, 0
        for name, m in self.linears:
            s = layer_sparsity_target(self.config.sparsity, name)
            Hinv, U = self._layer_U(H_dict.pop(name))
            W = m.weight.data.to(torch.float32)
            if self.mask_mode == "dynamic":
                Wc = _sequential_zero_dynamic(W, U, s, self.blocksize)
            else:
                Hdiag = Hinv.diagonal().clamp(min=1e-12)
                score = (W.to(torch.float64) * W.to(torch.float64) / Hdiag.unsqueeze(0)).to(torch.float32)
                mask = per_row_topk_mask(score, s)          # 1=keep
                Wc = _sequential_zero(W, U, mask, self.blocksize)
            m.weight.data.copy_(Wc.to(m.weight.data.dtype))
            del Hinv, U, Wc
            sp = (m.weight.data == 0).float().mean().item()
            pruned += int((m.weight.data == 0).sum().item())
            total += m.weight.data.numel()
            layer_reports.append({"layer": name, "sparsity": round(sp, 4)})
        del H_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    def score_dry(self, calib_batches, max_batches: int = 32) -> dict:
        """不落刀的敏感度评分：每层 OBS 理论删除损失 = 被删项 score 之和。

        返回 {layer_name: estimated_loss}（SGMix 的敏感度输入）。
        逐层处理、用完释放（3B 级模型的显存安全）。
        """
        assert calib_batches is not None
        H_dict = self.prepare(calib_batches, max_batches)
        sens = {}
        for name, m in self.linears:
            s = layer_sparsity_target(self.config.sparsity, name)
            Hinv = damp_inverse(H_dict.pop(name), self.percdamp)
            Hdiag = Hinv.diagonal().clamp(min=1e-12).to(torch.float32)
            w = m.weight.data.detach().to(torch.float32)
            score = w * w / Hdiag.unsqueeze(0)
            mask = per_row_topk_mask(score, s)
            sens[name] = float((score * (1 - mask)).sum().item())
            del Hinv, score, mask
        del H_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return sens
