"""自研 GPTQ：逐列量化 + Hessian 误差反馈（从零实现）。

借鉴关系：算法出自 Frantar et al., *GPTQ* (ICLR 2023)；本实现从零手写，
与 quant/rtn.py 的 RTN 共享 group_params（分组 scale/zero），
与 OBC 剪枝器共享 Hessian 引擎（utils/hessian.py）。

与 RTN 的本质区别：RTN 对每个权重独立四舍五入，舍入误差是无人管理的噪声；
GPTQ 在量化第 j 列时，把该列的舍入误差按 Hessian 度量传播给**尚未量化**的
后续列（通过 H⁻¹ 的 Cholesky 上三角 U），等于用二阶信息"预补偿"后面的权重
——同样的位宽，误差被吃掉了。分组 scale 在每组起点按当前（已补偿）权重
重新计算（官方同款行为）。
"""

import logging
import time

import torch

from ..utils import find_linears
from ..utils.hessian import collect_xtx, damp_inverse
from .rtn import group_params

logger = logging.getLogger(__name__)


def _gptq_layer(W: torch.Tensor, U: torch.Tensor, bits: int, group_size: int,
                symmetric: bool, blocksize: int = 128,
                act_order: bool = False, h_diag: torch.Tensor | None = None) -> torch.Tensor:
    """对单层权重做 GPTQ 量化（W: [out, in] float32，返回伪量化副本）。

    act_order=True（act-order/desc_act，官方代码特性、论文正文未含）：
    按激活对角量降序处理列——高激活列先量化、误差反馈给低激活列。
    需配 static groups（各组 scale/zero 由原始列序预计算），
    否则重排后的"组起点重估"会混入错误列。
    """
    n_out, n_in = W.shape
    qmax = 2 ** (bits - 1) - 1
    qpeak = 2 ** bits - 1
    qmin = -qmax if symmetric else 0   # 对称网格 [-qmax, qmax]，非对称 [0, qpeak]
    Wq = torch.empty_like(W)

    if act_order:
        assert h_diag is not None, "act_order 需要传入 H 对角线"
        order = torch.argsort(h_diag.to(torch.float64), descending=True)
        Wp = W[:, order]
        Up = U[order][:, order]
        # static groups：原始列序下预计算每列的 scale/zero
        S_all, Z_all = group_params(W, bits, group_size, symmetric)  # [n_out, n_in]
        Wq_p = torch.empty_like(Wp)
        Sslice = Zslice = None
        slice_start = 0
        for j in range(n_in):
            s = S_all[:, order[j]]
            z = Z_all[:, order[j]]
            d = Up[j, j]
            w = Wp[:, j]
            q = torch.clamp(torch.round(w / s) + z, qmin, qpeak)
            dq = (q - z) * s
            err = (w - dq) / d
            Wq_p[:, j] = dq
            if j + 1 < n_in:
                Wp[:, j + 1:] -= err.unsqueeze(1) * Up[j, j + 1:].unsqueeze(0)
        # 反重排回原始列序
        inv = torch.empty_like(order)
        inv[order] = torch.arange(n_in, device=order.device)
        Wq[:, order] = Wq_p
        return Wq

    Sslice = Zslice = None
    slice_start = 0

    for b0 in range(0, n_in, blocksize):
        b1 = min(b0 + blocksize, n_in)
        block_err = torch.zeros(b1 - b0, n_out, device=W.device)
        for j in range(b0, b1):
            # 分组参数：每组起点用当前（已补偿）权重重新计算
            if group_size and group_size > 0:
                if j % group_size == 0:
                    g_end = min(j + group_size, n_in)
                    Sslice, Zslice = group_params(W[:, j:g_end], bits, 0, symmetric)
                    slice_start = j
            elif j == 0:
                Sslice, Zslice = group_params(W, bits, 0, symmetric)

            s = Sslice[:, j - slice_start]
            z = Zslice[:, j - slice_start]
            d = U[j, j]
            w = W[:, j]
            q = torch.clamp(torch.round(w / s) + z, qmin, qpeak)
            dq = (q - z) * s
            err = (w - dq) / d
            Wq[:, j] = dq
            if j + 1 < b1:
                W[:, j + 1:b1] -= err.unsqueeze(1) * U[j, j + 1:b1].unsqueeze(0)
            block_err[j - b0] = err
        if b1 < n_in:
            W[:, b1:] -= block_err.T @ U[b0:b1, b1:]
    return Wq


class GPTQQuantizer:
    """逐层 GPTQ 伪量化（同 RTNQuantizer：保留原始权重，可 restore）。"""

    def __init__(self, model, config, percdamp: float = 0.01, blocksize: int = 128,
                 include: tuple = (), exclude: tuple = ("lm_head", "embed_out"),
                 act_order: bool = False):
        self.model = model
        self.config = config          # RTNConfig 兼容：bits/group_size/symmetric
        self.percdamp = percdamp
        self.blocksize = blocksize
        self.act_order = act_order
        self.linears = find_linears(model, include=include, exclude=exclude)
        self._backup: dict = {}

    def quantize_(self, calib_batches, max_batches: int = 32) -> dict:
        t0 = time.time()
        H_dict = collect_xtx(self.model, self.linears, calib_batches, max_batches)
        errs = []
        for name, m in self.linears:
            self._backup[name] = m.weight.data.detach().clone()
            W = m.weight.data.to(torch.float32)
            H = H_dict.pop(name)
            Hinv = damp_inverse(H, self.percdamp)
            U = torch.linalg.cholesky(Hinv, upper=True).to(torch.float32)
            del Hinv
            Wq = _gptq_layer(W, U, self.config.bits, self.config.group_size,
                             self.config.symmetric, self.blocksize,
                             act_order=self.act_order,
                             h_diag=H.diagonal().to(torch.float32) if self.act_order else None)
            del H, U
            err = (Wq - W).abs().mean().item()
            errs.append(err)
            m.weight.data.copy_(Wq.to(m.weight.data.dtype))
        report = {
            "method": "gptq-scratch",
            "bits": self.config.bits,
            "group_size": self.config.group_size,
            "symmetric": self.config.symmetric,
            "n_layers": len(self.linears),
            "mean_abs_weight_err": round(sum(errs) / max(len(errs), 1), 6),
            "duration_s": round(time.time() - t0, 2),
        }
        logger.info("[gptq-scratch] W%d g%s：%d 层，耗时 %.1fs",
                    self.config.bits, self.config.group_size or "ch",
                    report["n_layers"], report["duration_s"])
        return report

    def restore(self) -> None:
        for name, m in self.linears:
            if name in self._backup:
                m.weight.data.copy_(self._backup[name])
        self._backup.clear()
