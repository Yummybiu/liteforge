"""RTN（Round-To-Nearest）伪量化：从零实现，可在 CPU/GPU 上直接跑。

为什么值得自己写：
- RTN 是一切现代量化方法（GPTQ/AWQ/SmoothQuant）的基线，面试常被要求
  现场推导 scale/zero-point 的计算——这里就是可运行的参考实现；
- 伪量化（quantize→dequantize 回浮点）不需要反量化内核即可评估量化对
  质量的影响；真实加速需要整数内核（GPTQ/AWQ 打包格式 + 推理引擎），见 wrappers。

支持：
- 对称（symmetric）：只用 scale，常配 per-output-channel；
- 非对称（asymmetric）：scale + zero-point，常配 group-wise（如 group_size=128）；
- group_size=0 表示按输出通道（per-column of W^T，即 W 的每一行）分组。
"""

import logging
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..utils import find_linears

logger = logging.getLogger(__name__)


@dataclass
class RTNConfig:
    bits: int = 4
    group_size: int = 128   # 0 = per-output-channel
    symmetric: bool = False


def _quant_groups(w: torch.Tensor, bits: int, symmetric: bool):
    """对最后一维分组量化。w: (..., G)。返回 (dequant, err_absmean, scale)。"""
    qmax = 2 ** (bits - 1) - 1          # 对称: [-qmax, qmax]
    qmin, qpeak = 0, 2 ** bits - 1      # 非对称: [0, 2^b-1]

    if symmetric:
        scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        q = torch.clamp(torch.round(w / scale), -qmax, qmax)
        deq = q * scale
    else:
        wmin = w.amin(dim=-1, keepdim=True)
        wmax = w.amax(dim=-1, keepdim=True)
        scale = (wmax - wmin).clamp(min=1e-8) / qpeak
        zero = torch.round(-wmin / scale).clamp(qmin, qpeak)
        q = torch.clamp(torch.round(w / scale) + zero, qmin, qpeak)
        deq = (q - zero) * scale
    return deq, (deq - w).abs().mean().item(), scale


def quantize_tensor(w: torch.Tensor, bits: int, group_size: int, symmetric: bool) -> torch.Tensor:
    """量化再反量化一个 [n_out, n_in] 权重，返回同 dtype 的伪量化权重。"""
    orig_dtype = w.dtype
    wf = w.to(torch.float32)
    if group_size and group_size > 0:
        n_out, n_in = wf.shape
        if n_in % group_size != 0:
            pad = group_size - (n_in % group_size)
            wf = torch.nn.functional.pad(wf, (0, pad))
            deq, _, _ = _quant_groups(wf, bits, symmetric)
            deq = deq.reshape(n_out, n_in + pad)[..., :n_in]
        else:
            g = wf.reshape(n_out, n_in // group_size, group_size)
            deq, _, _ = _quant_groups(g, bits, symmetric)
            deq = deq.reshape(n_out, n_in)
    else:
        deq, _, _ = _quant_groups(wf, bits, symmetric)
    return deq.to(orig_dtype)


class RTNQuantizer:
    """对模型内全部（或指定）Linear 做伪量化，保留原始权重可 restore。"""

    def __init__(self, model: nn.Module, config: RTNConfig,
                 include: tuple = (), exclude: tuple = ("lm_head", "embed_out")):
        self.model = model
        self.config = config
        self.linears = find_linears(model, include=include, exclude=exclude)
        self._backup: dict = {}

    def quantize_(self) -> dict:
        t0 = time.time()
        errs = []
        for name, m in self.linears:
            self._backup[name] = m.weight.data.detach().clone()
            wq = quantize_tensor(
                m.weight.data, self.config.bits, self.config.group_size,
                self.config.symmetric,
            )
            err = (wq.to(torch.float32) - m.weight.data.to(torch.float32)).abs().mean().item()
            errs.append(err)
            m.weight.data.copy_(wq)
        report = {
            "method": "rtn",
            "bits": self.config.bits,
            "group_size": self.config.group_size,
            "symmetric": self.config.symmetric,
            "n_layers": len(self.linears),
            "mean_abs_weight_err": round(sum(errs) / max(len(errs), 1), 6),
            "duration_s": round(time.time() - t0, 2),
        }
        logger.info("[rtn] W%d g%s%s：%d 层，平均权重绝对误差 %.2e",
                    self.config.bits, self.config.group_size or "channel",
                    " 对称" if self.config.symmetric else " 非对称",
                    report["n_layers"], report["mean_abs_weight_err"])
        return report

    def restore(self) -> None:
        for name, m in self.linears:
            if name in self._backup:
                m.weight.data.copy_(self._backup[name])
        self._backup.clear()
