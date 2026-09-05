"""统一损失计价器（Unified Loss Meter）：LiteForge V2 的核心。

把"剪枝损失"与"量化损失"放进同一货币：校准集输出 L2。

数学：线性层 y = Wx，压缩引入 ΔW = W − W_c。校准输入 X 上的输出损失

    L = ||X(W−W_c)ᵀ||²_F = tr(ΔW · H · ΔWᵀ),   H = XᵀX

关键性质：**H 只依赖输入分布，不依赖压缩方法**——剪枝和量化用同一把尺子。
损失计算无需保留 X，只需每层一个 d×d 的 H（collect_xtx 复用）。

用法（概念）：
    table = measure_menus(model, calib_batches, menu, chunk=6)
    # → {"model.layers.0.mlp.down_proj": {"w4g128": {"bits_eff": 4.25, "loss": 0.13}, ...}, ...}

菜单（menu）约定：每层对每个压缩选项测一个 (bits_eff, loss)：
    "fp16"  : 16 bit, loss=0
    "w8g128"/"w4g128"/"w3g128"/"w2g128" : RTN 或 GPTQ 量化
    "p25"/"p50"/"p75" : OBC 剪枝（static 估计或 dynamic 实测）

有效比特口径（诚实记账）：分组非对称量化每个 group 存 1 个 scale + 1 个
zero（各 16bit），故 b_eff = b + 32/group_size；对称量化 b_eff = b + 16/group_size；
剪枝 b_eff = 16·(1−sparsity)。fp16 的 16 bit 不含 embedding/lm_head（未压缩）。
"""

import logging
import time

import torch

from .prune.base import per_row_topk_mask
from .prune.obc import _sequential_zero, _sequential_zero_dynamic
from .quant.rtn import group_params
from .utils import find_linears
from .utils.hessian import collect_xtx, damp_inverse

logger = logging.getLogger(__name__)

# ---- 菜单定义 ------------------------------------------------------------

def effective_bits_quant(bits: int, group_size: int, symmetric: bool) -> float:
    overhead = (16 if symmetric else 32) / group_size if group_size else 0.0
    return bits + overhead


def effective_bits_prune(sparsity: float) -> float:
    return 16.0 * (1.0 - sparsity)


DEFAULT_MENU = (
    {"name": "w8g128", "kind": "quant", "bits": 8, "group_size": 128},
    {"name": "w4g128", "kind": "quant", "bits": 4, "group_size": 128},
    {"name": "w3g128", "kind": "quant", "bits": 3, "group_size": 128},
    {"name": "w2g128", "kind": "quant", "bits": 2, "group_size": 128},
    {"name": "p25", "kind": "prune", "sparsity": 0.25},
    {"name": "p50", "kind": "prune", "sparsity": 0.50},
    {"name": "p75", "kind": "prune", "sparsity": 0.75},
)


# ---- 损失计算 ------------------------------------------------------------

def compression_loss(H: torch.Tensor, W_c: torch.Tensor, W: torch.Tensor) -> float:
    """tr(ΔW H ΔWᵀ)，ΔW = W − W_c。H 为 XᵀX（sum 形式，与 X 的行数一致缩放）。"""
    dW = (W.to(torch.float64) - W_c.to(torch.float64))
    return float(torch.trace(dW @ H.to(torch.float64) @ dW.T))


def quantize_copy(W: torch.Tensor, bits: int, group_size: int, symmetric: bool,
                  impl: str = "rtn", U: torch.Tensor | None = None) -> torch.Tensor:
    """在 W 的副本上应用量化，返回压缩后的权重（不动模型）。"""
    if impl == "gptq":
        from .quant.gptq import _gptq_layer
        return _gptq_layer(W.clone(), U, bits, group_size, symmetric)
    from .quant.rtn import quantize_tensor
    return quantize_tensor(W, bits, group_size, symmetric)


def prune_copy(W: torch.Tensor, sparsity: float, Hinv_diag: torch.Tensor,
               U: torch.Tensor, mode: str = "static") -> torch.Tensor:
    """在 W 的副本上应用剪枝（static=OBS 静态预选；dynamic=SparseGPT 块级重评分）。"""
    if mode == "dynamic":
        return _sequential_zero_dynamic(W.clone(), U, sparsity)
    score = W.to(torch.float64) * W.to(torch.float64) / Hinv_diag.unsqueeze(0).clamp(min=1e-12)
    mask = per_row_topk_mask(score.to(torch.float32), sparsity)
    return _sequential_zero(W.clone(), U, mask)


# ---- 分块测量主流程 -------------------------------------------------------

def _chunk_linears(linears, chunk: int):
    """把层列表按 chunk 个一组切分（控制 H 的同时驻留内存）。chunk=0 → 全量。"""
    if not chunk or chunk <= 0:
        yield linears
        return
    for i in range(0, len(linears), chunk):
        yield linears[i:i + chunk]


def measure_menus(
    model,
    linears,
    calib_batches,
    menu: tuple = DEFAULT_MENU,
    max_batches: int = 16,
    chunk: int = 0,
    percdamp: float = 0.01,
    quant_impl: str = "rtn",        # rtn（CPU 快）| gptq（需 U，较慢更准）
    prune_mode: str = "static",     # static（快，高稀疏度偏乐观）| dynamic（慢，准）
) -> dict:
    """逐块采集 H → 对每个 Linear 的每个菜单选项测 (bits_eff, loss)。

    返回 {layer_name: {option_name: {"bits_eff": float, "loss": float, "kind": str}}}
    模型权重不被修改（全部在副本上操作）。
    """
    device = next(model.parameters()).device
    table: dict = {}
    t0 = time.time()
    n_done = 0

    for chunk_i, linears_c in enumerate(_chunk_linears(linears, chunk)):
        H_dict = collect_xtx(model, linears_c, calib_batches, max_batches)
        for name, m in linears_c:
            H = H_dict.pop(name)
            W = m.weight.data.detach().to(torch.float32)
            Hinv = damp_inverse(H, percdamp)
            Hinv_diag = Hinv.diagonal()
            U = torch.linalg.cholesky(Hinv, upper=True).to(torch.float32)
            del Hinv

            opts = {"fp16": {"bits_eff": 16.0, "loss": 0.0, "kind": "keep"}}
            for item in menu:
                if item["kind"] == "quant":
                    Wc = quantize_copy(W, item["bits"], item["group_size"],
                                       symmetric=False, impl=quant_impl, U=U)
                    be = effective_bits_quant(item["bits"], item["group_size"], False)
                else:
                    Wc = prune_copy(W, item["sparsity"], Hinv_diag, U, mode=prune_mode)
                    be = effective_bits_prune(item["sparsity"])
                opts[item["name"]] = {
                    "bits_eff": round(be, 3),
                    "loss": compression_loss(H, Wc, W),
                    "kind": item["kind"],
                }
                del Wc
            table[name] = {"shape": list(W.shape), "options": opts}
            n_done += 1
            del H, U
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[lossmeter] chunk %d 完成（%d/%d 层）",
                    chunk_i, n_done, len(linears))

    logger.info("[lossmeter] 全部完成：%d 层 × %d 选项，耗时 %.1fs",
                len(table), 1 + len(menu), time.time() - t0)
    return table
