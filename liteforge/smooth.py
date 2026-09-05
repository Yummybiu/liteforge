"""SmoothQuant W8A8：从零实现（离线等价变换 + α 网格搜索 + 蒙特卡洛损失）。

文献：Xiao et al., *SmoothQuant* (mit-han-lab, arXiv:2211.10438)。
本实现从零构建，与仓库的统一损失货币（lossmeter）打通。

问题：LLM 激活有系统性离群通道（少数通道幅度极大），直接 W8A8 量化时
激活侧误差爆炸。解法：数学等价变换把离群"迁移"进权重——

    y = X·Wᵀ = (X·diag(1/s))·(W·diag(s))ᵀ,   s_j = act_max_j^α / w_max_j^(1-α)

α ∈ [0,1] 是跷跷板：α→1 激活更平滑（权重更难），α→0 反之。
权重 per-channel 对称 W8；激活 per-token 动态对称 A8（真实引擎口径）。

关键口径：W8A8 的损失必须计入**激活量化误差**——它依赖具体 token，
不能像 weight-only 那样用 H 的 trace 公式闭式算，故按校准 batch 蒙特卡洛
累加 ||Q(X')Q(W')ᵀ − X'W'ᵀ||²。这也是 α 搜索的目标函数。
"""

import logging
import time

import torch

from .utils.hessian import collect_xtx

logger = logging.getLogger(__name__)

QMAX = 127  # int8 对称


# ---- 基础量化算子 ---------------------------------------------------------

def quant_w8_perchannel(W: torch.Tensor) -> torch.Tensor:
    """权重 per-channel（每输出行）对称 W8，返回反量化 float。"""
    scale = W.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / QMAX
    return (torch.clamp(torch.round(W / scale), -QMAX, QMAX) * scale).to(W.dtype)


def quant_a8_pertoken(X: torch.Tensor) -> torch.Tensor:
    """激活 per-token（每行）动态对称 A8，返回反量化 float。X: (..., n_in)。"""
    scale = X.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / QMAX
    return (torch.clamp(torch.round(X / scale), -QMAX, QMAX) * scale).to(X.dtype)


def smooth_scale(w_max: torch.Tensor, act_max: torch.Tensor, alpha: float) -> torch.Tensor:
    """s_j = act_max_j^α / w_max_j^(1-α)（含 eps 防零）。w_max/act_max: [n_in]。"""
    eps = 1e-5
    return ((act_max.clamp(min=eps) ** alpha)
            / (w_max.clamp(min=eps) ** (1.0 - alpha)))


# ---- 激活统计 ------------------------------------------------------------

def collect_act_max(model, linears, calib_batches, max_batches: int = 16) -> dict:
    """逐层激活通道最大幅度 {layer: act_max[n_in]}（前向预钩子 running max）。"""
    device = next(model.parameters()).device
    act_max = {name: None for name, _ in linears}
    hooks = []

    def make_hook(lname):
        def pre_hook(module, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).abs().amax(dim=0)
            cur = act_max[lname]
            act_max[lname] = x if cur is None else torch.maximum(cur, x)
        return pre_hook

    try:
        for name, m in linears:
            hooks.append(m.register_forward_pre_hook(make_hook(name)))
        with torch.no_grad():
            for i, batch in enumerate(calib_batches):
                if i >= max_batches:
                    break
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                model(input_ids=batch.to(device))
    finally:
        for h in hooks:
            h.remove()
    return {k: v.to(torch.float32) if v is not None else None for k, v in act_max.items()}


# ---- W8A8 蒙特卡洛损失（α 搜索目标） ----------------------------------------

def _batch_w8a8_loss(x: torch.Tensor, W: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """单 batch 的 W8A8 输出误差平方和（等价空间中计算，数学上与原空间相同）。

    x: [n_tok, n_in]（该层本 batch 输入）；W: [n_out, n_in]；s: [n_in]。
    """
    x_s = x.to(torch.float32) / s
    W_s = W.to(torch.float32) * s.unsqueeze(0)
    xq = quant_a8_pertoken(x_s)
    wq = quant_w8_perchannel(W_s)
    y = x_s @ W_s.T
    yq = xq @ wq.T
    return (yq - y).pow(2).sum()


@torch.no_grad()
def w8a8_alpha_sweep(model, linears, calib_batches, alphas=(0.4, 0.5, 0.6, 0.7, 0.8),
                     max_batches: int = 8, include_no_smooth: bool = True) -> dict:
    """α 网格搜索。返回 {layer: {"by_alpha": {α: loss}, "no_smooth": loss, "best_alpha": α}}。

    两遍校准：第一遍采集 act_max；第二遍逐 α 蒙特卡洛（权重侧量化只做一次/α）。
    模型权重不被修改。层级并行度低（逐层钩子按批累加），精度换可行性。
    """
    device = next(model.parameters()).device
    t0 = time.time()
    act_max_map = collect_act_max(model, linears, calib_batches, max_batches)

    sweep = {name: {"by_alpha": {}} for name, _ in linears}
    # no_smooth 用 s=1 表达（等效"无迁移"），统一进循环
    configs = [("no_smooth", None)] + [(f"a{a}", a) for a in alphas] \
        if include_no_smooth else [(f"a{a}", a) for a in alphas]

    for cfg_name, alpha in configs:
        # 每层缓存平滑后的量化权重
        Wq_map = {}
        for name, m in linears:
            W = m.weight.data.detach().to(torch.float32)
            if alpha is None:
                s = torch.ones(W.shape[1], device=device, dtype=torch.float32)
            else:
                s = smooth_scale(W.abs().amax(dim=0), act_max_map[name], alpha)
            Wq_map[name] = (W, s)

        # 重放校准，钩子内逐 batch 累加损失
        losses = {name: 0.0 for name, _ in linears}
        hooks = []

        def make_hook(lname):
            def pre_hook(module, args):
                x = args[0].detach().reshape(-1, args[0].shape[-1]).to(torch.float32)
                W, s = Wq_map[lname]
                losses[lname] += float(_batch_w8a8_loss(x, W, s).item())
            return pre_hook

        try:
            for name, m in linears:
                hooks.append(m.register_forward_pre_hook(make_hook(name)))
            for i, batch in enumerate(calib_batches):
                if i >= max_batches:
                    break
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                model(input_ids=batch.to(device))
        finally:
            for h in hooks:
                h.remove()

        for name, _ in linears:
            if cfg_name == "no_smooth":
                sweep[name]["no_smooth"] = losses[name]
            else:
                sweep[name]["by_alpha"][alpha] = losses[name]

    for name, _ in linears:
        by = sweep[name]["by_alpha"]
        sweep[name]["best_alpha"] = min(by, key=by.get)

    logger.info("[smooth] α 扫描完成（%d 层 × %d 配置），耗时 %.1fs",
                len(linears), len(configs), time.time() - t0)
    return sweep


def apply_smoothquant(model, linears, calib_batches, alpha: float,
                      max_batches: int = 16) -> dict:
    """把最优平滑**写进权重**（W ← W·diag(s)）——之后按普通 W8A16/W8A8 部署。

    注意：真实 W8A8 引擎需要同时保存 s 并在运行时对激活除以 s（或融合进
    前一层）。本函数只做"权重侧"固化；激活侧变换由部署引擎承担，
    质量验证用 w8a8_alpha_sweep 的蒙特卡洛口径。返回每层的 s 供导出。
    """
    act_max_map = collect_act_max(model, linears, calib_batches, max_batches)
    scales = {}
    for name, m in linears:
        W = m.weight.data.detach().to(torch.float32)
        s = smooth_scale(W.abs().amax(dim=0), act_max_map[name], alpha)
        m.weight.data.copy_((W * s.unsqueeze(0)).to(m.weight.data.dtype))
        scales[name] = s.tolist()
    logger.info("[smooth] 已固化 α=%.2f 的平滑变换到 %d 层权重", alpha, len(scales))
    return scales
