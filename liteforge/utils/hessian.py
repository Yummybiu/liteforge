"""Hessian 工具：校准输入的 XᵀX 采集与阻尼求逆。

OBC / SparseGPT / GPTQ 同族方法共用：
- H = 2·E[x xᵀ]（实现里省略常数 2，等价缩放）；
- 阻尼 H += mean(diag(H))·percdamp·I（数值稳定，官方同款）；
- Hinv = (H + damp)⁻¹；GPTQ/OBC 的顺序误差补偿只用它的 Cholesky 上三角。
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def collect_xtx(
    model: nn.Module,
    linears,
    calib_batches,
    max_batches: int = 32,
) -> dict:
    """前向钩子逐层累计 XᵀX（float32 累加）。返回 {layer_name: H[n_in, n_in]}。"""
    device = next(model.parameters()).device
    H = {
        name: torch.zeros(m.in_features, m.in_features, dtype=torch.float32, device=device)
        for name, m in linears
    }
    hooks = []

    def make_hook(lname):
        def pre_hook(module, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).to(torch.float32)
            H[lname] += x.T @ x
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
                batch = batch.to(device)
                if batch.dim() == 2:
                    model(input_ids=batch)
                else:
                    model(input_ids=batch["input_ids"],
                          attention_mask=batch.get("attention_mask"))
    finally:
        for h in hooks:
            h.remove()
    return H


def damp_inverse(H: torch.Tensor, percdamp: float = 0.01):
    """阻尼 + 求逆。输入 [n, n]（任意精度），返回 float64 的 H⁻¹。"""
    H = H.to(torch.float64)
    dead = H.diagonal() == 0
    if dead.any():  # 死列处理：官方同款，用均值替代对角
        H[dead, dead] = H.diagonal().mean()
    damp = percdamp * H.diagonal().mean()
    H = H + damp * torch.eye(H.shape[0], dtype=torch.float64, device=H.device)
    return torch.cholesky_inverse(torch.linalg.cholesky(H))
