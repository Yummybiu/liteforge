"""剪枝基类：激活统计的钩子采集、掩码生成与施加、结果报告。

设计说明：
- 剪枝后的模型仍以**稠密权重**方式存储与推理（权重置零）。
  质量损失（困惑度）用本仓库的 eval 模块评估；实际加速需要稀疏推理内核
  （如 2:4 半结构化 + Ampere sparse tensor core、或稀疏 kernel 库），
  端到端速度收益见 deploy/ 下的部署基准。这是学术评估的标准做法，务必诚实区分。
- Pruner 可插拔：子类只需实现 `compute_scores()`，返回 {layer_name: score_tensor}。
  预留 slot：论文自定义剪枝方法实现为新的 BasePruner 子类即可接入全部评测/报告链路。
"""

import logging
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..utils import find_linears

logger = logging.getLogger(__name__)


@dataclass
class PruneConfig:
    sparsity: float = 0.5
    structure: str = "unstructured"  # unstructured | 2:4
    include: tuple = ()              # 只剪名字含这些子串的 Linear
    exclude: tuple = ("lm_head", "embed_out")


@dataclass
class PruneResult:
    method: str
    sparsity_target: float
    structure: str
    overall_sparsity: float
    duration_s: float
    layer_reports: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "sparsity_target": self.sparsity_target,
            "structure": self.structure,
            "overall_sparsity": self.overall_sparsity,
            "duration_s": round(self.duration_s, 2),
            "layer_reports": self.layer_reports,
        }


def per_row_topk_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    """非结构化：逐输出行按 score 保留 top-(1-sparsity)，返回 0/1 float 掩码。"""
    n_cols = score.shape[1]
    n_keep = max(1, int(round(n_cols * (1.0 - sparsity))))
    idx = score.topk(n_keep, dim=1).indices
    mask = torch.zeros_like(score)
    mask.scatter_(1, idx, 1.0)
    return mask


def group_2to4_mask(score: torch.Tensor) -> torch.Tensor:
    """2:4 半结构化：沿输入维每 4 个连续元素保留 score 最大的 2 个。"""
    n_out, n_cols = score.shape
    if n_cols % 4 != 0:
        raise ValueError(f"2:4 剪枝要求输入维是 4 的倍数，当前 {n_cols}")
    groups = score.view(n_out, n_cols // 4, 4)
    idx = groups.topk(2, dim=2).indices
    mask = torch.zeros_like(groups)
    mask.scatter_(2, idx, 1.0)
    return mask.view(n_out, n_cols)


def layer_sparsity_target(cfg_sparsity, name: str) -> float:
    """解析逐层稀疏度：标量直接返回；dict 按 layer 名查（缺省用 __default__）。"""
    if isinstance(cfg_sparsity, dict):
        return float(cfg_sparsity.get(name, cfg_sparsity.get("__default__", 0.5)))
    return float(cfg_sparsity)


def layer_sparsity(linear: nn.Linear) -> float:
    w = linear.weight.data
    return (w == 0).float().mean().item()


class BasePruner:
    """模板方法子类：实现 compute_scores(layer_inputs) 即可。

    layer_inputs: {name: (x_sum[n_in], n_tokens)} —— 该层输入在各特征维度上的
    平方和与 token 数（供 Wanda 类激活感知方法使用）；Magnitude 等方法忽略它。
    """

    method_name = "base"

    def __init__(self, model: nn.Module, config: PruneConfig):
        self.model = model
        self.config = config
        self.linears = find_linears(model, include=config.include, exclude=config.exclude)
        if not self.linears:
            raise RuntimeError("没有找到可剪枝的 nn.Linear 层")

    # ---- 激活统计（Wanda 需要） ------------------------------------------
    def collect_layer_inputs(self, calib_batches, max_batches: int = 32):
        """前向钩子采集每个 Linear 的输入平方和。calib_batches 产出 input_ids 张量。"""
        device = next(self.model.parameters()).device
        stats = {
            name: {"x_sum": torch.zeros(m.in_features, dtype=torch.float64, device=device),
                   "n": 0}
            for name, m in self.linears
        }
        hooks = []

        def make_hook(lname):
            def pre_hook(module, args):
                x = args[0]
                xf = x.detach().reshape(-1, x.shape[-1]).to(torch.float64)
                stats[lname]["x_sum"] += xf.pow(2).sum(dim=0)
                stats[lname]["n"] += xf.shape[0]
            return pre_hook

        try:
            for name, m in self.linears:
                hooks.append(m.register_forward_pre_hook(make_hook(name)))
            with torch.no_grad():
                for i, batch in enumerate(calib_batches):
                    if i >= max_batches:
                        break
                    if isinstance(batch, (list, tuple)):
                        batch = batch[0]
                    batch = batch.to(device)
                    if batch.dim() == 2:
                        self.model(input_ids=batch)
                    else:  # 已含 attention_mask 等
                        self.model(input_ids=batch["input_ids"],
                                   attention_mask=batch.get("attention_mask"))
        finally:
            for h in hooks:
                h.remove()

        layer_inputs = {
            name: (s["x_sum"], s["n"]) for name, s in stats.items()
        }
        return layer_inputs

    # ---- 子类实现 ---------------------------------------------------------
    def compute_scores(self, layer_inputs) -> dict:
        """返回 {layer_name: score_tensor(float32, 形状=该层权重形状)}。"""
        raise NotImplementedError

    # ---- 主流程 -----------------------------------------------------------
    def run(self, calib_batches=None, max_batches: int = 32) -> PruneResult:
        t0 = time.time()
        layer_inputs = {}
        if calib_batches is not None:
            logger.info("[%s] 采集校准激活统计 ...", self.method_name)
            layer_inputs = self.collect_layer_inputs(calib_batches, max_batches)

        scores = self.compute_scores(layer_inputs)
        layer_reports = []
        pruned_elems = total_elems = 0

        for name, m in self.linears:
            s = layer_sparsity_target(self.config.sparsity, name)
            w = m.weight.data
            score = scores[name].to(w.device)
            if self.config.structure == "2:4":
                mask = group_2to4_mask(score)
            else:
                mask = per_row_topk_mask(score, s)
            m.weight.data.mul_(mask.to(w.dtype))
            sp = (w == 0).float().mean().item()
            pruned_elems += int((w == 0).sum().item())
            total_elems += w.numel()
            layer_reports.append({"layer": name, "shape": list(w.shape),
                                  "sparsity": round(sp, 4)})

        result = PruneResult(
            method=self.method_name,
            sparsity_target=self.config.sparsity,
            structure=self.config.structure,
            overall_sparsity=round(pruned_elems / max(total_elems, 1), 4),
            duration_s=time.time() - t0,
            layer_reports=layer_reports,
        )
        logger.info("[%s] 剪枝完成：整体稀疏度 %.2f%%，耗时 %.1fs",
                    self.method_name, result.overall_sparsity * 100, result.duration_s)
        return result
