"""通用工具：随机种子、设备选择、模块过滤、JSON 读写、环境信息。"""

import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

_LOGGER_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _LOGGER_CONFIGURED = True


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    """preference: auto | cuda | cpu"""
    if preference == "cpu":
        return torch.device("cpu")
    if preference in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_linears(
    model: nn.Module,
    include: tuple = (),
    exclude: tuple = ("lm_head", "embed_out"),
) -> list:
    """返回 [(name, nn.Linear), ...]。

    include 非空时只保留名字中包含任一子串的层；
    exclude 中任一子串出现在名字里则跳过（默认排除输出头，剪枝/量化通常不动它）。
    """
    out = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(ex in name for ex in exclude):
            continue
        if include and not any(inc in name for inc in include):
            continue
        out.append((name, module))
    return out


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    linear = sum(m.weight.numel() for _, m in find_linears(model))
    return {"total_params": total, "linear_params": linear}


def model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024


def save_json(obj: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def torch_env_info() -> dict:
    return {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
