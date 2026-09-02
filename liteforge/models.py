"""模型与分词器加载（统一 dtype/device 处理）。"""

import logging

import torch

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "auto": "auto",
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def load_model_and_tokenizer(model_id: str, device: str = "auto",
                             dtype: str = "auto", trust_remote: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = DTYPE_MAP[dtype]
    kwargs = {"trust_remote_code": trust_remote}
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    tok_dtype = None if torch_dtype == "auto" else torch_dtype
    if torch_dtype == "auto" and torch.cuda.is_available():
        model = model.to(torch.bfloat16 if _bf16_ok() else torch.float16)
        tok_dtype = model.dtype
    model = model.to(_resolve_device(device))

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("模型 %s 加载完成：%s @ %s, dtype=%s",
                model_id, f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params",
                next(model.parameters()).device, next(model.parameters()).dtype)
    return model, tokenizer


def _resolve_device(preference: str):
    if preference == "cpu":
        return torch.device("cpu")
    if preference in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _bf16_ok() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(0)
    return cap[0] >= 8  # Ampere 及以上
