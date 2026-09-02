"""文本数据：WikiText-2 加载、任意文本文件加载、按 token 分块与校准批次。"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

BLOCK_SIZE = 2048


_WIKI_LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "cache", "wikitext",
)


def load_wikitext2(split: str = "test") -> str:
    """WikiText-2 raw（PPL 标准评测集）。

    优先本地 parquet（cache/wikitext/{split}.parquet，见 scripts/），
    否则在线下载（尊重 HF_ENDPOINT，如 hf-mirror.com）。
    """
    from datasets import load_dataset
    local = os.path.join(_WIKI_LOCAL_DIR, f"{split}.parquet")
    if os.path.exists(local):
        ds = load_dataset("parquet", data_files={split: local}, split=split)
    else:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(t for t in ds["text"] if t)


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_eval_text(spec: str) -> str:
    """spec: 'wikitext2[:split]' 或 'file:/path/to.txt'"""
    if spec.startswith("file:"):
        return load_text_file(spec[5:])
    parts = spec.split(":")
    split = parts[1] if len(parts) > 1 else "test"
    return load_wikitext2(split)


def tokenize_to_ids(tokenizer, text: str) -> list:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if isinstance(ids[0], list):  # batch 返回
        ids = ids[0]
    return list(ids)


def chunk_ids(ids: list, block_size: int = BLOCK_SIZE) -> list:
    n_blocks = len(ids) // block_size
    return [ids[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]


class BlockBatcher:
    """把 token 块按 batch_size 迭代为 input_ids 张量（评测与校准共用）。"""

    def __init__(self, tokenizer, text: str, block_size: int = BLOCK_SIZE,
                 batch_size: int = 4, max_blocks: int | None = None):
        ids = tokenize_to_ids(tokenizer, text)
        self.blocks = chunk_ids(ids, block_size)
        if max_blocks:
            self.blocks = self.blocks[:max_blocks]
        self.batch_size = batch_size
        logger.info("文本 %d tokens → %d 个 %d-token 块", len(ids), len(self.blocks), block_size)

    def __len__(self):
        return (len(self.blocks) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for i in range(0, len(self.blocks), self.batch_size):
            chunk = self.blocks[i:i + self.batch_size]
            yield torch.tensor(chunk, dtype=torch.long)
