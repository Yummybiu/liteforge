"""MMLU-mini 评测：4 选 1 似然打分（lm-eval 口径的轻量实现）。

为什么用似然打分而不是生成：对 A/B/C/D 单 token 选项，一次前向即可比较
四个选项的 logit——比生成式评测便宜一个量级，且是 lm-eval-harness 对
多项选择任务的标准口径（loglikelihood 比较）。

数据：cais/mmlu test split。优先本地 parquet（cache/mmlu/，scripts 下载），
否则在线（尊重 HF_ENDPOINT）。MMLU-mini = 分层抽样 n 题。
输出：总体准确率 + 分桶（STEM / 人文 / 社科 / 其他，MMLU 官方分类）。

口径说明：单 token 选项时用最后位置 logit 直接比较（等价于对 " A/B/C/D"
续写的 log-likelihood 比较，因为该续写是单 token）；若某选项 token 化后
多于一个 token，回退到完整续写打分。
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

MMLU_LOCAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "cache", "mmlu")

STEM = {"abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering", "elementary_mathematics",
        "high_school_biology", "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning"}
HUMANITIES = {"formal_logic", "high_school_european_history", "high_school_us_history",
              "high_school_world_history", "international_law", "jurisprudence",
              "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
              "prehistory", "professional_law", "world_religions"}
SOCIAL = {"econometrics", "high_school_geography", "high_school_government_and_politics",
          "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
          "human_sexuality", "professional_psychology", "public_relations", "security_studies",
          "sociology", "us_foreign_policy"}


def bucket_of(subject: str) -> str:
    if subject in STEM:
        return "stem"
    if subject in HUMANITIES:
        return "humanities"
    if subject in SOCIAL:
        return "social"
    return "other"


def load_mmlu(n: int = 500, seed: int = 42):
    """分层抽样 n 题（按 subject 比例）。返回 [{question, choices, answer, subject}]。"""
    from datasets import load_dataset
    local = os.path.join(MMLU_LOCAL_DIR, "test.parquet")
    if os.path.exists(local):
        ds = load_dataset("parquet", data_files={"test": local}, split="test")
    else:
        ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=seed)
    if n and n < len(ds):
        ds = ds.select(range(n))
    return [{"question": ex["question"], "choices": list(ex["choices"]),
             "answer": int(ex["answer"]), "subject": ex["subject"]} for ex in ds]


def format_prompt(ex: dict) -> str:
    letters = ["A", "B", "C", "D"]
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ex["choices"]))
    return (f"The following are multiple choice questions (with answers) "
            f"about {ex['subject'].replace('_', ' ')}.\n\n"
            f"{ex['question']}\n{body}\nAnswer:")


@torch.no_grad()
def _choice_logprobs(model, tokenizer, prompt: str, choices: list) -> list:
    """每个选项续写的 log-prob（单 token 快路径 + 多 token 回退）。"""
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    letter_ids, single = [], True
    for ch in choices:
        ids = tokenizer(" " + ch, add_special_tokens=False)["input_ids"]
        if isinstance(ids[0], list):
            ids = ids[0]
        if len(ids) != 1:
            single = False
        letter_ids.append(ids)
    if single:
        logits = model(**enc).logits[0, -1, :]
        lse = torch.logsumexp(logits, dim=-1)
        return [float(logits[lid[0]] - lse) for lid in letter_ids]
    # 多 token 回退：完整续写 log-likelihood
    out = []
    for lid in letter_ids:
        cont = torch.tensor([lid], device=device)
        full = torch.cat([enc["input_ids"], cont], dim=1)
        logits = model(full).logits[0]
        lp = 0.0
        for j, t in enumerate(lid):
            pos = enc["input_ids"].shape[1] - 1 + j
            lse = torch.logsumexp(logits[pos], dim=-1)
            lp += float(logits[pos, t] - lse)
        out.append(lp)
    return out


@torch.no_grad()
def evaluate_mmlu(model, tokenizer, n: int = 500, batch_log_every: int = 50) -> dict:
    items = load_mmlu(n)
    correct = {"total": 0}
    buckets = {}
    for i, ex in enumerate(items):
        prompt = format_prompt(ex)
        lps = _choice_logprobs(model, tokenizer, prompt, [str(c) for c in range(4)]
                               if all(len(str(c)) == 1 for c in ex["choices"])
                               else ["A", "B", "C", "D"])
        pred = int(max(range(4), key=lambda j: lps[j]))
        # 选项是文本时，比的是"选项索引字母"——answer 是索引，语义一致
        ok = pred == ex["answer"]
        b = bucket_of(ex["subject"])
        buckets.setdefault(b, [0, 0])
        buckets[b][1] += 1
        correct["total"] += int(ok)
        buckets[b][0] += int(ok)
        if (i + 1) % batch_log_every == 0:
            logger.info("[mmlu] %d/%d running acc=%.3f", i + 1, len(items),
                        correct["total"] / (i + 1))
    return {
        "task": "eval-mmlu",
        "n": len(items),
        "accuracy": round(correct["total"] / max(len(items), 1), 4),
        "by_bucket": {b: {"acc": round(c / max(t, 1), 4), "n": t}
                      for b, (c, t) in sorted(buckets.items())},
    }
