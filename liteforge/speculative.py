"""投机解码（Speculative Decoding）：从零实现，draft-target 贪心验证。

文献：Leviathan et al. (ICML 2023) / Chen et al. 2023。本实现针对贪心解码
（确定性），并附带采样版的正确性说明。

流程（贪心口径）：
1. draft（小模型）自回归贪心出 k 个 token；
2. target（大模型）对 [前缀 + k 个 draft] 做**一次** teacher-forcing 前向——
   位置 i 的 logits 恰好给出对第 i+1 个 token 的预测，一次前向校验 k 个
   draft 并免费多拿一个"红利"token；
3. 从左到右接受：位置 i 上 target argmax 与 draft 一致 → 接受；
   首个不一致处 → 采纳 target 的 token 并截断本轮，重新 draft。

正确性定理（贪心）：输出与 target 纯贪心**逐 token 相同**——验证不改变
分布，只改变算力来源。本仓库用单测直接证明该性质。

统计口径：α 接受率 = 接受的 draft token / 提交的 draft token；
每 target 前向产出 token 数（理论 [1, k+1]，均匀接受率 α 下期望 (1-α^(k+1))/(1-α)）。
"""

import logging
import time

import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def speculative_generate(target, draft, input_ids: torch.Tensor,
                         max_new_tokens: int, k: int = 4, eos_id: int | None = None):
    """贪心投机解码。返回 dict（含序列与全部统计）。两模型需同 tokenizer。"""
    device = input_ids.device
    ids = input_ids
    n_ctx = ids.shape[1]
    stats = {"draft_tokens": 0, "accepted": 0, "corrected": 0,
             "target_forwards": 0, "draft_forwards": 0, "steps": []}
    generated = 0
    t0 = time.time()

    while generated < max_new_tokens:
        # 1) draft 自回归 k 个
        drafted = []
        cur = ids
        for _ in range(min(k, max_new_tokens - generated)):
            logits = draft(cur).logits[:, -1, :]
            t = logits.argmax(dim=-1, keepdim=True)
            drafted.append(t)
            cur = torch.cat([cur, t], dim=1)
            stats["draft_tokens"] += 1
            stats["draft_forwards"] += 1

        # 2) target 单次前向校验 k 个 draft（+1 红利位置）
        full = torch.cat([ids] + drafted, dim=1) if drafted else ids
        tgt_logits = target(full).logits
        stats["target_forwards"] += 1

        # 3) 贪心验证：位置 n_ctx-1+i 的 argmax 应等于 drafted[i]
        tokens = []
        accepted_this = 0
        for i, t in enumerate(drafted):
            p = tgt_logits[:, n_ctx - 1 + i, :].argmax(dim=-1, keepdim=True)
            if torch.equal(p, t):
                tokens.append(p)
                accepted_this += 1
            else:
                tokens.append(p)      # 采纳 target 的修正 token
                stats["corrected"] += 1
                break
        else:
            # 全部接受 → 领取红利 token
            bonus = tgt_logits[:, n_ctx - 1 + len(drafted), :].argmax(dim=-1, keepdim=True)
            tokens.append(bonus)
        stats["accepted"] += accepted_this

        # 4) 拼接与终止
        n_add = 0
        for t in tokens:
            if generated >= max_new_tokens:
                break
            ids = torch.cat([ids, t], dim=1)
            generated += 1
            n_add += 1
            if eos_id is not None and int(t) == eos_id:
                break
        stats["steps"].append({"drafted": len(drafted), "accepted": accepted_this,
                               "emitted": n_add})
        n_ctx = ids.shape[1]
        if eos_id is not None and int(ids[0, -1]) == eos_id:
            break

    stats["wall_s"] = round(time.time() - t0, 3)
    stats["tokens_generated"] = generated
    stats["acceptance_rate"] = round(stats["accepted"] / max(stats["draft_tokens"], 1), 4)
    stats["tokens_per_target_forward"] = round(
        generated / max(stats["target_forwards"], 1), 3)
    return {"ids": ids, "stats": stats}


@torch.no_grad()
def greedy_generate_baseline(model, input_ids: torch.Tensor, max_new_tokens: int,
                             eos_id: int | None = None):
    """纯贪心基线（正确性对照 + 速度对照）。"""
    ids = input_ids
    generated = 0
    t0 = time.time()
    for _ in range(max_new_tokens):
        logits = model(ids).logits[:, -1, :]
        t = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, t], dim=1)
        generated += 1
        if eos_id is not None and int(t) == eos_id:
            break
    return {"ids": ids, "stats": {"tokens_generated": generated,
                                  "wall_s": round(time.time() - t0, 3)}}
