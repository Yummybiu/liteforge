# LiteForge 压缩报告卡 — 1.5B

> 自动生成于 27 条实验记录；全部数字可由仓库脚本一键复现。口径：剪枝/伪量化为方法学质量（稠密推理），部署速度单独由 vLLM 实测。

## 一、压缩-质量对照（WikiText-2 PPL）

| 方法 | 设置 | PPL ↓ |
|---|---|---|
| dense | - | 9.26 |
| wanda | 0.5/unstructured | 13.34 |
| rtn | W4/g128 | 10.37 |
| obc | 0.5/unstructured | 13.73 |
| gptq | W4/g128 | 9.91 |
| obc | 0.5/unstructured | 11.67 |

## 二、预算分配（V2）

- None @ target 2.25 bit：实际 2.250 bit，预测总损失 39476888624.9424，分布 {'w2g128': 196}
- None @ target 2.25 bit：实际 2.250 bit，预测总损失 39476888624.9425，分布 {'w2g128': 196}
- None @ target 2.25 bit：实际 - bit，预测总损失 1895186689.3312，分布 {'w4g128': 196}
- None @ target 2.5 bit：实际 2.500 bit，预测总损失 18051827756.6918，分布 {'w3g128': 52, 'w2g128': 143, 'w4g128': 1}
- None @ target 2.5 bit：实际 2.500 bit，预测总损失 18169166051.5033，分布 {'w3g128': 66, 'w2g128': 129, 'w4g128': 1}
- None @ target 2.5 bit：实际 - bit，预测总损失 1895186689.3312，分布 {'w4g128': 196}
- None @ target 2.75 bit：实际 2.750 bit，预测总损失 11166694909.9416，分布 {'w4g128': 11, 'w3g128': 95, 'w2g128': 90}
- None @ target 2.75 bit：实际 2.750 bit，预测总损失 11753156022.8558，分布 {'w3g128': 114, 'w2g128': 78, 'w4g128': 4}
- None @ target 2.75 bit：实际 - bit，预测总损失 1895186689.3312，分布 {'w4g128': 196}

### 预测 vs 实际（损失模型的可信度）

| 方案 | 预测损失 | 实际 PPL |
|---|---|---|
| alloc_qwen1.5B_t2.25_dp.json | 39476888624.9424 | 475939.43 |
| alloc_qwen1.5B_t2.25_unifw4.json | 1895186689.3312 | 9.83 |
| alloc_qwen1.5B_t2.5_dp.json | 18051827756.6918 | 92628.02 |
| alloc_qwen1.5B_t2.5_unifw4.json | 1895186689.3312 | 9.83 |
| alloc_qwen1.5B_t2.75_dp.json | 11166694909.9416 | 2777.20 |
| alloc_qwen1.5B_t2.75_unifw4.json | 1895186689.3312 | 9.83 |

## 三、SmoothQuant W8A8 α 扫描

```json
{
 "task": "smooth-alpha",
 "model": "F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-1.5B",
 "method": "smoothquant",
 "params": {
  "alphas": [
   0.4,
   0.5,
   0.6,
   0.7,
   0.8
  ],
  "calib_size": 8
 },
 "metrics": {
  "total_no_smooth": 122958292.99746704,
  "total_by_alpha": {
   "0.4": 67997515.35647583,
   "0.5": 44408300.517082214,
   "0.6": 33287061.702568054,
   "0.7": 30910115.030044556,
   "0.8": 36505971.74438095
  },
  "best_alpha": 0.7,
  "improvement_ratio": 3.98
 },
 "env": {
  "torch": "2.14.0+cu126",
  "cuda_available": true,
  "device_name": "NVIDIA RTX A5000",
  "timestamp": "2026-09-06 00:02:24"
 },
 "per_layer": {
  "model.layers.0.self_attn.q_proj": {
   "by_alpha": {
    "0.4": 270826.884765625,
    "0.5": 137189.4921875,
    "0.6": 111240.4208984375,
    "0.7": 187106.376953125,
    "0.8": 409068.9296875
   },
   "no_smooth": 567137.1328125,
   "best_alpha": 0.6
  },
  "model.layers.0.self_attn.k_proj": {
   "by_alpha": {
    "0.4": 5626.342346191406,
    "0.5": 4961.83203125,
    "0.6": 4508.4246826171875,
    "0.7": 4601.908447265625,
    "0.8": 5163.816833496094
   },
   "no_smooth": 114517.2998046875,
   "best_alpha": 0.6
  },
  "model.layers.0.self_attn.v_proj": {
   "by_alpha": {
    "0.4": 907.3454895019531,
    "0.5": 735.0914764404297,
    "0.6": 642.6500930786133,
    "0.7": 601.6292343139648,
    "0.8": 634.0177383422852
   },
   "no_smooth": 13013.872924804688,
   "best_alpha": 0.7
  },
  "model.layers.0.self_attn.o_proj": {
   "by_
```

## 四、投机解码

```json
{
 "task": "spec-bench",
 "model": "F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-1.5B",
 "method": "speculative",
 "params": {
  "draft": "F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-0.5B",
  "k": 8,
  "max_new": 256,
  "prompt_words": 5
 },
 "metrics": {
  "exact_vs_greedy": true,
  "acceptance_rate": 0.7986,
  "tokens_per_target_forward": 7.314,
  "speedup_wall": 0.97,
  "baseline_wall_s": 19.968,
  "spec_wall_s": 20.69
 },
 "env": {
  "torch": "2.14.0+cu126",
  "cuda_available": true,
  "device_name": "NVIDIA RTX A5000",
  "timestamp": "2026-09-06 00:06:21"
 }
}
```

## 五、MMLU-mini 下游

```json
{
 "task": "eval-mmlu",
 "model": "F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-1.5B",
 "method": "dense",
 "params": {
  "n": 500
 },
 "metrics": {
  "task": "eval-mmlu",
  "n": 500,
  "accuracy": 0.572,
  "by_bucket": {
   "humanities": {
    "acc": 0.522,
    "n": 182
   },
   "other": {
    "acc": 0.5647,
    "n": 85
   },
   "social": {
    "acc": 0.7156,
    "n": 109
   },
   "stem": {
    "acc": 0.5242,
    "n": 124
   }
  },
  "model": "F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-1.5B"
 },
 "env": {
  "torch": "2.14.0+cu126",
  "cuda_available": true,
  "device_name": "NVIDIA RTX A5000",
  "timestamp": "2026-09-06 00:03:55"
 }
}
```

