# LiteForge 压缩报告卡 — 0.5B

> 自动生成于 35 条实验记录；全部数字可由仓库脚本一键复现。口径：剪枝/伪量化为方法学质量（稠密推理），部署速度单独由 vLLM 实测。

## 一、压缩-质量对照（WikiText-2 PPL）

| 方法 | 设置 | PPL ↓ |
|---|---|---|
| dense | - | 13.21 |
| wanda | 0.5/unstructured | 25.52 |
| magnitude | 0.5/unstructured | 485.58 |
| rtn | W4/g128 | 15.69 |
| rtn | W8/g128 | 13.22 |
| wanda | 0.25/unstructured | 13.72 |
| wanda | 0.6/unstructured | 76.71 |
| wanda | 0.5/2:4 | 70.14 |
| rtn | W3/g128 | 51.37 |
| rtn | W4/g0 | 23.99 |
| gptq | W4/g128 | 14.13 |
| obc | 0.5/unstructured | 54.16 |
| gptq | W4/g128 | 14.13 |
| wanda-sgmix | - | 29.53 |
| wanda-sgmix | - | 34.97 |
| obc | 0.5/unstructured | 17.52 |
| dense | - | 13.03 |

## 二、预算分配（V2）

- None @ target 2.25 bit：实际 2.250 bit，预测总损失 10577913774.7179，分布 {'w2g128': 168}
- None @ target 2.25 bit：实际 2.250 bit，预测总损失 10577913774.7179，分布 {'w2g128': 168}
- None @ target 2.25 bit：实际 - bit，预测总损失 484261390.3060，分布 {'w4g128': 168}
- None @ target 2.5 bit：实际 2.500 bit，预测总损失 5287005678.6641，分布 {'w2g128': 100, 'w3g128': 68}
- None @ target 2.5 bit：实际 2.500 bit，预测总损失 5327600502.0053，分布 {'w2g128': 86, 'w3g128': 77, 'w4g128': 5}
- None @ target 2.5 bit：实际 - bit，预测总损失 484261390.3060，分布 {'w4g128': 168}
- None @ target 2.75 bit：实际 2.750 bit，预测总损失 3103355268.8842，分布 {'w3g128': 91, 'w4g128': 9, 'w2g128': 68}
- None @ target 2.75 bit：实际 2.750 bit，预测总损失 3119053730.6627，分布 {'w2g128': 64, 'w3g128': 97, 'p75': 2, 'w4g128': 5}
- None @ target 2.75 bit：实际 - bit，预测总损失 484261390.3060，分布 {'w4g128': 168}

### 预测 vs 实际（损失模型的可信度）

| 方案 | 预测损失 | 实际 PPL |
|---|---|---|
| alloc_qwen0.5B_t2.25_dp.json | 10577913774.7179 | 195903.56 |
| alloc_qwen0.5B_t2.25_unifw4.json | 484261390.3060 | 14.16 |
| alloc_qwen0.5B_t2.5_dp.json | 5287005678.6641 | 13934.12 |
| alloc_qwen0.5B_t2.5_unifw4.json | 484261390.3060 | 14.16 |
| alloc_qwen0.5B_t2.75_dp.json | 3103355268.8842 | 651.57 |
| alloc_qwen0.5B_t2.75_unifw4.json | 484261390.3060 | 14.16 |

## 三、SmoothQuant W8A8 α 扫描

```json
{
 "task": "smooth-alpha",
 "model": "F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-0.5B",
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
  "total_no_smooth": 23858744.322993964,
  "total_by_alpha": {
   "0.4": 11487921.32736659,
   "0.5": 8471586.017186224,
   "0.6": 6922536.552536875,
   "0.7": 6537716.478294373,
   "0.8": 7378890.340036809
  },
  "best_alpha": 0.7,
  "improvement_ratio": 3.65
 },
 "env": {
  "torch": "2.14.0+cu126",
  "cuda_available": true,
  "device_name": "NVIDIA RTX A5000",
  "timestamp": "2026-09-05 23:54:29"
 },
 "per_layer": {
  "model.layers.0.self_attn.q_proj": {
   "by_alpha": {
    "0.4": 3038.1517639160156,
    "0.5": 2447.3234252929688,
    "0.6": 2896.5906982421875,
    "0.7": 4019.108856201172,
    "0.8": 7716.232238769531
   },
   "no_smooth": 5714.3328857421875,
   "best_alpha": 0.5
  },
  "model.layers.0.self_attn.k_proj": {
   "by_alpha": {
    "0.4": 493.15329360961914,
    "0.5": 381.61231994628906,
    "0.6": 342.25257873535156,
    "0.7": 463.09156036376953,
    "0.8": 777.972785949707
   },
   "no_smooth": 1165.6306915283203,
   "best_alpha": 0.6
  },
  "model.layers.0.self_attn.v_proj": {
   "by_alpha": {
    "0.4": 4.015387713909149,
    "0.5": 3.1193143129348755,
    "0.6": 3.1976747512817383,
    "0.7": 3.225851595401764,
    "0.8": 4.034596741199493
   },
   "no_smooth": 23.33472180366516,
   "best_alpha": 0.5
  },
  "model.layers.0.self_att
```

## 五、MMLU-mini 下游

```json
{
 "task": "eval-mmlu",
 "model": "F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-0.5B",
 "method": "dense",
 "params": {
  "n": 500
 },
 "metrics": {
  "task": "eval-mmlu",
  "n": 500,
  "accuracy": 0.476,
  "by_bucket": {
   "humanities": {
    "acc": 0.4505,
    "n": 182
   },
   "other": {
    "acc": 0.5059,
    "n": 85
   },
   "social": {
    "acc": 0.5688,
    "n": 109
   },
   "stem": {
    "acc": 0.4113,
    "n": 124
   }
  },
  "model": "F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-0.5B"
 },
 "env": {
  "torch": "2.14.0+cu126",
  "cuda_available": true,
  "device_name": "NVIDIA RTX A5000",
  "timestamp": "2026-09-05 23:55:30"
 }
}
```

