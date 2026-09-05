# LiteForge 压缩报告卡 — 0.5B

> 自动生成于 16 条实验记录；全部数字可由仓库脚本一键复现。口径：剪枝/伪量化为方法学质量（稠密推理），部署速度单独由 vLLM 实测。

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

