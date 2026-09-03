# LiteForge 实验对照表

| 模型 | 方法 | 稀疏度/位宽 | PPL↓ | 生成 tok/s | 记录文件 |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | dense | - | 13.2064 | 13.9 | `20260903_011442_dense.json` |
| Qwen2.5-0.5B | gptq | W4/g128 | 14.1349 | - | `20260903_024052_gptq_scratch.json` |
| Qwen2.5-0.5B | magnitude | 0.5 | 485.5846 | - | `20260903_011734_prune_magnitude.json` |
| Qwen2.5-0.5B | obc | 0.5 | 17.5178 | - | `20260903_233326_prune_obc.json` |
| Qwen2.5-0.5B | rtn | W3/g128 | 51.3712 | - | `20260903_012501_rtn_w3.json` |
| Qwen2.5-0.5B | rtn | W4/g0 | 23.9901 | - | `20260903_012543_rtn_w4.json` |
| Qwen2.5-0.5B | rtn | W4/g128 | 15.6889 | - | `20260903_011818_rtn_w4.json` |
| Qwen2.5-0.5B | rtn | W8/g128 | 13.2184 | - | `20260903_011902_rtn_w8.json` |
| Qwen2.5-0.5B | wanda | 0.25 | 13.7198 | - | `20260903_012151_prune_wanda.json` |
| Qwen2.5-0.5B | wanda | 0.5(2:4) | 70.1408 | - | `20260903_012416_prune_wanda.json` |
| Qwen2.5-0.5B | wanda | 0.5 | 25.5227 | 13.4 | `20260903_011651_prune_wanda.json` |
| Qwen2.5-0.5B | wanda | 0.6 | 76.7116 | - | `20260903_012303_prune_wanda.json` |
| Qwen2.5-0.5B | wanda-sgmix | - | 34.9688 | 15.5 | `20260903_025646_sgmix.json` |
| Qwen2.5-1.5B | dense | - | 9.261 | 14.3 | `20260903_013237_dense.json` |
| Qwen2.5-1.5B | gptq | W4/g128 | 9.9066 | - | `20260903_055219_gptq_scratch.json` |
| Qwen2.5-1.5B | obc | 0.5 | 11.6737 | - | `20260903_235142_prune_obc.json` |
| Qwen2.5-1.5B | rtn | W4/g128 | 10.3747 | - | `20260903_013514_rtn_w4.json` |
| Qwen2.5-1.5B | wanda | 0.5 | 13.3361 | 14.5 | `20260903_013427_prune_wanda.json` |
| Qwen2.5-3B | dense | - | 8.0142 | - | `20260903_014830_dense.json` |
| Qwen2.5-3B | gptq | W4/g128 | 8.3432 | - | `20260904_043334_gptq_scratch.json` |
| Qwen2.5-3B | obc | 0.5 | 9.8108 | - | `20260904_030015_prune_obc.json` |
| Qwen2.5-3B | rtn | W4/g128 | 9.0027 | - | `20260903_015035_rtn_w4.json` |
| Qwen2.5-3B | wanda | 0.5 | 10.8894 | - | `20260903_014952_prune_wanda.json` |
