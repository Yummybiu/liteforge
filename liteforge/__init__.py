"""LiteForge：LLM 压缩-评测-部署一站式工具箱。

子模块：
- prune  : 结构/非结构化剪枝（Magnitude、Wanda，可插拔接口）
- quant  : 伪量化（RTN 从零实现）与 GPTQ/AWQ 薄封装
- eval   : 困惑度（滑窗）、前向/生成速度基准
- deploy : vLLM / llama.cpp 部署实测脚本
- report : 实验结果聚合与 trade-off 图
"""

__version__ = "0.3.0"
