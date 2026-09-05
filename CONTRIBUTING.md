# 贡献指南

欢迎 Issue 与 PR。本项目是一个人的研究型仓库，贡献前请读 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
（三层架构与诚实边界）和 [docs/decision_log.md](docs/decision_log.md)（设计决策）。

## 纪律（与仓库既有决策一致）

1. **口径诚实**：剪枝/伪量化的质量数字不得与部署速度混写；manifest 必须标注 pseudo 口径。
2. **测试先行**：新算法模块必须带离线单测（微型模型 + mock，不依赖下载）；
   数学声称（恒等式/最优性/精确性）要有定理级测试。
3. **决策留痕**：改动核心设计时在 docs/decision_log.md 追加条目——动机、
   替代方案、失败实验与负结果同样记录。
4. **结果 schema**：所有实验输出统一 JSON（task/model/method/params/metrics/env），
   report/card 依赖该 schema 聚合。

## 开发流程

```bash
make lint      # 死导入扫描
make test      # 52 项离线单测
make coverage  # 覆盖率
```

## 提交信息

约定式前缀：`feat:` / `fix:` / `docs:` / `exp:`（实验数据）。
