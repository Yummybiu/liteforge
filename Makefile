# LiteForge 开发与复现入口
# 用法: make <target>   （Windows Git Bash 自带 make；纯 CMD 用户看各 target 的命令）

.PHONY: test coverage smoke-cpu hero card lint clean

test:            ## 离线单测（≈22s，无 GPU/无网络）
	python -m pytest tests/ -q

coverage:        ## 单测 + 覆盖率
	python -m pytest tests/ -q --cov=liteforge --cov-report=term-missing

smoke-cpu:       ## 冒烟（需本地模型，CPU 慢）: bash scripts/run_smoke.sh
	bash scripts/run_smoke.sh

hero:            ## 从 results/*.json 再生 README 横幅图
	python scripts/make_hero.py

card:            ## 生成 0.5B 报告卡
	python -m liteforge.cli report-card --model-key 0.5B --out results/card_0.5B.md

lint:            ## 基础静态检查（编译 + 死导入扫描）
	python -m compileall -q liteforge/
	python scripts/dead_imports.py

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; rm -rf .pytest_cache
