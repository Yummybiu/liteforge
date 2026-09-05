FROM python:3.12-slim

# LiteForge 开发/评测镜像（CPU）；GPU 实验推荐实验室原生环境 + conda
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY liteforge ./liteforge
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e .

COPY tests ./tests
COPY examples ./examples
COPY scripts ./scripts

# 默认跑离线测试（≈20s）；覆盖式使用：
#   docker run --rm liteforge python examples/01_quickstart_ppl.py --model <model>
CMD ["python", "-m", "pytest", "tests/", "-q"]
