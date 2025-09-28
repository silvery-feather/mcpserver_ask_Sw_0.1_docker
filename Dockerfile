FROM python:3.12-slim

# faiss 的运行库
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 安装 uv（按 pyproject 安装依赖）
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}" \
    UV_LINK_MODE=copy

WORKDIR /app

# 先拷依赖声明，利用缓存
COPY pyproject.toml ./
RUN uv venv /app/.venv \
 && . /app/.venv/bin/activate \
 && (uv sync --frozen --no-install-project || uv sync --no-install-project)

# 再拷代码（包含 md/ 与 rag_cache/）
COPY . .

# 激活 venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# 暴露 SSE 端口
ENV PORT=8765
EXPOSE 8765

# 默认自动检查索引（你已带 rag_cache，正常也能直接跑）
CMD ["python", "main.py", "--auto"]
