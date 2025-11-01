FROM python:3.12-slim

RUN groupadd -g 1000 droidrun \
    && useradd -m -u 1000 -g 1000 -s /bin/bash droidrun

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    android-tools-adb \
    && rm -rf /var/lib/apt/lists/*

USER droidrun

WORKDIR /droidrun

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/home/droidrun/.local/bin:${PATH}"

COPY . .

RUN uv venv && \
    uv pip --no-cache-dir install .[google,anthropic,openai,deepseek,ollama,openrouter]

ENTRYPOINT [".venv/bin/droidrun"]

CMD ["setup"]