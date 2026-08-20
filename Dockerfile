FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY utils/ ./utils/
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY configs/ ./configs/
COPY data/ ./data/
COPY pyproject.toml .

RUN pip install --no-cache-dir .

RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
