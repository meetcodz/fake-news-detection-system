FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal dependency file for API deployment (no CUDA torch)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy source code and model artifacts
COPY src/ ./src/
COPY utils/ ./utils/
COPY app/ ./app/
COPY configs/ ./configs/
COPY models/ ./models/
COPY pyproject.toml .

# Install the package itself in non-editable mode
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
