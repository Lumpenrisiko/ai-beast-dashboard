FROM python:3.12-slim

# Install nvidia-smi (CUDA runtime)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default config
ENV DASHBOARD_HOST=0.0.0.0
ENV DASHBOARD_PORT=8083
ENV LM_STUDIO_URL=http://host.docker.internal:11434
ENV LM_STUDIO_LOG_DIR=
ENV LACT_ENABLED=false
ENV STATS_INTERVAL=2

EXPOSE 8083

CMD ["python3", "main.py"]
