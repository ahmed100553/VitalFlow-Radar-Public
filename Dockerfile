# VitalFlow-Radar Production Docker Image
# Optimized for Raspberry Pi (ARM64)

FROM python:3.11-slim-bookworm

# Labels
LABEL maintainer="ahmedsendel@gmail.com"
LABEL version="1.0.0"
LABEL description="VitalFlow-Radar Edge Server"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VITALFLOW_PORT=8080 \
    VITALFLOW_HOST=0.0.0.0 \
    VITALFLOW_DB=/data/vitalflow.db \
    VITALFLOW_LOG=/logs/server.log

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libatlas-base-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash vitalflow \
    && mkdir -p /app /data /logs \
    && chown -R vitalflow:vitalflow /app /data /logs

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY production/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=vitalflow:vitalflow . .

# Switch to non-root user
USER vitalflow

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')" || exit 1

# Default command
CMD ["python", "production/edge_server.py"]
