# ============================
# Stage 1: Build environment
# ============================
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a virtual environment
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt

# ============================
# Stage 2: Runtime environment
# ============================
FROM python:3.11-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Ensure venv is used
ENV PATH="/opt/venv/bin:$PATH"

# Copy only necessary files
COPY app.py train.py ./
COPY models/ ./models/
COPY requirements.txt .
COPY tests/ ./tests/

# Expose port
EXPOSE 8000

# Run with Gunicorn for production
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
