FROM python:3.11-slim

# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system deps (optional minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY train.py /app/train.py
COPY app.py /app/app.py

# Models folder (artifacts are created by train.py)
RUN mkdir -p /app/models

# Run training at build time (optional for local reproducibility)
# You can comment this if you prefer to train on host and then COPY models/
RUN python /app/train.py

# Expose Flask port
EXPOSE 8000

# Default command: serve with gunicorn for better container use
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]
