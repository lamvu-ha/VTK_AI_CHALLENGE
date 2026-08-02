# Dockerfile for AIC 2026 Participant Model Submission
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and dataset structures
COPY src/ ./src/
COPY main.py .

# Mount volumes for input test data and output submission
VOLUME ["/input", "/output"]

# Run model pipeline
CMD ["python", "main.py"]
