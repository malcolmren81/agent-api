# Phase 2 Agent Backend Dockerfile
# Multi-stage build for optimized production image

# Stage 1: Builder
FROM python:3.11 as builder

# Set working directory
WORKDIR /app

# Set PATH to include user-installed Python packages
ENV PATH="/root/.local/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and Prisma schema
COPY requirements.txt .
COPY prisma/ ./prisma/

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Generate Prisma client in builder stage so it's included in /root/.local
RUN prisma generate

# Stage 2: Runtime
FROM python:3.11

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH

# Set working directory
WORKDIR /app

# Install runtime dependencies including Node.js for Prisma, postgresql-client for migrations
# Use official Node.js binaries instead of NodeSource repository to avoid external dependency failures
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    libssl3 \
    openssl \
    xz-utils \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://nodejs.org/dist/v20.11.0/node-v20.11.0-linux-x64.tar.xz -o /tmp/node.tar.xz \
    && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 \
    && rm /tmp/node.tar.xz \
    && node --version \
    && npm --version

# Copy Python dependencies and Prisma binaries from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache /root/.cache

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY prisma/ ./prisma/
COPY palet8_agents/ ./palet8_agents/
COPY migrations/ ./migrations/

# Prisma client already generated in builder stage and copied via /root/.local
# No need to regenerate here - this speeds up the build

# Create necessary directories
RUN mkdir -p logs generated_images cache

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Create startup script that runs safe SQL migrations then starts the app
RUN echo '#!/bin/sh' > /app/start.sh && \
    echo 'echo "Running safe SQL migrations..."' >> /app/start.sh && \
    echo 'psql "$DATABASE_URL" < migrations/init.sql 2>&1 | grep -E "CREATE|ERROR|DETAIL|NOTICE" || true' >> /app/start.sh && \
    echo 'echo "Migrations complete. Starting server..."' >> /app/start.sh && \
    echo 'exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300' >> /app/start.sh && \
    chmod +x /app/start.sh

# Start the service with migrations
CMD ["/app/start.sh"]
