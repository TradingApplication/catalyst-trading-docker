u#!/bin/bash
# Fix TA-Lib build issues in Dockerfiles

echo "Fixing TA-Lib installation in Dockerfiles..."

# Fix Dockerfile.scanner
cat > Dockerfile.scanner << 'EOF'
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \dockerfile_scanner_fixed
    curl \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library with proper paths
RUN cd /tmp && \
    wget https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    cd / && \
    rm -rf /tmp/ta-lib* && \
    ldconfig

# Set library paths for TA-Lib
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python packages WITHOUT TA-Lib
RUN grep -v "TA-Lib\|ta-lib\|talib" requirements.txt > requirements_no_talib.txt || cp requirements.txt requirements_no_talib.txt
RUN pip install --no-cache-dir -r requirements_no_talib.txt

# Now install TA-Lib Python wrapper separately
RUN pip install --no-cache-dir TA-Lib==0.4.28

# Copy service files
COPY scanner_service.py .
COPY database_utils.py .

# Create directories
RUN mkdir -p /app/logs /app/data

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV SERVICE_NAME=scanner_service

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5001/health || exit 1

# Run service
CMD ["python", "scanner_service.py"]
EOF

# Fix Dockerfile.pattern
cat > Dockerfile.pattern << 'EOF'
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library with proper paths
RUN cd /tmp && \
    wget https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    cd / && \
    rm -rf /tmp/ta-lib* && \
    ldconfig

# Set library paths for TA-Lib
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python packages WITHOUT TA-Lib
RUN grep -v "TA-Lib\|ta-lib\|talib" requirements.txt > requirements_no_talib.txt || cp requirements.txt requirements_no_talib.txt
RUN pip install --no-cache-dir -r requirements_no_talib.txt

# Now install TA-Lib Python wrapper separately
RUN pip install --no-cache-dir TA-Lib==0.4.28

# Copy service files
COPY pattern_service.py .
COPY database_utils.py .

# Create directories
RUN mkdir -p /app/logs /app/data /app/models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV SERVICE_NAME=pattern_service

# Expose port
EXPOSE 5002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5002/health || exit 1

# Run service
CMD ["python", "pattern_service.py"]
EOF

# Fix Dockerfile.technical
cat > Dockerfile.technical << 'EOF'
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library with proper paths
RUN cd /tmp && \
    wget https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    cd / && \
    rm -rf /tmp/ta-lib* && \
    ldconfig

# Set library paths for TA-Lib
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python packages WITHOUT TA-Lib
RUN grep -v "TA-Lib\|ta-lib\|talib" requirements.txt > requirements_no_talib.txt || cp requirements.txt requirements_no_talib.txt
RUN pip install --no-cache-dir -r requirements_no_talib.txt

# Now install TA-Lib Python wrapper separately
RUN pip install --no-cache-dir TA-Lib==0.4.28

# Copy service files
COPY technical_service.py .
COPY database_utils.py .

# Create directories
RUN mkdir -p /app/logs /app/data /app/models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV SERVICE_NAME=technical_service

# Expose port
EXPOSE 5003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5003/health || exit 1

# Run service
CMD ["python", "technical_service.py"]
EOF

echo "✓ Fixed Dockerfiles created!"
echo ""
echo "Now rebuild the failed services:"
echo "docker-compose build --no-cache scanner-service"
echo "docker-compose build --no-cache pattern-service"
echo "docker-compose build --no-cache technical-service"