# RetailPulse – Dockerfile
# Builds a lightweight container that runs the Streamlit dashboard.
#
# Build:   docker build -t retailpulse .
# Run:     docker run -p 8501:8501 retailpulse
# Open:    http://localhost:8501

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install OS-level build tools needed by some Python packages (e.g. prophet, statsmodels)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer so re-builds are fast
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size
# prophet is optional; remove it from requirements.txt if you hit build issues
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the full project into the container
COPY . .

# Create directories that the pipeline writes to at runtime
RUN mkdir -p data/raw data/processed models reports

# The Streamlit dashboard listens on port 8501
EXPOSE 8501

# Health check — Streamlit exposes a health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command: generate demo data then launch the dashboard
# You can override this with: docker run retailpulse python run_pipeline.py
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
