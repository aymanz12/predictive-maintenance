# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including git for DVC if needed later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src src
COPY api api
COPY app app
# Note: 'models' will be pulled by DVC in CI/CD before this build context, 
# ensuring they are present locally during build.
COPY models models

# Expose ports for API (8000) and Gradio (7860)
EXPOSE 8000
EXPOSE 7860

# Copy and set entrypoint script
COPY docker/entrypoint.sh .
RUN chmod +x entrypoint.sh

# Environment variables
ENV PYTHONPATH=/app

CMD ["./entrypoint.sh"]
