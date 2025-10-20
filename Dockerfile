FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EVC_USE_TK=0

WORKDIR /app

# Install system deps if needed (add as required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY src/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy app source
COPY src/ /app/

EXPOSE 8080

# Use PORT env if provided by the platform (Render sets $PORT)
CMD ["sh", "-c", "streamlit run web_app.py --server.address=0.0.0.0 --server.port ${PORT:-8080}"]


