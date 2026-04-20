FROM python:3.11-slim

# Tạo non-root user cho security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài Python deps trước -> Docker cache được lớp này
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Tất cả entrypoint scripts cần executable
RUN chmod +x entrypoint_api.sh entrypoint_worker.sh entrypoint_beat.sh entrypoint_streamlit.sh \
    && ( [ -f entrypoint.sh ] && chmod +x entrypoint.sh || true )

# Chủ sở hữu appuser
RUN chown -R appuser:appuser /app

USER appuser

# Expose cả FastAPI (8000) và Streamlit (8501)
# Từng service trong docker-compose sẽ set command riêng (entrypoint_*.sh)
EXPOSE 8000 8501

# Không set ENTRYPOINT -> docker-compose sẽ override bằng command:
