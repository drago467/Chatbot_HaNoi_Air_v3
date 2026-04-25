FROM python:3.11-slim

# Tạo non-root user cho security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài Python deps trước → Docker layer cache được giữ khi code thay đổi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source
COPY . .

# Entrypoint scripts cần executable (sau R15 gỡ Celery → chỉ còn api + streamlit)
RUN chmod +x entrypoint_api.sh entrypoint_streamlit.sh

# Chủ sở hữu appuser (non-root best practice)
RUN chown -R appuser:appuser /app

USER appuser

# Expose cả FastAPI (8000) và Streamlit (8501)
# docker-compose.yml override CMD từng service với entrypoint_api.sh / entrypoint_streamlit.sh
EXPOSE 8000 8501

# Không set ENTRYPOINT → docker-compose quyết định command theo service.
# Default nếu chạy docker run đơn: khởi động FastAPI.
CMD ["./entrypoint_api.sh"]
