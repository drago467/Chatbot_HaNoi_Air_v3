#!/bin/bash
# Entrypoint cho Streamlit container (thin UI client — không init DB, API lo)
set -e

echo "[entrypoint_streamlit] Starting Streamlit on port 8501..."
exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
