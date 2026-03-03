#!/bin/bash
set -e

echo "Waiting for PostgreSQL to be ready..."
until pg_isready -h postgres -U postgres; do
    sleep 1
done

echo "PostgreSQL is ready. Initializing database..."
python -m app.db.init_db

echo "Starting Streamlit..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
