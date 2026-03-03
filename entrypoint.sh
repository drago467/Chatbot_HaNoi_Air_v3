#!/bin/bash
set -e

echo "Waiting for PostgreSQL to be ready..."
max_retries=30
counter=0
until pg_isready -h postgres -U postgres; do
    sleep 1
    counter=$((counter + 1))
    if [ $counter -ge $max_retries ]; then
        echo "PostgreSQL not available after $max_retries seconds"
        exit 1
    fi
done

echo "PostgreSQL is ready. Initializing database..."
python -m app.db.init_db

echo "Starting Streamlit..."
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
