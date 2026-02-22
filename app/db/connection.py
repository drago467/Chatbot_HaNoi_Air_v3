import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Get a connection to the Postgres database using environment variables."""
    # Priority: POSTGRES_URL, then individual variables
    postgres_url = os.getenv("POSTGRES_URL")
    if postgres_url:
        return psycopg2.connect(postgres_url)
    
    return psycopg2.connect(
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        database=os.getenv("POSTGRES_DB")
    )
