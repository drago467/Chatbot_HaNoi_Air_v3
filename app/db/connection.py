import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
import threading

load_dotenv()

# Global connection pool
_connection_pool = None
_pool_lock = threading.Lock()


def get_connection_pool(minconn=1, maxconn=10):
    """Get or create the global connection pool (thread-safe)."""
    global _connection_pool
    
    if _connection_pool is None:
        with _pool_lock:
            # Double-check after acquiring lock
            if _connection_pool is None:
                postgres_url = os.getenv("DATABASE_URL")
                if postgres_url:
                    _connection_pool = pool.ThreadedConnectionPool(
                        minconn=minconn,
                        maxconn=maxconn,
                        dsn=postgres_url
                    )
                else:
                    _connection_pool = pool.ThreadedConnectionPool(
                        minconn=minconn,
                        maxconn=maxconn,
                        user=os.getenv("POSTGRES_USER"),
                        password=os.getenv("POSTGRES_PASSWORD"),
                        host=os.getenv("POSTGRES_HOST"),
                        port=os.getenv("POSTGRES_PORT"),
                        database=os.getenv("POSTGRES_DB")
                    )
    
    return _connection_pool


def get_db_connection():
    """Get a connection from the pool.
    
    Note: Timezone should be set at query level using:
    SET TIMEZONE = 'Asia/Ho_Chi_Minh';
    """
    return get_connection_pool().getconn()


def release_connection(conn):
    """Release a connection back to the pool."""
    if conn:
        try:
            # Rollback any dirty state before returning to pool
            conn.rollback()
        except Exception:
            pass
        get_connection_pool().putconn(conn)


def close_all_connections():
    """Close all connections in the pool (thread-safe)."""
    global _connection_pool
    with _pool_lock:
        if _connection_pool:
            _connection_pool.closeall()
            _connection_pool = None
