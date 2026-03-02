"""Database Access Layer - Query helpers with RealDictCursor."""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Import from connection.py to avoid duplication
from app.db.connection import get_db_connection

load_dotenv()


def query(sql: str, params: tuple = None) -> List[Dict[str, Any]]:
    """Execute a query and return results as list of dictionaries.
    
    Args:
        sql: SQL query with %s placeholders for params
        params: Tuple of parameters for the query
        
    Returns:
        List of dictionaries with column names as keys
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


def query_one(sql: str, params: tuple = None) -> Optional[Dict[str, Any]]:
    """Execute a query and return a single result as dictionary.
    
    Args:
        sql: SQL query with %s placeholders for params
        params: Tuple of parameters for the query
        
    Returns:
        Single dictionary or None if no result
    """
    results = query(sql, params)
    return results[0] if results else None


def execute(sql: str, params: tuple = None) -> int:
    """Execute a query and return number of affected rows.
    
    Args:
        sql: SQL query with %s placeholders for params
        params: Tuple of parameters for the query
        
    Returns:
        Number of affected rows
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount
