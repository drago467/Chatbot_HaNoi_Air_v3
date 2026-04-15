"""Timezone utilities for handling ICT (UTC+7) timezone."""

from datetime import datetime
from typing import Optional
import pytz

ICT = pytz.timezone('Asia/Ho_Chi_Minh')
UTC = pytz.UTC


def to_ict(utc_dt: datetime) -> Optional[datetime]:
    """Convert UTC datetime to ICT (UTC+7).
    
    Args:
        utc_dt: datetime in UTC
        
    Returns:
        datetime in ICT timezone
    """
    if utc_dt is None:
        return None
    if utc_dt.tzinfo is None:
        utc_dt = UTC.localize(utc_dt)
    return utc_dt.astimezone(ICT)


def to_utc(ict_dt: datetime) -> Optional[datetime]:
    """Convert ICT datetime to UTC.
    
    Args:
        ict_dt: datetime in ICT
        
    Returns:
        datetime in UTC timezone
    """
    if ict_dt is None:
        return None
    if ict_dt.tzinfo is None:
        ict_dt = ICT.localize(ict_dt)
    return ict_dt.astimezone(UTC)


def now_ict() -> datetime:
    """Get current datetime in ICT timezone.
    
    Returns:
        Current datetime in ICT
    """
    return datetime.now(ICT)


def now_utc() -> datetime:
    """Get current datetime in UTC.
    
    Returns:
        Current datetime in UTC
    """
    return datetime.now(UTC)


def format_ict(utc_dt: datetime, fmt: str = "%H:%M ngày %d/%m/%Y") -> str:
    """Format UTC datetime as ICT string.
    
    Args:
        utc_dt: datetime in UTC
        fmt: format string
        
    Returns:
        Formatted string in ICT
    """
    if utc_dt is None:
        return ""
    ict_dt = to_ict(utc_dt)
    return ict_dt.strftime(fmt)


# SQL helpers for timezone
# Use in queries: WHERE ts_utc::timestamp AT TIME ZONE 'Asia/Ho_Chi_Minh' > ...
# Or set timezone: SET TIMEZONE 'Asia/Ho_Chi_Minh';
