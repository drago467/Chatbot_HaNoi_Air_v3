"""Timezone utilities for handling ICT (UTC+7) timezone."""

from datetime import datetime
from typing import Optional
import pytz

ICT = pytz.timezone('Asia/Ho_Chi_Minh')
UTC = pytz.UTC

_WEEKDAYS_VI = {
    0: "Thứ Hai", 1: "Thứ Ba", 2: "Thứ Tư", 3: "Thứ Năm",
    4: "Thứ Sáu", 5: "Thứ Bảy", 6: "Chủ Nhật",
}


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


def format_ict(utc_dt: datetime, fmt: Optional[str] = None) -> str:
    """Format UTC datetime as ICT string.

    Args:
        utc_dt: datetime in UTC
        fmt: format string. Nếu None, trả mặc định
            "HH:MM Thứ X DD/MM/YYYY" (có weekday để LLM không tự tính sai).

    Returns:
        Formatted string in ICT
    """
    if utc_dt is None:
        return ""
    ict_dt = to_ict(utc_dt)
    if fmt is None:
        weekday = _WEEKDAYS_VI[ict_dt.weekday()]
        return f"{ict_dt.strftime('%H:%M')} {weekday} {ict_dt.strftime('%d/%m/%Y')}"
    return ict_dt.strftime(fmt)


# SQL helpers for timezone
# Use in queries: WHERE ts_utc::timestamp AT TIME ZONE 'Asia/Ho_Chi_Minh' > ...
# Or set timezone: SET TIMEZONE 'Asia/Ho_Chi_Minh';
