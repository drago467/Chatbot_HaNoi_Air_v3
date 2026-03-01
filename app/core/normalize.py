"""Text normalization utilities for Vietnamese location names."""

import unicodedata


def normalize_name(name: str) -> str:
    """Normalize Vietnamese location name for reliable search.

    Goals:
    - lowercase
    - remove Vietnamese diacritics
    - normalize whitespace
    - map 'đ/Đ' -> 'd'

    Examples:
    - "Phường Đống Đa" -> "phuong dong da"
    - "Xã Dương Hòa"   -> "xa duong hoa"
    """
    if not name:
        return ""

    s = name.strip().lower()
    s = s.replace("đ", "d")

    # Remove combining marks
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

    # Collapse whitespace
    s = " ".join(s.split())
    return s
