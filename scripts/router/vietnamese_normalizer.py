"""
Vietnamese text normalizer for SLM Router.

Handles:
- Telex (no-diacritic) → proper Vietnamese Unicode
- Location name canonicalization (Cau Giay → Cầu Giấy)
- Whitespace normalization + NFC form
- Augmentation: generate no-diacritic variants for training robustness

Usage:
    from vietnamese_normalizer import normalize_vn, strip_diacritics

    text = normalize_vn("bay gio thoi tiet cau giay the nao")
    # → "Bây giờ thời tiết Cầu Giấy thế nào"
"""

import re
import unicodedata
from pathlib import Path

# ── Hanoi districts & wards (canonical names) ──────────────────────────────
HANOI_DISTRICTS = {
    # quận nội thành
    "ba dinh": "Ba Đình", "hoan kiem": "Hoàn Kiếm", "tay ho": "Tây Hồ",
    "long bien": "Long Biên", "cau giay": "Cầu Giấy", "dong da": "Đống Đa",
    "hai ba trung": "Hai Bà Trưng", "hoang mai": "Hoàng Mai",
    "thanh xuan": "Thanh Xuân", "nam tu liem": "Nam Từ Liêm",
    "bac tu liem": "Bắc Từ Liêm", "ha dong": "Hà Đông",
    # huyện ngoại thành
    "gia lam": "Gia Lâm", "dong anh": "Đông Anh", "soc son": "Sóc Sơn",
    "thanh tri": "Thanh Trì", "thuong tin": "Thường Tín",
    "phu xuyen": "Phú Xuyên", "ung hoa": "Ứng Hòa",
    "my duc": "Mỹ Đức", "chuong my": "Chương Mỹ",
    "thanh oai": "Thanh Oai", "quoc oai": "Quốc Oai",
    "thach that": "Thạch Thất", "phuc tho": "Phúc Thọ",
    "dan phuong": "Đan Phượng", "hoai duc": "Hoài Đức",
    "me linh": "Mê Linh", "son tay": "Sơn Tây", "ba vi": "Ba Vì",
}

# ── Common Vietnamese phrases (sorted by length desc for longest-match) ──
TELEX_PHRASES = {
    # Thời gian
    "bay gio": "bây giờ", "hom nay": "hôm nay", "ngay mai": "ngày mai",
    "hom qua": "hôm qua", "cuoi tuan": "cuối tuần", "tuan nay": "tuần này",
    "thang nay": "tháng này", "nam ngoai": "năm ngoái",
    "sang nay": "sáng nay", "trua nay": "trưa nay",
    "chieu nay": "chiều nay", "toi nay": "tối nay", "dem nay": "đêm nay",
    "sang mai": "sáng mai", "chieu mai": "chiều mai",
    "may ngay toi": "mấy ngày tới", "tuan toi": "tuần tới",
    # Thời tiết
    "thoi tiet": "thời tiết", "du bao": "dự báo",
    "nhiet do": "nhiệt độ", "do am": "độ ẩm",
    "suong mu": "sương mù", "ap suat": "áp suất",
    "tia uv": "tia UV", "tam nhin": "tầm nhìn",
    "diem suong": "điểm sương", "chi so": "chỉ số",
    # Hiện tượng
    "mua": "mưa", "nang": "nắng", "gio": "gió",
    "giong": "giông", "loc": "lốc",
    "suong": "sương", "may": "mây", "tuyet": "tuyết",
    "sam set": "sấm sét", "mua dong": "mưa dông",
    "mua phun": "mưa phùn", "mua rao": "mưa rào",
    "co bao": "có bão", "con bao": "cơn bão",
    # Trạng thái
    "lanh": "lạnh", "nong": "nóng", "am": "ẩm", "kho": "khô",
    "mat": "mát", "oi": "oi", "oi buc": "oi bức",
    "ret": "rét", "ret dam": "rét đậm", "ret hai": "rét hại",
    # Hoạt động
    "chay bo": "chạy bộ", "di dao": "đi dạo",
    "di choi": "đi chơi", "tuoi cay": "tưới cây",
    "di xe dap": "đi xe đạp",
    # Động từ / phụ từ phổ biến
    "the nao": "thế nào", "nhu the nao": "như thế nào",
    "bao nhieu": "bao nhiêu", "bao nhieu do": "bao nhiêu độ",
    "co khong": "có không",
    "duoc khong": "được không", "dang": "đang",
    "co the": "có thể", "can": "cần", "nen": "nên",
    "khong": "không", "co": "có", "di": "đi",
    "duoc": "được", "manh": "mạnh", "nhe": "nhẹ",
    "do": "độ", "o": "ở",
    # Đại từ / liên từ
    "o do": "ở đó", "the con": "thế còn", "con": "còn",
    "nhi": "nhỉ", "ha": "hả", "vay": "vậy",
    # Thành phố
    "ha noi": "Hà Nội", "da nang": "Đà Nẵng",
    "ho chi minh": "Hồ Chí Minh",
    # POI nổi tiếng
    "ho guom": "Hồ Gươm", "lang bac": "Lăng Bác",
    "noi bai": "Nội Bài", "my dinh": "Mỹ Đình",
    "royal city": "Royal City", "ho tay": "Hồ Tây",
}

# Merge districts into phrases
TELEX_PHRASES.update(HANOI_DISTRICTS)

# Pre-sort by length descending → longest match first
_SORTED_PHRASES = sorted(TELEX_PHRASES.items(), key=lambda x: -len(x[0]))

# Pre-compile regex patterns for performance
_PATTERNS = [(re.compile(r"\b" + re.escape(k) + r"\b"), v) for k, v in _SORTED_PHRASES]

# Multi-space cleanup
_MULTI_SPACE = re.compile(r"\s+")


def _has_vietnamese_diacritics(text: str) -> bool:
    """Check if text already contains Vietnamese diacritical marks."""
    nfd = unicodedata.normalize("NFD", text)
    for c in nfd:
        if unicodedata.category(c) == "Mn":  # combining mark = diacritic
            return True
    if "đ" in text or "Đ" in text:
        return True
    return False


def normalize_vn(text: str) -> str:
    """
    Normalize Vietnamese text:
    - If input already has diacritics → only NFC normalize + whitespace cleanup
    - If input is Telex / no-diacritic → apply full phrase replacement

    Args:
        text: raw user input (possibly Telex / no-diacritic)

    Returns:
        Normalized Vietnamese text
    """
    if not text or not text.strip():
        return text

    t = unicodedata.normalize("NFC", text.strip())

    # If already proper Vietnamese with diacritics → minimal normalization
    if _has_vietnamese_diacritics(t):
        return _MULTI_SPACE.sub(" ", t).strip()

    # Telex / no-diacritic input → full replacement pipeline
    lower = t.lower()

    for pattern, replacement in _PATTERNS:
        lower = pattern.sub(replacement, lower)

    # Clean whitespace
    result = _MULTI_SPACE.sub(" ", lower).strip()

    # Capitalize first letter
    if result:
        result = result[0].upper() + result[1:]

    return result


def strip_diacritics(text: str) -> str:
    """
    Remove Vietnamese diacritics → ASCII-like Telex output.
    Used for data augmentation: generate no-diacritic variants.

    Args:
        text: proper Vietnamese text

    Returns:
        Text without diacritics (lowercase)
    """
    nfd = unicodedata.normalize("NFD", text)
    # Remove combining marks (diacritics)
    ascii_text = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    # Handle special Vietnamese characters
    replacements = {"đ": "d", "Đ": "D"}
    for old, new in replacements.items():
        ascii_text = ascii_text.replace(old, new)
    return ascii_text.lower()


def augment_with_telex(record: dict, input_field: str = "input") -> list[dict]:
    """
    Given a training record with proper Vietnamese input,
    return [original, telex_variant] for data augmentation.

    Only creates variant if original actually has diacritics.

    Args:
        record: training record dict
        input_field: key containing user input text

    Returns:
        List of 1 or 2 records (original + optional telex variant)
    """
    original_text = record.get(input_field, "")
    telex_text = strip_diacritics(original_text)

    if telex_text == original_text.lower():
        # Already ASCII / no diacritics → no variant needed
        return [record]

    import copy
    variant = copy.deepcopy(record)
    variant[input_field] = telex_text
    return [record, variant]


# ── Self-test ──
if __name__ == "__main__":
    tests = [
        ("bay gio thoi tiet cau giay the nao", "Bây giờ thời tiết Cầu Giấy thế nào"),
        ("hom nay ha noi co mua khong", "Hôm nay Hà Nội có mưa không"),
        ("ngay mai dong da nong bao nhieu do", "Ngày mai Đống Đa nóng bao nhiêu độ"),
        ("gio o hoang mai manh khong", "Gió ở Hoàng Mai mạnh không"),
        ("co bao o ha noi khong", "Có bão ở Hà Nội không"),
        ("du bao thoi tiet thanh xuan chieu nay", "Dự báo thời tiết Thanh Xuân chiều nay"),
        ("sang mai di chay bo duoc khong", "Sáng mai đi chạy bộ được không"),
        ("nhiet do thach that tuan nay the nao", "Nhiệt độ Thạch Thất tuần này thế nào"),
        # Already-proper Vietnamese → should not break
        ("Thời tiết Hà Nội hôm nay?", "Thời tiết Hà Nội hôm nay?"),
    ]

    passed = 0
    for inp, expected in tests:
        result = normalize_vn(inp)
        status = "✅" if result == expected else "❌"
        if status == "❌":
            print(f"{status} Input:    '{inp}'")
            print(f"   Expected: '{expected}'")
            print(f"   Got:      '{result}'")
        else:
            print(f"{status} '{inp}' → '{result}'")
            passed += 1

    print(f"\n{passed}/{len(tests)} tests passed")

    # Test strip_diacritics
    print("\n--- strip_diacritics ---")
    print(strip_diacritics("Thời tiết Cầu Giấy hôm nay thế nào?"))
    # → "thoi tiet cau giay hom nay the nao?"
