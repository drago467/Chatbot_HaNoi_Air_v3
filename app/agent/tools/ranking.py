"""Ranking tools — district_ranking, ward_ranking_in_district."""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Tool: get_district_ranking ==============

class GetDistrictRankingInput(BaseModel):
    metric: str = Field(
        default="nhiet_do",
        description="Chỉ số: nhiet_do, do_am, gio, mua, uvi, ap_suat, diem_suong, may"
    )
    order: str = Field(default="cao_nhat", description="Thứ tự: cao_nhat hoặc thap_nhat")
    limit: int = Field(default=5, description="Số lượng kết quả (1-30)")


@tool(args_schema=GetDistrictRankingInput)
def get_district_ranking(metric: str = "nhiet_do", order: str = "cao_nhat", limit: int = 5) -> dict:
    """Xếp hạng các QUẬN/HUYỆN theo chỉ số thời tiết.

    DÙNG KHI: "quận nào nóng nhất?", "nơi nào mưa nhiều nhất?",
    "xếp hạng nhiệt độ các quận", "đâu gió giật mạnh nhất?".
    Chỉ số hỗ trợ: nhiet_do, do_am, gio, mua, uvi, ap_suat, diem_suong, may.
    Trả về: top N quận/huyện sắp xếp theo chỉ số.
    """
    from app.dal.weather_aggregate_dal import get_district_rankings
    from app.agent.tools.output_builder import build_district_ranking_output
    return build_district_ranking_output(get_district_rankings(metric, order, limit))


# ============== Tool: get_ward_ranking_in_district ==============

class GetWardRankingInput(BaseModel):
    district_name: str = Field(description="Tên quận/huyện (ví dụ: Cầu Giấy, Đống Đa)")
    metric: str = Field(default="nhiet_do", description="Chỉ số: nhiet_do, do_am, gio, uvi")
    order: str = Field(default="cao_nhat", description="Thứ tự: cao_nhat hoặc thap_nhat")
    limit: int = Field(default=10, description="Số lượng kết quả (1-30)")


@tool(args_schema=GetWardRankingInput)
def get_ward_ranking_in_district(
    district_name: str, metric: str = "nhiet_do",
    order: str = "cao_nhat", limit: int = 10
) -> dict:
    """Xếp hạng các PHƯỜNG/XÃ trong một quận/huyện theo chỉ số thời tiết.

    DÙNG KHI: "phường nào nóng nhất ở Cầu Giấy?", "xếp hạng độ ẩm ở Đống Đa",
    "đâu UV cao nhất trong quận?".
    Chỉ số hỗ trợ: nhiet_do, do_am, gio, uvi.
    Trả về: top N phường/xã trong quận sắp xếp theo chỉ số.
    """
    from app.dal.weather_aggregate_dal import get_ward_rankings_in_district

    from app.agent.tools.output_builder import build_ward_ranking_output, build_error_output
    # Resolve district name → district_id using scoped resolution
    from app.dal.location_dal import resolve_location_scoped
    resolved = resolve_location_scoped(district_name, target_scope="district")
    if resolved.get("status") == "not_found":
        return build_error_output({"error": "location_not_found", "message": f"Không tìm thấy quận/huyện: {district_name}"})
    if resolved.get("level") != "district":
        return build_error_output({"error": "location_not_district", "message": f"Địa điểm không phải quận/huyện: {district_name}"})

    district_id = resolved["data"].get("district_id")
    if not district_id:
        return build_error_output({"error": "no_district_id", "message": f"Thiếu district_id cho {district_name}"})

    return build_ward_ranking_output(get_ward_rankings_in_district(district_id, metric, order, limit))
