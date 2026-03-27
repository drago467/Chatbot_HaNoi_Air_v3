"""Ranking tools — district_ranking, ward_ranking_in_district."""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============== Tool: get_district_ranking ==============

class GetDistrictRankingInput(BaseModel):
    metric: str = Field(
        default="nhiet_do",
        description="Chi so: nhiet_do, do_am, gio, mua, uvi, ap_suat, diem_suong, may"
    )
    order: str = Field(default="cao_nhat", description="Thu tu: cao_nhat hoac thap_nhat")
    limit: int = Field(default=5, description="So luong ket qua (1-30)")


@tool(args_schema=GetDistrictRankingInput)
def get_district_ranking(metric: str = "nhiet_do", order: str = "cao_nhat", limit: int = 5) -> dict:
    """Xep hang cac QUAN/HUYEN theo chi so thoi tiet.

    DUNG KHI: "quan nao nong nhat?", "noi nao mua nhieu nhat?",
    "xep hang nhiet do cac quan", "dau gust manh nhat?".
    Chi so ho tro: nhiet_do, do_am, gio, mua, uvi, ap_suat, diem_suong, may.
    Tra ve: top N quan/huyen sap xep theo chi so.
    """
    from app.dal.weather_aggregate_dal import get_district_rankings
    return get_district_rankings(metric, order, limit)


# ============== Tool: get_ward_ranking_in_district ==============

class GetWardRankingInput(BaseModel):
    district_name: str = Field(description="Ten quan/huyen (vi du: Cau Giay, Dong Da)")
    metric: str = Field(default="nhiet_do", description="Chi so: nhiet_do, do_am, gio, uvi")
    order: str = Field(default="cao_nhat", description="Thu tu: cao_nhat hoac thap_nhat")
    limit: int = Field(default=10, description="So luong ket qua (1-30)")


@tool(args_schema=GetWardRankingInput)
def get_ward_ranking_in_district(
    district_name: str, metric: str = "nhiet_do",
    order: str = "cao_nhat", limit: int = 10
) -> dict:
    """Xep hang cac PHUONG/XA trong mot quan/huyen theo chi so thoi tiet.

    DUNG KHI: "phuong nao nong nhat o Cau Giay?", "xep hang do am o Dong Da",
    "dau UV cao nhat trong quan?".
    Chi so ho tro: nhiet_do, do_am, gio, uvi.
    Tra ve: top N phuong/xa trong quan sap xep theo chi so.
    """
    from app.dal.weather_aggregate_dal import get_ward_rankings_in_district

    # Resolve district name if needed
    from app.dal.location_dal import resolve_location
    resolved = resolve_location(district_name)
    if resolved.get("level") == "district":
        district_name = resolved["data"]["district_name_vi"]
    elif resolved.get("level") == "ward":
        district_name = resolved["data"].get("district_name_vi", district_name)

    return get_ward_rankings_in_district(district_name, metric, order, limit)
