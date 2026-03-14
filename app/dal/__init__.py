"""DAL - Data Access Layer for Weather Chatbot.

Exports:
- weather_dal: get_current_weather, get_hourly_forecast, get_daily_forecast, get_weather_history
- weather_aggregate_dal: get_district_weather, get_city_weather (district/city level)
- weather_helpers: wind_deg_to_vietnamese, wind_speed_to_beaufort, get_uv_status, etc.
- weather_knowledge_dal: detect_hanoi_weather_phenomena, get_seasonal_average, compare_with_seasonal
- location_dal: resolve_location, get_ward_by_id, search_wards
- activity_dal: get_activity_advice
- alerts_dal: get_weather_alerts
- comparison_dal: compare_weather, compare_with_yesterday
- timezone_utils: to_ict, to_utc, now_ict
"""

from app.dal.weather_dal import (
    get_current_weather,
    get_hourly_forecast,
    get_daily_forecast,
    get_weather_history,
    get_weather_range,
)

from app.dal.weather_aggregate_dal import (
    get_district_current_weather,
    get_district_hourly_forecast,
    get_district_daily_forecast,
    get_city_current_weather,
    get_city_hourly_forecast,
    get_city_daily_forecast,
    get_all_districts_current_weather,
)

from app.dal.weather_helpers import (
    wind_deg_to_vietnamese,
    wind_speed_to_beaufort,
    wind_beaufort_vietnamese,
    get_uv_status,
    get_dew_point_status,
    get_pressure_status,
    get_feels_like_status,
    weather_main_to_vietnamese,
)

from app.dal.weather_knowledge_dal import (
    detect_hanoi_weather_phenomena,
    get_seasonal_average,
    compare_with_seasonal,
    get_weather_summary_text,
)

from app.dal.location_dal import (
    resolve_location,
    get_ward_by_id,
    get_all_wards,
    get_districts,
    search_wards,
)

from app.dal.activity_dal import (
    get_activity_advice,
    get_activity_advice_detailed,
)

from app.dal.alerts_dal import (
    get_weather_alerts,
    get_all_district_alerts,
)

from app.dal.comparison_dal import (
    compare_weather,
    compare_with_yesterday,
)

from app.dal.timezone_utils import (
    to_ict,
    to_utc,
    now_ict,
    now_utc,
    format_ict,
)

__all__ = [
    # Weather DAL
    "get_current_weather",
    "get_hourly_forecast",
    "get_daily_forecast",
    "get_weather_history",
    "get_weather_range",
    # Weather Aggregate DAL (district/city level)
    "get_district_current_weather",
    "get_district_hourly_forecast",
    "get_district_daily_forecast",
    "get_city_current_weather",
    "get_city_hourly_forecast",
    "get_city_daily_forecast",
    "get_all_districts_current_weather",
    # Helpers
    "wind_deg_to_vietnamese",
    "wind_speed_to_beaufort",
    "wind_beaufort_vietnamese",
    "get_uv_status",
    "get_dew_point_status",
    "get_pressure_status",
    "get_feels_like_status",
    "weather_main_to_vietnamese",
    # Knowledge
    "detect_hanoi_weather_phenomena",
    "get_seasonal_average",
    "compare_with_seasonal",
    "get_weather_summary_text",
    # Location
    "resolve_location",
    "get_ward_by_id",
    "get_all_wards",
    "get_districts",
    "search_wards",
    # Activity
    "get_activity_advice",
    "get_activity_advice_detailed",
    # Alerts
    "get_weather_alerts",
    "get_all_district_alerts",
    # Comparison
    "compare_weather",
    "compare_with_yesterday",
    # Timezone
    "to_ict",
    "to_utc",
    "now_ict",
    "now_utc",
    "format_ict",
]
