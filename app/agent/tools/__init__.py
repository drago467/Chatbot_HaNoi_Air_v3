"""Tools package — 25+ LangGraph agent tools cho weather chatbot.

Chia theo domain:
- core.py: resolve_location, get_current_weather, get_weather_alerts
- forecast.py: hourly_forecast, daily_forecast, rain_timeline, best_time
- history.py: weather_history, daily_summary, weather_period
- compare.py: compare_weather, compare_with_yesterday, seasonal_comparison
- ranking.py: district_ranking, ward_ranking
- insight.py: temperature_trend, comfort_index, weather_change_alert,
              detect_phenomena, clothing_advice, activity_advice
- insight_new.py: 6 tool mới (UV safe, pressure trend, daily rhythm,
                  humidity timeline, sunny periods, multi-metric comparison)
"""

from app.agent.tools.core import (
    resolve_location,
    get_current_weather,
    get_weather_alerts,
)
from app.agent.tools.forecast import (
    get_hourly_forecast,
    get_daily_forecast,
    get_rain_timeline,
    get_best_time,
)
from app.agent.tools.history import (
    get_weather_history,
    get_daily_summary,
    get_weather_period,
)
from app.agent.tools.compare import (
    compare_weather,
    compare_with_yesterday,
    get_seasonal_comparison,
)
from app.agent.tools.ranking import (
    get_district_ranking,
    get_ward_ranking_in_district,
)
from app.agent.tools.insight import (
    detect_phenomena,
    get_temperature_trend,
    get_comfort_index,
    get_weather_change_alert,
    get_clothing_advice,
    get_activity_advice,
)
from app.agent.tools.insight_new import (
    get_uv_safe_windows,
    get_pressure_trend,
    get_daily_rhythm,
    get_humidity_timeline,
    get_sunny_periods,
    get_district_multi_compare,
)


# TOOLS list — tất cả tools để đăng ký với LangGraph agent
TOOLS = [
    # Core (3)
    resolve_location,
    get_current_weather,
    get_weather_alerts,
    # Forecast (4)
    get_hourly_forecast,
    get_daily_forecast,
    get_rain_timeline,
    get_best_time,
    # History (3)
    get_weather_history,
    get_daily_summary,
    get_weather_period,
    # Compare (3)
    compare_weather,
    compare_with_yesterday,
    get_seasonal_comparison,
    # Ranking (2)
    get_district_ranking,
    get_ward_ranking_in_district,
    # Insight (6)
    detect_phenomena,
    get_temperature_trend,
    get_comfort_index,
    get_weather_change_alert,
    get_clothing_advice,
    get_activity_advice,
    # Insight New (6)
    get_uv_safe_windows,
    get_pressure_trend,
    get_daily_rhythm,
    get_humidity_timeline,
    get_sunny_periods,
    get_district_multi_compare,
]

__all__ = [t.name for t in TOOLS] + ["TOOLS"]
