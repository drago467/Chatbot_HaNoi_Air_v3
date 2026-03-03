"""Weather thresholds - KTTV Vietnam standards and other thresholds."""

# KTTV Viet Nam Standards
KTTV_THRESHOLDS = {
    # Temperature (Celsius)
    "RET_DAM": 15,           # Ttb <= 15°C, ≥ 2 ngày
    "RET_HAI": 13,           # Ttb <= 13°C
    "NANG_NONG": 35,         # Tmax >= 35°C
    "NANG_NONG_GAY_GAT": 37, # Tmax >= 37°C
    "NANG_NONG_DB": 39,      # Tmax >= 39°C
    
    # Rain (mm/24h)
    "MUA_TO": 50,            # 50-100mm/24h
    "MUA_RAT_TO": 100,       # 100-200mm/24h
    
    # Humidity (%)
    "NOM_AM_HUMIDITY": 95,   # humidity >= 95%
    "NOM_AM_HUMIDITY_MEDIUM": 85,  # humidity >= 85%
    "NOM_AM_DEW_DIFF": 2,    # dew_point - temp < 2°C
    "NOM_AM_DEW_DIFF_MEDIUM": 3,  # dew_point - temp < 3°C
    
    # Visibility (meters)
    "SUONG_MU_VISIBILITY": 1000,  # < 1000m
}

# Other thresholds
THRESHOLDS = {
    # UV Index
    "UV_LOW": 2,
    "UV_MODERATE": 5,
    "UV_HIGH": 7,
    "UV_VERY_HIGH": 10,
    "UV_EXTREME": 11,
    
    # Wind (m/s)
    "WIND_CALM": 0.5,
    "WIND_LIGHT": 1.5,
    "WIND_MODERATE": 3.3,
    "WIND_STRONG": 10,
    "WIND_DANGEROUS": 20,
    
    # Rain probability
    "POP_LOW": 0.3,
    "POP_LIKELY": 0.5,
    "POP_VERY_LIKELY": 0.8,
    
    # Humidity (%)
    "HUMIDITY_LOW": 30,
    "HUMIDITY_HIGH": 80,
    
    # Dew Point (Celsius)
    "DEW_POINT_COMFORTABLE": 15,
    "DEW_POINT_NONG_AM": 18,
    "DEW_POINT_OI_BUC": 21,
    
    # Temperature (Celsius)
    "TEMP_EXTREME_COLD": 10,
    "TEMP_COLD": 15,
    "TEMP_COMFORTABLE_MIN": 20,
    "TEMP_COMFORTABLE_MAX": 28,
    "TEMP_HOT": 35,
    "TEMP_EXTREME_HOT": 38,
    
    # Pressure (hPa)
    "PRESSURE_LOW": 1000,
    "PRESSURE_NORMAL": 1013,
    "PRESSURE_HIGH": 1020,
    "PRESSURE_VERY_HIGH": 1030,
}

# Vietnamese status mappings
TEMP_STATUS = {
    "extreme_cold": "Rất lạnh",
    "cold": "Lạnh",
    "cool": "Mát lạnh",
    "comfortable": "Thoải mái",
    "warm": "Nóng nhẹ",
    "hot": "Nóng",
    "extreme_hot": "Rất nóng",
}

WEATHER_DESCRIPTIONS = {
    "Clear": "Trời quang",
    "Clouds": "Trời có mây",
    "Rain": "Có mưa",
    "Drizzle": "Mưa phun",
    "Thunderstorm": "Có giông",
    "Mist": "Có sương mù",
    "Fog": "Sương mù dày",
}
