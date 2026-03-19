"""
Run Aggregation - Entry point for weather data aggregation.

This script should be run AFTER weather data ingestion is complete.
It aggregates ward-level data to district and city levels.

Usage:
    python -m app.scripts.run_aggregation

Or in Python:
    from app.scripts.run_aggregation import run_aggregation
    results = run_aggregation()
"""

import sys
import argparse
from datetime import datetime

from app.scripts.aggregate_weather import (
    aggregate_district_hourly,
    aggregate_city_hourly,
    aggregate_district_daily,
    aggregate_city_daily,
)


def run_aggregation(data_kinds_hourly=None, data_kinds_daily=None):
    """Run weather aggregation for specified data kinds.
    
    Args:
        data_kinds_hourly: List of hourly data kinds to aggregate. 
                          Default: ['current', 'forecast', 'history']
        data_kinds_daily: List of daily data kinds to aggregate.
                         Default: ['forecast', 'history']
    
    Returns:
        Dict with results for each aggregation task
    """
    if data_kinds_hourly is None:
        data_kinds_hourly = ['current', 'forecast', 'history']
    if data_kinds_daily is None:
        data_kinds_daily = ['forecast', 'history']
    
    results = {}
    start_time = datetime.now()
    
    print(f"Starting aggregation at {start_time}")
    print(f"Hourly data kinds: {data_kinds_hourly}")
    print(f"Daily data kinds: {data_kinds_daily}")
    print()
    
    # Hourly aggregations
    print("=== Hourly Aggregation ===")
    for dk in data_kinds_hourly:
        print(f"  Aggregating district_hourly ({dk})...")
        results[f"district_hourly_{dk}"] = aggregate_district_hourly(dk)
        print(f"    Result: {results[f'district_hourly_{dk}']}")
        
        print(f"  Aggregating city_hourly ({dk})...")
        results[f"city_hourly_{dk}"] = aggregate_city_hourly(dk)
        print(f"    Result: {results[f'city_hourly_{dk}']}")
    
    print()
    
    # Daily aggregations
    print("=== Daily Aggregation ===")
    for dk in data_kinds_daily:
        print(f"  Aggregating district_daily ({dk})...")
        results[f"district_daily_{dk}"] = aggregate_district_daily(dk)
        print(f"    Result: {results[f'district_daily_{dk}']}")
        
        print(f"  Aggregating city_daily ({dk})...")
        results[f"city_daily_{dk}"] = aggregate_city_daily(dk)
        print(f"    Result: {results[f'city_daily_{dk}']}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print()
    print(f"=== Completed in {duration:.2f} seconds ===")
    
    # Summary
    success_count = sum(1 for r in results.values() if r.get("status") == "ok")
    error_count = sum(1 for r in results.values() if r.get("status") == "error")
    
    print(f"Success: {success_count}, Errors: {error_count}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run weather aggregation")
    parser.add_argument(
        "--hourly-only", 
        action="store_true", 
        help="Only run hourly aggregation"
    )
    parser.add_argument(
        "--daily-only", 
        action="store_true", 
        help="Only run daily aggregation"
    )
    parser.add_argument(
        "--data-kinds",
        nargs="+",
        choices=['current', 'forecast', 'history'],
        help="Specific data kinds to aggregate"
    )
    
    args = parser.parse_args()
    
    # Determine what to run
    if args.hourly_only:
        hourly_kinds = args.data_kinds or ['current', 'forecast', 'history']
        daily_kinds = []
    elif args.daily_only:
        hourly_kinds = []
        daily_kinds = args.data_kinds or ['forecast', 'history']
    else:
        hourly_kinds = args.data_kinds or ['current', 'forecast', 'history']
        daily_kinds = args.data_kinds or ['forecast', 'history']
    
    results = run_aggregation(hourly_kinds, daily_kinds)
    
    # Exit with error if any failed
    errors = [k for k, v in results.items() if v.get("status") == "error"]
    if errors:
        print(f"ERROR: Some aggregations failed: {errors}")
        sys.exit(1)
    
    print("All aggregations completed successfully!")


if __name__ == "__main__":
    main()
