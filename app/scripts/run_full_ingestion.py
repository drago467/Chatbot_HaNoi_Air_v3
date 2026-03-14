"""
Full Ingestion Pipeline - Runs both weather ingestion AND aggregation.

This script provides a complete pipeline:
1. Ingest weather data from OpenWeather API
2. Aggregate to district and city levels

Usage:
    python -m app.scripts.run_full_ingestion
    
    # Or with options
    python -m app.scripts.run_full_ingestion --weather-only --current-only
"""

import sys
import argparse
from datetime import datetime
import subprocess
import os

# Change to project directory
os.chdir('/c/Users/X1 gen 9/Downloads/Chatbot_HanoiAir')


def run_command(cmd, description):
    """Run a command and print results."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Full Weather Ingestion Pipeline")
    
    # Ingestion options (passed to ingest script)
    parser.add_argument("--weather-only", action="store_true", 
                       help="Only ingest weather (no air pollution)")
    parser.add_argument("--air-only", action="store_true", 
                       help="Only ingest air pollution (no weather)")
    parser.add_argument("--days", type=int, default=14, 
                       help="Number of days for history (default: 14)")
    parser.add_argument("--history-only", action="store_true", 
                       help="Only ingest history data")
    parser.add_argument("--current-only", action="store_true", 
                       help="Only ingest current data")
    parser.add_argument("--forecast-only", action="store_true", 
                       help="Only ingest forecast data")
    
    # Aggregation options
    parser.add_argument("--skip-aggregation", action="store_true", 
                       help="Skip aggregation step")
    parser.add_argument("--hourly-only", action="store_true", 
                       help="Only aggregate hourly data")
    parser.add_argument("--daily-only", action="store_true", 
                       help="Only aggregate daily data")
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    print(f"Starting Full Ingestion Pipeline at {start_time}")
    print(f"Python: {sys.executable}")
    
    success = True
    
    # Step 1: Ingestion
    if not args.history_only or not args.current_only or not args.forecast_only:
        # Build ingestion command
        cmd = [sys.executable, "-m", "app.scripts.ingest_openweather_async"]
        
        if args.weather_only:
            cmd.append("--weather-only")
        if args.air_only:
            cmd.append("--air-only")
        if args.days:
            cmd.extend(["--days", str(args.days)])
        if args.history_only:
            cmd.append("--history-only")
        if args.current_only:
            cmd.append("--current-only")
        if args.forecast_only:
            cmd.append("--forecast-only")
        
        cmd_str = " ".join(cmd)
        success = run_command(cmd_str, "STEP 1: WEATHER INGESTION")
        
        if not success:
            print("WARNING: Ingestion had errors, but continuing with aggregation...")
    
    # Step 2: Aggregation
    if not args.skip_aggregation:
        # Build aggregation command
        cmd = [sys.executable, "-m", "app.scripts.run_aggregation"]
        
        if args.hourly_only:
            cmd.append("--hourly-only")
        if args.daily_only:
            cmd.append("--daily-only")
        
        cmd_str = " ".join(cmd)
        success = run_command(cmd_str, "STEP 2: DATA AGGREGATION")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*60}")
    print(f"Pipeline completed in {duration:.2f} seconds")
    print(f"{'='*60}")
    
    if success:
        print("✓ All steps completed successfully!")
    else:
        print("⚠ Some steps had warnings or errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
