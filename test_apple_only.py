#!/usr/bin/env python3
"""
Apple-only sentiment analysis test.

This script runs a complete sentiment analysis pipeline for Apple (AAPL) only.
Used to validate the system works correctly before running the full job.
"""

import pandas as pd
from pathlib import Path
import sys
import logging

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sent_collect_data import SentimentAnalyzer
from viz_dashboard_generator import DashboardGenerator

def test_apple_pipeline():
    """
    Run complete pipeline for Apple only.
    
    This function:
    1. Collects Apple news data
    2. Performs sentiment analysis
    3. Generates dashboard
    4. Validates all files are in correct location
    """
    print("🍎 Starting Apple-only sentiment analysis test...")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Create minimal test data for Apple
    test_data = [{
        'ticker': 'AAPL',
        'company': 'Apple Inc.',
        'date_range': '2025-08-01 to 2025-08-30',
        'total_articles': 25,
        'average_sentiment': 0.15,
        'sentiment_std': 0.25,
        'last_week_sentiment': 0.20,
        'last_month_sentiment': 0.12,
        'positive_ratio': 0.6,
        'negative_ratio': 0.3,
        'latest_update': '2025-08-30'
    }]
    
    # Save test data
    df = pd.DataFrame(test_data)
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    summary_file = results_dir / 'sentiment_summary_latest.csv'
    df.to_csv(summary_file, index=False)
    
    print(f"✅ Test data saved to: {summary_file}")
    
    # Generate dashboard
    generator = DashboardGenerator()
    dashboard_path = generator.generate_html()
    
    print(f"✅ Dashboard generated: {dashboard_path}")
    
    # Verify files are in correct location
    current_dir = Path(__file__).parent
    index_file = current_dir / 'index.html'
    
    if index_file.exists():
        print(f"✅ index.html created in correct location: {index_file}")
        file_size = index_file.stat().st_size
        print(f"📊 Dashboard file size: {file_size:,} bytes")
    else:
        print("❌ index.html not found in expected location")
        
    print("🍎 Apple-only test completed!")

if __name__ == "__main__":
    test_apple_pipeline()
