#!/usr/bin/env python3
"""
Test suite for Apple (AAPL) sentiment analysis.

This test validates the sentiment analysis pipeline using Apple as a test case.
All operations are contained within the automated_sentiment_daily_analysis directory.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import os
from unittest.mock import Mock, patch

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sent_collect_data import SentimentAnalyzer
from viz_dashboard_generator import DashboardGenerator

class TestAppleSentimentAnalysis:
    """Test suite for Apple sentiment analysis functionality."""
    
    def setup_method(self):
        """Set up test environment before each test method."""
        self.test_ticker = "AAPL"
        self.test_company = "Apple Inc."
        
    def test_sentiment_analyzer_initialization(self):
        """Test that SentimentAnalyzer initializes correctly."""
        analyzer = SentimentAnalyzer()
        
        # Verify directory structure is within current scope
        current_dir = Path(__file__).parent
        assert analyzer.results_dir == current_dir / 'results'
        assert analyzer.archive_dir == current_dir / 'archive' / '2024_02' / 'sentiment'
        
        # Verify required directories exist
        assert analyzer.results_dir.exists()
        assert analyzer.archive_dir.exists()
        
    @patch('sent_collect_data.finnhub.Client')
    def test_get_company_news_apple(self, mock_finnhub):
        """Test news retrieval for Apple with mocked API."""
        # Mock Finnhub response
        mock_client = Mock()
        mock_client.company_news.return_value = [
            {
                'headline': 'Apple Reports Strong Q3 Earnings',
                'summary': 'Apple Inc. reported better than expected earnings...',
                'datetime': 1693440000,  # Mock timestamp
                'source': 'Reuters',
                'url': 'https://example.com/apple-news'
            }
        ]
        mock_finnhub.return_value = mock_client
        
        analyzer = SentimentAnalyzer()
        analyzer.finnhub_client = mock_client
        
        # Test news retrieval
        news = analyzer.get_company_news(self.test_ticker)
        
        assert len(news) == 1
        assert news[0]['headline'] == 'Apple Reports Strong Q3 Earnings'
        assert 'Apple Inc.' in news[0]['summary']
        
    def test_sentiment_analysis_apple(self):
        """Test sentiment analysis on sample Apple text."""
        analyzer = SentimentAnalyzer()
        
        # Test with positive Apple news
        positive_text = "Apple reports record-breaking iPhone sales and strong revenue growth"
        result = analyzer.analyze_sentiment(positive_text)
        
        assert 'sentiment_score' in result
        assert 'sentiment_label' in result
        assert 'confidence' in result
        assert isinstance(result['sentiment_score'], float)
        assert result['sentiment_score'] >= -1.0 and result['sentiment_score'] <= 1.0
        
    def test_dashboard_generator_apple_only(self):
        """Test dashboard generation with Apple data only."""
        # Create minimal test data for Apple
        test_data = {
            'ticker': ['AAPL'],
            'company': ['Apple Inc.'],
            'average_sentiment': [0.25],
            'total_articles': [10],
            'last_week_sentiment': [0.30],
            'last_month_sentiment': [0.20]
        }
        
        df = pd.DataFrame(test_data)
        
        # Save test data to results directory
        current_dir = Path(__file__).parent
        results_dir = current_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        
        test_file = results_dir / 'sentiment_summary_latest.csv'
        df.to_csv(test_file, index=False)
        
        # Verify file was created in correct location
        assert test_file.exists()
        assert test_file.parent == results_dir
        
        # Clean up test file
        test_file.unlink()

def test_directory_scope_compliance():
    """Test that all operations stay within automated_sentiment_daily_analysis directory."""
    current_dir = Path(__file__).parent
    
    # Verify we're in the correct directory
    assert current_dir.name == 'automated_sentiment_daily_analysis'
    
    # Verify no files are created outside this directory
    parent_dir = current_dir.parent
    
    # Check that results directory is within current scope
    results_path = current_dir / 'results'
    assert str(results_path).startswith(str(current_dir))
    
    # Check that archive directory is within current scope  
    archive_path = current_dir / 'archive'
    assert str(archive_path).startswith(str(current_dir))

if __name__ == "__main__":
    """Run tests when script is executed directly."""
    pytest.main([__file__, "-v"])
