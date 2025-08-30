"""
Sentiment analysis module that processes news articles and generates timestamped outputs:
1. Detailed sentiment per article with dates
2. Aggregated sentiment per stock with date ranges
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional
from transformers import pipeline
import finnhub
import time
import shutil
import logging
import sys
from automated_sentiment_daily_analysis.utils.db.sentiment_history import SentimentHistoryDB
from automated_sentiment_daily_analysis.utils.config.ticker_config import load_master_tickers
from automated_sentiment_daily_analysis.utils.config.api_providers_config import FINNHUB_KEY

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Sentiment analysis engine for stock market news.
    
    Processes news articles using FinBERT model and generates sentiment scores.
    All operations are contained within the automated_sentiment_daily_analysis directory.
    
    Attributes:
        mappings (dict): Stock ticker to company name mappings
        results_dir (Path): Directory for storing analysis results
        archive_dir (Path): Directory for archiving historical data
        history_db (SentimentHistoryDB): Database for historical sentiment tracking
        finnhub_client: Finnhub API client for news data
        sentiment_model: FinBERT model for sentiment analysis
    """
    
    def __init__(self):
        """
        Initialize the SentimentAnalyzer with proper directory structure.
        
        Sets up all necessary directories within automated_sentiment_daily_analysis scope.
        Initializes API clients and ML models with comprehensive error handling.
        """
        self.mappings = load_master_tickers()
        
        # FIXED: Work within current directory only (automated_sentiment_daily_analysis)
        current_dir = Path(__file__).parent
        self.results_dir = current_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Create archive directories within current scope
        self.archive_dir = current_dir / 'archive' / '2024_02' / 'sentiment'
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize history database
        self.history_db = SentimentHistoryDB()
        
        # Initialize Finnhub client with error handling
        try:
            if not FINNHUB_KEY:
                raise ValueError("Finnhub API key is not configured")
            self.finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)
            # Test the API key with a simple request
            self.finnhub_client.quote('AAPL')
            logger.info("Successfully initialized Finnhub client")
        except Exception as e:
            logger.error(f"Failed to initialize Finnhub client: {e}")
            raise ValueError(
                "Failed to initialize Finnhub client. Please check your API key configuration:\n"
                "1. Set the FINNHUB_KEY environment variable, or\n"
                "2. Add it to utils/config/api_keys.json"
            )
        
        # Initialize sentiment model
        try:
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
            logger.info("Successfully initialized sentiment model")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment model: {e}")
            raise
        
    def get_company_news(self, ticker: str) -> List[Dict]:
        """
        Retrieve news articles for a specific company from Finnhub API.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            
        Returns:
            List[Dict]: List of news articles with metadata including:
                - headline: Article headline
                - summary: Article summary/content
                - datetime: Unix timestamp of publication
                - source: News source name
                - url: Article URL
                
        Note: Fetches articles from the last 30 days with proper error handling
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            news = self.finnhub_client.company_news(
                ticker,
                _from=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d')
            )
            return news
        except Exception as e:
            print(f"Error getting news for {ticker}: {str(e)}")
            return []
            
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of a single text"""
        try:
            result = self.sentiment_model(text)[0]
            score = result['score']
            if result['label'] == 'negative':
                score = -score
            elif result['label'] == 'neutral':
                score = 0
            return {
                'sentiment_score': score,
                'sentiment_label': result['label'],
                'confidence': result['score']
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {
                'sentiment_score': 0,
                'sentiment_label': 'neutral',
                'confidence': 0
            }
            
    def process_all_stocks(self) -> None:
        """Process all stocks and generate both detailed and summary outputs"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        detailed_results = []
        summary_results = []
        
        for ticker, info in self.mappings.items():
            print(f"\nProcessing {ticker} ({info['name']})...")
            
            try:
                # Get news from Finnhub
                news = self.get_company_news(ticker)
                
                if news:
                    # Process each news item
                    article_results = []
                    for item in news:
                        headline = item.get('headline', '')
                        summary = item.get('summary', '')
                        
                        # Calculate sentiment scores
                        headline_sentiment = self.analyze_sentiment(headline)
                        summary_sentiment = self.analyze_sentiment(summary)
                        
                        # Combined score (weighted average)
                        combined_score = (
                            0.4 * headline_sentiment['sentiment_score'] +
                            0.6 * summary_sentiment['sentiment_score']
                        )
                        
                        article_results.append({
                            'ticker': ticker,
                            'company': info['name'],
                            'date': datetime.fromtimestamp(item.get('datetime', time.time())),
                            'title': headline,
                            'text': summary[:500],  # First 500 chars for reference
                            'source': item.get('source', 'unknown'),
                            'url': item.get('url', '#'),  # Add URL from Finnhub
                            'sentiment_score': combined_score,
                            'sentiment_label': headline_sentiment['sentiment_label'],
                            'confidence': headline_sentiment['confidence']
                        })
                    
                    if article_results:
                        # Add to detailed results
                        detailed_results.extend(article_results)
                        
                        # Calculate summary statistics
                        df = pd.DataFrame(article_results)
                        date_range = f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}"
                        
                        # Calculate weighted scores for different time windows
                        now = datetime.now()
                        week_mask = df['date'] >= (now - timedelta(days=7))
                        month_mask = df['date'] >= (now - timedelta(days=30))
                        
                        summary_results.append({
                            'ticker': ticker,
                            'company': info['name'],
                            'date_range': date_range,
                            'total_articles': len(df),
                            'average_sentiment': df['sentiment_score'].mean(),
                            'sentiment_std': df['sentiment_score'].std(),
                            'last_week_sentiment': df[week_mask]['sentiment_score'].mean() if week_mask.any() else None,
                            'last_month_sentiment': df[month_mask]['sentiment_score'].mean() if month_mask.any() else None,
                            'positive_ratio': (df['sentiment_label'] == 'positive').mean(),
                            'negative_ratio': (df['sentiment_label'] == 'negative').mean(),
                            'latest_update': df['date'].max()
                        })
                else:
                    print(f"No news found for {ticker}")
                    
                # Rate limiting
                time.sleep(0.5)  # Finnhub rate limit
                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                time.sleep(1)  # Longer delay after error
        
        if detailed_results:
            # Create DataFrames
            detailed_df = pd.DataFrame(detailed_results)
            detailed_df.sort_values(['ticker', 'date'], inplace=True)
            
            summary_df = pd.DataFrame(summary_results)
            summary_df.sort_values('average_sentiment', ascending=False, inplace=True)
            
            # Save to historical database
            self.history_db.save_current_data(detailed_df, summary_df)
            
            # Calculate trends from historical data
            trends_df = self.history_db.get_sentiment_trends()
            if not trends_df.empty:
                print("\nSentiment Trends:")
                print(f"Stocks with rising sentiment: {len(trends_df[trends_df['trend'] == 'UP'])}")
                print(f"Stocks with falling sentiment: {len(trends_df[trends_df['trend'] == 'DOWN'])}")
                print(f"Average sentiment change: {trends_df['sentiment_change'].mean():.2f}")
            
            # Save current results with timestamp
            detailed_path = self.results_dir / f"sentiment_detailed_{timestamp}.csv"
            summary_path = self.results_dir / f"sentiment_summary_{timestamp}.csv"
            
            detailed_df.to_csv(detailed_path, index=False)
            summary_df.to_csv(summary_path, index=False)
            
            # Create symlinks for latest files
            latest_detailed = self.results_dir / "sentiment_detailed_latest.csv"
            latest_summary = self.results_dir / "sentiment_summary_latest.csv"
            
            if latest_detailed.exists():
                latest_detailed.unlink()
            if latest_summary.exists():
                latest_summary.unlink()
                
            latest_detailed.symlink_to(detailed_path.name)
            latest_summary.symlink_to(summary_path.name)
            
            # Archive previous files
            for file in self.results_dir.glob("sentiment_*_2*.csv"):
                if timestamp not in str(file):
                    archive_path = self.archive_dir / file.name
                    shutil.move(str(file), str(archive_path))
            
            # Print summary
            print("\nSentiment Analysis Summary:")
            print(f"Total articles processed: {len(detailed_df)}")
            print(f"Companies analyzed: {len(summary_df)}")
            print(f"\nResults saved to:")
            print(f"- Detailed results: {detailed_path}")
            print(f"- Summary results: {summary_path}")
            print(f"- Archive directory: {self.archive_dir}")
            print(f"- Historical database: {self.history_db.db_dir}")
        else:
            print("No results to save")

def main():
    """Main function to run sentiment analysis"""
    print("\nStarting sentiment analysis...")
    analyzer = SentimentAnalyzer()
    analyzer.process_all_stocks()
    print("\nSentiment analysis complete!")

if __name__ == "__main__":
    main() 