"""
Dashboard generation module that creates an interactive HTML report.
Includes sentiment analysis, market data, and Bayesian predictions with historical trends.
"""

import os
import webbrowser
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
import shutil
import json

# Correct package imports
from automated_sentiment_daily_analysis.utils.config.ticker_config import load_master_tickers
from automated_sentiment_daily_analysis.utils.db.sentiment_history import SentimentHistoryDB

class DashboardGenerator:
    def __init__(self):
        self.mappings = load_master_tickers()
        # Set output directory to automated_sentiment_daily_analysis
        self.output_dir = Path(__file__).parent
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize history database
        self.history_db = SentimentHistoryDB()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_master_data(self) -> pd.DataFrame:
        """Load latest sentiment data"""
        try:
            # Load sentiment data
            sentiment_path = self.output_dir / "sentiment_summary_latest.csv"
            if not sentiment_path.exists():
                raise FileNotFoundError(f"Sentiment data not found at {sentiment_path}")
                
            df = pd.read_csv(sentiment_path)
            if df.empty:
                raise ValueError("Sentiment data file is empty")
            
            # Load detailed articles data
            detailed_path = self.output_dir / "sentiment_detailed_latest.csv"
            if detailed_path.exists():
                articles_df = pd.read_csv(detailed_path)
                articles_df['date'] = pd.to_datetime(articles_df['date'])
                
                # Group articles by ticker and convert to list of dictionaries
                articles_by_ticker = {}
                for ticker in df['ticker'].unique():
                    ticker_articles = articles_df[articles_df['ticker'] == ticker].to_dict('records')
                    if ticker_articles:  # Only add if there are articles
                        articles_by_ticker[ticker] = ticker_articles
                
                # Add articles to main dataframe
                df['articles'] = df['ticker'].map(lambda x: articles_by_ticker.get(x, []))
            
            # Create a DataFrame with all tickers from master list
            master_df = pd.DataFrame(list(self.mappings.items()), columns=['ticker', 'info'])
            master_df['company'] = master_df['info'].apply(lambda x: x['name'])
            master_df['sector'] = master_df['info'].apply(lambda x: x.get('sector', 'N/A'))
            master_df = master_df.drop('info', axis=1)
            
            # Merge with sentiment data to ensure all tickers are included
            df = pd.merge(master_df, 
                         df.drop('company', axis=1, errors='ignore'), 
                         on=['ticker'], 
                         how='left')
            
            # Load historical trends if available
            try:
                trends_df = self.history_db.get_sentiment_trends()
                if not trends_df.empty:
                    df = pd.merge(
                        df,
                        trends_df[['ticker', 'sentiment_change', 'trend', 'days_of_history']],
                        on='ticker',
                        how='left'
                    )
            except Exception as e:
                self.logger.warning(f"Could not load historical trends: {e}")
            
            # Log data summary
            total_tickers = len(df)
            tickers_with_data = len(df[df['average_sentiment'].notna()])
            missing_tickers = sorted(df[df['average_sentiment'].isna()]['ticker'].tolist())
            
            self.logger.info(f"Total tickers in master list: {total_tickers}")
            self.logger.info(f"Tickers with sentiment data: {tickers_with_data}")
            self.logger.info(f"Tickers without data ({total_tickers - tickers_with_data}): {', '.join(missing_tickers)}")
            self.logger.info(f"Columns: {', '.join(df.columns)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading sentiment data: {e}")
            raise

    def get_data_funnel_stats(self, df: pd.DataFrame) -> dict:
        """Calculate statistics for the data processing funnel"""
        # Basic funnel stats
        stats = {
            'total_stocks': len(self.mappings),
            'stocks_with_news': len(df[df['total_articles'].notna()]),
            'total_articles': df['total_articles'].sum() if 'total_articles' in df.columns else 0,
            'stocks_with_sentiment': len(df[df['average_sentiment'].notna()]),
        }
        
        # Add sentiment summary statistics
        sentiment_df = df[df['average_sentiment'].notna()]
        if not sentiment_df.empty:
            stats.update({
                'avg_sentiment': sentiment_df['average_sentiment'].mean(),
                'positive_stocks': len(sentiment_df[sentiment_df['average_sentiment'] > 0.5]),
                'negative_stocks': len(sentiment_df[sentiment_df['average_sentiment'] < -0.5]),
                'neutral_stocks': len(sentiment_df[(sentiment_df['average_sentiment'] >= -0.5) & (sentiment_df['average_sentiment'] <= 0.5)]),
                'trending_up': len(sentiment_df[sentiment_df['trend'] == 'UP']) if 'trend' in sentiment_df.columns else 0,
                'trending_down': len(sentiment_df[sentiment_df['trend'] == 'DOWN']) if 'trend' in sentiment_df.columns else 0,
                'most_positive': sentiment_df.nlargest(1, 'average_sentiment').iloc[0] if len(sentiment_df) > 0 else None,
                'most_negative': sentiment_df.nsmallest(1, 'average_sentiment').iloc[0] if len(sentiment_df) > 0 else None,
                'highest_coverage': sentiment_df.nlargest(1, 'total_articles').iloc[0] if len(sentiment_df) > 0 else None
            })
        
        return stats
            
    def generate_article_page(self, ticker: str, company: str, articles: list, timestamp: str) -> Path:
        """Generate a dedicated HTML page for a stock's articles"""
        # Validate input
        if not isinstance(articles, list) or not articles:
            return None
            
        # Filter out any invalid articles and ensure proper data types
        valid_articles = []
        for article in articles:
            if isinstance(article, dict) and all(k in article for k in ['title', 'date', 'source', 'sentiment_score']):
                try:
                    valid_article = {
                        'title': str(article['title']),
                        'date': pd.to_datetime(article['date']).strftime('%Y-%m-%d'),
                        'source': str(article['source']),
                        'sentiment_score': float(article['sentiment_score']),
                        'text': str(article.get('text', ''))[:500],  # Limit text length
                        'url': str(article.get('url', '#'))  # Get URL with fallback
                    }
                    valid_articles.append(valid_article)
                except (ValueError, TypeError):
                    continue
                    
        if not valid_articles:
            return None
            
        # Sort articles by date (newest first)
        valid_articles.sort(key=lambda x: x['date'], reverse=True)
        
        # Calculate statistics
        sentiment_scores = [a['sentiment_score'] for a in valid_articles]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        positive_count = sum(1 for s in sentiment_scores if s > 0.2)
        negative_count = sum(1 for s in sentiment_scores if s < -0.2)
        neutral_count = len(sentiment_scores) - positive_count - negative_count
        
        html_template = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>{company} ({ticker}) - Articles</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                :root {{
                    --primary-color: #000000;
                    --secondary-color: #666666;
                    --accent-color: #d5001c;
                    --text-color: #2c3e50;
                    --border-color: #e0e0e0;
                    --hover-color: #f8f9fa;
                }}
                
                body {{ 
                    font-family: "Porsche Next", "Segoe UI", Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    color: var(--text-color);
                    background-color: #f5f6fa;
                    line-height: 1.6;
                }}
                
                .header {{
                    background: linear-gradient(135deg, #000 0%, #1a1a1a 100%);
                    padding: 25px 0;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                    position: sticky;
                    top: 0;
                    z-index: 1000;
                    backdrop-filter: blur(10px);
                }}
                
                .header::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: radial-gradient(circle at 30% 50%, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0) 70%);
                    pointer-events: none;
                }}
                
                .header-content {{
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 0 30px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                
                .header-title {{
                    font-size: 28px;
                    font-weight: 300;
                    color: #fff;
                    margin: 0;
                    letter-spacing: 1px;
                }}
                
                .back-btn {{
                    background: rgba(255,255,255,0.1);
                    border: 1px solid rgba(255,255,255,0.2);
                    padding: 10px 20px;
                    color: #fff;
                    text-decoration: none;
                    font-size: 14px;
                    border-radius: 4px;
                    transition: all 0.3s ease;
                    backdrop-filter: blur(5px);
                }}
                
                .back-btn:hover {{
                    background: rgba(255,255,255,0.2);
                    transform: translateY(-2px);
                }}
                
                .container {{
                    max-width: 1400px;
                    margin: 40px auto;
                    padding: 0 30px;
                }}
                
                .stats-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 40px;
                }}
                
                .stat-card {{
                    background: #fff;
                    border-radius: 12px;
                    padding: 25px;
                    text-align: center;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                    transition: transform 0.3s ease;
                }}
                
                .stat-card:hover {{
                    transform: translateY(-5px);
                }}
                
                .stat-value {{
                    font-size: 32px;
                    font-weight: 300;
                    color: var(--primary-color);
                    margin-bottom: 10px;
                }}
                
                .stat-label {{
                    font-size: 14px;
                    color: var(--secondary-color);
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                
                .articles-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                    gap: 30px;
                    margin-top: 40px;
                }}
                
                .article-card {{
                    background: #fff;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                    transition: all 0.3s ease;
                }}
                
                .article-card:hover {{
                    transform: translateY(-5px);
                }}
                
                .article-content {{
                    padding: 25px;
                }}
                
                .article-title {{
                    font-size: 18px;
                    font-weight: 400;
                    margin: 0 0 15px;
                    line-height: 1.4;
                }}
                
                .article-title a {{
                    color: var(--primary-color);
                    text-decoration: none;
                    transition: color 0.2s ease;
                }}
                
                .article-title a:hover {{
                    color: var(--accent-color);
                }}
                
                .article-meta {{
                    display: flex;
                    align-items: center;
                    gap: 20px;
                    margin-bottom: 15px;
                    font-size: 14px;
                    color: var(--secondary-color);
                }}
                
                .article-sentiment {{
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-weight: 500;
                    font-size: 14px;
                }}
                
                .article-sentiment.positive {{
                    background: rgba(40,167,69,0.1);
                    color: #28a745;
                }}
                
                .article-sentiment.negative {{
                    background: rgba(220,53,69,0.1);
                    color: #dc3545;
                }}
                
                .article-sentiment.neutral {{
                    background: rgba(108,117,125,0.1);
                    color: #6c757d;
                }}
                
                .article-summary {{
                    font-size: 14px;
                    color: var(--text-color);
                    line-height: 1.6;
                    margin-top: 15px;
                    opacity: 0.8;
                }}
                
                @media (max-width: 768px) {{
                    .header-content {{
                        padding: 0 20px;
                    }}
                    
                    .header-title {{
                        font-size: 20px;
                    }}
                    
                    .container {{
                        padding: 0 20px;
                        margin: 20px auto;
                    }}
                    
                    .articles-grid {{
                        grid-template-columns: 1fr;
                        gap: 20px;
                    }}
                    
                    .stat-card {{
                        padding: 20px;
                    }}
                    
                    .stat-value {{
                        font-size: 28px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="header-content">
                    <h1 class="header-title">{company} ({ticker})</h1>
                    <a href="sentiment_report_latest.html" class="back-btn">← Back to Dashboard</a>
                </div>
            </div>
            
            <div class="container">
                <div class="stats-container">
                    <div class="stat-card">
                        <div class="stat-value">{len(valid_articles)}</div>
                        <div class="stat-label">Total Articles</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{positive_count}</div>
                        <div class="stat-label">Positive Articles</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{negative_count}</div>
                        <div class="stat-label">Negative Articles</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{avg_sentiment:.2f}</div>
                        <div class="stat-label">Average Sentiment</div>
                    </div>
                </div>
                
                <div class="articles-grid">
        '''
        
        # Add articles
        for article in valid_articles:
            sentiment_class = 'positive' if article['sentiment_score'] > 0.2 else 'negative' if article['sentiment_score'] < -0.2 else 'neutral'
            
            # Get article text if available, otherwise use empty string
            article_text = article.get('text', '')
            if article_text:
                article_text = f"{article_text[:200]}..."
            
            html_template += f'''
                    <div class="article-card {sentiment_class}">
                        <div class="article-content">
                            <h3 class="article-title">
                                <a href="{article['url']}" target="_blank" rel="noopener noreferrer">{article['title']}</a>
                            </h3>
                            <div class="article-meta">
                                <span>{article['date']}</span>
                                <span>{article['source']}</span>
                                <span class="article-sentiment {sentiment_class}">
                                    {article['sentiment_score']:.2f}
                                </span>
                            </div>
                            <div class="article-summary">{article_text}</div>
                        </div>
                    </div>
            '''
        
        html_template += '''
                </div>
            </div>
        </body>
        </html>
        '''
        
        # Save the article page
        output_path = self.output_dir / f"articles_{ticker}_{timestamp}.html"
        with open(output_path, 'w') as f:
            f.write(html_template)
            
        # Create symlink for latest version
        latest_path = self.output_dir / f"articles_{ticker}_latest.html"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(output_path.name)
        
        return output_path

    def generate_html(self) -> Path:
        """Generate HTML report from sentiment data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Load master data
            df = self.load_master_data()
            
            # Generate individual article pages for stocks with data
            for _, row in df.iterrows():
                if isinstance(row.get('articles'), list) and row['articles']:
                    try:
                        self.generate_article_page(
                            row['ticker'],
                            row['company'],
                            row['articles'],
                            timestamp
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to generate article page for {row['ticker']}: {e}")
            
            # Calculate the overall date range
            date_ranges = df[df['date_range'].notna()]['date_range'].str.split(' to ', expand=True)
            if not date_ranges.empty:
                start_date = min(date_ranges[0])
                end_date = max(date_ranges[1])
                date_range = f"{start_date} to {end_date}"
            else:
                date_range = 'N/A'
            
            # Get funnel statistics
            funnel_stats = self.get_data_funnel_stats(df)
            
            # Split data into stocks with and without sentiment
            has_data = df.dropna(subset=['average_sentiment'])
            no_data = df[~df.index.isin(has_data.index)]
            
            # Generate HTML
            html_template = r'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Market Sentiment Analysis</title>
                <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
                <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
                <link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css">
                <style>
                    :root {
                        --primary-color: #e0e0e0;
                        --secondary-color: #a0a0a0;
                        --accent-color: #404040;
                        --text-color: #2c3e50;
                        --border-color: #404040;
                        --hover-color: #f8f9fa;
                        --dark-bg: #1a1a1a;
                        --header-gradient: linear-gradient(135deg, #000000, #1a1a1a);
                        --modal-bg: rgba(0, 0, 0, 0.95);
                        --modal-content-bg: #1a1a1a;
                        --link-color: #4a90e2;
                        --link-hover: #357abd;
                    }
                    
                    body { 
                        font-family: 'Segoe UI', Arial, sans-serif;
                        margin: 0;
                        padding: 0;
                        color: var(--text-color);
                        background-color: #f5f6fa;
                    }
                    
                    .container {
                        max-width: 1400px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    
                    .header {
                        background: linear-gradient(135deg, #000 0%, #1a1a1a 100%);
                        padding: 25px 0;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                        position: sticky;
                        top: 0;
                        z-index: 1000;
                        backdrop-filter: blur(10px);
                    }
                    
                    .header::before {
                        content: '';
                        position: absolute;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background: radial-gradient(circle at 30% 50%, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0) 70%);
                        pointer-events: none;
                    }
                    
                    .header-content {
                        max-width: 1400px;
                        margin: 0 auto;
                        padding: 0 30px;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    }
                    
                    .header-title {
                        font-family: "Porsche Next", "Segoe UI", Arial, sans-serif;
                        font-size: 28px;
                        font-weight: 300;
                        letter-spacing: 1px;
                        color: #fff;
                        margin: 0;
                    }
                    
                    .sync-status {
                        display: flex;
                        flex-direction: column;
                        align-items: flex-end;
                        font-size: 12px;
                        color: rgba(255,255,255,0.8);
                        margin-left: 20px;
                    }
                    
                    .sync-status span {
                        margin: 2px 0;
                        padding: 4px 8px;
                        background: rgba(255,255,255,0.1);
                        border-radius: 4px;
                        backdrop-filter: blur(5px);
                    }
                    
                    .sync-status .sync-success {
                        background: rgba(40,167,69,0.3);
                        color: #28a745;
                    }
                    
                    .sync-status .sync-warning {
                        background: rgba(255,193,7,0.3);
                        color: #ffc107;
                    }
                    
                    .instant-report-btn:disabled {
                        background: rgba(255,255,255,0.05);
                        color: rgba(255,255,255,0.5);
                        cursor: not-allowed;
                        transform: none;
                    }
                    
                    .update-data-btn {
                        background: rgba(0,200,0,0.2);
                        border: 1px solid rgba(0,200,0,0.3);
                    }
                    
                    .update-data-btn:hover {
                        background: rgba(0,200,0,0.3);
                    }
                    
                    .update-data-btn:disabled {
                        background: rgba(255,255,255,0.05);
                        color: rgba(255,255,255,0.5);
                        cursor: not-allowed;
                        transform: none;
                    }
                    
                    .header-controls {
                        display: flex;
                        align-items: center;
                    }
                    
                    .card {
                        background: #fff;
                        border-radius: 12px;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                        margin-bottom: 40px;
                        overflow: hidden;
                        transition: transform 0.3s ease;
                    }
                    
                    .card:hover {
                        transform: translateY(-5px);
                    }
                    
                    .card-header {
                        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                        padding: 25px 30px;
                        border-bottom: 1px solid rgba(0,0,0,0.1);
                    }
                    
                    .card-title {
                        font-family: "Porsche Next", "Segoe UI", Arial, sans-serif;
                        font-size: 24px;
                        font-weight: 300;
                        color: #000;
                        margin: 0;
                        letter-spacing: 0.5px;
                    }
                    
                    table.dataTable {
                        margin: 20px 0 !important;
                        border-spacing: 0;
                        border: none;
                    }
                    
                    table.dataTable thead th {
                        background: #000;
                        color: #fff;
                        font-weight: 400;
                        padding: 15px 20px;
                        border: none;
                        letter-spacing: 0.5px;
                    }
                    
                    table.dataTable tbody td {
                        padding: 15px 20px;
                        border-bottom: 1px solid rgba(0,0,0,0.05);
                        font-size: 14px;
                        transition: background 0.2s ease;
                    }
                    
                    table.dataTable tbody tr:hover td {
                        background: rgba(0,0,0,0.02);
                    }
                    
                    .stock-link {
                        cursor: pointer;
                        position: relative;
                        color: inherit;
                        text-decoration: none;
                        display: inline-block;
                    }
                    
                    .stock-link:after {
                        content: '';
                        position: absolute;
                        width: 100%;
                        height: 1px;
                        bottom: -2px;
                        left: 0;
                        background-color: #000;
                        transform: scaleX(0);
                        transform-origin: left;
                        transition: transform 0.3s ease;
                    }
                    
                    .stock-link:hover:after {
                        transform: scaleX(1);
                    }
                    
                    .stock-link:hover {
                        color: #000;
                    }
                    
                    .trend-symbol {
                        display: inline-block;
                        padding: 4px 8px;
                        border-radius: 4px;
                        font-size: 12px;
                        font-weight: 500;
                        margin-left: 8px;
                        min-width: 28px;
                        text-align: center;
                    }
                    
                    @media (max-width: 768px) {
                        .header-content {
                            padding: 0 20px;
                        }
                        
                        .header-title {
                            font-size: 20px;
                        }
                        
                        .card {
                            margin: 20px 10px;
                            border-radius: 8px;
                        }
                        
                        .card-header {
                            padding: 20px;
                        }
                    }
                </style>
            </head>
            <body>
                <!-- Hidden timestamp for sync verification -->
                <div id="dashboard-timestamp" style="display: none;" data-timestamp="{timestamp}" data-generated="{readable_timestamp}"></div>
                <div class="header">
                    <div class="header-content">
                        <h1 class="header-title">Market Sentiment Analysis</h1>
                        <div class="header-controls">
                            <div class="sync-status" id="sync-status">
                                <span id="local-timestamp">🕐 Loading...</span>
                                <span id="sync-indicator">⏳ Checking sync...</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="container">
                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">Sentiment Overview</h2>
                        </div>
                        <table id="sentiment-table" class="display">
                            <thead>
                                <tr>
                                    <th>Ticker</th>
                                    <th>Company</th>
                                    <th>Last Week</th>
                                    <th>Last Month</th>
                                    <th>Articles</th>
                                    <th>Sentiment Change</th>
                                    <th>Trend</th>
                                </tr>
                            </thead>
                            <tbody>
            '''
            
            # Add modal template
            html_template += """
                <div id="articleModal" class="modal">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h3 class="modal-title"></h3>
                            <button class="modal-close">&times;</button>
                        </div>
                        <div class="modal-body">
                            <ul class="article-list"></ul>
                        </div>
                    </div>
                </div>
            """
            
            # Store articles data
            articlesData = {}
            
            # Add rows for stocks with data
            for _, row in has_data.iterrows():
                # Format sentiment values without trend symbols
                sentiment_7d = f"{row.get('last_week_sentiment', ''):.2f}" if pd.notna(row.get('last_week_sentiment')) else ''
                sentiment_30d = f"{row.get('last_month_sentiment', ''):.2f}" if pd.notna(row.get('last_month_sentiment')) else ''
                
                # Add historical trend info
                if pd.notna(row.get('sentiment_change')):
                    sentiment_change = f"{row['sentiment_change']:.2f}"
                    trend_class = 'trend-up' if row['trend'] == 'UP' else 'trend-down'
                    sentiment_trend = f"<span class='{trend_class}'>{row['trend']}</span>"
                else:
                    sentiment_change = ''
                    sentiment_trend = ''
                
                # Store articles data for this stock
                articles_data = []
                if 'articles' in row:
                    for article in row['articles']:
                        articles_data.append({
                            'title': article['title'],
                            'date': pd.to_datetime(article['date']).strftime('%Y-%m-%d') if isinstance(article['date'], str) else article['date'].strftime('%Y-%m-%d'),
                            'source': article['source'],
                            'sentiment': article['sentiment_score']
                        })
                
                html_template += f"""
                    <script>
                        articlesData['{row['ticker']}'] = [
                            {','.join([
                                f"""{{
                                    "title": "{str(article.get('title', '')).replace('"', '\\"') if pd.notna(article.get('title')) else ''}",
                                    "date": "{article.get('date', '')}",
                                    "source": "{article.get('source', '')}",
                                    "sentiment": {article.get('sentiment', 0)},
                                    "url": "{article.get('url', '#')}"
                                }}"""
                                for article in articles_data
                            ])}
                        ];
                    </script>
                    <tr>
                        <td class="text">{row['ticker']}</td>
                        <td class="text">
                            <span class="stock-link" onclick="showArticles('{row['ticker']}', '{row['company']}')">{row['company']}</span>
                        </td>
                        <td class="number">{sentiment_7d}</td>
                        <td class="number">{sentiment_30d}</td>
                        <td class="number">{int(row['total_articles']) if pd.notna(row.get('total_articles')) else ''}</td>
                        <td class="number">{sentiment_change}</td>
                        <td class="text">{sentiment_trend}</td>
                    </tr>
                """
            
            html_template += """
                    </tbody>
                </table>
                
                <div class="section-header">Stocks without Sentiment Data</div>
                <table id="missing-data-table" class="display">
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Company</th>
                            <th>Sector</th>
                            <th>Missing Data</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            # Add rows for stocks without data
            for _, row in no_data.iterrows():
                missing = []
                if pd.isna(row.get('average_sentiment')):
                    missing.append('Sentiment')
                    
                html_template += f"""
                    <tr>
                        <td>{row['ticker']}</td>
                        <td>{row['company']}</td>
                        <td>{self.mappings[row['ticker']]['sector']}</td>
                        <td>{', '.join(missing)}</td>
                    </tr>
                """
            
            html_template += """
                    </tbody>
                </table>
                </div>
                <div class="card" id="articles-section" style="display: none;">
                    <div class="card-header">
                        <h2 class="card-title">Articles</h2>
                        <div class="selected-company"></div>
                    </div>
                    <div class="articles-container">
                        <div class="articles-list"></div>
                    </div>
                </div>
                <div class="card" style="margin-top: 30px;">
                    <div class="card-header">
                        <h2 class="card-title">Legend & Information</h2>
                    </div>
                    <div style="padding: 20px;">
                        <h3 style="font-size: 16px; margin-bottom: 15px;">Trend Indicators</h3>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                            <div>
                                <h4 style="font-size: 14px; margin-bottom: 10px;">Weekly & Monthly Trends</h4>
                                <ul style="list-style: none; padding: 0;">
                                    <li style="margin-bottom: 8px;">
                                        <span class="trend-symbol trend-up">U</span>
                                        <span style="margin-left: 10px">Up: Sentiment improved by >5%</span>
                                    </li>
                                    <li style="margin-bottom: 8px;">
                                        <span class="trend-symbol trend-down">D</span>
                                        <span style="margin-left: 10px">Down: Sentiment declined by >5%</span>
                                    </li>
                                    <li style="margin-bottom: 8px;">
                                        <span class="trend-symbol trend-stable">S</span>
                                        <span style="margin-left: 10px">Stable: Change within ±5%</span>
                                    </li>
                                    <li style="margin-bottom: 8px;">
                                        <span class="trend-symbol trend-new">N</span>
                                        <span style="margin-left: 10px">New: No previous data for comparison</span>
                                    </li>
                                </ul>
                            </div>
                            <div>
                                <h4 style="font-size: 14px; margin-bottom: 10px;">Time Windows</h4>
                                <ul style="list-style: none; padding: 0;">
                                    <li style="margin-bottom: 8px;">
                                        <strong>Last Week:</strong> Average sentiment of articles from past 7 days
                                    </li>
                                    <li style="margin-bottom: 8px;">
                                        <strong>Last Month:</strong> Average sentiment of articles from past 30 days
                                    </li>
                                </ul>
                            </div>
                            <div>
                                <h4 style="font-size: 14px; margin-bottom: 10px;">Sentiment Scoring</h4>
                                <ul style="list-style: none; padding: 0;">
                                    <li style="margin-bottom: 8px;">
                                        Range: -1.0 (most negative) to +1.0 (most positive)
                                    </li>
                                    <li style="margin-bottom: 8px;">
                                        Calculated using FinBERT model on headlines (40%) and content (60%)
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <script>
                // Modal functionality
                const modal = document.getElementById('articleModal');
                const modalTitle = modal.querySelector('.modal-title');
                const modalBody = modal.querySelector('.article-list');
                const closeBtn = modal.querySelector('.modal-close');
                const articlesSection = document.getElementById('articles-section');
                const articlesList = document.querySelector('.articles-list');
                const selectedCompany = document.querySelector('.selected-company');
                
                function showArticles(ticker, company) {
                    window.location.href = `articles_${ticker}_latest.html`;
                }
                
                // Synchronization checking functionality
                async function checkSyncStatus() {
                    const localTimestamp = document.getElementById('local-timestamp');
                    const syncIndicator = document.getElementById('sync-indicator');
                    const timestampElement = document.getElementById('dashboard-timestamp');
                    
                    // Show local dashboard generation timestamp
                    const localGenerated = timestampElement.getAttribute('data-generated');
                    const localTimestampId = timestampElement.getAttribute('data-timestamp');
                    localTimestamp.textContent = `🕐 Local: ${localGenerated}`;
                    
                    try {
                        // Check GitHub Pages version by fetching the dashboard HTML
                        const response = await fetch('https://theemeraldnetwork.github.io/Kalimera/' + Date.now());
                        
                        if (response.ok) {
                            const htmlText = await response.text();
                            
                            // Extract timestamp from GitHub dashboard HTML
                            const timestampMatch = htmlText.match(/data-timestamp="([^"]+)"/);
                            const generatedMatch = htmlText.match(/data-generated="([^"]+)"/);
                            
                            if (timestampMatch && generatedMatch) {
                                const githubTimestampId = timestampMatch[1];
                                const githubGenerated = generatedMatch[1];
                                
                                // Compare timestamp IDs
                                if (localTimestampId === githubTimestampId) {
                                    syncIndicator.className = 'sync-success';
                                    syncIndicator.textContent = `✅ GitHub: ${githubGenerated} - SYNCED`;
                                } else {
                                                                         // Parse timestamps to calculate difference
                                     const localDate = new Date(localTimestampId.replace(/(\\d{4})(\\d{2})(\\d{2})_(\\d{2})(\\d{2})(\\d{2})/, '$1-$2-$3T$4:$5:$6'));
                                     const githubDate = new Date(githubTimestampId.replace(/(\\d{4})(\\d{2})(\\d{2})_(\\d{2})(\\d{2})(\\d{2})/, '$1-$2-$3T$4:$5:$6'));
                                    const timeDiff = Math.abs(localDate - githubDate) / (1000 * 60); // minutes
                                    
                                    syncIndicator.className = 'sync-warning';
                                    if (localDate > githubDate) {
                                        syncIndicator.textContent = `⚠️ GitHub: ${githubGenerated} - ${Math.round(timeDiff)}min behind`;
                                    } else {
                                        syncIndicator.textContent = `⚠️ Local newer than GitHub by ${Math.round(timeDiff)}min`;
                                    }
                                }
                            } else {
                                syncIndicator.className = 'sync-warning';
                                syncIndicator.textContent = '⚠️ GitHub: Old version detected (no timestamp)';
                            }
                        } else {
                            throw new Error(`HTTP ${response.status}`);
                        }
                    } catch (error) {
                        console.error('Error checking GitHub sync:', error);
                        syncIndicator.className = 'sync-warning';
                        syncIndicator.textContent = '❌ GitHub: Cannot check sync';
                    }
                }
                
                // Initialize sync checking on page load
                document.addEventListener('DOMContentLoaded', () => {
                    checkSyncStatus();
                    // Re-check every 2 minutes
                    setInterval(checkSyncStatus, 120000);
                });
                
                closeBtn.onclick = () => modal.classList.remove('show');
                window.onclick = (e) => {
                    if (e.target === modal) modal.classList.remove('show');
                };
                
                // Close on escape key
                document.addEventListener('keydown', (e) => {
                    if (e.key === 'Escape') modal.classList.remove('show');
                });
                
                $(document).ready(function() {
                    $('#sentiment-table').DataTable({
                        order: [[2, 'desc']],
                        pageLength: 25,
                        scrollY: '60vh',
                        scrollCollapse: true,
                        responsive: true,
                        columnDefs: [
                            {
                                targets: [0, 1, 6],
                                className: 'text'
                            },
                            { 
                                targets: [2, 3, 5],
                                className: 'number',
                                render: function(data, type, row) {
                                    if (type === 'sort') {
                                        let match = data.match(/([-+]?[0-9]*\\.?[0-9]+)/);
                                        return match ? parseFloat(match[0]) : -999999;
                                    }
                                    return data;
                                }
                            },
                            { 
                                targets: [4],
                                className: 'number'
                            }
                        ],
                        language: {
                            search: "Filter:",
                            lengthMenu: "Show _MENU_ companies",
                            info: "Displaying _START_ to _END_ of _TOTAL_ companies",
                            infoEmpty: "No companies to display",
                            infoFiltered: "(filtered from _MAX_ total companies)",
                            paginate: {
                                first: "«",
                                last: "»",
                                next: "›",
                                previous: "‹"
                            }
                        },
                        dom: '<"table-controls"lf>rt<"table-footer"ip>'
                    });
                    
                    $('#missing-data-table').DataTable({
                        order: [[0, 'asc']],
                        pageLength: 10,
                        scrollY: '30vh',
                        scrollCollapse: true,
                        responsive: true,
                        language: {
                            search: "Filter:",
                            lengthMenu: "Show _MENU_ companies",
                            info: "Displaying _START_ to _END_ of _TOTAL_ companies",
                            infoEmpty: "No companies to display",
                            infoFiltered: "(filtered from _MAX_ total companies)",
                            paginate: {
                                first: "«",
                                last: "»",
                                next: "›",
                                previous: "‹"
                            }
                        }
                    });
                    
                    // Keyboard shortcuts
                    $(document).keydown(function(e) {
                        // Ctrl/Cmd + F for search
                        if ((e.ctrlKey || e.metaKey) && e.keyCode == 70) {
                            e.preventDefault();
                            $('.dataTables_filter input').first().focus();
                        }
                        // Ctrl/Cmd + R for refresh
                        if ((e.ctrlKey || e.metaKey) && e.keyCode == 82) {
                            e.preventDefault();
                            window.location.reload();
                        }
                    });
                });
            </script>
        </body>
        </html>
        """
            
            # Get current time in UTC
            utc_now = datetime.now(ZoneInfo('UTC'))
            # Convert to CET
            cet_now = utc_now.astimezone(ZoneInfo('Europe/Paris'))
            
            # Format timestamps
            readable_timestamp = cet_now.strftime('%Y-%m-%d %H:%M:%S CET')
            
            # Populate template with timestamp data using string replacement
            html_content = html_template.replace('{timestamp}', timestamp)
            html_content = html_content.replace('{readable_timestamp}', readable_timestamp)
            
            # Save HTML file with timestamp
            output_path = self.output_dir / "index.html"
            with open(output_path, 'w') as f:
                f.write(html_content)
                
            # Create symlink for latest report
            latest_path = self.output_dir / "sentiment_report_latest.html"
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(output_path.name)
            
            # No archiving needed - we only keep index.html and latest symlink
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            raise
            
    def generate_dashboard(self) -> None:
        """Generate and open the HTML dashboard"""
        try:
            html_path = self.generate_html()
            webbrowser.open(f'file://{html_path.absolute()}')
            self.logger.info(f"\nDashboard generated and opened: {html_path}")
            self.logger.info(f"Archive directory: {self.archive_dir}")
        except Exception as e:
            self.logger.error(f"Failed to generate dashboard: {e}")
            raise

def main():
    """Main function to generate dashboard"""
    print("\nStarting dashboard generation...")
    generator = DashboardGenerator()
    generator.generate_dashboard()
    print("\nDashboard generation complete!")

if __name__ == "__main__":
    main() 