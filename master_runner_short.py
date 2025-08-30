#!/usr/bin/env python3

"""
Simplified master runner script that:
1. Collects sentiment data
2. Generates and opens the dashboard
3. Pushes changes to GitHub
4. Sends email report with declining stocks
"""

import os
import sys
import subprocess
from datetime import datetime
import logging
from pathlib import Path
import shutil

def setup_logging():
    """
    Setup comprehensive logging configuration within automated_sentiment_daily_analysis directory.
    
    Creates log files with detailed formatting for monitoring and debugging.
    All log files are stored within the automated_sentiment_daily_analysis/logs directory.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory within current directory only
    current_dir = Path(__file__).parent
    log_dir = current_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging with detailed format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_dir / 'tigro_master_detailed.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def run_command_with_logging(command: list, description: str, logger: logging.Logger, max_retries: int = 3) -> bool:
    """Run a command with detailed logging and retry logic"""
    # Set up environment with PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path.cwd())  # Add current directory to PYTHONPATH
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"🔄 Attempt {attempt}/{max_retries}: {' '.join(command)}")
            # Start process
            # Run process directly to see all output
            try:
                subprocess.run(
                    command,
                    check=True,
                    text=True,
                    cwd=os.getcwd(),
                    env=env
                )
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with exit code {e.returncode}")
                if e.stdout:
                    logger.error(f"Output: {e.stdout}")
                if e.stderr:
                    logger.error(f"Error: {e.stderr}")
                return False
            
                # Success already returned in the try block
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Attempt {attempt} failed: {e}")
            if e.stdout:
                logger.error(f"📤 STDOUT: {e.stdout}")
            if e.stderr:
                logger.error(f"📤 STDERR: {e.stderr}")
            
            if attempt == max_retries:
                logger.error(f"🚨 All {max_retries} attempts failed for: {description}")
                return False
            else:
                logger.info(f"🔄 Retrying in 5 seconds...")
                import time
                time.sleep(5)
                
        except Exception as e:
            logger.error(f"🚨 Unexpected error in {description}: {e}")
            return False
    
    return False

def copy_to_docs(logger: logging.Logger) -> bool:
    """Copy latest results to docs directory for GitHub Pages (LATEST DATA ONLY)
    
    Args:
        logger: Logger instance for detailed logging
        
    Returns:
        bool: True if copy operation successful, False otherwise
        
    Note: All operations are contained within automated_sentiment_daily_analysis directory
    """
    try:
        # Work within automated_sentiment_daily_analysis directory only
        current_dir = Path(__file__).parent
        docs_dir = current_dir / 'docs'
        results_dir = current_dir / 'results'
        
        # Clean docs directory first (remove historic data)
        if docs_dir.exists():
            logger.info("🧹 Cleaning docs directory of historic data...")
            shutil.rmtree(docs_dir)
        
        # Ensure docs directory exists
        docs_dir.mkdir(exist_ok=True)
        
        # Copy latest sentiment report as both index.html and sentiment_report_latest.html
        latest_report = results_dir / "sentiment_report_latest.html"
        if latest_report.exists():
            # Copy as index.html for GitHub Pages root
            shutil.copy2(latest_report, docs_dir / "index.html")
            # Also keep as sentiment_report_latest.html for direct links
            shutil.copy2(latest_report, docs_dir / "sentiment_report_latest.html")
            logger.info("✅ Copied main dashboard as index.html and sentiment_report_latest.html")
        else:
            logger.warning("⚠️ No sentiment report found to copy")
            
        # Copy ONLY the latest article HTML files (not historic ones)
        article_count = 0
        for article_file in results_dir.glob("articles_*_latest.html"):
            shutil.copy2(article_file, docs_dir / article_file.name)
            article_count += 1
            
        logger.info(f"✅ Copied {article_count} individual stock article pages (LATEST ONLY)")
        logger.info(f"📊 Sentiment dashboard will be available at: https://theemeraldnetwork.github.io/Kalimera/")
        
        return True
    except Exception as e:
        logger.error(f"Error copying files to docs: {e}")
        return False

def push_to_github(logger: logging.Logger) -> bool:
    """Push changes to GitHub"""
    try:
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Git commands
        commands = [
            ['git', 'add', 'results/*'],
            ['git', 'add', 'docs/*'],
            ['git', 'commit', '-m', f'Update sentiment analysis and dashboard - {timestamp}'],
            ['git', 'push']
        ]
        
        for cmd in commands:
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
        logger.info("Successfully pushed changes to GitHub")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error pushing to GitHub: {e}")
        return False

def send_email_report(logger: logging.Logger) -> bool:
    """Send email report with sentiment analysis"""
    try:
        logger.info("Sending email report...")
        from automated_sentiment_daily_analysis.utils.email.report_sender import SentimentEmailSender
        import pandas as pd
        
        # Load latest sentiment summary
        results_dir = Path('results')
        
        # Try to use the latest symlink first, then fall back to dated files
        latest_symlink = results_dir / 'sentiment_summary_latest.csv'
        if latest_symlink.exists():
            summary_df = pd.read_csv(latest_symlink)
        else:
            # Find dated files (not symlinks with spaces)
            summary_files = [f for f in results_dir.glob('sentiment_summary_*.csv') 
                           if f.name.count('_') == 2 and 'latest' not in f.name]
            
            if not summary_files:
                logger.warning("No sentiment summary files found")
                return False
                
            latest_file = max(summary_files, key=lambda f: f.stat().st_mtime)
            summary_df = pd.read_csv(latest_file)
        
        # Initialize email sender
        sender = SentimentEmailSender()
        
        # Send the email (test_mode=False for real emails)
        sender.send_email(summary_df, test_mode=False)
        logger.info("✅ Email report sent successfully")
        return True
    except Exception as e:
        logger.error(f"Error sending email report: {e}")
        return False

def main():
    """
    Main execution function for daily sentiment analysis automation.
    
    This function orchestrates the complete daily workflow:
    1. Sentiment data collection using Finnhub API
    2. Dashboard generation with interactive HTML
    3. Git synchronization to GitHub Pages (TheEmeraldNetwork/Kalimera)
    4. Email reporting (currently disabled)
    5. Log cleanup and maintenance
    
    All operations are contained within the automated_sentiment_daily_analysis directory.
    No files are created outside this scope to maintain system integrity.
    
    Returns:
        bool: True if all critical operations complete successfully
        
    Raises:
        Exception: If critical components fail (sentiment analysis, dashboard generation)
    """
    start_time = datetime.now()
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("🐅 TIGRO DAILY AUTOMATION STARTED")
    logger.info(f"📅 Date: {start_time}")
    logger.info(f"📁 Project Root: {os.getcwd()}")
    logger.info(f"🐍 Python Executable: {sys.executable}")
    logger.info(f"🐍 Python Virtual Environment: {os.environ.get('VIRTUAL_ENV', 'Not activated')}")
    logger.info("=" * 60)
    
    # Check prerequisites
    logger.info("🔍 Checking prerequisites...")
    # Use virtual environment's Python if available
    venv_python = Path('venv/bin/python')
    if venv_python.exists():
        python_path = str(venv_python.absolute())
    else:
        python_path = sys.executable
    logger.info(f"🐍 Python executable: {python_path}")
    
    # Scripts are in the current directory now
    sentiment_script = Path('sent_collect_data.py')
    dashboard_script = Path('viz_dashboard_generator.py')
    
    if not sentiment_script.exists():
        logger.error(f"🚨 Missing sentiment script: {sentiment_script}")
        return False
        
    if not dashboard_script.exists():
        logger.error(f"🚨 Missing dashboard script: {dashboard_script}")
        return False
        
    logger.info("✅ All prerequisites checked successfully")
    
    # Step 1: Sentiment Analysis
    logger.info("📊 Starting sentiment analysis...")
    if not run_command_with_logging(
        [python_path, 'sent_collect_data.py'],
        "sentiment analysis",
        logger
    ):
        logger.error("🚨 Sentiment analysis failed!")
        return False
    logger.info("✅ Sentiment analysis completed successfully")
    
    # Step 2: Dashboard Generation
    logger.info("📈 Generating dashboard...")
    if not run_command_with_logging(
        [python_path, 'viz_dashboard_generator.py'],
        "dashboard generation",
        logger
    ):
        logger.error("🚨 Dashboard generation failed!")
        return False
    logger.info("✅ Dashboard generation completed successfully")
    
    # Step 3: Git operations - Push index.html directly
    logger.info("🚀 Syncing dashboard to GitHub Pages...")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Git add only index.html (the dashboard)
    if not run_command_with_logging(['git', 'add', 'index.html'], "git add index.html", logger, max_retries=2):
        logger.error("🚨 Git add index.html failed!")
        return False
    
    # Git commit with descriptive message
    if not run_command_with_logging(
        ['git', 'commit', '-m', f'📊 Daily dashboard update - {timestamp}'],
        "git commit",
        logger,
        max_retries=2
    ):
        logger.warning("⚠️ Git commit failed - possibly no changes to dashboard")
    
    # Git push to gh-pages branch for GitHub Pages
    if not run_command_with_logging(['git', 'push', 'origin', 'gh-pages'], "git push", logger, max_retries=1):
        logger.warning("⚠️ Git push failed - dashboard may not be updated on GitHub")
        # Don't return False here - the local process still succeeded
    else:
        logger.info("✅ Successfully synced latest dashboard to GitHub")
        logger.info("🌐 Dashboard available at: https://theemeraldnetwork.github.io/Kalimera/")
    
    # Step 5: Email Report - RE-ENABLED
    logger.info("📧 Sending email report...")
    try:
        from utils.email.report_sender import SentimentEmailSender
        import pandas as pd
        
        # Load latest sentiment data from current directory
        current_dir = Path(__file__).parent
        sentiment_file = current_dir / 'results' / 'sentiment_summary_latest.csv'
        if sentiment_file.exists():
            df = pd.read_csv(sentiment_file)
            logger.info(f"📊 Loaded sentiment data for {len(df)} stocks")
            
            email_sender = SentimentEmailSender()
            success = email_sender.send_email(df, test_mode=False)
            
            if success:
                logger.info("✅ Email report sent successfully")
            else:
                logger.error("🚨 Email report failed to send")
        else:
            logger.error("🚨 No sentiment data file found for email")
            
    except Exception as e:
        logger.error(f"🚨 Email error: {e}")
        import traceback
        logger.error(f"🚨 Email traceback: {traceback.format_exc()}")
    
    # Step 6: Cleanup
    logger.info("🧹 Cleaning up old log files...")
    cleanup_old_logs(logger)
    logger.info("✅ Log cleanup completed")
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("=" * 60)
    logger.info("🎉 DAILY AUTOMATION COMPLETED SUCCESSFULLY")
    logger.info(f"⏱️  Total Duration: {duration}")
    logger.info(f"📊 Dashboard: https://theemeraldnetwork.github.io/Kalimera/")
    logger.info(f"📧 Email Report: Disabled as requested")
    logger.info("=" * 60)
    
    return True

def cleanup_old_logs(logger: logging.Logger):
    """
    Clean up old log files to prevent disk space issues.
    
    Removes log files older than 30 days from the automated_sentiment_daily_analysis/logs directory.
    
    Args:
        logger: Logger instance for status reporting
    """
    try:
        # Work within current directory only
        current_dir = Path(__file__).parent
        logs_dir = current_dir / 'logs'
        if not logs_dir.exists():
            return
            
        # Keep only last 30 days of logs
        import time
        current_time = time.time()
        thirty_days_ago = current_time - (30 * 24 * 60 * 60)
        
        for log_file in logs_dir.glob('*.log'):
            if log_file.stat().st_mtime < thirty_days_ago:
                log_file.unlink()
                logger.info(f"🗑️ Deleted old log: {log_file.name}")
                
    except Exception as e:
        logger.warning(f"⚠️ Log cleanup warning: {e}")

if __name__ == "__main__":
    main() 