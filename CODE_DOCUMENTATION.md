# Code Documentation - Automated Sentiment Daily Analysis

## 📁 Directory Structure
```
automated_sentiment_daily_analysis/
├── master_runner_short.py          # Main orchestration script
├── sent_collect_data.py            # Sentiment data collection
├── viz_dashboard_generator.py      # Dashboard generation
├── test_apple_sentiment.py         # Test suite
├── results/                        # Analysis outputs
├── logs/                          # Execution logs
├── archive/                       # Historical data
├── utils/                         # Utility modules
└── venv/                          # Virtual environment
```

## 🔍 Core Components

### 1. master_runner_short.py
**Purpose**: Main orchestration script for daily automation
**Key Functions**:
- `setup_logging()`: Configures logging within current directory
- `run_command_with_logging()`: Executes subprocesses with retry logic
- `copy_to_docs()`: Copies results to docs directory for GitHub Pages
- `main()`: Orchestrates complete workflow

**Recent Fixes**:
- ✅ All paths now relative to automated_sentiment_daily_analysis
- ✅ Git push targets TheEmeraldNetwork/Kalimera gh-pages branch
- ✅ Email service disabled as requested
- ✅ Comprehensive documentation added

### 2. sent_collect_data.py
**Purpose**: Collects news and performs sentiment analysis
**Key Classes**:
- `SentimentAnalyzer`: Main analysis engine

**Key Methods**:
- `get_company_news()`: Retrieves news from Finnhub API
- `analyze_sentiment()`: Processes text through FinBERT model
- `process_all_stocks()`: Orchestrates complete analysis pipeline

**Recent Fixes**:
- ✅ Directory paths fixed to stay within current scope
- ✅ Comprehensive docstrings added
- ✅ Error handling improved

### 3. viz_dashboard_generator.py
**Purpose**: Generates interactive HTML dashboard
**Key Classes**:
- `DashboardGenerator`: Dashboard creation engine

**Recent Fixes**:
- ✅ Output path changed to write index.html in current directory
- ✅ Syntax warning fixed (regex escape sequence)
- ✅ Timezone handling improved for CET display

## 🧪 Testing Framework

### Test Coverage
- ✅ SentimentAnalyzer initialization
- ✅ Apple news retrieval (mocked)
- ✅ Sentiment analysis functionality
- ✅ Dashboard generation with Apple data
- ✅ Directory scope compliance verification

### Running Tests
```bash
source venv/bin/activate
python -m pytest test_apple_sentiment.py -v
```

## 🔒 Security & Dependencies

### Virtual Environment
- Location: `./venv/`
- Python Version: 3.13.3
- Key Packages:
  - pandas 2.3.2
  - transformers 4.55.4
  - torch 2.8.0
  - finnhub-python 2.4.24
  - pytest 8.4.1

### API Security
- Finnhub API key stored in `utils/config/api_keys.json`
- Email credentials in `utils/email/email_config.json`

## 🚀 Deployment

### GitHub Pages Integration
- Repository: TheEmeraldNetwork/Kalimera
- Branch: gh-pages
- URL: https://theemeraldnetwork.github.io/Kalimera/

### Automation Schedule
- Frequency: Daily at 7 AM CET
- Logs: Stored in `./logs/` directory
- Email: Currently disabled

## 📊 Data Flow
1. Finnhub API → News Collection
2. FinBERT Model → Sentiment Analysis
3. Data Processing → CSV Files in `./results/`
4. Dashboard Generation → `./index.html`
5. Git Push → GitHub Pages
6. Email Report → Disabled

## 🐛 Known Issues
- ⚠️ Syntax warning in viz_dashboard_generator.py (FIXED)
- ⚠️ Email service disabled per user request
- ⚠️ Archive functionality simplified for scope compliance

## 📝 Maintenance Notes
- Log retention: 30 days
- All operations contained within automated_sentiment_daily_analysis
- No external directory dependencies
- Self-contained virtual environment
