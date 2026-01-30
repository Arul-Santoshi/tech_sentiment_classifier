# Tech Sentiment Classifier

A sentiment analysis tool for technology and AI news articles using DistilBERT.

## Features

- Fetches technology and AI news articles from NewsAPI
- Classifies sentiment (positive/negative/neutral) using pre-trained DistilBERT
- Generates sentiment scores on a -1 to +1 scale
- Creates visualizations of sentiment trends over time
- Exports results to CSV

## Installation

```bash
# Clone the repository
git clone https://github.com/Arul-Santoshi/tech_sentiment_classifier.git
cd tech_sentiment_classifier

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Get a free API key from [NewsAPI](https://newsapi.org/register)
2. Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
3. Add your API key to `.env`:
   ```
   NEWSAPI_KEY=your_api_key_here
   ```

## Usage

### Run the Full Pipeline

```bash
# Using environment variable
python main.py

# Or pass API key directly
python main.py --api-key YOUR_API_KEY

# Customize article count and output files
python main.py --articles 500 --output-csv results.csv --output-plot sentiment.png
```

### Test the Classifier

```bash
# Run tests on sample headlines
python test_classifier.py

# Interactive mode
python test_classifier.py --interactive

# Test custom headlines
python test_classifier.py "AI stocks reach all-time high" "Tech layoffs continue"
```

### Command Line Options

```
--api-key       NewsAPI API key (or set NEWSAPI_KEY env var)
--articles      Target number of articles to fetch (default: 1000)
--output-csv    Output CSV file path (default: tech_sentiment_results.csv)
--output-plot   Output plot file path (default: sentiment_analysis.png)
--skip-fetch    Skip fetching and use existing CSV file
--input-csv     Input CSV file to use with --skip-fetch
```

## Output

### CSV Columns

| Column | Description |
|--------|-------------|
| date | Publication date (YYYY-MM-DD) |
| headline | Article headline |
| source | News source name |
| sentiment_label | Classification: positive, negative, or neutral |
| sentiment_score | Score from -1 (very negative) to +1 (very positive) |

### Visualizations

The tool generates:
- Daily average sentiment score (line plot with trend)
- Overall sentiment distribution (pie chart)
- Sentiment counts over time (stacked bar chart)
- Sentiment score distribution (histogram)
- Sentiment by news source (horizontal bar chart)

## Project Structure

```
tech_sentiment_classifier/
├── main.py                 # Main pipeline script
├── news_fetcher.py         # NewsAPI integration
├── sentiment_classifier.py # DistilBERT sentiment analysis
├── visualization.py        # Plotting and statistics
├── test_classifier.py      # Test script for validation
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variable template
└── README.md              # This file
```

## Model Details

Uses `distilbert-base-uncased-finetuned-sst-2-english` from Hugging Face:
- Pre-trained on SST-2 (Stanford Sentiment Treebank)
- Binary classification adapted to three-class output
- Sentiment thresholds: positive (>0.2), negative (<-0.2), neutral (between)

## License

MIT License
