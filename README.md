# financial_sentiment_analysis
Predicting stock price moves using Natural Language Processing (NLP) on the FNSPID dataset. Featuring sentiment scoring with TextBlob/VADER and technical analysis with TA-Lib.
# Portfolio Sentiment & Market Analysis Pipeline

## Project Overview
I developed this automated pipeline for Nova Financial Solutions to quantify the impact of financial news on stock returns. The system analyzes sentiment across a portfolio of tech equities (AAPL, AMZN, GOOG, META, NVDA).

## Technical Implementation
- **NLP:** Used VADER lexicon to process thousands of financial headlines.
- **Indicators:** Calculated 20-day SMA and Daily % Returns.
- **Scalability:** The script dynamically detects and processes all CSV files in the `/data` directory.

## Final Results
- **Portfolio Correlation:** 0.0170
- **Visualization:** [Include your portfolio_correlation_plot.png here]

## How to Run
1. Ensure your ticker CSVs are in the `/data` folder.
2. Run `python final_analysis_pipeline.py`.
