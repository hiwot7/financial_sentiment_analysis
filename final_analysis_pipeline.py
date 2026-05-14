import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# --- TASK 1: DATA LOADING & NLP ---
print("Step 1: Starting Portfolio Sentiment Analysis...")
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

news = pd.read_csv('raw_analyst_ratings.csv')
news['sentiment'] = news['headline'].apply(
    lambda x: sia.polarity_scores(str(x))['compound'])
news['date'] = pd.to_datetime(news['date'], errors='coerce').dt.date

# --- TASK 2: MULTI-STOCK PROCESSING ---
print("Step 2: Processing all stock files in /data...")
data_folder = 'data'
all_files = glob.glob(os.path.join(data_folder, "*.csv"))

combined_stocks = []

for file in all_files:
    stock_name = os.path.basename(file).replace('.csv', '')
    df = pd.read_csv(file)

    # FIX: Handle missing 'Adj Close' by falling back to 'Close'
    close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

    # Calculate Indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['daily_return'] = df[close_col].pct_change() * 100
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['Stock'] = stock_name

    combined_stocks.append(df)

portfolio_df = pd.concat(combined_stocks)
print(f"Task 2 Complete: Processed {len(all_files)} stocks successfully!")

# --- TASK 3: CORRELATION ANALYSIS ---
print("Step 3: Calculating Portfolio-wide Correlation...")
daily_news = news.groupby('date')['sentiment'].mean().reset_index()
merged = pd.merge(daily_news, portfolio_df, left_on='date', right_on='Date')

correlation = merged['sentiment'].corr(merged['daily_return'])

print("\n" + "="*45)
print(f"PORTFOLIO CORRELATION RESULT: {correlation:.4f}")
print("="*45)

# Save Visualization
plt.figure(figsize=(10, 6))
sns.regplot(data=merged, x='sentiment', y='daily_return',
            scatter_kws={'alpha': 0.3}, color='navy')
plt.title(
    f'Portfolio Sentiment vs Daily Returns (Correlation: {correlation:.2f})')
plt.savefig('portfolio_correlation_plot.png')
print("Visual plot saved as 'portfolio_correlation_plot.png'")
