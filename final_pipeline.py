import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. SETUP & DATA LOADING
print("Step 1: Loading Data...")
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# Using absolute paths based on your folder structure
base_path = os.getcwd()
stock_path = os.path.join(base_path, 'data', 'AAPL.csv')
news_path = os.path.join(base_path, 'data', 'raw_analyst_ratings.csv')

try:
    stock_df = pd.read_csv(stock_path)
    news_df = pd.read_csv(news_path)
    print("Files found and loaded!")
except Exception as e:
    print(f"Error finding files: {e}")
    # Backup: try current directory
    stock_df = pd.read_csv('AAPL.csv')
    news_df = pd.read_csv('raw_analyst_ratings.csv')

# 2. ANALYSIS
print("Step 2: Running Sentiment and Technical Analysis...")
# Sentiment
news_df['sentiment'] = news_df['headline'].apply(
    lambda x: sia.polarity_scores(x)['compound'])
news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce').dt.date

# Technical Indicators (Manual Pandas)
stock_df['SMA_20'] = stock_df['Close'].rolling(window=20).mean()
stock_df['returns'] = stock_df['Adj Close'].pct_change() * 100
stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date

# 3. CORRELATION
print("Step 3: Calculating Correlation...")
daily_news = news_df.groupby('date')['sentiment'].mean().reset_index()
merged = pd.merge(daily_news, stock_df, left_on='date', right_on='Date')
correlation = merged['sentiment'].corr(merged['returns'])

# 4. RESULTS
print("-" * 30)
print(f"NOVA FINANCIAL ANALYSIS COMPLETE")
print(f"Pearson Correlation Score: {correlation:.4f}")
print("-" * 30)

# 5. SAVE PLOT
plt.figure(figsize=(10, 6))
sns.regplot(data=merged, x='sentiment', y='returns')
plt.title(f'Sentiment vs Returns (Corr: {correlation:.2f})')
plt.savefig('correlation_plot.png')
print("Plot saved as 'correlation_plot.png'. Open it to see your results!")
