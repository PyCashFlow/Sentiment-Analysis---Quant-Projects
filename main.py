import os
import matplotlib.pyplot as plt
import pandas as pd
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure the static/charts directory exists
os.makedirs('static/charts', exist_ok=True)

# URL for Finviz stock pages
finviz_url = 'https://finviz.com/quote.ashx?t='

# Stocks to analyze
tickers = ['BSCO', 'GOLD', 'NVDA']

# Request headers
headers = {'user-agent': 'my-app'}

# Dictionary to hold news tables for each ticker
news_tables = {}

# Fetch news tables for each ticker
for ticker in tickers:
    try:
        url = finviz_url + ticker
        req = Request(url=url, headers=headers)
        response = urlopen(req)
        html = BeautifulSoup(response, features='html.parser')
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# Initialize sentiment analyzer
vader = SentimentIntensityAnalyzer()

# Parse news headlines and extract relevant data
parsed_data = []

for ticker, news_table in news_tables.items():
    if news_table:
        for row in news_table.findAll('tr'):
            title = row.a.text
            compound_score = vader.polarity_scores(title)['compound']
            parsed_data.append([ticker, title, compound_score])

# Creating DataFrame from parsed data
df = pd.DataFrame(parsed_data, columns=['ticker', 'title', 'compound'])

# Check if DataFrame is empty
if df.empty:
    print("No relevant news headlines found.")
else:
    # Group by ticker and calculate mean of the 'compound' column
    avg_compound_scores = df.groupby('ticker')['compound'].mean()
    avg_compound_scores = avg_compound_scores.reset_index()
    avg_compound_scores.columns = ['ticker', 'average_compound']

    # Save the average compound scores to a CSV file
    avg_compound_scores.to_csv('average_compound_scores.csv', index=False)

    # Debugging output
    print("\nAverage Compound Scores:")
    print(avg_compound_scores)

    # Plotting the sentiment scores
    plt.figure(figsize=(10, 6))
    mean_df = df.groupby('ticker')['compound'].mean()
    mean_df.plot(kind='bar', color=['blue', 'green', 'red'], title='Sentiment Analysis of Stock News')
    plt.xlabel('Ticker')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig('static/charts/sentiment_chart.png')  # Save in static directory
    plt.close()

    # Save DataFrame to a CSV file
    df.to_csv('sentiment_results.csv', index=False)
