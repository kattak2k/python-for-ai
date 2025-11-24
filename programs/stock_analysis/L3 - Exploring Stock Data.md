# Exploring Stock Market Data

This notebook provides step-by-step instructions for replicating what Brian did in the third demo, to analyze Apple's stock closing prices between October 24, 2024 and October 23, 2025. 

Please follow along, and feel free to play with different variations too! 

## Step 1: Define the Dates and Ticker Variables

These variables will be used at the end to generate a summary of the insights.


```python
start_date = '2024-10-24'
end_date = '2025-10-23'
ticker = 'AAPL'
```

## Step 2: Download Apple Stock Data

- Open Jupyter chat by clicking on the chat bubble icon on the left sidebar of Jupyter Lab
- Create a new chat by clicking `+Chat` (choose any name for the file)
- In the chat window, attach the Markdown file `yfinance_docs.md` using the `@` symbol
  - Type `@`, then select `file` from the autocomplete menu
  - You'll see a list of available files in the lesson's directory, choose `yfinance_docs.md` 
- To download Apple stock data, use a prompt like this:
   > Use yfinance to download Apple (AAPL) stock data for this period:
   > - start date: October 24, 2024
   > - end date: October 23, 2025
   >
   > Save the returned results in a DataFrame called `aapl`
- Transfer the generated code to the cell below and run it


```python

```

<span style="color:green; font-weight:bold;">Note:</span> If you see a message saying that cached data was used due to rate limit errors, this is a temporary issue in this learning environment that occurs when too many requests are sent simultaneously. You can run the notebook locally if you wish to download data for different stocks.

- Print the columns of the DataFrame `aapl`: `aapl.columns`


```python

```

- In the chat window, ask how you can flatten the columns of the DataFrame using a prompt like this:
  > The DataFrame aapl has multiIndexed columns [('Close','AAPL'),('High','AAPL'),('Low','AAPL'),('Open','AAPL'),('Volume','AAPL')]. Flatten the columns by removing 'AAPL'.


```python

```

## Step 3: Calculate Basic Statistics & Metrics

- In the same chat window, use a prompt like this to calculate the basic descriptive statistics of the DataFrame:
  > Display the shape and statistical summary of the DataFrame aapl.


```python

```

- To calculate the total return, use a prompt like this:
  > Use the Close column of DataFrame aapl to find the total return in percentage (`total_return`) based on the start price and end price.


```python

```

## Step 4: Visualize the Closing Price

- To visualize Apple's Closing price, use a prompt like this:
  > Create a line chart showing the closing price trend using the column 'Close' of the DataFrame `aapl`.
  >
  > Use matplotlib to create a professional-looking chart with:
  > - Clear title and axis labels
  > - Grid for readability
  > - Appropriate colors and styling


```python

```

- To find the dates that correspond to the peak and lowest prices, use a prompt like this:
  > Use the Close column of aapl dataFrame to find and print:
  > - the peak date (in a variable called `peak_date`) that corresponds to the maximum closing price `peak_price`
  > - the lowest date (in a variable called `lowest_date`) that corresponds to the minimum closing price `lowest_price`
  >
  > Update the above code to show the peak and low prices in the line chart.


```python

```

- To find the context related to the peak and lowest date, use a prompt like this:
  > For the `peak_date` and `lowest_date`, search for related Apple news using Serper. The Serper API key is saved in a .env file. Store the snippets of the found articles in a json string `news_snippets` that has these fields: peak_date, lowest_date, peak_news_snippets, lowest_news_snippets.

<span style="color:green; font-weight:bold;">Note:</span> The `SERPER_API_KEY` variable is already defined in this environment, you do not need to create  an `.env` file. 


```python

```

## Step 5: Analyze Volatility

- To calculate the signal's volatility, use a prompt like this: 
  >In the DataFrame aapl, find the overall volatility in percentage using the column Close. Volatility is the standard deviation of the daily percentage changes. Save the result in a variable called `volatility`.


```python

```

- To find and plot the rolling volatility, use a prompt like this:
  > Calculate the rolling volatility as the as 20-day standard deviation of the daily percentage change and plot it. Identify days of high volatility where volatility is greater than mean + std. Save the days of high volatility in a DataFrame called `high_vol_days`.


```python

```

## Step 6: Report Generation

- To generate a report summarizing the insights, use a prompt like this:
  
  > Use gpt-4.1-mini to generate a summary that takes in these variables:
  > - ticker: stock ticker (string)
  > - start_date: analysis starting period (string)
  > - end_date: analysis end period (string)
  > - numerical metrics: total_return & volatility (in percentage)
  > - peak_date, peak_price
  > - lowest_date, lowest_price
  > - high_vol_days: pandas DataFrame showing high volatility days
  > - news_snippets: string containing snippet of news for the peak and lowest dates
  >
  > The OpenAI API key is stored in the .env file. The variables are already defined in the notebook.


```python

```
