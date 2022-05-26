# Sourcing the data for all the stocks of S&P500
# Load libraries
import pandas as pd
import bs4 as bs
import datetime as dt
import os
import pandas_datareader.data as web
import pickle
import requests
import csv
# Yahoo for dataReader
import yfinance as yf
yf.pdr_override()

import warnings
warnings.filterwarnings('ignore')

# Load dataset
# Data fetching - Nasdaq 100

def save_tickers():
    Nasdaqurl = "https://en.wikipedia.org/wiki/Nasdaq-100#External_links"
    request_Nasdaq = requests.get(Nasdaqurl)
    soup = bs.BeautifulSoup(request_Nasdaq.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable', 'id':'constituents'})
    header = table.findAll("th")
    if header[1].text.rstrip() != "Ticker":
        raise Exception("Can't parse website's table!")
        # Retrieve the values in the table
    tickers = []
    rows = table.findAll("tr")
    for row in rows:
        fields = row.findAll("td")
        if fields:
            ticker = fields[1].text.rstrip()
            tickers.append(str(ticker))
    print('Tickers \n', tickers)
    print('Number of Tickers \n', len(tickers))
    return tickers

#save_tickers()


def get_data_from_yahoo():
    # tickers = save_tickers()
    ticker = "^IXIC" # Nasdaq Composite
    start = dt.datetime(2020, 1, 1)
    end = dt.datetime.now()
    dataset = yf.download(ticker, start=start, end=end)
    dataset.to_csv("NasdaqData.csv")
    return dataset.to_csv("NasdaqData.csv")


get_data_from_yahoo()
