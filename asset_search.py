import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging

np.float_ = np.float64

from prophet import Prophet
from asset_search_config import *
import os

# Setup logging
logging.basicConfig(filename='asset_search.log', level=logging.INFO)

# Ensure cmdstanpy is set as the backend for Prophet
from prophet import Prophet
import cmdstanpy

TEMP_FILE = 'temp_final_assets.txt'

def try_fetch_ticker(symbol, retries=3, delay=1):
    """
    Try fetching ticker information with multiple strategies for symbol conversion.
    """
    strategies = [
        lambda s: s,  # Original symbol
        lambda s: s.replace('$', '-P'),  # Replace $ with -P (preferred stock)
        lambda s: s.replace('$', '-'),  # Replace $ with -
        lambda s: s.replace('$', ''),  # Remove $ entirely
        lambda s: s.replace('$', '.'),  # Replace $ with .
        lambda s: s.replace('.', '-'),  # Replace . with -
        lambda s: s.replace('.', ''),  # Remove . entirely
    ]

    for strategy in strategies:
        modified_symbol = strategy(symbol)
        for attempt in range(retries):
            try:
                ticker = yf.Ticker(modified_symbol)
                if ticker.info:
                    logging.info(f"Successfully fetched data for {modified_symbol}")
                    return ticker, modified_symbol
            except Exception as e:
                logging.info(f"Error fetching {modified_symbol} (attempt {attempt + 1}/{retries}): {e}")
                time.sleep(delay)
        logging.info(f"Failed to fetch data for {modified_symbol} after {retries} retries.")

    logging.error(f"Failed to fetch data for {symbol} after trying all conversions.")
    return None, symbol


def fetch_assets_data():
    """
    Fetch and store asset data from Yahoo Finance.
    Separate data by asset types and store in CSV files.
    """
    stock_symbols_df = pd.read_csv('stock_symbols.csv')
    stock_symbols = stock_symbols_df['Symbol'].dropna().tolist()

    failed_symbols = []

    for symbol in stock_symbols:
        ticker, final_symbol = try_fetch_ticker(symbol)

        if not ticker:
            print(f"Failed to fetch data for {symbol} after trying all conversions.")
            failed_symbols.append(symbol)
            continue

        info = ticker.info
        asset_type = info.get('quoteType', 'Unknown')

        # Filtering by country if COUNTRY_FILTER is defined
        if COUNTRY_FILTER and info.get('country') not in COUNTRY_FILTER:
            continue

        if asset_type in ASSET_TYPE_RATIO.keys():
            data = {
                'ticker': final_symbol,
                'asset_type': asset_type,
                'pe_ratio': info.get('forwardPE', None),
                'market_cap': info.get('marketCap', None),
                'dividend_yield': info.get('dividendYield', None),
                'avg_volume': info.get('averageVolume', 0),  # Fallback to 0 if missing
                'analyst_coverage': info.get('numberOfAnalystOpinions', 0),  # Fallback to 0 if missing
                'institutional_ownership': info.get('heldPercentInstitutions', 0),  # Fallback to 0 if missing
                'country': info.get('country', 'Unknown')  # Fallback to 'Unknown' if missing
            }

            # Save to corresponding CSV file by asset type
            csv_filename = f'{asset_type}.csv'
            df = pd.DataFrame([data])
            df.to_csv(csv_filename, mode='a', header=not os.path.exists(csv_filename), index=False)

    # Save failed symbols to a separate file
    if failed_symbols:
        with open('failed_symbols.txt', 'w') as f:
            for symbol in failed_symbols:
                f.write(f"{symbol}\n")
        print(f"Failed to fetch data for {len(failed_symbols)} symbols. Check 'failed_symbols.txt' for details.")
    else:
        print("Successfully fetched data for all symbols.")


def filter_by_pe_ratio(df):
    """
    Filter assets by P/E ratio.
    """
    return df[(df['pe_ratio'].fillna(0) >= PE_RATIO['min']) & (df['pe_ratio'].fillna(float('inf')) <= PE_RATIO['max'])]


def filter_by_popularity(df):
    """
    Filter assets by popularity.
    Popularity is based on trading volume, analyst coverage, and institutional ownership.
    """
    df['popularity'] = (df['avg_volume'].fillna(0) * 0.4) + \
                       (df['analyst_coverage'].fillna(0) * 0.3) + \
                       (df['institutional_ownership'].fillna(0) * 0.3)

    if STOCK_POPULARITY == 'High':
        threshold = df['popularity'].quantile(0.75)
    elif STOCK_POPULARITY == 'Medium':
        threshold = df['popularity'].quantile(0.50)
    else:  # Low
        threshold = df['popularity'].quantile(0.25)

    return df[df['popularity'] >= threshold]


def apply_additional_filters(df):
    """
    Apply additional filters as specified in the config.
    """
    if 'market_cap_min' in ADDITIONAL_FILTERS:
        df = df[df['market_cap'].fillna(0) >= ADDITIONAL_FILTERS['market_cap_min']]
    if 'dividend_yield_min' in ADDITIONAL_FILTERS:
        df = df[df['dividend_yield'].fillna(0) >= ADDITIONAL_FILTERS['dividend_yield_min']]
    return df


def perform_forecasting(ticker, years=FORECAST_YEARS):
    """
    Perform forecasting using Prophet for a given ticker.
    Return forecasted values.
    """
    try:
        df = yf.download(ticker, period="5y")['Adj Close'].reset_index()
        df.rename(columns={'Date': 'ds', 'Adj Close': 'y'}, inplace=True)

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=years * 365)
        forecast = model.predict(future)

        return forecast[['ds', 'yhat']]
    except Exception as e:
        print(f"Forecasting failed for {ticker}: {e}")
        return None


def select_top_assets(df, asset_type, top_n):
    """
    Select the top N assets based on Prophet forecasting.
    """
    forecasts = {}
    for ticker in df['ticker']:
        forecast = perform_forecasting(ticker)
        if forecast is not None:
            final_value = forecast['yhat'].iloc[-1]
            forecasts[ticker] = final_value

    top_assets = sorted(forecasts, key=forecasts.get, reverse=True)[:top_n]
    return top_assets


def save_progress(asset_list):
    """
    Save progress to a temporary file.
    """
    with open(TEMP_FILE, 'w') as f:
        for asset in asset_list:
            f.write(f"{asset}\n")


def process_assets():
    """
    Process assets by fetching, filtering, forecasting, and selecting the final list.
    """
    # Fetch and store asset data
    fetch_assets_data()

    final_assets = []

    for asset_type, ratio in ASSET_TYPE_RATIO.items():
        csv_filename = f'{asset_type}.csv'
        if not os.path.exists(csv_filename):
            print(f"No data available for asset type {asset_type}, skipping.")
            continue

        df = pd.read_csv(csv_filename)

        # Apply filters
        df = filter_by_pe_ratio(df)
        df = filter_by_popularity(df)
        df = apply_additional_filters(df)

        # Select top N assets based on the ratio
        top_n = int(TOTAL_ASSETS * ratio)
        top_assets = select_top_assets(df, asset_type, top_n)
        final_assets.extend(top_assets)

        # Save progress to a temporary file
        save_progress(final_assets)

    # Write final asset list to the output file
    with open(OUTPUT_FILE, 'w') as f:
        for asset in final_assets:
            f.write(f"{asset}\n")

    print(f"Final list of assets written to {OUTPUT_FILE}")

    # Clean up the temporary file
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)


if __name__ == '__main__':
    # Attempt to resume from the last progress
    if os.path.exists(TEMP_FILE):
        with open(TEMP_FILE, 'r') as f:
            final_assets = [line.strip() for line in f.readlines()]
    else:
        final_assets = []

    process_assets()
