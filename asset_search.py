import pandas as pd
import yfinance as yf
import numpy as np

np.float_ = np.float64
from prophet import Prophet
from asset_search_config import TOTAL_ASSETS, ASSET_TYPE_RATIO, PE_RATIO, STOCK_POPULARITY, COUNTRY_FILTER, \
    ADDITIONAL_FILTERS, OUTPUT_FILE, FORECAST_YEARS, PROPHET_PARAMS


def convert_symbol(symbol):
    """ Convert special characters in the symbol to a more query-friendly format. """
    return symbol.replace('$', '_').replace('.', '-')


def fetch_historical_data(symbol, period='2y'):
    """ Fetch historical data for the given symbol using yfinance. """
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        if not hist.empty:
            return hist
    except Exception as e:
        with open('fetch_errors.log', 'a') as f:
            f.write(f"Error fetching historical data for {symbol}: {e}\n")
    return None


def forecast_performance(df, forecast_years, params):
    """ Use Prophet to forecast the future performance of the asset. """
    df = df.reset_index()
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

    # Remove timezone information from the 'ds' column
    df['ds'] = df['ds'].dt.tz_localize(None)

    model = Prophet(**params)
    model.fit(df[['ds', 'y']])
    future = model.make_future_dataframe(periods=forecast_years * 365)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(forecast_years * 365)


def select_top_forecasted_assets(financial_df, forecast_years, params):
    """ Filter the top assets based on Prophet forecasted performance. """
    forecasted_data = []
    for index, row in financial_df.iterrows():
        symbol = row['Symbol']
        hist_data = fetch_historical_data(symbol)
        if hist_data is not None:
            forecast = forecast_performance(hist_data, forecast_years, params)
            final_value = forecast['yhat'].iloc[-1]
            current_value = hist_data['Close'].iloc[-1]  # Get the most recent closing price

            # Calculate expected growth percentage
            growth_percentage = ((final_value - current_value) / current_value) * 100

            forecasted_data.append({
                'Symbol': symbol,
                'Current Value': current_value,
                'Forecasted Value': final_value,
                'Expected Growth (%)': growth_percentage,
                'PE Ratio': row['PE Ratio'],
                'Market Cap': row['Market Cap'],
                'Type': row['Type'],
                'Country': row['Country'],
                'Volume': row['Volume'],
                'Forecast Reason': f"Forecasted to reach {final_value:.2f} in {FORECAST_YEARS} years based on historical performance. Expected growth of {growth_percentage:.2f}%.",
            })

    forecasted_df = pd.DataFrame(forecasted_data)
    return forecasted_df.sort_values(by='Forecasted Value', ascending=False)


# Step 1: Read the stock symbols from the CSV
symbols_df = pd.read_csv('stock_symbols.csv')
symbols = symbols_df['Symbol'].dropna().apply(convert_symbol).tolist()

# Step 2: Fetch financial data
financial_data = []
for symbol in symbols:
    try:
        stock = yf.Ticker(symbol)
        data = stock.info
        financial_data.append({
            'Symbol': symbol,
            'PE Ratio': data.get('trailingPE'),
            'Market Cap': data.get('marketCap'),
            'Price': data.get('regularMarketPrice'),
            'Type': data.get('quoteType'),
            'Country': data.get('country'),
            'Volume': data.get('regularMarketVolume'),
        })
    except Exception as e:
        with open('fetch_errors.log', 'a') as f:
            f.write(f"Error fetching data for {symbol}: {e}\n")

financial_df = pd.DataFrame(financial_data)

# Step 3: Ensure correct data types
financial_df['PE Ratio'] = pd.to_numeric(financial_df['PE Ratio'], errors='coerce')
financial_df['Market Cap'] = pd.to_numeric(financial_df['Market Cap'], errors='coerce')
financial_df['Volume'] = pd.to_numeric(financial_df['Volume'], errors='coerce')

# Drop rows where P/E Ratio is NaN
financial_df.dropna(subset=['PE Ratio'], inplace=True)

# Step 4: Apply Filters

# Filter based on P/E Ratio
financial_df = financial_df[
    (financial_df['PE Ratio'] >= PE_RATIO['min']) &
    (financial_df['PE Ratio'] <= PE_RATIO['max'])
]

# Filter based on country if specified
if COUNTRY_FILTER:
    financial_df = financial_df[financial_df['Country'].isin(COUNTRY_FILTER)]

# Filter based on stock popularity (trading volume)
volume_thresholds = {
    'High': financial_df['Volume'].quantile(0.75),
    'Medium': financial_df['Volume'].median(),
    'Low': financial_df['Volume'].quantile(0.25)
}

# Allow filtering by multiple popularity levels
volume_filter = financial_df['Volume'] >= min(volume_thresholds[pop] for pop in STOCK_POPULARITY)
financial_df = financial_df[volume_filter]

# Additional filters
for key, value in ADDITIONAL_FILTERS.items():
    financial_df = financial_df[financial_df[key] == value]

# Step 5: Forecasting and Select Top Assets

forecasted_assets = select_top_forecasted_assets(financial_df, FORECAST_YEARS, PROPHET_PARAMS)

# Step 6: Apply Asset Type Ratio to Final Selection

selected_assets = pd.DataFrame()

for asset_type, ratio in ASSET_TYPE_RATIO.items():
    if ratio > 0:
        count = max(int(TOTAL_ASSETS * ratio), 1)  # Ensure at least 1 asset is selected if ratio > 0
        filtered_df = forecasted_assets[forecasted_assets['Type'] == asset_type]
        top_assets = filtered_df.head(count)
        selected_assets = pd.concat([selected_assets, top_assets])

# If not enough assets are selected (e.g., due to missing data), repeat selection to fill the quota
while len(selected_assets) < TOTAL_ASSETS:
    remaining_count = TOTAL_ASSETS - len(selected_assets)
    additional_assets = forecasted_assets[~forecasted_assets['Symbol'].isin(selected_assets['Symbol'])].head(
        remaining_count)
    selected_assets = pd.concat([selected_assets, additional_assets])

# Step 7: Add Detailed Reasoning for Selection

def generate_reason(row):
    """ Generate detailed reason for selecting this asset. """
    reason = f"Selected based on forecasted value of {row['Forecasted Value']:.2f}. "
    reason += f"Current Value: {row['Current Value']:.2f}, Expected Growth: {row['Expected Growth (%)']:.2f}%. "
    reason += f"PE Ratio: {row['PE Ratio']}, Market Cap: {row['Market Cap']}. "
    reason += row['Forecast Reason']
    return reason

selected_assets['Reason'] = selected_assets.apply(generate_reason, axis=1)

# Step 8: Save the selected assets to a CSV file
selected_assets.to_csv(OUTPUT_FILE, index=False)

print(f"Top {TOTAL_ASSETS} assets saved to '{OUTPUT_FILE}'")
