# Configuration for asset search and filtering

# Total number of assets to include in the final list
TOTAL_ASSETS = 20

# Ratio of asset types in the final portfolio.
# The sum of all ratios should equal 1.
# Adjust the ratio as needed for each asset type.
ASSET_TYPE_RATIO = {
    'ETF': 0.4,
    'Stock': 0.4,
    'Bond': 0.1,
    'MutualFund': 0.1
}

# P/E Ratio filter (Price-to-Earnings Ratio)
# P/E Ratio thresholds - specify a minimum and/or maximum value
PE_RATIO = {
    'min': 10,  # Minimum P/E Ratio
    'max': 30   # Maximum P/E Ratio
}

# Stock popularity filter based on trading volume
# Options: 'High', 'Medium', 'Low'
STOCK_POPULARITY = 'High'

# Country filter - List of ISO 3166-1 alpha-2 country codes
# Example: ['US', 'CA', 'GB'] for United States, Canada, and Great Britain.
# Leave empty for a global search.
COUNTRY_FILTER = ['US']

# Additional filters based on financial metrics
# Add any additional filters as needed. Leave empty or None if not used.
ADDITIONAL_FILTERS = {
    'market_cap_min': 1e9,  # Minimum market cap (in dollars)
    'dividend_yield_min': 0.02,  # Minimum dividend yield (as a decimal, e.g., 0.02 for 2%)
    'pe_ratio_max': 25,  # Maximum P/E ratio, redundant with PE_RATIO but can be used as a different criterion
}

# Forecasting configuration using Prophet
# Number of years to forecast the performance of assets
FORECAST_YEARS = 5

# Prophet forecasting parameters
# Leave empty or default for Prophet to use its default settings.
PROPHET_PARAMS = {
    'daily_seasonality': False,
    'yearly_seasonality': True,
    'weekly_seasonality': False
}

# Name of the output file to store the final list of assets
OUTPUT_FILE = 'final_list.txt'