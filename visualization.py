import matplotlib.pyplot as plt
import yfinance as yf
import re
import os

def parse_backtest_results(file_path):
    """Parse backtest results from the text file."""
    with open(file_path, 'r') as file:
        data = file.read()

    strategies = re.split(r'\nStrategy:', data)
    parsed_data = {}

    for strategy in strategies:
        if not strategy.strip():
            continue

        strategy_name_match = re.search(r'^\s*(.*)', strategy)
        strategy_name = strategy_name_match.group(1).strip() if strategy_name_match else "Unknown"

        purchase_info = re.search(r'Initial cash: \d+\nPurchase date: (.+)\nCurrent date: (.+)', strategy)
        purchase_date = purchase_info.group(1) if purchase_info else None
        current_date = purchase_info.group(2) if purchase_info else None

        asset_info = re.findall(r'(\w+) \| Weight: (\d+\.\d+)', strategy)
        start_values = re.findall(r'(\w+) \| Weight: \d+\.\d+ \| Purchase price: \$.+ \| QTY: .+ \| Total value: \$(\d+\.\d+)', strategy)
        end_values = re.findall(r'(\w+) \| Change over period: [+-]?\d+\.\d+% \| Total value: \$(\d+\.\d+)', strategy)
        money_spent_match = re.search(r'Money spent: \$(\d+\.\d+)', strategy)
        money_spent = float(money_spent_match.group(1)) if money_spent_match else None

        assets = {symbol: float(weight) for symbol, weight in asset_info}
        start_values_dict = {symbol: float(value) for symbol, value in start_values}
        end_values_dict = {symbol: float(value) for symbol, value in end_values}

        parsed_data[strategy_name] = {
            'purchase_date': purchase_date,
            'current_date': current_date,
            'assets': assets,
            'start_values': start_values_dict,
            'end_values': end_values_dict,
            'money_spent': money_spent
        }

    return parsed_data

def get_full_company_names(symbols):
    """Fetch full company names for given stock symbols."""
    company_names = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            company_names[symbol] = ticker.info['longName']
        except:
            company_names[symbol] = symbol  # Fallback to symbol if name not available

    return company_names

def plot_pie_chart(portfolio, strategy, company_names, output_folder):
    """Plot pie chart of the portfolio allocation with full company names."""
    if not portfolio:
        print(f"No assets found for {strategy}, skipping pie chart.")
        return

    labels = [company_names.get(ticker, ticker) for ticker in portfolio.keys()]
    sizes = [weight for weight in portfolio.values()]

    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f'Portfolio Allocation for {strategy}')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend(labels, loc="best")

    # Save the figure as a jpg file
    output_path = os.path.join(output_folder, f"{strategy}_portfolio_allocation.jpg")
    plt.savefig(output_path, format='jpg')
    plt.close()

def plot_portfolio_growth(portfolio, strategy, start_date, end_date, output_folder):
    """Plot year-to-year growth of each asset in the portfolio."""
    if not portfolio:
        print(f"No assets found for {strategy}, skipping growth chart.")
        return

    plt.figure(figsize=(14, 7))

    for ticker in portfolio.keys():
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                print(f"No data for {ticker}, skipping.")
                continue
            data['Adj Close'].plot(label=ticker)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    plt.title(f'Year-to-Year Portfolio Growth for {strategy}')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend(loc='best')
    plt.grid(True)

    # Save the figure as a jpg file
    output_path = os.path.join(output_folder, f"{strategy}_portfolio_growth.jpg")
    plt.savefig(output_path, format='jpg')
    plt.close()


def plot_total_portfolio_value_growth(portfolio, strategy, cash, start_date, end_date, output_folder):
    """Plot the total portfolio value growth over time based on initial investments in money."""
    if not portfolio:
        print(f"No assets found for {strategy}, skipping total portfolio value growth chart.")
        return

    # Dictionary to store total portfolio value over time
    portfolio_values = None

    for ticker, weight in portfolio.items():
        try:
            # Fetch data from yfinance
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                print(f"No data for {ticker}, skipping.")
                continue

            # Calculate the initial investment amount in money for this asset
            initial_investment = cash * weight

            # Calculate the number of shares bought
            initial_price = data['Adj Close'].iloc[0]
            qty = initial_investment / initial_price

            # Calculate the value of this asset over time
            asset_values = qty * data['Adj Close']

            # Add this asset's values to the total portfolio values
            if portfolio_values is None:
                portfolio_values = asset_values
            else:
                portfolio_values += asset_values

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    # Plot the total portfolio value over time
    if portfolio_values is not None and not portfolio_values.empty:
        plt.figure(figsize=(14, 7))
        portfolio_values.plot(label='Total Portfolio Value')

        plt.title(f'Total Portfolio Value Growth for {strategy}')
        plt.xlabel('Date')
        plt.ylabel('Total Portfolio Value (in currency)')
        plt.legend(loc='best')
        plt.grid(True)

        # Save the figure as a jpg file
        output_path = os.path.join(output_folder, f"{strategy}_total_portfolio_value_growth.jpg")
        plt.savefig(output_path, format='jpg')
        plt.close()

        print(f"Successfully created {strategy}_total_portfolio_value_growth.jpg")
    else:
        print(f"No valid portfolio data found for {strategy}, unable to plot total portfolio value growth.")


def plot_combined_pie_chart(start_values, end_values, strategy, company_names, output_folder):
    """Plot combined pie charts for the starting and ending portfolio values."""
    # Ensure both start_values and end_values have the same set of keys (assets)
    all_assets = set(start_values.keys()).union(end_values.keys())

    # Fill missing values with 0 to ensure matching lengths
    start_values_filled = {asset: start_values.get(asset, 0) for asset in all_assets}
    end_values_filled = {asset: end_values.get(asset, 0) for asset in all_assets}

    labels = [company_names.get(ticker, ticker) for ticker in all_assets]
    start_sizes = [start_values_filled[ticker] for ticker in all_assets]
    end_sizes = [end_values_filled[ticker] for ticker in all_assets]

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    axs[0].pie(start_sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    axs[0].set_title(f'{strategy} - 2014')
    axs[0].axis('equal')

    axs[1].pie(end_sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    axs[1].set_title(f'{strategy} - 2024')
    axs[1].axis('equal')

    plt.suptitle(f'Portfolio Value Comparison for {strategy}')
    plt.legend(labels, loc="best")

    # Save the figure as a jpg file
    output_path = os.path.join(output_folder, f"{strategy}_portfolio_comparison.jpg")
    plt.savefig(output_path, format='jpg')
    plt.close()

if __name__ == '__main__':
    # Parse the backtest results file
    results_file = 'backtesting_results.txt'
    parsed_results = parse_backtest_results(results_file)

    # Create a folder for storing the images
    output_folder = 'media_files'
    os.makedirs(output_folder, exist_ok=True)

    for strategy, data in parsed_results.items():
        portfolio = data['assets']
        purchase_date = data['purchase_date']
        current_date = data['current_date']
        start_values = data['start_values']
        end_values = data['end_values']
        money_spent = data['money_spent']

        # Fetch company names for the portfolio
        company_names = get_full_company_names(portfolio.keys())

        # Ensure the strategy name is valid and replace spaces with underscores for filenames
        strategy = strategy.replace(" ", "_").replace("/", "_")

        # Plot portfolio allocation pie chart
        plot_pie_chart(portfolio, strategy, company_names, output_folder)

        # Plot portfolio growth over time
        plot_portfolio_growth(portfolio, strategy, purchase_date, current_date, output_folder)

        # Plot combined pie charts for start and end values
        plot_combined_pie_chart(start_values, end_values, strategy, company_names, output_folder)

        # Plot total revenue charts for start and end values
        plot_total_portfolio_value_growth(portfolio, strategy, money_spent, purchase_date, current_date, output_folder)

    # Verify that all files were created
    expected_files = [
        'Sharpe_Ratio_portfolio_allocation.jpg',
        'Sharpe_Ratio_portfolio_growth.jpg',
        'Sharpe_Ratio_portfolio_comparison.jpg',
        'Pure_Return_portfolio_allocation.jpg',
        'Pure_Return_portfolio_growth.jpg',
        'Pure_Return_portfolio_comparison.jpg',
        'Minimal_Volatility_portfolio_allocation.jpg',
        'Minimal_Volatility_portfolio_growth.jpg',
        'Minimal_Volatility_portfolio_comparison.jpg',
        'Black-Litterman_portfolio_allocation.jpg',
        'Black-Litterman_portfolio_growth.jpg',
        'Black-Litterman_portfolio_comparison.jpg',
        'Sharpe_Ratio_total_portfolio_value_growth.jpg',
        'Pure_Return_total_portfolio_value_growth.jpg',
        'Minimal_Volatility_total_portfolio_value_growth.jpg',
        'Black-Litterman_total_portfolio_value_growth.jpg'
    ]

    missing_files = [f for f in expected_files if not os.path.exists(os.path.join(output_folder, f))]

    if missing_files:
        print("The following files were not created as expected:")
        for file in missing_files:
            print(file)
    else:
        print("All files were created successfully.")
