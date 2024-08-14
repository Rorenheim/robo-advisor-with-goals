import yfinance as yf
import datetime
import os

def backtest_portfolio(cash, purchase_date, current_date, portfolio, strategy):
    print(f"Initial cash: {cash}")
    print(f"Purchase date: {purchase_date}")
    print(f"Current date: {current_date}")
    print("")

    total_value_at_purchase = 0
    investments = []

    # Calculate start date for a wider window (3 months before purchase date)
    start_date = (datetime.datetime.strptime(purchase_date, '%Y-%m-%d') - datetime.timedelta(days=90)).strftime('%Y-%m-%d')

    # Get prices within a 3-month window before the purchase date
    for ticker, weight in portfolio.items():
        if weight == 0:
            continue  # Skip if weight is zero to avoid unnecessary API calls and errors
        try:
            data = yf.download(ticker, start=start_date, end=purchase_date)
            if not data.empty:
                # Use the closest available data point before or on the purchase_date
                purchase_price = data['Adj Close'].iloc[-1]
                qty = (cash * weight) // purchase_price
                investment_value = qty * purchase_price
                total_value_at_purchase += investment_value
                investments.append((ticker, weight, purchase_price, qty, investment_value))
            else:
                print(f"No data found for {ticker} in the window {start_date} to {purchase_date}. Skipping.")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}. Skipping.")

    leftover_cash = cash - total_value_at_purchase

    result_str = ""
    result_str += f"-----------------------------------------------\n"
    result_str += f"----- Portfolio Backtest Results ({strategy}) ----\n"
    result_str += f"-----------------------------------------------\n\n"

    result_str += f"{purchase_date}:\n"
    if investments:
        for inv in investments:
            result_str += f"{inv[0]} | Weight: {inv[1]:.4f} | Purchase price: ${inv[2]:.2f} | QTY: {inv[3]} | Total value: ${inv[4]:.2f}\n"

        result_str += f"\nMoney spent: ${total_value_at_purchase:.2f}\n"
        result_str += f"Money left: ${leftover_cash:.2f}\n\n"
    else:
        result_str += "No valid investments made due to missing historical data.\n"

    total_current_value = 0
    gains_losses = []

    # Calculate start date for a wider window for current prices
    start_current_date = (datetime.datetime.strptime(current_date, '%Y-%m-%d') - datetime.timedelta(days=3)).strftime('%Y-%m-%d')

    # Get current prices within a 3-day window around the current date
    for inv in investments:
        try:
            data = yf.download(inv[0], start=start_current_date, end=current_date)
            if not data.empty:
                # Use the last available data point within the range
                current_price = data['Adj Close'].iloc[-1]
                current_value = inv[3] * current_price
                gain_loss = current_value - inv[4]
                total_current_value += current_value
                gains_losses.append((inv[0], current_value, gain_loss, (gain_loss / inv[4]) * 100))
            else:
                print(f"No current data found for {inv[0]}. Skipping.")
        except Exception as e:
            print(f"Error fetching current data for {inv[0]}: {e}. Skipping.")

    result_str += f"{current_date}:\n"
    if gains_losses:
        for gl in gains_losses:
            result_str += f"{gl[0]} | Change over period: {gl[3]:+.2f}% | Total value: ${gl[1]:.2f} | Gain/Loss: ${gl[2]:+.2f}\n"

        result_str += f"\nTotal value: ${total_current_value:.2f}\n"
        result_str += f"Total gain/loss: ${total_current_value - total_value_at_purchase:.2f}\n"
        if total_value_at_purchase > 0:
            result_str += f"Total change over time: {((total_current_value - total_value_at_purchase) / total_value_at_purchase) * 100:.2f}%\n"
        else:
            result_str += "Total change over time: N/A (No valid investments)\n"
    else:
        result_str += "No valid current values to calculate gain/loss.\n"

    print(result_str)

    # Save the results to a file
    results_dir = "backtest_results"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"backtest_results_{strategy}.txt"), "w") as file:
        file.write(result_str)

if __name__ == '__main__':
    from sample_results import results  # Import results from sample_results.py

    cash = 10000  # Initial cash in EUR
    purchase_date = "2014-06-06"  # Purchase date
    current_date = datetime.datetime.today().strftime('%Y-%m-%d')  # Current date

    # Backtest for each portfolio strategy in the results
    for strategy, data in results.items():
        if isinstance(data, tuple):
            portfolio = {ticker: float(weight) for ticker, weight in data[1]}
        elif isinstance(data, dict):
            portfolio = data

        print(f"\nStrategy: {strategy}\n")
        backtest_portfolio(cash, purchase_date, current_date, portfolio, strategy)
