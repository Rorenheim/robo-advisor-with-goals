import yfinance as yf
import datetime

def backtest_portfolio(cash, purchase_date, current_date, portfolio, output_file):
    with open(output_file, 'a') as file:
        file.write(f"Initial cash: {cash}\n")
        file.write(f"Purchase date: {purchase_date}\n")
        file.write(f"Current date: {current_date}\n\n")

        total_value_at_purchase = 0
        investments = []

        # Calculate start date for a wider window (3 months before purchase date)
        start_date = (datetime.datetime.strptime(purchase_date, '%Y-%m-%d') - datetime.timedelta(days=90)).strftime('%Y-%m-%d')

        # Get prices within a 3-month window before the purchase date
        for ticker, weight in portfolio.items():
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
                    file.write(f"No data found for {ticker} in the window {start_date} to {purchase_date}. Skipping.\n")
            except Exception as e:
                file.write(f"Error fetching data for {ticker}: {e}. Skipping.\n")

        leftover_cash = cash - total_value_at_purchase

        file.write("-----------------------------------------------\n")
        file.write("----- Portfolio Backtest Results ----\n")
        file.write("-----------------------------------------------\n\n")

        file.write(f"{purchase_date}:\n")
        if investments:
            for inv in investments:
                file.write(f"{inv[0]} | Weight: {inv[1]:.4f} | Purchase price: ${inv[2]:.2f} | QTY: {inv[3]} | Total value: ${inv[4]:.2f}\n")

            file.write(f"\nMoney spent: ${total_value_at_purchase:.2f}\n")
            file.write(f"Money left: ${leftover_cash:.2f}\n\n")
        else:
            file.write("No valid investments made due to missing historical data.\n")

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
                    file.write(f"No current data found for {inv[0]}. Skipping.\n")
            except Exception as e:
                file.write(f"Error fetching current data for {inv[0]}: {e}. Skipping.\n")

        file.write(f"{current_date}:\n")
        if gains_losses:
            for gl in gains_losses:
                file.write(f"{gl[0]} | Change over period: {gl[3]:+.2f}% | Total value: ${gl[1]:.2f} | Gain/Loss: ${gl[2]:+.2f}\n")

            file.write(f"\nTotal value: ${total_current_value:.2f}\n")
            file.write(f"Total gain/loss: ${total_current_value - total_value_at_purchase:.2f}\n")
            if total_value_at_purchase > 0:
                file.write(f"Total change over time: {((total_current_value - total_value_at_purchase) / total_value_at_purchase) * 100:.2f}%\n\n")
            else:
                file.write("Total change over time: N/A (No valid investments)\n\n")
        else:
            file.write("No valid current values to calculate gain/loss.\n")

if __name__ == '__main__':
    from sample_results import results  # Import results from sample_results.py

    cash = 10000  # Initial cash in EUR
    purchase_date = "2014-06-06"  # Purchase date
    current_date = datetime.datetime.today().strftime('%Y-%m-%d')  # Current date

    # Specify the output file
    output_file = "backtesting_results.txt"

    # Clear the file if it exists
    open(output_file, 'w').close()

    # Backtest for each portfolio strategy in the results
    for strategy, data in results.items():
        if isinstance(data, tuple):
            portfolio = {ticker: float(weight) for ticker, weight in data[1]}
        elif isinstance(data, dict):
            portfolio = data

        with open(output_file, 'a') as file:
            file.write(f"\nStrategy: {strategy}\n")
        backtest_portfolio(cash, purchase_date, current_date, portfolio, output_file)
