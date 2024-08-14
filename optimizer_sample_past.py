import numpy as np
from roboadvisor.optimizer_past import PortfolioOptimizerPast

def format_portfolio(portfolio):
    """Helper function to format portfolio data."""
    formatted_portfolio = [
        (asset, round(float(weight), 4)) for asset, weight in portfolio[1]
    ]
    return portfolio[0], formatted_portfolio, round(float(portfolio[2]), 4), round(float(portfolio[3]), 4), round(float(portfolio[4]), 4)

if __name__ == '__main__':
    assets = ['TLT', 'SPY', 'GDX', 'AAPL', 'FXI', 'GLD', 'VDE', 'UUP', 'VT', 'IYF', 'EWI', 'TIP']
    optimizer = PortfolioOptimizerPast(assets, portfolio_size=5, end_date='2014-06-06', max_pos=0.30, min_pos=0.05)

    # Store results in a dictionary
    results = {}

    # Optimize for Sharpe Ratio
    optimizer.optimize_for_sharpe()
    results['Sharpe Ratio'] = format_portfolio(optimizer.best_sharpe_portfolio_)

    # Optimize for Pure Return
    optimizer.optimize_for_return()
    results['Pure Return'] = format_portfolio(optimizer.best_return_portfolio_)

    # Optimize for Minimal Volatility
    optimizer.optimize_for_volatility()
    results['Minimal Volatility'] = format_portfolio(optimizer.best_vol_portfolio_)

    # Example custom P, Q, omega matrices for Black-Litterman model
    P = np.array([[1, 0, -1] + [0] * (len(assets) - 3)])  # Example custom P matrix
    Q = np.array([0.03])  # Example custom Q vector
    omega = np.array([[0.02]])  # Example custom omega matrix

    # Optimize for Black-Litterman Model
    optimal_portfolio_bl = optimizer.optimize_for_black_litterman_model(P=P, Q=Q, omega=omega)

    # Round 'Black-Litterman' portfolio to 4 decimals and format it as a dictionary
    rounded_bl_portfolio = {asset: round(float(weight), 4) for asset, weight in optimal_portfolio_bl.items()}
    results['Black-Litterman'] = rounded_bl_portfolio

    # Save the results to a file (e.g., sample_results.py)
    with open("sample_results.py", "w") as file:
        # Ensure numpy is imported and write the results
        file.write("import numpy as np\n\n")
        file.write("results = {\n")
        for key, value in results.items():
            if isinstance(value, tuple):
                file.write(f"    '{key}': (np.array({value[0].tolist()}), {value[1]}, {value[2]}, {value[3]}, {value[4]}),\n")
            else:
                file.write(f"    '{key}': {value},\n")
        file.write("}\n")
