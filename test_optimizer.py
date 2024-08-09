from roboadvisor.optimizer import PortfolioOptimizer

# Define a list of assets (tickers) to consider for the portfolio
assets = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA', 'FB', 'BRK.B', 'V', 'JNJ', 'WMT']

# Define the goal, years, and starting cash
goal = 20000
years = 10
starting_cash = 2000

# Initialize the PortfolioOptimizer
optimizer = PortfolioOptimizer(assets=assets, goal=goal, years=years, starting_cash=starting_cash)

# Run the optimization to achieve the goal
optimized_portfolio = optimizer.optimize_for_goal()

# Print the results
print("Optimized Portfolio:")
for asset in optimized_portfolio:
    print(asset)
