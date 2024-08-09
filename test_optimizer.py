from roboadvisor.optimizer import PortfolioOptimizer

# Define your parameters
assets = ['AAPL', 'GOOGL']
goal = 20000
years = 10
starting_cash = 2000

# Instantiate and use the optimizer
optimizer = PortfolioOptimizer(assets=assets, goal=goal, years=years, starting_cash=starting_cash)


# Run the optimization to achieve the goal
optimized_portfolio = optimizer.optimize_for_goal()

# Print the results
print("Optimized Portfolio:")
for asset in optimized_portfolio:
    print(asset)
