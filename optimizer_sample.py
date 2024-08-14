import numpy as np
from roboadvisor.optimizer import PortfolioOptimizer

if __name__=='__main__':
    assets = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'INTC', 'KO', 'JNJ', 'BRK', 'PG', 'SPY', 'QQQ', 'EEM', 'VTI', 'EFA', 'IWM', 'VEA', 'VWO', 'IVV', 'AGG', 'XOM', 'CSCO', 'NVDA']
    optimizer = PortfolioOptimizer(assets, portfolio_size=5, max_pos = 0.30, min_pos = 0.05)

