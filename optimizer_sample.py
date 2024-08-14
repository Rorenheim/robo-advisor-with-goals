import numpy as np
from roboadvisor.optimizer import PortfolioOptimizer

if __name__=='__main__':
    assets=['TLT','SPY','GDX','AAPL','FXI','GLD','VDE','UUP','VT','IYF','EWI','TIP']
    optimizer = PortfolioOptimizer(assets, portfolio_size=5)

    # Optional custom P, Q, omega matrices
P = np.array([[1, 0, -1] + [0] * (len(assets) - 3)])  # Example custom P matrix
Q = np.array([0.03])  # Example custom Q vector
omega = np.array([[0.02]])  # Example custom omega matrix

# Run Black-Litterman optimization with or without custom P, Q, omega
optimal_portfolio = optimizer.optimize_for_black_litterman_model(P=P, Q=Q, omega=omega)
