import numpy as np

results = {
    'Sharpe Ratio': (np.array([0.1495, 0.3, 0.3, 0.172, 0.0785]), [('AAPL', 0.1495), ('MSFT', 0.3), ('GOOG', 0.3), ('AMZN', 0.172), ('META', 0.0785)], 22.1866, 16.7851, 1.3218),
    'Pure Return': (np.array([0.0, 1.0, 0.0, 0.0, 0.0]), [('MSFT', 0.0), ('GOOG', 1.0), ('AMZN', 0.0), ('META', 0.0), ('INTC', 0.0)], 30.1385, 22.1297, 1.3619),
    'Minimal Volatility': (np.array([0.1726, 0.1869, 0.2527, 0.0769, 0.3109]), [('AAPL', 0.1726), ('MSFT', 0.1869), ('GOOG', 0.2527), ('AMZN', 0.0769), ('INTC', 0.3109)], 17.0039, 15.4262, 1.1023),
    'Black-Litterman': {'AAPL': 0.3158, 'INTC': 0.3158, 'AMZN': 0.226, 'MSFT': 0.0898, 'META': 0.0526},
}
