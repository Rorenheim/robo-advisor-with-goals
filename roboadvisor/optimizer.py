# optimizer.py
'''
@author: Kevin Vecmanis
'''

# Standard Python library imports
import matplotlib
import datetime
import pandas as pd
import numpy as np
import time
import logging
import random
from itertools import combinations
import scipy.optimize as optimize
from operator import itemgetter
from numpy.linalg import inv

# 3rd party imports
import yfinance as yf

class PortfolioOptimizer:
    '''
    This class object receives a list of assets (tickers) and a portfolio size
    and returns the optimal portfolio optimized for Sharpe, Pure Return, and
    Volatility

    Parameters:
    -----------
        assets : list
            A list of stock tickers that the optimizer should choose from to build the portfolio.
        risk_tolerance: float, optional (default=5.0)
            A number on a scale of 1.0 to 10.0 that indicates the acceptable risk level.
        portfolio_size: int, optional (default=5)
            The number of assets that should be in the final optimal portfolio.
        max_iters: int, optional (default=None)
            The number of times the portfolio simulation should be run by the optimizer.
        print_init : bool, optional (default=True)
            Whether or not to print the portfolio metrics after initialization.
        max_pos : float, optional (default=1.0)
            The maximum weight that one asset can occupy in a portfolio.
        min_pos : float, optional (default=0.0)
            The minimum weight that one asset can occupy in a portfolio.

    Attributes:
    -----------
        asset_basket_ - the list of assets in its entirety
        asset_errors_ - the number of stock tickers that weren't found or had no data.
        cov_matrix_results - list of the covariance matrices for each unique asset combination.
        return_matrix_results - list of the return matrices for each unique asset combination.
        asset_combo_list - list of all the unique asset combinations.
        max_iters_ - number of portfolio combinations to analyze
        portfolio_size_ - the number of assets that can be used in the optimal portfolio
        assets_ - instantiation of an attribute to be used during optimization
        risk_tolerance - maximum volatility client is willing to incur. 1.0 = 100%
        raw_asset_data - a master copy of the adjusted close dataframe from our data query.
        sim_iterations - the number of random portfolio weights to simulate for each asset combination.
        sim_packages - a master queue of all the asset combinations and corresponding data to be analyzed.
        _sharpe_ - local variable for storing and passing sharpe score
        _port_return_ - local variable for storing and passing portfolio return
        _port_vol_ - local variable for storing and passing portfolio volatility
        portfolio_stats_ - list portfolio stats for a given asset combo and weight.
        sharpe_scores_ - comprehensive matrix of simulation results for sharpe optimization
        return_scores_ - comprehensive matrix of simulation results for return optimization
        vol_scores - comprehensive matrix of simulation results for volatility optimization

    Methods:
    --------
        _fetch_data - Get data from yfinance using the list of assets in asset_basket
        _plot_asset_prices - plot the normalized adjusted closes for all the assets.
        portfolio_simulation - simulate and plot markowitz bullet for one specified asset combination.
        portfolio_stats - calculates performance metrics for one set of weights on one asset combination.
        optimize_for_sharpe - Finds the optimal portfolio that provides best Sharpe ratio
        optimize_for_return - Finds the optimal portfolio that provides the best Return.
        optimize_for_volatility - Finds the optimal portfolio that provides the smallest volatility.
    '''

    def __init__(self,
                 assets,
                 risk_tolerance=5.0,
                 portfolio_size=5,
                 max_iters=None,
                 print_init=True,
                 max_pos=1.0,
                 min_pos=0.0):

        '''
        Initiation calls four functions and instantiates several attributes.
        '''
        matplotlib.use('PS')
        self.max_pos_ = max_pos
        self.min_pos_ = min_pos
        self.print_init_ = print_init
        self.asset_basket_ = assets
        self.max_iters_ = max_iters
        self.portfolio_size_ = portfolio_size
        self.assets_ = assets
        self.num_assets_ = portfolio_size
        self.risk_tolerance_ = risk_tolerance
        self.sim_iterations_ = 2500
        self._fetch_data()
        self.optimize_for_sharpe()
        self.optimize_for_return()
        self.optimize_for_volatility()
        self.optimize_for_black_litterman_model()

    def _fetch_data(self):
        '''
        This function fetches data from yfinance using the list of assets in asset_basket
        and declares additional class attributes pertaining to the data needed for analysis.

        asset_errors_
        asset_combos_
        raw_asset_data
        sim_packages

        Returns:
            None
        '''
        start = time.time()
        self.asset_errors_ = []
        self.cov_matrix_results = []
        self.return_matrix_results = []
        self.asset_combo_list = []
        df = pd.DataFrame()
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')

        for asset in self.asset_basket_:
            try:
                temp = yf.download(asset, start="2000-01-01", end=end_date)
                if not temp.empty:
                    temp = temp[['Adj Close']]  # Use Adjusted Close prices
                    temp.columns = [f"{asset}_Adj_Close"]
                    if df.empty:
                        df = temp
                    else:
                        df = pd.merge(df, temp, how='outer', left_index=True, right_index=True)
                else:
                    print(f"No data found for asset: {asset}")
                    self.asset_errors_.append(asset)
            except Exception as e:
                print(f"Error fetching data for {asset}: {e}")
                self.asset_errors_.append(asset)

        if df.empty:
            print("No valid data fetched for any asset.")
            return

        df = df.dropna()
        features = [col for col in df.columns if "Adj_Close" in col]
        if not features:
            print("No adjusted close prices found.")
            return

        df = df[features]
        self.raw_asset_data = df.copy()
        self.asset_combos_ = list([combo for combo in combinations(features, self.portfolio_size_)])
        print(f'Number of unique asset combinations: {len(self.asset_combos_)}')

        if self.max_iters_ is None:
            self.max_iters_ = len(self.asset_combos_)

        elif len(self.asset_combos_) < self.max_iters_:
            self.max_iters_ = len(self.asset_combos_)

        print(f'Analyzing {self.max_iters_} of {len(self.asset_combos_)} asset combinations...')

        self.sim_packages = []
        for i in range(self.max_iters_):
            assets = list(self.asset_combos_[i])
            filtered_df = df[assets].copy()
            returns = np.log(filtered_df / filtered_df.shift(1))
            return_matrix = returns.mean() * 252  # Generate annualized return matrix
            cov_matrix = returns.cov() * 252  # Generate annualized covariance matrix
            self.num_assets_ = len(assets)
            self.sim_packages.append([assets, cov_matrix, return_matrix])

        print('Omitted assets:', self.asset_errors_)
        print(f'Time to fetch data: {time.time() - start:.2f} seconds')

    def portfolio_simulation(self):
        '''
        Runs a simulation by randomly selecting portfolio weights a specified
        number of times (iterations), returns the list of results and plots
        all the portfolios as well.

        Returns:
        -------
            port_returns: array, array of all the simulated portfolio returns.
            port_vols: array, array of all the simulated portfolio volatilities.
        '''
        start = time.time()
        iterations = self.sim_iterations_
        self.simulation_results = []

        # Take a copy of simulation packages so that the original copy isn't altered
        sim_packages = self.sim_packages.copy()

        # Loop through each return and covariance matrix from all the asset combos.
        for _ in range(len(self.sim_packages)):
            # Pop a simulation package and load returns, cov_matrix, and asset list from it.
            sim = sim_packages.pop()
            returns = np.array(sim[2])
            cov_matrix = np.array(sim[1])
            assets = sim[0]

            port_sharpes = []
            port_returns = []
            port_vols = []

            for _ in range(iterations):
                weights = np.random.dirichlet(np.ones(self.num_assets_), size=1)
                weights = weights[0]
                ret = np.sum(returns * weights)
                vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                port_returns.append(ret)
                port_vols.append(vol)
                port_sharpes.append(ret / vol)

            # Declare additional class attributes from the results
            port_returns = np.array(port_returns)
            port_vols = np.array(port_vols)
            port_sharpes = np.array(port_sharpes)
            self.simulation_results.append([assets, port_returns, port_vols, port_sharpes])

        print('---')
        print(f'Time to simulate portfolios: {time.time() - start:.2f} seconds')
        print('---')

    def portfolio_stats(self, weights):
        '''
        We can gather the portfolio performance metrics for a specific set of weights.
        This function will be important because we'll want to pass it to an optimization
        function - either Hyperopt or Scipy SCO to get the portfolio with the best
        desired characteristics.

        Note: Sharpe ratio here uses a risk-free short rate of 0.

        Parameters:
        ----------
            weights: array, asset weights in the portfolio.

        Returns:
        --------
            array, portfolio statistics - mean, volatility, sharp ratio.
        '''
        returns = self.return_matrix_
        cov_matrix = self.cov_matrix_

        # Convert to array in case list was passed instead.
        weights = np.array(weights)
        port_return = np.sum(returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = port_return / port_vol
        self._sharpe_ = sharpe
        self._port_return_ = port_return
        self._port_vol_ = port_vol

        stats = [port_return, port_vol, sharpe]
        self.portfolio_stats_ = np.array(stats)

        return np.array(stats)

    def optimize_for_sharpe(self):
        '''Optimization function to optimize on Sharpe Ratio.
        '''
        min_con = self.min_pos_
        max_con = self.max_pos_
        num_assets = self.portfolio_size_

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((min_con, max_con) for x in range(num_assets))
        initializer = num_assets * [1. / num_assets,]
        sim_packages = self.sim_packages.copy()

        def _maximize_sharpe(weights):
            self.portfolio_stats(weights)
            sharpe = self._sharpe_
            return -sharpe

        self.sharpe_scores_ = []

        for _ in range(len(sim_packages)):
            sim = sim_packages.pop()
            self.return_matrix_ = np.array(sim[2])
            self.cov_matrix_ = np.array(sim[1])
            self.assets_ = sim[0]

            optimal_sharpe = optimize.minimize(
                _maximize_sharpe,
                initializer,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
            )

            optimal_sharpe_weights_ = optimal_sharpe['x'].round(4)
            optimal_sharpe_stats_ = self.portfolio_stats(optimal_sharpe_weights_)

            # Here we just strip out the 'Adj_Close' tag from the asset list
            x = self.assets_
            asset_list = []
            for i in range(len(x)):
                temp = x[i].split('_')
                asset_list.append(temp[0])

            optimal_sharpe_portfolio_ = list(zip(asset_list, list(optimal_sharpe_weights_)))
            self.sharpe_scores_.append([optimal_sharpe_weights_,
                                        optimal_sharpe_portfolio_,
                                        round(optimal_sharpe_stats_[0] * 100, 4),
                                        round(optimal_sharpe_stats_[1] * 100, 4),
                                        round(optimal_sharpe_stats_[2], 4)])

        self.sharpe_scores_ = sorted(self.sharpe_scores_, key=itemgetter(4), reverse=True)
        self.best_sharpe_portfolio_ = self.sharpe_scores_[0]
        temp = self.best_sharpe_portfolio_

        print('-----------------------------------------------')
        print('----- Portfolio Optimized for Sharpe Ratio ----')
        print('-----------------------------------------------')
        print('')
        print(*temp[1], sep='\n')
        print('')
        print(f'Optimal Portfolio Return: {temp[2]}')
        print(f'Optimal Portfolio Volatility: {temp[3]}')
        print(f'Optimal Portfolio Sharpe Ratio: {temp[4]}')
        print('')
        print('')

    def optimize_for_return(self):
        '''Function to optimize purely on return.
        '''
        num_assets = self.portfolio_size_
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for x in range(num_assets))
        initializer = num_assets * [1. / num_assets,]
        sim_packages = self.sim_packages.copy()

        def _maximize_return(weights):
            self.portfolio_stats(weights)
            port_return = self._port_return_
            return -port_return

        self.return_scores_ = []
        for _ in range(len(sim_packages)):
            sim = sim_packages.pop()
            self.return_matrix_ = np.array(sim[2])
            self.cov_matrix_ = np.array(sim[1])
            self.assets_ = sim[0]

            optimal_return = optimize.minimize(
                _maximize_return,
                initializer,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
            )

            optimal_return_weights_ = optimal_return['x'].round(4)
            optimal_return_stats_ = self.portfolio_stats(optimal_return_weights_)

            # Here we just strip out the 'Adj_Close' tag from the asset list
            x = self.assets_
            asset_list = []
            for i in range(len(x)):
                temp = x[i].split('_')
                asset_list.append(temp[0])

            optimal_return_portfolio_ = list(zip(asset_list, list(optimal_return_weights_)))
            self.return_scores_.append([optimal_return_weights_,
                                        optimal_return_portfolio_,
                                        round(optimal_return_stats_[0] * 100, 4),
                                        round(optimal_return_stats_[1] * 100, 4),
                                        round(optimal_return_stats_[2], 4)])

        self.return_scores_ = sorted(self.return_scores_, key=itemgetter(2), reverse=True)
        self.best_return_portfolio_ = self.return_scores_[0]
        temp = self.best_return_portfolio_

        if self.print_init_:
            print('-----------------------------------------------')
            print('----- Portfolio Optimized for Pure Return ----')
            print('-----------------------------------------------')
            print('')
            print(*temp[1], sep='\n')
            print('')
            print(f'Optimal Portfolio Return: {temp[2]}')
            print(f'Optimal Portfolio Volatility: {temp[3]}')
            print(f'Optimal Portfolio Sharpe Ratio: {temp[4]}')
            print('')
            print('')

    def optimize_for_volatility(self):
        '''Function to optimize on volatility only (risk)
        '''
        num_assets = self.portfolio_size_
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for x in range(num_assets))
        initializer = num_assets * [1. / num_assets,]
        sim_packages = self.sim_packages.copy()

        def _minimize_volatility(weights):
            self.portfolio_stats(weights)
            port_vol = self._port_vol_
            return port_vol

        self.vol_scores_ = []
        for _ in range(len(sim_packages)):
            sim = sim_packages.pop()
            self.return_matrix_ = np.array(sim[2])
            self.cov_matrix_ = np.array(sim[1])
            self.assets_ = sim[0]

            optimal_vol = optimize.minimize(
                _minimize_volatility,
                initializer,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
            )

            optimal_vol_weights_ = optimal_vol['x'].round(4)
            optimal_vol_stats_ = self.portfolio_stats(optimal_vol_weights_)

            # Here we just strip out the 'Adj_Close' tag from the asset list
            x = self.assets_
            asset_list = []
            for i in range(len(x)):
                temp = x[i].split('_')
                asset_list.append(temp[0])

            optimal_vol_portfolio_ = list(zip(asset_list, list(optimal_vol_weights_)))
            self.vol_scores_.append([optimal_vol_weights_,
                                     optimal_vol_portfolio_,
                                     round(optimal_vol_stats_[0] * 100, 4),
                                     round(optimal_vol_stats_[1] * 100, 4),
                                     round(optimal_vol_stats_[2], 4)])

        self.vol_scores_ = sorted(self.vol_scores_, key=itemgetter(3))
        self.best_vol_portfolio_ = self.vol_scores_[0]
        temp = self.best_vol_portfolio_

        if self.print_init_:
            print('-----------------------------------------------------')
            print('----- Portfolio Optimized for Minimal Volatility ----')
            print('-----------------------------------------------------')
            print('')
            print(*temp[1], sep='\n')
            print('')
            print(f'Optimal Portfolio Return: {temp[2]}')
            print(f'Optimal Portfolio Volatility: {temp[3]}')
            print(f'Optimal Portfolio Sharpe Ratio: {temp[4]}')
            print('')
            print('')

    def implied_rets(self, risk_aversion, sigma, w):
        '''Calculate implied returns based on risk aversion, covariance matrix, and weights.'''
        implied_rets = risk_aversion * sigma.dot(w).squeeze()
        return implied_rets

    def optimize_for_black_litterman_model(self, P=None, Q=None, omega=None):
        '''Function to optimize the portfolio using the Black-Litterman model.'''
        # Configure logging
        logging.basicConfig(filename='black_litterman_optimization.log', level=logging.INFO)
        logging.info("Starting Black-Litterman optimization...")

        # Step 1: Calculate market equilibrium returns
        market_weights = np.array([1.0 / self.raw_asset_data.shape[1]] * self.raw_asset_data.shape[1])
        market_cov = self.raw_asset_data.cov()
        market_return = np.dot(market_weights, self.raw_asset_data.mean())
        market_var = np.dot(market_weights.T, np.dot(market_cov, market_weights))
        risk_aversion = market_return / market_var
        logging.info(f"Market return: {market_return}, Market variance: {market_var}, Risk aversion: {risk_aversion}")

        implied_equilibrium_returns = self.implied_rets(risk_aversion, market_cov, market_weights)
        logging.info(f"Implied equilibrium returns: {implied_equilibrium_returns}")

        # Step 2: Define views and uncertainties
        if P is None:
            P = np.array([[1, -1] + [0] * (self.raw_asset_data.shape[1] - 2)])  # Default View matrix
        if Q is None:
            Q = np.array([0.05])  # Default View returns
        if omega is None:
            tau = 0.025
            omega = np.diag(np.diag(P.dot(tau * market_cov).dot(P.T)))  # Default Uncertainty in views
        logging.info(f"View matrix P: {P}, View returns Q: {Q}, Uncertainty omega: {omega}")

        # Step 3: Combine views with market equilibrium using Black-Litterman model
        tau = 0.025
        sigma_scaled = market_cov * tau
        BL_return_vector = implied_equilibrium_returns + sigma_scaled.dot(P.T).dot(
            inv(P.dot(sigma_scaled).dot(P.T) + omega).dot(Q - P.dot(implied_equilibrium_returns))
        )
        logging.info(f"Black-Litterman adjusted returns: {BL_return_vector}")

        # Step 4: Optimize portfolio using adjusted returns
        inverse_cov = pd.DataFrame(inv(market_cov.values), index=market_cov.columns, columns=market_cov.index)
        initial_weights = inverse_cov.dot(BL_return_vector)
        initial_weights = initial_weights / sum(initial_weights)

        def objective(weights):
            port_return = np.sum(weights * BL_return_vector)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(market_cov, weights)))
            sharpe = port_return / port_vol
            return -sharpe

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((self.min_pos_, self.max_pos_) for _ in range(len(BL_return_vector)))
        result = optimize.minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        optimal_weights = result.x

        # Step 5: Strip out '_Adj_Close' suffix and keep only top 'portfolio_size' assets
        tickers = [ticker.replace('_Adj_Close', '') for ticker in self.raw_asset_data.columns]
        sorted_weights = sorted(zip(tickers, optimal_weights), key=lambda x: -x[1])[:self.portfolio_size_]

        # Ensure the sum of the top weights is 1
        final_weights = np.array([weight for _, weight in sorted_weights])
        final_weights = final_weights / final_weights.sum()
        sorted_weights = [(ticker, weight) for (ticker, _), weight in zip(sorted_weights, final_weights)]

        # Select the corresponding returns and covariance matrix for the chosen assets
        selected_tickers = [ticker for ticker, _ in sorted_weights]
        selected_return_vector = BL_return_vector[[ticker + '_Adj_Close' for ticker in selected_tickers]]
        selected_cov_matrix = market_cov.loc[selected_return_vector.index, selected_return_vector.index]

        # Calculate portfolio metrics based on selected assets
        optimal_portfolio_return = np.dot(final_weights, selected_return_vector)
        optimal_portfolio_volatility = np.sqrt(np.dot(final_weights.T, np.dot(selected_cov_matrix, final_weights)))
        optimal_portfolio_sharpe_ratio = optimal_portfolio_return / optimal_portfolio_volatility

        # Log the results
        logging.info(f"Black-Litterman optimized portfolio weights: {dict(sorted_weights)}")

        # Print the results
        print('-----------------------------------------------------')
        print('----- Portfolio Optimized for Black-Litterman ----')
        print('-----------------------------------------------------')
        for asset, weight in sorted_weights:
            print(f"('{asset}', np.float64({weight:.4f}))")
        print('')
        print(f'Optimal Portfolio Return: {optimal_portfolio_return:.4f}')
        print(f'Optimal Portfolio Volatility: {optimal_portfolio_volatility:.4f}')
        print(f'Optimal Portfolio Sharpe Ratio: {optimal_portfolio_sharpe_ratio:.4f}')

        # Return the optimized portfolio weights
        return dict(sorted_weights)