import numpy as np
import cupy as cp
import pandas as pd
import yfinance as yf
from itertools import combinations
from joblib import Parallel, delayed
import scipy.optimize as optimize
import time

class PortfolioOptimizer:
    def __init__(self, assets, risk_tolerance=5.0, portfolio_size=5, max_iters=None, print_init=True, max_pos=1.0, min_pos=0.0):
        self.max_pos_ = max_pos
        self.min_pos_ = min_pos
        self.print_init_ = print_init
        self.asset_basket_ = assets
        self.max_iters_ = max_iters
        self.portfolio_size_ = portfolio_size
        self.num_assets_ = portfolio_size
        self.risk_tolerance_ = risk_tolerance
        self.sim_iterations_ = 2500

        start_time = time.time()
        print("Starting data fetch...")
        self._fetch_data()
        print(f"Data fetched in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        print("Starting Sharpe optimization...")
        self.optimize_for_sharpe()
        print(f"Sharpe optimization completed in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        print("Starting return optimization...")
        self.optimize_for_return()
        print(f"Return optimization completed in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        print("Starting volatility optimization...")
        self.optimize_for_volatility()
        print(f"Volatility optimization completed in {time.time() - start_time:.2f} seconds")
        
    def _fetch_data(self):
        def download_asset(asset):
            try:
                temp = yf.download(asset, start="2000-01-01", end="2024-06-06")
                if not temp.empty:
                    temp = temp[['Adj Close']] 
                    temp.columns = [f"{asset}_Adj_Close"]
                    return temp
                else:
                    return None
            except Exception as e:
                return None

        df_list = Parallel(n_jobs=4)(delayed(download_asset)(asset) for asset in self.asset_basket_)
        df = pd.concat([d for d in df_list if d is not None], axis=1)

        df = df.dropna()
        features = [col for col in df.columns if "Adj_Close" in col]
        self.raw_asset_data = df[features].copy()
        self.asset_combos_ = list(combinations(features, self.portfolio_size_))
        
        if self.max_iters_ is None:
            self.max_iters_ = len(self.asset_combos_)

        elif len(self.asset_combos_) < self.max_iters_:
            self.max_iters_ = len(self.asset_combos_)

        batch_size = 10000  # Adjust batch size as needed
        self.sim_packages = []
        for i in range(0, self.max_iters_, batch_size):
            batch_combos = self.asset_combos_[i:i + batch_size]
            batch_sim_packages = Parallel(n_jobs=-1)(delayed(self._process_combination)(combo) for combo in batch_combos)
            self.sim_packages.extend(batch_sim_packages)
            print(f"Processed {i + len(batch_combos)} combinations...")

    def _process_combination(self, combo):
        assets = list(combo)
        filtered_df = self.raw_asset_data[assets].copy()
        returns = cp.array(np.log(filtered_df / filtered_df.shift(1)))  
        return_matrix = cp.mean(returns, axis=0) * 252  
        cov_matrix = cp.cov(returns.T) * 252 
        return [assets, cov_matrix, return_matrix]

    def portfolio_simulation(self):
        iterations = self.sim_iterations_
        self.simulation_results = []
        
        sim_packages = self.sim_packages.copy()
        
        for _ in range(len(self.sim_packages)):
            sim = sim_packages.pop()
            returns = cp.array(sim[2])
            cov_matrix = cp.array(sim[1])
            assets = sim[0]
                       
            port_sharpes = []
            port_returns = []
            port_vols = []
                        
            for _ in range(iterations):
                weights = cp.random.dirichlet(cp.ones(self.num_assets_), size=1)[0]
                ret = cp.sum(returns * weights)
                vol = cp.sqrt(cp.dot(weights.T, cp.dot(cov_matrix, weights)))
                port_returns.append(cp.asnumpy(ret))
                port_vols.append(cp.asnumpy(vol))
                port_sharpes.append(cp.asnumpy(ret / vol))
            
            self.simulation_results.append([assets, np.array(port_returns), np.array(port_vols), np.array(port_sharpes)])
     
    def portfolio_stats(self, weights):
        returns = cp.array(self.return_matrix_)
        cov_matrix = cp.array(self.cov_matrix_)
        
        weights = cp.array(weights)
        port_return = cp.sum(returns * weights)
        port_vol = cp.sqrt(cp.dot(weights.T, cp.dot(cov_matrix, weights)))
        sharpe = port_return / port_vol
        self._sharpe_ = cp.asnumpy(sharpe)
        self._port_return_ = cp.asnumpy(port_return)
        self._port_vol_ = cp.asnumpy(port_vol)     
        
        stats = [self._port_return_, self._port_vol_, self._sharpe_]        
        self.portfolio_stats_ = np.array(stats)
        
        return np.array(stats)

    def optimize_for_sharpe(self):
        min_con = self.min_pos_
        max_con = self.max_pos_  
        num_assets = self.portfolio_size_     

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((min_con, max_con) for x in range(num_assets))
        initializer = np.clip(np.random.dirichlet(np.ones(num_assets)), min_con, max_con)
        sim_packages = self.sim_packages.copy()
            
        def _maximize_sharpe(weights):     
            self.portfolio_stats(weights)
            sharpe = self._sharpe_         
            return -sharpe
           
        self.sharpe_scores_ = []

        for _ in range(len(sim_packages)):
            sim = sim_packages.pop()
            self.return_matrix_ = sim[2]
            self.cov_matrix_ = sim[1]
            self.assets_ = sim[0]
            
            optimal_sharpe = optimize.minimize(
                _maximize_sharpe,
                initializer,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9}
            )
            
            optimal_sharpe_weights_ = np.clip(optimal_sharpe['x'].round(4), min_con, max_con)
            optimal_sharpe_stats_ = self.portfolio_stats(optimal_sharpe_weights_)
            
            asset_list = [x.split('_')[0] for x in self.assets_]
            
            optimal_sharpe_portfolio_ = list(zip(asset_list, list(optimal_sharpe_weights_)))
            self.sharpe_scores_.append([optimal_sharpe_weights_,
                                        optimal_sharpe_portfolio_,
                                        round(optimal_sharpe_stats_[0] * 100, 4),
                                        round(optimal_sharpe_stats_[1] * 100, 4),
                                        round(optimal_sharpe_stats_[2], 4)])
        
        self.sharpe_scores_ = sorted(self.sharpe_scores_, key=itemgetter(4), reverse=True)
        self.best_sharpe_portfolio_ = self.sharpe_scores_[0]
        
        print('-----------------------------------------------')
        print('----- Portfolio Optimized for Sharpe Ratio ----')
        print('-----------------------------------------------')
        print('')
        print(*self.best_sharpe_portfolio_[1], sep='\n')
        print('')
        print(f'Optimal Portfolio Return: {self.best_sharpe_portfolio_[2]}')
        print(f'Optimal Portfolio Volatility: {self.best_sharpe_portfolio_[3]}')
        print(f'Optimal Portfolio Sharpe Ratio: {self.best_sharpe_portfolio_[4]}')
        print('')
        print('')

    def optimize_for_return(self):
        num_assets = self.portfolio_size_       
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for x in range(num_assets))
        initializer = np.clip(np.random.dirichlet(np.ones(num_assets)), 0, 1)
        sim_packages = self.sim_packages.copy()
         
        def _maximize_return(weights): 
            self.portfolio_stats(weights)
            port_return = self._port_return_
            return -port_return
        
        self.return_scores_ = []
        for _ in range(len(sim_packages)):
            sim = sim_packages.pop()
            self.return_matrix_ = sim[2]
            self.cov_matrix_ = sim[1]
            self.assets_ = sim[0]
            
            optimal_return = optimize.minimize(
                _maximize_return,
                initializer,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9}
            )
            
            optimal_return_weights_ = np.clip(optimal_return['x'].round(4), 0, 1)
            optimal_return_stats_ = self.portfolio_stats(optimal_return_weights_)
            
            asset_list = [x.split('_')[0] for x in self.assets_]
            optimal_return_portfolio_ = list(zip(asset_list, list(optimal_return_weights_)))
            self.return_scores_.append([optimal_return_weights_,
                                        optimal_return_portfolio_,
                                        round(optimal_return_stats_[0] * 100, 4),
                                        round(optimal_return_stats_[1] * 100, 4),
                                        round(optimal_return_stats_[2], 4)])
        
        self.return_scores_ = sorted(self.return_scores_, key=itemgetter(2), reverse=True)
        self.best_return_portfolio_ = self.return_scores_[0]
        
        if self.print_init_:
            print('-----------------------------------------------')
            print('----- Portfolio Optimized for Pure Return ----')
            print('-----------------------------------------------')
            print('')
            print(*self.best_return_portfolio_[1], sep='\n')
            print('')
            print(f'Optimal Portfolio Return: {self.best_return_portfolio_[2]}')
            print(f'Optimal Portfolio Volatility: {self.best_return_portfolio_[3]}')
            print(f'Optimal Portfolio Sharpe Ratio: {self.best_return_portfolio_[4]}')
            print('')
            print('')
    
    def optimize_for_volatility(self):
        num_assets = self.portfolio_size_       
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for x in range(num_assets))
        initializer = np.clip(np.random.dirichlet(np.ones(num_assets)), 0, 1)
        sim_packages = self.sim_packages.copy()
        
        def _minimize_volatility(weights):           
            self.portfolio_stats(weights)
            port_vol = self._port_vol_     
            return port_vol
        
        self.vol_scores_ = []
        for _ in range(len(sim_packages)):
            sim = sim_packages.pop()
            self.return_matrix_ = sim[2]
            self.cov_matrix_ = sim[1]
            self.assets_ = sim[0]
            
            optimal_vol = optimize.minimize(
                _minimize_volatility,
                initializer,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9}
            )
            
            optimal_vol_weights_ = np.clip(optimal_vol['x'].round(4), 0, 1)
            optimal_vol_stats_ = self.portfolio_stats(optimal_vol_weights_)
            
            asset_list = [x.split('_')[0] for x in self.assets_]
            optimal_vol_portfolio_ = list(zip(asset_list, list(optimal_vol_weights_)))
            self.vol_scores_.append([optimal_vol_weights_,
                                     optimal_vol_portfolio_,
                                     round(optimal_vol_stats_[0] * 100, 4),
                                     round(optimal_vol_stats_[1] * 100, 4),
                                     round(optimal_vol_stats_[2], 4)])
        
        self.vol_scores_ = sorted(self.vol_scores_, key=itemgetter(3))
        self.best_vol_portfolio_ = self.vol_scores_[0]

        if self.print_init_:      
            print('-----------------------------------------------------')
            print('----- Portfolio Optimized for Minimal Volatility ----')
            print('-----------------------------------------------------')
            print('')
            print(*self.best_vol_portfolio_[1], sep='\n')
            print('')
            print(f'Optimal Portfolio Return: {self.best_vol_portfolio_[2]}')
            print(f'Optimal Portfolio Volatility: {self.best_vol_portfolio_[3]}')
            print(f'Optimal Portfolio Sharpe Ratio: {self.best_vol_portfolio_[4]}')
            print('')
            print('') 
