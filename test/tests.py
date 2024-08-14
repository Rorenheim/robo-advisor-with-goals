class PortfolioOptimizer:
    def __init__(self, assets, risk_tolerance=5.0, portfolio_size=5, max_iters=None, print_init=True, max_pos=1.0, min_pos=0.0):
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
        if self.raw_asset_data is not None:
            self.optimize_for_sharpe()
            self.optimize_for_return()
            self.optimize_for_volatility()
            self.optimize_for_black_litterman_model()

    def _fetch_data(self):
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
                    temp = temp[['Adj Close']]
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
            self.raw_asset_data = None
            return

        df = df.dropna()
        features = [col for col in df.columns if "Adj_Close" in col]
        if not features:
            print("No adjusted close prices found.")
            self.raw_asset_data = None
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
            return_matrix = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            self.num_assets_ = len(assets)
            self.sim_packages.append([assets, cov_matrix, return_matrix])

        print('Omitted assets:', self.asset_errors_)
        print(f'Time to fetch data: {time.time() - start:.2f} seconds')

    def optimize_for_sharpe(self):
        if not self.sim_packages:
            print("No data available for optimization.")
            return
        # Optimization logic...

    def optimize_for_return(self):
        if not self.sim_packages:
            print("No data available for optimization.")
            return
        # Optimization logic...

    def optimize_for_volatility(self):
        if not self.sim_packages:
            print("No data available for optimization.")
            return
        # Optimization logic...

    def optimize_for_black_litterman_model(self, P=None, Q=None, omega=None):
        if self.raw_asset_data is None:
            print("No data available for Black-Litterman optimization.")
            return
        # Optimization logic...