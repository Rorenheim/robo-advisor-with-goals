# The Ultimate Open Robo-Advisor Project

*Originally developed by Kevin Vecmanis (VanAurum open robo-advisor)*

This open-source initiative provides the necessary tools for building optimal Markowitz portfolios and determining the best rebalancing strategy, considering transaction costs. With the financial world moving towards zero-cost ETFs, this project aims to empower individuals with industry-leading portfolio management tools at no cost.

## Mission

The mission of this project is to place powerful portfolio management tools into the hands of individuals at zero cost, helping users create and manage optimal investment portfolios.

## Files and Descriptions

### 1. **optimizer.py**
This file contains the main `PortfolioOptimizer` class, which is used to optimize a portfolio based on various criteria such as Sharpe ratio, pure return, volatility, and the Black-Litterman model. The class utilizes historical data fetched from Yahoo Finance and offers methods for fetching data, simulating portfolios, calculating portfolio statistics, and optimizing for different performance metrics.

### 2. **optimizer_past.py**
This module is a modified version of the `PortfolioOptimizer` class, specifically designed for backtesting with historical data. It includes an additional parameter, `end_date`, to define the end of the historical period for backtesting purposes. The methods in this class are similar to those in `optimizer.py` but tailored for backtesting scenarios.

### 3. **optimizer_sample.py**
This script provides a sample implementation of the `PortfolioOptimizer` class. It demonstrates how to initialize the optimizer with a set of assets and parameters and includes an optional custom implementation of the Black-Litterman model using custom P, Q, and omega matrices.

### 4. **optimizer_sample_past.py**
This script offers a sample implementation of the `PortfolioOptimizerPast` class, used for backtesting. It includes helper functions to format portfolio data and stores the results of different optimization strategies. The script also demonstrates how to optimize a portfolio using the Sharpe ratio, pure return, minimal volatility, and the Black-Litterman model.

### 5. **backtesting.py**
This script is designed to backtest portfolios based on historical data. It uses the results generated from optimizer scripts and simulates the performance of portfolios from a purchase date to the current date. The script provides detailed outputs, including initial investments, portfolio values, and gains or losses over time. It also saves the results to text files for further analysis.

### 6. **visualization.py**
This module includes functions to parse backtesting results and generate visualizations such as pie charts, portfolio growth charts, and combined value comparisons. The charts are saved as image files, allowing for easy review and presentation of portfolio performance over time.

### 7. **sample_results.py**
This file contains sample results from various portfolio optimization strategies, including Sharpe ratio, pure return, minimal volatility, and Black-Litterman. The results are stored in a dictionary format, making them easy to access and use for backtesting or further analysis.

## Installation Guide

To set up and run the portfolio optimization tools in this repository, follow these steps:

### Prerequisites

Ensure you have the following installed on your system:

- **Python 3.7+**
- **pip** (Python package installer)

### Step 1: Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/Rorenheim/robo-advisor-yfinance-markowitz-black-litterman.git

### Step 2: Navigate to the Repository Directory
Change into the directory where the repository was cloned:

```bash
cd repository-name
```

### Step 3: Create a Virtual Environment (Optional but Recommended)
Create and activate a virtual environment to manage dependencies:

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 4: Install Required Python Packages
Install the necessary packages:

```bash
pip install matplotlib pandas numpy yfinance scipy
```

### Step 5: Running the Scripts

**Portfolio Optimization**
To optimize a portfolio, you can run the sample script:

```bash
python optimizer_sample.py
```

**Backtesting**
To backtest the optimized portfolio strategies:

```bash
python backtesting.py
```

**Visualization**
To generate visualizations of the portfolio performance:

```bash
python visualization.py
```

### Step 6: Customize and Extend
You can customize the scripts by modifying the asset lists, optimization parameters, and other settings in the respective Python files. The scripts are modular and can be adapted to different use cases, such as optimizing portfolios with different asset classes or using alternative data sources.

### Step 7: Deactivate the Virtual Environment
Once you're done, you can deactivate the virtual environment:

```bash
deactivate
```

### Step 8: Update the Repository
If you make changes to the scripts, commit your changes and push them to the GitHub repository:

```bash
git add .
git commit -m "Describe your changes here"
git push origin main
```
