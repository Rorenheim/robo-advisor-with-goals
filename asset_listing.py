import pandas as pd
from yahoo_fin import stock_info as si


def normalize_symbol(symbol):
    """
    Normalize stock symbols by replacing $ with - and handling special suffixes.
    """
    # Replace $ with - for preferred stocks or specific classes
    if '$' in symbol:
        symbol = symbol.replace('$', '-')

    # If the symbol ends with .U, normalize it to -U
    if symbol.endswith('.U'):
        symbol = symbol.replace('.U', '-U')

    return symbol


def fetch_stock_symbols():
    """
    Fetch stock symbols from major US exchanges and combine them into a single set.
    """
    # Gather stock symbols from major US exchanges
    df1 = pd.DataFrame(si.tickers_sp500())
    df2 = pd.DataFrame(si.tickers_nasdaq())
    df3 = pd.DataFrame(si.tickers_dow())
    df4 = pd.DataFrame(si.tickers_other())

    # Convert DataFrame to list, then to sets
    sym1 = set(normalize_symbol(symbol) for symbol in df1[0].values.tolist())
    sym2 = set(normalize_symbol(symbol) for symbol in df2[0].values.tolist())
    sym3 = set(normalize_symbol(symbol) for symbol in df3[0].values.tolist())
    sym4 = set(normalize_symbol(symbol) for symbol in df4[0].values.tolist())

    # Join the 4 sets into one. Because it's a set, there will be no duplicate symbols
    symbols = set.union(sym1, sym2, sym3, sym4)

    # Some stocks are 5 characters. Those stocks with the suffixes listed below are not of interest.
    suffixes_to_exclude = ['W', 'R', 'P', 'Q']
    filtered_symbols = set()

    for symbol in symbols:
        if len(symbol) <= 4 or symbol[-1] not in suffixes_to_exclude:
            filtered_symbols.add(symbol)

    return list(filtered_symbols)


def save_symbols(symbols):
    """
    Save the stock symbols to a CSV file.
    """
    df = pd.DataFrame(symbols, columns=["Symbol"])
    df.to_csv('stock_symbols.csv', index=False)
    print(f"All symbols saved to 'stock_symbols.csv'.")


if __name__ == '__main__':
    # Fetch all stock symbols from exchanges
    symbols = fetch_stock_symbols()
    print(f"Total symbols fetched: {len(symbols)}")

    # Save the symbols to a CSV file
    save_symbols(symbols)
