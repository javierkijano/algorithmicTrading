import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


api_base_url = 'https://paper-api.alpaca.markets' #for paper trading
api_key_id = 'PKEDQO1QF8VXGFKNCGD4'
api_secret_key = '57liVkeUcXdRawLDlHtXUSXRTX2FGUjhBNPaazXS'

api = tradeapi.REST(key_id=api_key_id, secret_key=api_secret_key, base_url=api_base_url, api_version='v2') # or use ENV Vars shown below


assets = api.list_assets()
assets = [asset for asset in assets if asset.shortable and asset.tradable and asset.marginable and asset.easy_to_borrow]
exchanges = list(set([asset.exchange for asset in assets]))
symbols = [asset.symbol for asset in assets][:20]

# get market information about the symbols
barset = api.get_barset(symbols, 'day', limit=365)

# unpacking information into close, volume dictionary
data_symbols = [[[bar.t.date() for bar in barset[symbol]], [bar.c for bar in barset[symbol]], [bar.v for bar in barset[symbol]]] for symbol in symbols]
series_symbols_close = [ pd.Series(data=data_symbol[1], index=data_symbol[0]) for data_symbol in data_symbols]
series_symbols_volume = [ pd.Series(data=data_symbol[2], index=data_symbol[0]) for data_symbol in data_symbols]
df_data_close = pd.DataFrame(data=dict(zip(symbols, series_symbols_close)))
df_data_volume = pd.DataFrame(data=dict(zip(symbols, series_symbols_volume)))
df_data = dict(zip(['close', 'volume'], [df_data_close, df_data_volume]))

# calculate close_diff
df_data['close_diff'] = df_data['close'].diff(periods=1)
na_indices = np.where(df_data['close_diff'].isna())
for i, j in zip(na_indices[0], na_indices[1]):
    df_data['close_diff'].iat[i,j] = 0


# calculate close_diff_pct
df_data['close_diff_pct'] = df_data['close'].pct_change(periods=1)
na_indices = np.where(df_data['close_diff_pct'].isna())
for i, j in zip(na_indices[0], na_indices[1]):
    df_data['close_diff_pct'].iat[i,j] = 0
df_data['close_diff_pct'] = df_data['close_diff_pct'] + 1

# calculate cumulatives
df_data['close_diff_cum'] = df_data['close_diff'].cumsum()
df_data['close_diff_pct_cum'] = df_data['close_diff_pct'].cumprod()

# checks
# df_data['close'].iloc[0] + df_data['close_diff_cum'].iloc[-1]




# calculate highest and less correlated assets
df_data_corr = df_data['close_diff_pct'].corr()
for symbol_i, _ in enumerate(symbols):
    for symbol_j, _ in enumerate(symbols):
        if symbol_i >= symbol_j:
            df_data_corr.iloc[symbol_i][symbol_j] = np.nan
high_correlated_pairs = np.where(abs(df_data_corr) > 0.6)
high_correlated_pairs = [(symbols[high_correlated_pairs[0][i]], symbols[high_correlated_pairs[1][i]]) for i in range(len(high_correlated_pairs[0]))]
less_correlated_pairs = np.where(abs(df_data_corr) < 0.1)
less_correlated_pairs = [(symbols[less_correlated_pairs[0][i]], symbols[less_correlated_pairs[1][i]]) for i in range(len(less_correlated_pairs[0]))]


df_data['close']['TWTR', 'VTI'].plot()







# # flatten
# import itertools
# list(itertools.chain.from_iterable(temp))
#
# temp = [x.index for x in series_symbols_close]
#
#
# df_data['close']
#


# account = api.get_account()
# api.list_positions()
# pd.DataFrame.fr

#
# df = pd.DataFrame([data for barset in barsets[symbols],columns=['A','B','C'])
#
#
# barset = api.get_barset('AAPL', 'day', limit=5)
#
#
#
# # See how much AAPL moved in that timeframe.
# week_open = aapl_bars[0].o
# week_close = aapl_bars[-1].c
# percent_change = (week_close - week_open) / week_open * 100
# print('AAPL moved {}% over the last 5 days'.format(percent_change))
#
# # Submit a market order to buy 1 share of Apple at market price
# api.submit_order(
#     symbol='AAPL',
#     qty=1,
#     side='buy',
#     type='market',
#     time_in_force='gtc'
# )
#
