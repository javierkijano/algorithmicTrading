import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

print(__doc__)
from collections import defaultdict

import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import pickle



# checks
#df_data['close'].iloc[0] + df_data['close_diff_cum'].iloc[-1]




class Simulator:

    def __init__(self, periodType='day', periods=365, rm_cache=True):

        api_base_url = 'https://paper-api.alpaca.markets'  # for paper trading
        api_key_id = 'PKEDQO1QF8VXGFKNCGD4'
        api_secret_key = '57liVkeUcXdRawLDlHtXUSXRTX2FGUjhBNPaazXS'

        self.api = tradeapi.REST(key_id=api_key_id, secret_key=api_secret_key, base_url=api_base_url,
                            api_version='v2')  # or use ENV Vars shown below

        if rm_cache:
            assets = self.api.list_assets()
            assets = [asset for asset in assets if
                      asset.shortable and asset.tradable and asset.marginable and asset.easy_to_borrow]
            exchanges = list(set([asset.exchange for asset in assets]))
            symbols = [asset.symbol for asset in assets][:20]

            # get market information about the symbols
            barset = self.api.get_barset(symbols, periodType, limit=periods)

            # unpacking information into close, volume dictionary
            data_symbols = [[[bar.t.date() for bar in barset[symbol]], [bar.c for bar in barset[symbol]],
                             [bar.v for bar in barset[symbol]]] for symbol in symbols]
            series_symbols_close = [pd.Series(data=data_symbol[1], index=data_symbol[0]) for data_symbol in data_symbols]
            series_symbols_volume = [pd.Series(data=data_symbol[2], index=data_symbol[0]) for data_symbol in data_symbols]
            df_data_close = pd.DataFrame(data=dict(zip(symbols, series_symbols_close)))
            df_data_volume = pd.DataFrame(data=dict(zip(symbols, series_symbols_volume)))
            df_data = dict(zip(['close', 'volume'], [df_data_close, df_data_volume]))

            # calculate derived metrics
            self._calculateDerivedMetrics()

            # save results in object
            pickle.dump(df_data, open("df_data.dat", "wb"))
        else:
            df_data = pickle.load(open("df_data.dat", "rb"))
            symbols = df_data['close'].columns.tolist()

        self.df_data = df_data
        self.symbols = symbols
        self.setPortfolioAllocation = np.zeros_like(self.symbols)




    def setPortfolio(self):
        # calculate highest and less correlated assets
        df_data_corr = self.df_data['close_diff_pct'].corr()
        for symbol_i, _ in enumerate(self.symbols):
            for symbol_j, _ in enumerate(self.symbols):
                if symbol_i >= symbol_j:
                    df_data_corr.iloc[symbol_i][symbol_j] = np.nan
        #high_correlated_pairs = np.where(abs(df_data_corr) > 0.6)
        #high_correlated_pairs = [(self.symbols[high_correlated_pairs[0][i]], self.symbols[high_correlated_pairs[1][i]]) for i in
        #                         range(len(high_correlated_pairs[0]))]
        less_correlated_pairs = np.where(abs(df_data_corr) < 0.1)
        less_correlated_pairs = [(self.symbols[less_correlated_pairs[0][i]], self.symbols[less_correlated_pairs[1][i]]) for i in
                                 range(len(less_correlated_pairs[0]))]

    def setPorfolio2(self, num_symbols, plot=False):

        corr = self.df_data['close'].corr(method="spearman").to_numpy()
        corr_linkage = hierarchy.ward(corr)

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            dendro = hierarchy.dendrogram(corr_linkage, labels=self.symbols, ax=ax1,
                                          leaf_rotation=90)
            dendro_idx = np.arange(0, len(dendro['ivl']))
            ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
            ax2.set_xticks(dendro_idx)
            ax2.set_yticks(dendro_idx)
            ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
            ax2.set_yticklabels(dendro['ivl'])
            fig.tight_layout()
            plt.show()

        # extract 5 representative symbols from clustering
        cluster_ids = hierarchy.fcluster(corr_linkage, num_symbols, criterion='maxclust')
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        selected_symbols =  [self.symbols[v[0]] for v in cluster_id_to_feature_ids.values()]

        # drop symbols from datasets
        symbols_to_drop = list(set(self.symbols) - set(selected_symbols))
        for metric in self.df_data:
            self.df_data[metric].drop(columns=symbols_to_drop, inplace=True)
        self.symbols = selected_symbols

    def _calculateDerivedMetrics(self):
        # calculate close_diff
        self.df_data['close_diff'] = self.df_data['close'].diff(periods=1)
        na_indices = np.where(self.df_data['close_diff'].isna())
        for i, j in zip(na_indices[0], na_indices[1]):
            self.df_data['close_diff'].iat[i, j] = 0

        # calculate close_diff_pct
        self.df_data['close_diff_pct'] = self.df_data['close'].pct_change(periods=1)
        na_indices = np.where(self.df_data['close_diff_pct'].isna())
        for i, j in zip(na_indices[0], na_indices[1]):
            self.df_data['close_diff_pct'].iat[i, j] = 0
        self.df_data['close_diff_pct'] = self.df_data['close_diff_pct'] + 1

        # calculate cumulatives
        self.df_data['close_diff_cum'] = self.df_data['close_diff'].cumsum()
        self.df_data['close_diff_pct_cum'] = self.df_data['close_diff_pct'].cumprod()

    def setInitialBalance(self, balance=10000):

        self.balance = balance

    def setPortfolioAllocation(self, balance_pct=0.1, portfolioAllocation=None):

        self.portfolioAllocation = np.repeat(self.balance*balance_pct/len(self.symbols), len(self.symbols))

    def simulate(self, start_date, end_date):
        pass

    def step(self):
        pass


class MontecarloSimulation:

    def __init__(self):
        pass

    def simulate(self, start_date, periods, num_samples=100):
        pass

simulator =Simulator(rm_cache=False)
simulator.setPorfolio2(5)

a=0







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
