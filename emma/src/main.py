import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

print(__doc__)
from collections import defaultdict

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import pickle
from datetime import *
import pygmo

# checks
#df_data['close'].iloc[0] + df_data['close_diff_cum'].iloc[-1]




class Simulator:

    def __init__(self, balance=10000, periodType='day', periods=365, rm_cache=True):

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
        self.portfolioAllocation = np.repeat(0, len(self.symbols))
        self.portfolioValue = 0
        self.balance = balance
        self.equity = 0
        self.simulations = []

    def setPortfolio2(self):
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

    def setPorfolio(self, num_symbols, plot=False):

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

#    def setInitialBalance(self, balance=10000):
#
#        self.balance = balance

    def setPortfolioAllocation(self, balance_pct=0.1, portfolioAllocation=[]):

        self.balance = self.balance + self.portfolioValue
        if len(portfolioAllocation) == 0:
            self.portfolioAllocation = np.repeat(self.balance*balance_pct/len(self.symbols), len(self.symbols))
        else:
            self.portfolioAllocation = portfolioAllocation
        self.portfolioValue = sum(self.portfolioAllocation)
        self.balance = self.balance - self.portfolioValue
        self.equity = self.balance + self.portfolioValue

    def getStrategyPortfolioAllocation(self):

        return self.portfolioAllocation

    def montecarlo_simulate(self, start_date, end_date, num_montecarlo_runs=500, conf_interval=(10,90), plot=False):

        tsGenerator = SyntheticTimeSeriesGenerator(self.df_data['close_diff_pct'])
        l_df_ts = tsGenerator.generate(start_date, end_date, num_timeseries=num_montecarlo_runs, method='bootstrapping')

        dates = self.df_data['close'].loc[start_date:end_date].index
        df_simulations = pd.DataFrame(data=None, index=dates, columns=['run_'+str(i) for i in range(num_montecarlo_runs)])

        for run_i in range(num_montecarlo_runs):
            if run_i % 10 == 0:
                pass
                # print('... artificial timeseries generation (%d/%d)' % (run_i,num_montecarlo_runs))
            strategyPortfolioAllocation = self.getStrategyPortfolioAllocation()
            df_ts = l_df_ts[run_i]
            df_ts = df_ts.cumprod()
            s_portfolioValue = np.sum(df_ts/df_ts.iloc[0]*strategyPortfolioAllocation, axis=1)
            df_simulations['run_'+str(run_i)] = s_portfolioValue.values

        df_real = self.simulatePortfolioValue(start_date, end_date, plot=False).to_frame()
        df_conf_interval = pd.DataFrame(data=np.percentile(df_simulations, q=conf_interval, axis=1).transpose(),
                                        columns=['lowerConf', 'upperConf'], index=df_simulations.index)
        df_simulations_predicted = pd.concat([df_simulations.mean(axis=1).to_frame(), df_conf_interval], axis=1)
        df_simulations_predicted.columns = ['mean'] + df_conf_interval.columns.to_list()
        if plot:
            df_real.plot()
            df_simulations_predicted['mean'].plot()
            plt.fill_between(df_simulations_predicted.index,
                             df_simulations_predicted['lowerConf'],
                             df_simulations_predicted['upperConf'],
                             facecolor='green', alpha=0.2, interpolate=True)
            plt.show()
            #df_simulations.iloc[-1].plot.hist()
            #plt.show()
            print('... expected mean = %f' % np.mean(df_simulations.iloc[-1]))
            np.percentile(df_simulations.iloc[-1], q=[5, 10, 25, 50, 75, 90, 95])
            a=0
        return (df_real, df_simulations_predicted)

    def simulatePortfolio(self, start_date, end_date, plot=False):

        dates = self.df_data['close'].loc[start_date:end_date].index
        strategyPortfolioAllocation = self.getStrategyPortfolioAllocation()
        df_real = self.df_data['close_diff_pct_cum'].loc[dates]
        df_real = df_real/df_real.iloc[0]*strategyPortfolioAllocation

        if plot:
            df_real.plot()
            plt.show()

        return df_real

    def simulatePortfolioValue(self, start_date, end_date, plot=False):

        dates = self.df_data['close'].loc[start_date:end_date].index
        strategyPortfolioAllocation = self.getStrategyPortfolioAllocation()
        df_real = self.df_data['close_diff_pct_cum'].loc[dates]
        df_real = np.sum(df_real / df_real.iloc[0] * strategyPortfolioAllocation, axis=1)

        if plot:
            df_real.plot()
            plt.show()

        return df_real


class SyntheticTimeSeriesGenerator:

    def __init__(self, df_data):

        self.df_data = df_data

    def generate(self, start_date, end_date, num_timeseries=None, method='bootstrapping'):

        if method == 'bootstrapping':

            l_df_ts = self.generate_bootstrapping(start_date, end_date, num_timeseries=num_timeseries)

        elif method == 'meloinvento':

            l_df_ts = self.generate_bootstrapping(start_date, end_date, num_timeseries=num_timeseries)

        else:
            pass

        return l_df_ts

    def generate_bootstrapping(self, start_date, end_date, num_timeseries):

        dates = self.df_data.loc[start_date:end_date].index
        l_df_ts = []
        for ts_i in range(num_timeseries):
            sample_dates = np.random.choice(dates, len(dates), replace=True)
            df_ts_i = self.df_data.loc[sample_dates]
            l_df_ts.append(df_ts_i)
        return l_df_ts



class MOoptimizationProblemDefinition:

    def __init__(self, simulator, start_date, end_date, num_montecarlo_runs=50):

        self.simulator = simulator
        self._start_date = start_date
        self._end_date = end_date
        self._bounds = (len(self.simulator.symbols)*[0], len(self.simulator.symbols)*[simulator.equity])
        self._num_montecarlo_runs = num_montecarlo_runs

    # Define objectives
    def fitness(self, x):

        x = x/np.sum(x)*self.simulator.portfolioValue
        self.simulator.setPortfolioAllocation(portfolioAllocation=x)
        df_real, df_simulations_predicted = self.simulator.montecarlo_simulate(self._start_date, self._end_date, self._num_montecarlo_runs)
        f1 = np.sum(df_simulations_predicted['upperConf']-df_simulations_predicted['lowerConf'], axis=0)
        f2 = np.sum(np.absolute(df_real-df_simulations_predicted['mean']), axis=0)
        return [f1, f2]

    # Return number of objectives
    def get_nobj(self):

        return 2

    # inequality constraints
    def get_nic(self):
        return 0

    # equality constraints
    def get_nec(self):

        return 0

    # Return bounds of decision variables
    def get_bounds(self):
        return self._bounds

    # Return function name
    def get_name(self):
        return "Porfolio optimization"


simulator = Simulator(balance=5000, rm_cache=False)
simulator.setPorfolio(5)
simulator.setPortfolioAllocation(balance_pct=0.2)
# simulator.simulatePortfolio(start_date=date(2020,2,1), end_date=date(2020,3,30), plot=True)
# simulator.simulatePortfolioValue(start_date=date(2020,2,1), end_date=date(2020,3,30), plot=True)
# simulator.montecarlo_simulate(start_date=date(2020,2,1), end_date=date(2020,3,30), plot=True)

prob = pygmo.problem(MOoptimizationProblemDefinition(simulator, start_date=date(2020,3,1), end_date=date(2020,3,31), num_montecarlo_runs=10))

# create population
pop = pygmo.population(prob, size=20*4)
# select algorithm
algo = pygmo.algorithm(pygmo.nsga2(gen=50))
algo.set_verbosity(1)

# run optimization

pop = algo.evolve(pop)
pickle.dump(pop, open("optimizationPopulation.dat", "wb"))

#ndf, dl, dc, ndl = pygmo.fast_non_dominated_sorting(pop.get_f())
#pygmo.plot_non_dominated_fronts(pop.get_f())
#plt.plot()

if False:
    pop = pickle.load(open("optimizationPopulation.dat", "rb"))
    pygmo.plot_non_dominated_fronts(pop.get_f())
    best_index = np.where(np.all(pop.get_f() < [4000,1000], axis=1))[0][0]
    best_portfolioAllocation = pop.get_x()[best_index]
    best_portfolioAllocation = best_portfolioAllocation/np.sum(best_portfolioAllocation)*simulator.portfolioValue
    simulator.setPortfolioAllocation(portfolioAllocation=best_portfolioAllocation)
    simulator.montecarlo_simulate(start_date=date(2020,3,1), end_date=date(2020,3,31), num_montecarlo_runs=50, plot=True)




# extract results
#fits, vectors = pop.get_f(), pop.get_x()
# extract and print non-dominated fronts
#ndf, dl, dc, ndr = pygmo.fast_non_dominated_sorting(fits)
#print(ndf)

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
