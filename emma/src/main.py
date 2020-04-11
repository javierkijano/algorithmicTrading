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
            symbols = [asset.symbol for asset in assets][:200]

            #set([getattr(asset, 'class') for asset in assets])

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
            df_data = self._calculateDerivedMetrics(df_data)

            # save results in object
            pickle.dump(df_data, open("../data/df_data.dat", "wb"))
        else:
            df_data = pickle.load(open("../data/df_data.dat", "rb"))
            symbols = df_data['close'].columns.tolist()

        self.df_data = df_data
        self.symbols = symbols
        self.portfolioAllocation = np.repeat(0, len(self.symbols))
        self.portfolioAllocationDirection = np.repeat('long', len(self.symbols))
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

    def _calculateDerivedMetrics(self, df_data):
        # calculate close_diff
        df_data['close_diff'] = df_data['close'].diff(periods=1)
        na_indices = np.where(df_data['close_diff'].isna())
        for i, j in zip(na_indices[0], na_indices[1]):
            df_data['close_diff'].iat[i, j] = 0

        # calculate close_diff_pct
        df_data['close_diff_pct'] = df_data['close'].pct_change(periods=1)
        na_indices = np.where(df_data['close_diff_pct'].isna())
        for i, j in zip(na_indices[0], na_indices[1]):
            df_data['close_diff_pct'].iat[i, j] = 0
        df_data['close_diff_pct'] = df_data['close_diff_pct'] + 1

        # calculate cumulatives
        df_data['close_diff_cum'] = df_data['close_diff'].cumsum()
        df_data['close_diff_pct_cum'] = df_data['close_diff_pct'].cumprod()
        return df_data

#    def setInitialBalance(self, balance=10000):
#
#        self.balance = balance

    def setPortfolioAllocation(self, balance_pct=0.1, portfolioAllocation=[], portfolioAllocationDirection=[]):

        if len(portfolioAllocationDirection) == 0:
            self.portfolioAllocationDirection = len(self.symbols)*['long']
        else:
            self.portfolioAllocationDirection = portfolioAllocationDirection

        self.balance = self.balance + self.portfolioValue
        if len(portfolioAllocation) == 0:
            self.portfolioAllocation = np.repeat(self.balance*balance_pct/len(self.symbols), len(self.symbols))
        else:
            self.portfolioAllocation = portfolioAllocation
        self.portfolioValue = sum(self.portfolioAllocation)
        self.balance = self.balance - self.portfolioValue
        self.equity = self.balance + self.portfolioValue

    def getStrategyPortfolioAllocation(self):

        return self.portfolioAllocation, self.portfolioAllocationDirection

    def montecarlo_simulate(self, train_dates, predict_dates, num_montecarlo_runs=500, conf_interval=(5,95), plot=False):

        tsGenerator = SyntheticTimeSeriesGenerator(self.df_data['close_diff_pct'])
        l_df_ts_inc = tsGenerator.generate(train_dates, predict_dates, num_timeseries=num_montecarlo_runs, method='bootstrapping')

        dates = self.df_data['close'].loc[predict_dates[0]:predict_dates[1]].index
        df_simulations = pd.DataFrame(data=None, index=dates, columns=['run_'+str(i) for i in range(num_montecarlo_runs)])

        for run_i in range(num_montecarlo_runs):
            if run_i % 10 == 0:
                pass
                # print('... artificial timeseries generation (%d/%d)' % (run_i,num_montecarlo_runs))
            strategyPortfolioAllocation, strategyPortfolioAllocationDirection = self.getStrategyPortfolioAllocation()
            df_ts_inc = l_df_ts_inc[run_i]
            df_ts_inc = (df_ts_inc - 1) *\
                    [1 if self.portfolioAllocationDirection[symbol_i] == 'long' else -1 for symbol_i, _ in enumerate(self.symbols)] *\
                    strategyPortfolioAllocation
            df_ts_cum = strategyPortfolioAllocation + np.cumsum(df_ts_inc)
            s_portfolioValue = np.sum(df_ts_cum, axis=1)
            df_simulations['run_'+str(run_i)] = s_portfolioValue.values

        df_porfolioValue_cum, _ = self.simulatePortfolioValue(train_dates[0], predict_dates[1], plot=False)
        df_porfolioValue_cum = df_porfolioValue_cum/df_porfolioValue_cum.loc[df_porfolioValue_cum.loc[predict_dates[0]:].index[0]]*self.portfolioValue
        df_conf_interval = pd.DataFrame(data=np.percentile(df_simulations, q=conf_interval, axis=1).transpose(),
                                        columns=['lowerConf', 'upperConf'], index=df_simulations.index)
        df_simulations_predicted = pd.concat([df_simulations.mean(axis=1).to_frame(), df_conf_interval], axis=1)
        df_simulations_predicted.columns = ['mean'] + df_conf_interval.columns.to_list()
        if plot:
            df_porfolioValue_cum.plot()
            df_simulations_predicted['mean'].plot()
            plt.fill_between(df_simulations_predicted.index,
                             df_simulations_predicted['lowerConf'],
                             df_simulations_predicted['upperConf'],
                             facecolor='green', alpha=0.2, interpolate=True)
            plt.xticks(rotation=45)
            plt.show()
        return (df_porfolioValue_cum, df_simulations_predicted)

    def simulatePortfolio(self, start_date, end_date, plot=False):

        dates = self.df_data['close'].loc[start_date:end_date].index
        strategyPortfolioAllocation, strategyPortfolioDirection = self.getStrategyPortfolioAllocation()
        df_portfolioAllocation_inc = \
            (self.df_data['close_diff_pct'].loc[dates] - 1) *\
            [1 if self.portfolioAllocationDirection[symbol_i] == 'long' else -1 for symbol_i, _ in enumerate(self.symbols)] * \
            strategyPortfolioAllocation
        df_portfolioAllocation_cum = strategyPortfolioAllocation + np.cumsum(df_portfolioAllocation_inc)

        if plot:
            df_portfolioAllocation_cum.plot()
            plt.xticks(rotation=45)
            plt.show()
        return (df_portfolioAllocation_cum, df_portfolioAllocation_inc)

    def simulatePortfolioValue(self, start_date, end_date, plot=False):

        df_portfolioAllocation_cum, df_portfolioAllocation_inc = self.simulatePortfolio(start_date, end_date, plot=False)
        df_portfolioValue_inc = np.sum(df_portfolioAllocation_inc, axis=1)
        df_portfolioValue_cum = np.sum(df_portfolioAllocation_cum, axis=1)
        if plot:
            df_portfolioValue_cum.plot()
            plt.xticks(rotation=45)
            plt.show()
        return (df_portfolioValue_cum, df_portfolioValue_inc)


class SyntheticTimeSeriesGenerator:

    def __init__(self, df_data):

        self.df_data = df_data

    def generate(self, train_dates, predict_dates, num_timeseries=None, method='bootstrapping'):

        if method == 'bootstrapping':

            l_df_ts = self.generate_bootstrapping(train_dates, predict_dates, num_timeseries=num_timeseries)

        elif method == 'meloinvento':

            l_df_ts = self.generate_bootstrapping(train_dates, predict_dates, num_timeseries=num_timeseries)

        else:
            pass

        return l_df_ts

    def generate_bootstrapping(self, train_dates, predict_dates, num_timeseries):

        train_dates = self.df_data.loc[train_dates[0]:train_dates[1]].index
        predict_dates = self.df_data.loc[predict_dates[0]:predict_dates[1]].index
        l_df_ts = []
        for ts_i in range(num_timeseries):
            sample_dates = np.random.choice(train_dates, len(predict_dates), replace=True)
            df_ts_i = self.df_data.loc[sample_dates]
            df_ts_i.index = np.sort(sample_dates)
            a=0
            l_df_ts.append(df_ts_i)
        return l_df_ts

class MOoptimization:

    class ProblemDefinition:

        def __init__(self, simulator, start_date, end_date, num_montecarlo_runs=50):
            self.simulator = simulator
            self._start_date = start_date
            self._end_date = end_date
            self._bounds = (
            len(self.simulator.symbols) * [-simulator.equity], len(self.simulator.symbols) * [simulator.equity])
            self._num_montecarlo_runs = num_montecarlo_runs

        # Define objectives
        def fitness(self, x):
            portfolioAllocationDirection = ['long' if i == 1 else 'short' for i in np.sign(x)]
            x = x / np.sign(x)
            x = x / np.sum(x) * self.simulator.portfolioValue
            self.simulator.setPortfolioAllocation(portfolioAllocation=x,
                                                  portfolioAllocationDirection=portfolioAllocationDirection)

            df_real_cum, _ = self.simulator.simulatePortfolioValue(self._start_date, self._end_date)
            f1 = -(df_real_cum.iloc[-1] - df_real_cum.iloc[0])
            f2 = np.sum(np.absolute((df_real_cum - np.linspace(df_real_cum.iloc[0], df_real_cum.iloc[-1], len(df_real_cum.index)))**2))

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

    def __init__(
            self, simulator, start_date, end_date,
            generations=100, crossover=0.8, mutation=0.1,
            num_montecarlo_runs=50):


        # create optimization problem
        prob = pygmo.problem(self.ProblemDefinition(
            simulator,
            start_date=start_date,
            end_date=end_date,
            num_montecarlo_runs=num_montecarlo_runs))
        # create population
        self.pop = pygmo.population(prob, size=20 * 4)
        # select algorithm
        self.algo = pygmo.algorithm(pygmo.nsga2(gen=generations, cr=crossover, m=mutation))

    def optimize(self, verbose=0):

        self.algo.set_verbosity(verbose)
        # run optimization
        self.pop = self.algo.evolve(self.pop)

        f = np.vectorize(lambda x: 'long' if x == 1 else 'short')
        best_portfolioAllocationDirections = f(np.sign(self.pop.get_x()))
        pop_x = np.abs(self.pop.get_x())
        best_portfolioAllocations = pop_x / np.sum(pop_x, axis=1)[:, None] * simulator.portfolioValue
        best_fitnesses = self.pop.get_f()
        self.optimizationData = {
            'best_fitnesses': best_fitnesses,
            'best_portfolioAllocations': best_portfolioAllocations,
            'best_portfolioAllocationDirections': best_portfolioAllocationDirections}
        pickle.dump(self.optimizationData, open("../data/optimizationPopulation.dat", "wb"))
        return self.optimizationData

    def getBestResult(self, dim_importance=(1,1), percentil=.9, plot=False):

        self.optimizationData = pickle.load(open("../data/optimizationPopulation.dat", "rb"))
        # choose best_index
        #normalize: take into account that it goes the other way round
        best_fitnesses_norm = self.optimizationData['best_fitnesses'] - np.max(self.optimizationData['best_fitnesses'], axis=0)
        best_fitnesses_norm = best_fitnesses_norm/np.min(best_fitnesses_norm, axis=0)
        best_fitnesses_norm = best_fitnesses_norm*dim_importance/np.sum(dim_importance)
        best_fitnesses_norm_aux = np.sum(best_fitnesses_norm, axis=1)
        best_index = np.argsort(best_fitnesses_norm_aux)
        aux = int(np.floor(len(best_index)*(percentil)))
        if aux >= len(best_index):
            aux = aux - 1
        best_index = best_index[aux]
        best_portfolioAllocation = self.optimizationData['best_portfolioAllocations'][best_index]
        best_portfolioAllocationDirection = self.optimizationData['best_portfolioAllocationDirections'][best_index]
        best_portfolioAllocationFitness = self.optimizationData['best_fitnesses'][best_index]
        if plot:
            pygmo.plot_non_dominated_fronts(self.optimizationData['best_fitnesses'])
            plt.plot(
                best_portfolioAllocationFitness[0], best_portfolioAllocationFitness[1], 'ro',
                markersize=10)
            plt.show()
        return best_portfolioAllocation, best_portfolioAllocationDirection




simulator = Simulator(balance=5000, periodType='day', periods=365, rm_cache=False)
simulator.setPorfolio(5)
simulator.setPortfolioAllocation(balance_pct=0.2)

l_portfolioValue = [simulator.portfolioValue]
train_period = [date(2020,2,15), date(2020,2,28)]
predict_period = [date(2020,3,1), date(2020,3,31)]
delta = timedelta(days=1)
while train_period[1] < predict_period[1]:

    print('... Train period (%s / %s) --> Prediction (%s)' % (str(train_period[0]), str(train_period[1]), str(train_period[1] + delta)))
    train_period[0] = train_period[0] + delta
    train_period[1] = train_period[1] + delta
    try:
        simulator.df_data['close'].loc[train_period[1] + delta]
    except:
        continue
    # optimization
    optimization = MOoptimization(
        simulator, train_period[0],  train_period[1],
        generations=50, crossover=0.8, mutation=0.1,
        num_montecarlo_runs=50)
    optimization.optimize()
    best_portfolioAllocation, best_portfolioAllocationDirection = optimization.getBestResult(
        dim_importance=(0.2,0.8),
        percentil=.95,
        plot=False)
    # perform simulation
    simulator.setPortfolioAllocation(
        portfolioAllocation=best_portfolioAllocation,
        portfolioAllocationDirection=best_portfolioAllocationDirection)
    _, portfolioValue = simulator.simulatePortfolioValue(
        start_date=train_period[1]+delta, end_date=train_period[1]+delta)

    # # plot it
    # simulator.simulatePortfolioValue(
    #     start_date=train_period[0], end_date=predict_period[1], plot=True)

    l_portfolioValue.append(portfolioValue.iloc[0])
    pickle.dump(np.cumsum(l_portfolioValue), open("../data/simulation_strategy.dat", "wb"))


# perform simulation
simulator.montecarlo_simulate(
    train_dates=train_period,
    predict_dates=predict_period,
    num_montecarlo_runs=50,
    plot=True)


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
