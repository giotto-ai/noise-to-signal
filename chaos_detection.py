# General import
import pandas as pd
import numpy as np
import os
import sys

# Giotto (use 'pip install giotto-learn' if you want to install it)
import gtda as gt
import gtda.time_series as ts
import gtda.diagrams as diag
import gtda.homology as hl


# Plotting
import matplotlib.pyplot as plt
from plotting import plot_diagram, plot_landscapes
from plotting import plot_betti_surfaces, plot_betti_curves
from plotting import plot_point_cloud

# Miscellaneous
from itertools import product
import plotly.express as px
from pandarallel import pandarallel # to parallelize pandas functions
from scipy.fftpack import rfft
from functools import reduce
import openml
from openml.datasets.functions import get_dataset
#######################################################################

# A helper function
def concat_dfs(dfs):
    return reduce(lambda x, y: pd.concat([x, y]), dfs)


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def create_pred_df(pred, y_test, threshold=0.5, rolling_param=300):
    """
    INPUT:
        pred: array
        y_test: list
        threshold: float
        rolling_param: int

    OUTPUT:
        pred_df: pandas DataFrame
    """

    pred_df = pd.DataFrame()
    pred_df['pred'] = pred
    pred_df['ref'] = y_test
    pred_df['rolling'] = pred_df['pred'].rolling(rolling_param).mean()
    pred_df['indicator'] = pred_df['rolling'] > threshold
    pred_df = pred_df.dropna()
    return pred_df


def plot_results(results, noise_level, x_tick_labels=None, return_figure=False, fig_size=(20, 15)):
    """
    INPUT:
        results: array
        noise_level: list
        x_tick_labels: list
        return_figure: boolean
        figsize: in cm, tuple
    """

    fig, ax = plt.subplots(figsize=(cm2inch(fig_size)))
    ax.plot(noise_level, results[:,0], '-', label='all features')
    ax.plot(noise_level, results[:,1], '-', label='TDA features only')
    ax.plot(noise_level, results[:,2], '-', label='no TDA features')
    # only features from the dynamical system itself
    ax.plot(noise_level, results[:,3], '-', label='intrinsic features only')
    ax.set_xlabel('Signal to Noise Ratio')
    ax.set_ylabel('Balanced Accuracy')
    if x_tick_labels is not None:
        ax.set_xticks(np.arange(0, len(x_tick_labels) / 10., 1 / 10))
        ax.set_xticklabels([str(x) for x in np.round(x_tick_labels, 1)])
    ax.legend()
    if return_figure == True:
        return fig


def plot_predictions(pred, y_test):
    """
    INPUT:
        pred: list
        y_test: list

    OUTPUT:
        None
    """

    pred_df = create_pred_df(pred, y_test)
    fig= plt.figure(figsize=(20, 10))
    plt.plot(y_test, c='blue', label='truth')
    plt.plot(pred_df['indicator'], c='red', alpha=0.5, label='prediction', linestyle='solid')
    plt.plot(pred_df['rolling'], c='green', alpha=0.5, label='rolling', linestyle='solid')
    plt.legend(loc='center left')


def convert_to_SNR(time_series, noise_level):
    """
    INPUT:
        time_series: list of time series
        noise_level: list of noise levels

    OUTPUT:
        SNR: list with noise levels converted to signal-to-noise ratio
    """

    return [(ts['x'] ** 2).mean() / (noise_level ** 2) for ts in time_series[0]][0]


def tda_diagrams(path,
                 embedding_time_delay,
                 embedding_dimension,
                 window_width,
                 window_stride,
                 homology_dim=2,
                 return_betti_surface=False):
    """
    INPUT:
        path: int (number to OpenML dataset)
        embedder_time_delay: int
        embedding_dimension: int
        window_width: int
        window_stride: int
        homology_dim: int
        return_betti_surface: boolean

    OUTPUT:
        X_scaled: persistence diagrams
        df_betti_list: List of Betti curve DataFrames
    """

    df = get_dataset(path)
    df = df.get_data()[0]
    df.rename({'label': 'y', 'coord_0': 'x'}, axis='columns', inplace=True)
    df['idx'] = np.arange(len(df))

    embedder = ts.TakensEmbedding(parameters_type='search', dimension=embedding_dimension,
                                  time_delay=embedding_time_delay, n_jobs=-1)
    embedder.fit(df['x'])
    embedder_time_delay = embedder.time_delay_
    embedder_dimension = embedder.dimension_

    print('Optimal embedding time delay based on mutual information: ', embedder_time_delay)
    print('Optimal embedding dimension based on false nearest neighbors: ', embedder_dimension)

    X_embedded, y_embedded = embedder.transform_resample(df['x'], df['y'])
    sliding_window = ts.SlidingWindow(width=window_width, stride=window_stride)
    sliding_window.fit(X_embedded, y_embedded)

    X_windows, y_windows = sliding_window.transform_resample(X_embedded, y_embedded)

    homology_dimensions = [0, 1, 2]
    persistenceDiagram = hl.VietorisRipsPersistence(metric='euclidean', max_edge_length=10,
                                                    homology_dimensions=homology_dimensions, n_jobs=-1)

    X_diagrams = persistenceDiagram.fit_transform(X_windows[:])
    diagram_scaler = diag.Scaler()
    diagram_scaler.fit(X_diagrams)
    X_scaled = diagram_scaler.transform(X_diagrams)

    persistent_entropy = diag.PersistenceEntropy()
    X_persistent_entropy = persistent_entropy.fit_transform(X_scaled)

    betti_curves = diag.BettiCurve()
    betti_curves.fit(X_scaled)
    X_betti_curves = betti_curves.transform(X_scaled)

    df_betti_list = []
    for i in homology_dimensions:
        df_betti_list.append(pd.DataFrame(X_betti_curves[:, i, :]))

    if return_betti_surface==True:
        return (X_scaled, df_betti_list, X_betti_curves)
    else:
        return (X_scaled, df_betti_list)


def num_relevant_holes(X_scaled, homology_dim, theta=0.7):
    """
    INPUT:
        X_scaled: scaled persistence diagrams, numpy array
        homology_dim: dimension of the homology to consider, integer
        theta: value between 0 and 1 to be used to calculate the threshold, float

    OUTPUT:
        n_rel_holes: list of the number of relevant holes in each time window
    """

    n_rel_holes = []

    for i in range(X_scaled.shape[0]):
        persistence_table = pd.DataFrame(X_scaled[i], columns=['birth', 'death', 'homology'])
        persistence_table['lifetime'] = persistence_table['death'] - persistence_table['birth']
        threshold = persistence_table[persistence_table['homology'] == homology_dim]['lifetime'].max() * theta
        n_rel_holes.append(persistence_table[(persistence_table['lifetime'] > threshold)
                                             & (persistence_table['homology'] == homology_dim)].shape[0])
    return n_rel_holes


def average_lifetime(X_scaled, homology_dim):
    """
    INPUT:
        X_scaled: scaled persistence diagrams, numpy array
        homology_dim: dimension of the homology to consider, integer

    OUTPUT:
        avg_lifetime_list: list of average lifetime for each time window
    """

    avg_lifetime_list = []

    for i in range(X_scaled.shape[0]):
        persistence_table = pd.DataFrame(X_scaled[i], columns=['birth', 'death', 'homology'])
        persistence_table['lifetime'] = persistence_table['death'] - persistence_table['birth']
        avg_lifetime_list.append(persistence_table[persistence_table['homology']
                                                   == homology_dim]['lifetime'].mean())

    return avg_lifetime_list


def betti_surface_feature(df_betti, betti_rolling=1):
    """
    INPUT:
        df_betti: pandas dataframe for the betti surface
        betti_rolling: rolling_parameter, integer

    OUTPUT:
        mean along the epsilon axis of the non-zero elements of the betti surface
    """

    return df_betti.groupby(df_betti.index).apply(lambda g: find_mean_nonzero(g)).rolling(betti_rolling).mean()


def betti_surface_argmax(df_betti):
    """
    INPUT:
        df_betti: pandas dataframe for the betti surface

    OUTPUT:
        argmax along the epsilon axis
    """

    return np.argmax(np.array(df_betti), axis=1)


def get_persistent_entropy(X_scaled, homology_dim=0):
    """
    INPUT:
        X_scaled: scaled persistence diagrams, numpy array
        homology_dim: dimension of the homology to consider, integer

    OUTPUT:
        persistent_entropy: array
    """

    persistent_entropy = diag.PersistenceEntropy()
    return persistent_entropy.fit_transform(X_scaled)


def calculate_amplitude_feature(X_scaled, metric='wasserstein', order=2):
    """
    INPUT:
        X_scaled: scaled persistence diagrams, numpy array
        metric: Either 'wasserstein' (default), 'landscape', 'betti', 'bottleneck' or 'heat'
        order: integer

    OUTPUT:
        amplitude: vector with the values for the amplitude feature
    """

    amplitude = diag.Amplitude(metric=metric, order=order)
    return amplitude.fit_transform(X_scaled)


def create_non_tda_features(path,
                            fourier_window_size=[],
                            rolling_mean_size=[],
                            rolling_max_size=[],
                            rolling_min_size=[],
                            mad_size=[],
                            fourier_coefficients=[]):
    """
    INPUT:
        path: int (number to OpenML dataset)
        fourier_window_size: a list of window sizes. Note: min must be > max(fourier_coefficients)
        rolling_mean_size: a list of window sizes
        rolling_max_shift: a list of window sizes
        rolling_min_shift: a list of window sizes
        mad_size: a list of window sizes
        fourier_coefficients: a list of all fourier coefficients to include.
                              Note: max must be < min(fourier_window_size)
    OUTPUT:
        df: pandas dataframe with columns:
            max_... for rolling max features
            min_... for rolling min features
            mean_... for rolling mean features
            mad_... for rolling mad features
            fourier_... for fourier coefficients
    """

    df = get_dataset(path)
    df = df.get_data()[0]
    df.rename({'label': 'y', 'coord_0': 'x', 'coord_1': 'x_dot'}, axis='columns', inplace=True)

    pandarallel.initialize()

    for r in rolling_max_size:
        df['max_' + str(r)] = df['x'].rolling(r).max()
    for r in rolling_mean_size:
        df['mean_' + str(r)] = df['x'].rolling(r).mean()
    for r in rolling_min_size:
        df['min_' + str(r)] = df['x'].rolling(r).min()
    for r in mad_size:
        df['mad_' + str(r)] = df['x'] - df['x'].rolling(r).min()
    if (not fourier_coefficients and fourier_window_size) or (not fourier_window_size and fourier_coefficients):
        print('Need to specify the fourier coeffcients and the window size')
    for n in fourier_coefficients:
        df[f'fourier_w_{n}'] = df['x'].rolling(fourier_window_size).parallel_apply(lambda x: rfft(x)[n],
                                                                                   raw=False)
    # Remove all rows with NaNs
    df.dropna(axis='rows', inplace=True)
    return df


#Helper function
def find_mean_nonzero(g):
    if g.to_numpy().nonzero()[1].any():
        return g.to_numpy().nonzero()[1].mean()
    else:
        return 0


def create_all_features(path, noise_level, return_betti_surface=False):
    """
    INPUT:
        path: int (number to OpenML dataset)
        noise_level: list with all noise levels
        return_betti_surface: boolean

    OUTPUT:
        df: all features in a dataframe OR
        df, X_betti_curves: df and Betti curves

    """

    df = create_non_tda_features(path=path,
                                 rolling_max_size=[10, 20, 50],
                                 rolling_min_size=[10, 20, 50],
                                 rolling_mean_size=[10, 20, 50],
                                 fourier_coefficients=[1,2],
                                 fourier_window_size=40)
    df['idx'] = np.arange(len(df))

    window_stride = 50
    diagrams = tda_diagrams(path=path,
                            embedding_dimension=14,
                            embedding_time_delay=5,
                            window_width=100,
                            window_stride=window_stride,
                            return_betti_surface=return_betti_surface)
    X_scaled = diagrams[0]
    df_betti_list = diagrams[1]
    if return_betti_surface == True:
        X_betti_curves = diagrams[2]

    num_holes_feature = num_relevant_holes(X_scaled, homology_dim=0, theta=0.7)
    avg_lifetime_feature = average_lifetime(X_scaled, homology_dim=0)
    betti_feature = []
    for dim in range(3):
        betti_feature.append(betti_surface_feature(df_betti_list[dim], betti_rolling=1))
    amplitude_feature = calculate_amplitude_feature(X_scaled, metric='wasserstein', order=2)

    length = len(np.array([[x] * window_stride for x in num_holes_feature]).flatten())
    df.drop(df[df['idx'] < (df.shape[0] - length)].index, axis='rows', inplace=True)
    df['num_holes'] = np.array([[x] * window_stride for x in num_holes_feature]).flatten()
    df['avg_lifetime'] = np.array([[x] * window_stride for x in avg_lifetime_feature]).flatten()
    for dim in range(3):
        df[f'betti_{dim}'] = np.array([[x] * window_stride for x in betti_surface_feature(df_betti_list[dim])]).flatten()
    for dim in [1,2]:
        df[f'betti_argmax_{dim}'] = np.array([[x] * window_stride for x in betti_surface_argmax(df_betti_list[dim])]).flatten()
    df['amplitude'] = np.array([[x] * window_stride for x in amplitude_feature]).flatten()

    df.drop('idx', axis = 'columns', inplace=True)
    if return_betti_surface == True:
        return df, X_betti_curves
    else:
        return df


if __name__ == '__main__':
    """
    This only works if the raw data of the Duffing system are available. 
    """
    main_dir = 'duffing_raw'
    noise_level = sys.argv[1:]
    noise_level = [x for x in noise_level]
    for n in noise_level:
        n_sets = 14
        fts_train = concat_dfs([create_all_features(os.path.join(main_dir,
                                                             f'dataset_{itr}',
                                                             f'duffing_{n}.pickle'), noise_level=n)
                                for itr in range(n_sets)])
        fts_train.to_pickle('train_'+str(n)+'.pickle')

        fts_test = concat_dfs([create_all_features(os.path.join(main_dir,
                                                             f'dataset_{itr}',
                                                             f'duffing_{n}.pickle'), noise_level=n)
                    for itr in range(n_sets, 20)])
        fts_test.to_pickle('test_'+str(n)+'.pickle')
