import typing
import numpy as np
from matplotlib import pyplot as plt
import itertools
from typing import Literal
from scipy import stats
from bisect import bisect_right


def get_cdf(kde, u,*, x0 = None, y0 = None):
    if (x0 is None) == (y0 is None):
        raise ValueError("Only one of x0, y0 can be None")
    if x0 is not None:
        cdf = np.cumsum(kde.evaluate(np.vstack([np.ones(u.shape) * x0, u])))
    else:
        cdf = np.cumsum(kde.evaluate(np.vstack([u, np.ones(u.shape) * y0])))
    assert cdf[-1] > 0, f"{x0=}, {y0=}"
    cdf = cdf / cdf[-1]
    return cdf


def pack(val, min=0, max=1):
    if val < min:
        return min
    if val > max:
        return max

    return val


def get_quantile(cdf, u, q):
    q_pos = bisect_right(cdf, q)
    q_pos = pack(q_pos, min=0, max=u.shape[0]-1)
    return u[q_pos]


def plot_precision(y_act: np.ndarray, y_pred: np.ndarray, quantiles: list[float] = None,
                   plt_mode: Literal['raw', 'subtraction', 'division'] = 'raw',
                   plotter_mode: Literal['plot', 'scatter'] = 'plot') -> None:
    quantiles = [0.05, 0.5, 0.95] if quantiles is None else quantiles

    kde = stats.gaussian_kde(np.vstack([y_act, y_pred]))
    u = np.linspace(min(y_act), max(y_act), num=10_000, endpoint=True)

    if plotter_mode == 'scatter':
        pred_sp = y_pred
    else:
        pred_sp = np.linspace(min(y_pred), max(y_pred), num=250)

    quantiles_sp = np.zeros((len(quantiles), len(pred_sp)))
    for j, pred in enumerate(pred_sp):
        cdf = get_cdf(kde, u, y0=pred)
        for i, q in enumerate(quantiles):
            quantiles_sp[i, j] = get_quantile(cdf, u, q)

    modifier = {'raw':         (lambda x: x),
                'subtraction': (lambda x: x - pred_sp),
                'division':    (lambda x: x / pred_sp)}

    plotter = {'plot':    (lambda x, y: plt.plot(x, y)),
               'scatter': (lambda x, y: plt.scatter(x, y))}

    for i, q in enumerate(quantiles):
        plotter[plotter_mode](pred_sp, modifier[plt_mode](quantiles_sp[i, :]))

    plt.legend([f"{q}-quantile" for q in quantiles])


def plot_recall(y_act: np.ndarray, y_pred: np.ndarray, quantiles: list[float] = None,
                   plt_mode: Literal['raw', 'subtraction', 'division'] = 'raw',
                   plotter_mode: Literal['plot', 'scatter'] = 'plot') -> None:
    return plot_precision(y_pred, y_act, quantiles, plt_mode, plotter_mode)