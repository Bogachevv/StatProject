__all__ = ['plot_precision', 'plot_recall']

import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
from typing import Literal, Callable, Tuple
from scipy import stats, integrate
from bisect import bisect_right


def _pack(val, min_v=0, max_v=1):
    return min(max(val, min_v), max_v)


def _get_quantile(cdf: ndarray, u: ndarray, q: float):
    q_pos = bisect_right(cdf, q)
    q_pos = _pack(q_pos, min_v=0, max_v=u.shape[0] - 1)
    return u[q_pos]


def _prod(a: ndarray, b: ndarray) -> ndarray:
    r"""
    Returned array ordered by a
    :return: :math:`a \times b = [(x, \, y) \: | \: (x \in a) \wedge (y \in b)\]`
    """
    return np.vstack([np.repeat(a, len(b)), np.tile(b, len(a))])


def _calc_mean(pdf: ndarray, u: ndarray) -> float:
    return integrate.trapezoid(pdf * u, u)


def _sort_pair(x: ndarray, y: ndarray) -> None:
    """
    Sort arrays x, y by x-order
    :param x: 1D array, MUTABLE
    :param y: 1D array, MUTABLE
    :return:
    """
    ast = np.argsort(x)
    x[:] = x[ast]
    y[:] = y[ast]


def _get_plotter(plotter_mode: Literal['plot', 'scatter'], ax) -> Callable:
    if plotter_mode == 'plot':
        return ax.plot
    if plotter_mode == 'scatter':
        return ax.scatter
    raise NotImplemented


def _get_modifier(mode: Literal['raw', 'subtraction', 'division'], param: ndarray | None) -> Callable:
    if mode == 'raw':
        return lambda x: x
    if mode == 'subtraction':
        return lambda x: x - param
    if mode == 'division':
        assert np.all(param != 0.0)
        return lambda x: x / param
    raise NotImplemented


def plot_precision(y_act: np.ndarray, y_pred: np.ndarray, quantiles: list[float] = None,
                   plt_mode: Literal['raw', 'subtraction', 'division'] = 'raw',
                   plotter_mode: Literal['plot', 'scatter'] = 'plot',
                   ax: plt.axes = None,
                   quality: int = 1_000,
                   resolution: int = 100) -> None:
    quantiles = [0.05, 0.5, 0.95] if quantiles is None else quantiles
    ax = plt.axes() if ax is None else ax

    pred_sp = np.linspace(np.min(y_pred), np.max(y_pred), num=resolution)
    quantiles_sp = np.zeros((len(quantiles), pred_sp.shape[0]))
    mean_sp = np.zeros_like(pred_sp)

    points = _prod(pred_sp, np.linspace(np.min(y_act), np.max(y_act), num=quality))

    kde_evs = (stats.
               gaussian_kde(np.vstack([y_act, y_pred])).
               evaluate(points).
               reshape((-1, quality)))

    points = (points.
              transpose().
              reshape((resolution, quality, -1))
              [:, :, 1].
              copy())

    for j, pred in enumerate(pred_sp):
        u = points[j, :]
        cdf = integrate.cumulative_trapezoid(kde_evs[j, :], u, initial=0.0)
        mean_sp[j] = _calc_mean(kde_evs[j, :] / cdf[-1], u)
        cdf /= cdf[-1]

        for i, q in enumerate(quantiles):
            quantiles_sp[i, j] = _get_quantile(cdf, u, q)

    modifier = _get_modifier(plt_mode, pred_sp)
    plotter = _get_plotter(plotter_mode, ax)

    for i, q in enumerate(quantiles):
        plotter(pred_sp, modifier(quantiles_sp[i, :]), label=f"{q}-quantile")

    plotter(pred_sp, mean_sp, label="mean", c='r')

    ax.legend()


def plot_recall(y_act: np.array, y_pred: np.array, quantiles: list[float] = None,
                plt_mode: Literal['raw', 'subtraction', 'division'] = 'raw',
                plotter_mode: Literal['plot', 'scatter'] = 'plot',
                ax: plt.axes = None) -> None:
    return plot_precision(y_pred, y_act, quantiles, plt_mode, plotter_mode, ax)
