import numpy as np
from numpy import ndarray
from scipy import integrate
from matplotlib import pyplot as plt
from typing import Literal, Callable
from scipy import stats
from bisect import bisect_right
from fastkde import fastKDE


def _pack(val, min_v=0, max_v=1):
    return min(max(val, min_v), max_v)


def _calc_mean(pdf: ndarray, u: ndarray) -> float:
    return integrate.trapezoid(pdf * u, u)


def _get_cdf(pdf: np.array, arg_sp: np.ndarray):
    cdf = integrate.cumulative_trapezoid(
        y=pdf,
        x=arg_sp,
        initial=0.0
    )

    return cdf


def _get_quantile(cdf, u, q):
    q_pos = bisect_right(cdf, q)
    q_pos = _pack(q_pos, min_v=0, max_v=u.shape[0] - 1)
    return u[q_pos]


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
                   ax: plt.axes = None) -> None:

    quantiles = [0.05, 0.5, 0.95] if quantiles is None else quantiles
    ax = plt.axes() if ax is None else ax

    pOfYGivenX, axes = fastKDE.conditional(inputVars=y_act, conditioningVars=y_pred)  # pdf of (act | pred)
    pred_ax = axes[0][(pOfYGivenX.mask[0, :] == False) & (axes[0] >= min(y_pred)) & (axes[0] <= max(y_pred))]
    act_ax = axes[1]
    cond_pdf = pOfYGivenX[:, (pOfYGivenX.mask[0, :] == False) & (axes[0] >= min(y_pred)) & (axes[0] <= max(y_pred))].data

    quantiles_sp = np.zeros((len(quantiles), cond_pdf.shape[1]))
    mean_sp = np.zeros_like(pred_ax)

    for j in range(cond_pdf.shape[1]):
        cdf = _get_cdf(cond_pdf[:, j], act_ax)
        mean_sp[j] = _calc_mean(cond_pdf[:, j] / cdf[-1], act_ax)
        for i, q in enumerate(quantiles):
            quantiles_sp[i, j] = _get_quantile(cdf, act_ax, q)

    modifier = _get_modifier(plt_mode, pred_ax)
    plotter = _get_plotter(plotter_mode, ax)

    for i, q in enumerate(quantiles):
        plotter(pred_ax, modifier(quantiles_sp[i, :]), label=f"{q}-quantile")

    plotter(pred_ax, mean_sp, label="mean", c='r')

    ax.legend()


def plot_recall(y_act: np.ndarray, y_pred: np.ndarray, quantiles: list[float] = None,
                plt_mode: Literal['raw', 'subtraction', 'division'] = 'raw',
                plotter_mode: Literal['plot', 'scatter'] = 'plot',
                ax: plt.axes = None) -> None:
    return plot_precision(y_pred, y_act, quantiles, plt_mode, plotter_mode, ax)

