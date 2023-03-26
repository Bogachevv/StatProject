import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from typing import Literal
from scipy import stats
from bisect import bisect_right
from fastkde import fastKDE


def pack(val, min=0, max=1):
    if val < min:
        return min
    if val > max:
        return max

    return val


def get_cdf(pdf: np.array, arg_sp: np.ndarray):
    cdf = np.zeros(pdf.shape)
    for i in range(pdf.shape[0]):
        cdf[i] = integrate.trapz(pdf[0:i], arg_sp[0:i])

    return cdf


def get_quantile(cdf, u, q):
    q_pos = bisect_right(cdf, q)
    q_pos = pack(q_pos, min=0, max=u.shape[0] - 1)
    return u[q_pos]


def plot_precision(y_act: np.ndarray, y_pred: np.ndarray, quantiles: list[float] = None,
                   plt_mode: Literal['raw', 'subtraction', 'division'] = 'raw',
                   plotter_mode: Literal['plot', 'scatter'] = 'plot',
                   ax: plt.axes = None) -> None:
    quantiles = [0.05, 0.5, 0.95] if quantiles is None else quantiles
    ax = plt.axes() if ax is None else ax

    pOfYGivenX, axes = fastKDE.conditional(inputVars=y_act, conditioningVars=y_pred)  # pdf of (act | pred)
    pred_ax = axes[0][pOfYGivenX.mask[0, :] == False]
    act_ax = axes[1]
    cond_pdf = pOfYGivenX[:, pOfYGivenX.mask[0, :] == False].data

    quantiles_sp = np.zeros((len(quantiles), cond_pdf.shape[1]))
    for j in range(cond_pdf.shape[1]):
        cdf = get_cdf(cond_pdf[:, j], act_ax)
        for i, q in enumerate(quantiles):
            quantiles_sp[i, j] = get_quantile(cdf, act_ax, q)


    modifier = {'raw': (lambda x: x),
                'subtraction': (lambda x: x - pred_ax),
                'division': (lambda x: x / pred_ax)}

    plotter = {'plot':    (lambda x, y: ax.plot(x, y)),
               'scatter': (lambda x, y: ax.scatter(x, y))}

    for i, q in enumerate(quantiles):
        plotter[plotter_mode](pred_ax, modifier[plt_mode](quantiles_sp[i, :]))

    ax.legend([f"{q}-quantile" for q in quantiles])


def plot_recall(y_act: np.ndarray, y_pred: np.ndarray, quantiles: list[float] = None,
                plt_mode: Literal['raw', 'subtraction', 'division'] = 'raw',
                plotter_mode: Literal['plot', 'scatter'] = 'plot',
                ax: plt.axes = None) -> None:
    return plot_precision(y_pred, y_act, quantiles, plt_mode, plotter_mode, ax)

