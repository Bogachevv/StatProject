import numpy as np
from matplotlib import pyplot as plt
from typing import Literal, Callable, Tuple
from scipy import stats, integrate
from bisect import bisect_right


def extend(arr: np.array, gen: Callable[[float, float, int], float]) -> np.array:
    new_arr = np.zeros(arr.shape[0] * 2 - 1)
    for i in range(arr.shape[0] - 1):
        new_arr[2*i] = arr[i]
        new_arr[2*i + 1] = gen(arr[i], arr[i+1], i)
    new_arr[-1] = arr[-1]

    assert new_arr.shape[0] == 2*arr.shape[0] - 1
    return new_arr


def get_cdf(pdf, left: float, right: float, n: int = 7, int_eps: float = 0.01, arg_delta: float = 0.01) -> Tuple[np.array, np.array]:
    arg_sp_gen = lambda a_prev, a_next, i: (a_prev + a_next) / 2

    old_arg_sp = np.linspace(left, right, num=n)
    new_arg_sp = extend(old_arg_sp, arg_sp_gen)
    old_val_sp = np.array([pdf(x) for x in old_arg_sp]).flatten()
    # old_val_sp = pdf(old_arg_sp)
    new_val_sp = extend(old_val_sp, lambda a_prev, a_next, i: pdf(new_arg_sp[2 * i + 1]))
    old_int = integrate.cumulative_trapezoid(old_val_sp, old_arg_sp, initial=0)
    new_int = integrate.cumulative_trapezoid(new_val_sp, new_arg_sp, initial=0)

    while True:
        delta_int = max(abs(new_int[2*i] - old_int[i]) for i in range(n))
        net_resolution = (new_arg_sp[-1] - new_arg_sp[0]) / len(new_arg_sp)
        arg_eps = arg_delta / max(new_val_sp)
        if (delta_int < int_eps) and (net_resolution < arg_eps):
            break
        n = 2 * n - 1
        old_arg_sp = new_arg_sp.copy()
        # new_arg_sp = extend(old_arg_sp, arg_sp_gen)
        new_arg_sp = np.linspace(left, right, num=2*n - 1)
        old_val_sp = new_val_sp.copy()
        new_val_sp = extend(old_val_sp, lambda a_prev, a_next, i: pdf(new_arg_sp[2 * i + 1]))
        old_int = new_int.copy()
        new_int = integrate.cumulative_trapezoid(new_val_sp, new_arg_sp, initial=0)

    # print(n)
    assert new_int[-1] > 0
    cdf = new_int / new_int[-1]

    return new_arg_sp, cdf


def pack(val, min=0, max=1):
    if val < min:
        return min
    if val > max:
        return max

    return val


def get_quantile(cdf, u, q):
    q_pos = bisect_right(cdf, q)
    q_pos = pack(q_pos, min=0, max=u.shape[0] - 1)
    return u[q_pos]


def plot_precision(y_act: np.array, y_pred: np.array, quantiles: list[float] = None,
                   plt_mode: Literal['raw', 'subtraction', 'division'] = 'raw',
                   plotter_mode: Literal['plot', 'scatter'] = 'plot',
                   ax: plt.axes = None) -> None:
    quantiles = [0.05, 0.5, 0.95] if quantiles is None else quantiles
    ax = plt.axes() if ax is None else ax

    if plotter_mode == 'plot':
        ast = np.argsort(y_pred)
        y_act = y_act[ast]
        y_pred = y_pred[ast]

    kde = stats.gaussian_kde(np.vstack([y_act, y_pred]))
    quantiles_sp = np.zeros((len(quantiles), len(y_pred)))

    left_bound = min(y_act)
    right_bound = max(y_act)
    for j, pred in enumerate(y_pred):
        u, cdf = get_cdf(lambda x: kde((x, pred))[0], left_bound, right_bound,
                         arg_delta=min(min(quantiles), 1 - max(quantiles)), int_eps=1e-3)
        for i, q in enumerate(quantiles):
            quantiles_sp[i, j] = get_quantile(cdf, u, q)

    modifier = {'raw': (lambda x: x),
                'subtraction': (lambda x: x - y_pred),
                'division': (lambda x: x / y_pred)}

    plotter = {'plot':    (lambda x, y: ax.plot(x, y)),
               'scatter': (lambda x, y: ax.scatter(x, y))}

    for i, q in enumerate(quantiles):
        plotter[plotter_mode](y_pred, modifier[plt_mode](quantiles_sp[i, :]))

    ax.legend([f"{q}-quantile" for q in quantiles])


def plot_recall(y_act: np.array, y_pred: np.array, quantiles: list[float] = None,
                plt_mode: Literal['raw', 'subtraction', 'division'] = 'raw',
                plotter_mode: Literal['plot', 'scatter'] = 'plot',
                ax: plt.axes = None) -> None:
    return plot_precision(y_pred, y_act, quantiles, plt_mode, plotter_mode, ax)

