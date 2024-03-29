import numpy as np
from matplotlib import pyplot as plt
import itertools

modes_list = ['raw', 'subtraction', 'division']
plotter_modes_list = ['plot', 'scatter']


def get_slice(center: float, sample: np.ndarray, res_dim: int, delta: float = 0.05, mode='absolute') -> np.ndarray:
    """
    :param center: slice center
    :param sample: [y, y^] sample
    :param res_dim: res_dim in {0, 1}
    :param delta: slice radius
    :param mode: 'absolute' or 'relative'
    :return:
        if res_dim == 0: function returns y: y^ in (center-delta, center+delta);
                   else: function returns y^: y in (center-delta, center+delta)
    """
    available_modes = ['absolute', 'relative']
    mode = mode.lower()
    if mode not in available_modes:
        raise ValueError(f"Incorrect mode value(mode={mode}):\nAvailable modes: {', '.join(available_modes)}")
    if mode == 'relative':
        if (delta > 1) or (delta < 0):
            raise ValueError(f"Incorrect delta value(mode={mode}) in relative mode:\nDelta must be in [0, 1]")
        delta *= delta
    if (len(sample.shape) != 2) or (sample.shape[0] != 2):
        raise ValueError(f"sample must be in R^(2xn)")

    if res_dim not in [0, 1]:
        raise ValueError(f"dim must be in {{0, 1}}")

    filter_dim = 1 - res_dim
    slc = sample[res_dim, :][(sample[filter_dim, :] < center + delta) & (sample[filter_dim, :] >= center - delta)]
    return slc if len(slc) else np.zeros(1)


def plot(arg_sp: np.ndarray, val_sp: np.ndarray, quantiles: list[float], plotter_mode: str, ax: plt.axes):
    if plotter_mode == 'plot':
        color_map = 'rgbk'
        line_styles = ['-', '--', '-.', ':']
        styles = (itertools.product(line_styles, color_map))
        colors = itertools.cycle(''.join(reversed(style)) for style in styles)
        for i, (q, c) in enumerate(zip(quantiles, colors)):
            plot_arr = np.vstack((arg_sp, val_sp[:, i]))
            plot_arr = plot_arr[:, np.argsort(plot_arr[0, :])]
            ax.plot(plot_arr[0, :], plot_arr[1, :], c)
    else:
        for i, q in enumerate(quantiles):
            ax.scatter(arg_sp, val_sp[:, i])
    # ax.xlabel("Predicted")
    # ax.ylabel("Actual")
    ax.legend(quantiles)


def plot_precision(y_act: np.ndarray, y_pred: np.ndarray, quantiles: list[float] = None, plt_mode: str = 'raw',
                   plotter_mode: str = 'plot', delta: float = 0.05, delta_mode='absolute',
                   ax: plt.axes = None) -> None:
    """
    :param y_act: sample of actual values
    :param y_pred: sample of predicted values
    :param quantiles: quantiles list. Default: [0.05, 0.25, 0.50, 0.75, 0.95]
    :param plt_mode: plt_mode must be in ['raw', 'subtraction', 'division']
    :param plotter_mode: plotter_mode must be in ['plot', 'scatter']
    :param delta: slice radius
    :param delta_mode: 'absolute' or 'relative'
    :param ax: pyplot axis
    """
    ax = plt.axes() if ax is None else ax
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95] if quantiles is None else quantiles
    arg_sp = y_pred
    sample = np.vstack((y_act, y_pred))
    val_sp = np.array([np.quantile(get_slice(arg, sample, 0, delta, delta_mode), quantiles)
                       for arg in arg_sp])

    plt_mode = plt_mode.lower()
    if plt_mode not in modes_list:
        raise ValueError(
            f"Incorrect plt_mode value(plt_mode={plt_mode}):\nAvailable plot modes: {', '.join(modes_list)}")
    if plt_mode == 'subtraction':
        for i in range(len(quantiles)):
            val_sp[:, i] -= y_act
    elif plt_mode == 'division':
        for i in range(len(quantiles)):
            val_sp[:, i] /= y_act

    plotter_mode = plotter_mode.lower()
    if plotter_mode not in plotter_modes_list:
        raise ValueError(
            f"Incorrect plotter_mode value(plotter_mode={plotter_mode}):\n"
            f"Available plotter modes: {', '.join(plotter_modes_list)}")

    plot(arg_sp, val_sp, quantiles, plotter_mode, ax)


def plot_recall(y_act: np.ndarray, y_pred: np.ndarray, quantiles: list[float] = None, plt_mode: str = 'raw',
                plotter_mode: str = 'plot', delta: float = 0.05, delta_mode='absolute',
                ax: plt.axes = None) -> None:
    """
    :param y_act: sample of actual values
    :param y_pred: sample of predicted values
    :param quantiles: quantiles list. Default: [0.05, 0.25, 0.50, 0.75, 0.95]
    :param plt_mode: plt_mode must be in ['raw', 'subtraction', 'division']
    :param plotter_mode: plotter_mode must be in ['plot', 'scatter']
    :param delta: slice radius
    :param delta_mode: 'absolute' or 'relative'
    :param ax: pyplot axis
    """
    return plot_precision(y_pred, y_act, quantiles, plt_mode, plotter_mode, delta, delta_mode, ax)
