import numpy as np
from matplotlib import pyplot as plt
import itertools


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


def plot_precision(y_pred: np.ndarray, y_act: np.ndarray, quantiles: list[float] = None, plt_mode: str = 'raw',
                   plotter_mode: str = 'plot', delta: float = 0.05, delta_mode='absolute') -> None:
    """
    :param y_pred: sample of predicted values
    :param y_act: sample of actual values
    :param quantiles: quantiles list. Default: [0.05, 0.25, 0.50, 0.75, 0.95]
    :param plt_mode: plt_mode must be in ['raw', 'subtraction', 'division']
    :param plotter_mode: plotter_mode must be in ['plot', 'scatter']
    :param delta: slice radius
    :param delta_mode: 'absolute' or 'relative'
    """
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95] if quantiles is None else quantiles
    arg_sp = y_pred
    sample = np.vstack((y_act, y_pred))
    val_sp = np.array([np.quantile(get_slice(arg, sample, 0, delta, delta_mode), quantiles)
                       for arg in arg_sp])

    modes_list = ['raw', 'subtraction', 'division']
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

    plotter_modes_list = ['plot', 'scatter']
    plotter_mode = plotter_mode.lower()
    if plotter_mode not in plotter_modes_list:
        raise ValueError(
            f"Incorrect plotter_mode value(plotter_mode={plotter_mode}):\n"
            f"Available plotter modes: {', '.join(plotter_modes_list)}")

    if plotter_mode == 'plot':
        color_map = 'rgbk'
        line_styles = ['-', '--', '-.', ':']
        styles = (itertools.product(line_styles, color_map))
        colors = itertools.cycle(''.join(reversed(style)) for style in styles)
        for i, (q, c) in enumerate(zip(quantiles, colors)):
            plot_arr = np.vstack((arg_sp, val_sp[:, i]))
            plot_arr = plot_arr[:, np.argsort(plot_arr[0, :])]
            plt.plot(plot_arr[0, :], plot_arr[1, :], c)
    else:
        for i, q in enumerate(quantiles):
            plt.scatter(arg_sp, val_sp[:, i])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.legend(quantiles)


def plot_recall(y_pred: np.ndarray, y_act: np.ndarray, quantiles: list[float] = None, plt_mode: str = 'raw',
                plotter_mode: str = 'plot', delta: float = 0.05, delta_mode='absolute') -> None:
    """
    :param y_pred: sample of predicted values
    :param y_act: sample of actual values
    :param quantiles: quantiles list. Default: [0.05, 0.25, 0.50, 0.75, 0.95]
    :param plt_mode: plt_mode must be in ['raw', 'subtraction', 'division']
    :param plotter_mode: plotter_mode must be in ['plot', 'scatter']
    :param delta: slice radius
    :param delta_mode: 'absolute' or 'relative'
    """
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95] if quantiles is None else quantiles
    arg_sp = y_act
    sample = np.vstack((y_act, y_pred))
    val_sp = np.vstack(tuple(np.quantile(get_slice(arg, sample, 1, delta, delta_mode), quantiles)
                             for arg in arg_sp))

    modes_list = ['raw', 'subtraction', 'division']
    plt_mode = plt_mode.lower()
    if plt_mode not in modes_list:
        raise ValueError(
            f"Incorrect plt_mode value(plt_mode={plt_mode}):\nAvailable plot modes: {', '.join(modes_list)}")
    if plt_mode == 'subtraction':
        for i in range(len(quantiles)):
            val_sp[:, i] -= y_pred
    elif plt_mode == 'division':
        for i in range(len(quantiles)):
            val_sp[:, i] /= y_pred

    plotter_modes_list = ['plot', 'scatter']
    plotter_mode = plotter_mode.lower()
    if plotter_mode not in plotter_modes_list:
        raise ValueError(
            f"Incorrect plotter_mode value(plotter_mode={plotter_mode}):\n"
            f"Available plotter modes: {', '.join(plotter_modes_list)}")

    if plotter_mode == 'plot':
        color_map = 'rgbk'
        line_styles = ['-', '--', '-.', ':']
        styles = (itertools.product(line_styles, color_map))
        colors = itertools.cycle(''.join(reversed(style)) for style in styles)
        for i, (q, c) in enumerate(zip(quantiles, colors)):
            plot_arr = np.vstack((arg_sp, val_sp[:, i]))
            plot_arr = plot_arr[:, np.argsort(plot_arr[0, :])]
            plt.plot(plot_arr[0, :], plot_arr[1, :], c)
    else:
        for i, q in enumerate(quantiles):
            plt.scatter(arg_sp, val_sp[:, i])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend(quantiles)

