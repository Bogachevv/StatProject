import numpy as np
from matplotlib import pyplot as plt
import typing
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
                   dot_c: int = 1_000, delta: float = 0.05, delta_mode='absolute') -> None:
    """
    :param y_pred: sample of predicted values
    :param y_act: sample of actual values
    :param quantiles: quantiles list. Default: [0.05, 0.25, 0.50, 0.75, 0.95]
    :param plt_mode: plt_mode must be in ['raw', 'subtraction', 'division']
    :param dot_c:
    :param delta: slice radius
    :param delta_mode: 'absolute' or 'relative'
    """
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95] if quantiles is None else quantiles
    arg_sp = np.linspace(np.min(y_act), np.max(y_act), num=dot_c, endpoint=True)
    sample = np.vstack((y_pred, y_act))
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

    color_map = 'rgbk'
    line_styles = ['-', '--', '-.', ':']
    styles = (itertools.product(line_styles, color_map))
    colors = itertools.cycle(''.join(reversed(style)) for style in styles)
    for i, (q, c) in enumerate(zip(quantiles, colors)):
        plt.plot(arg_sp, val_sp[:, i], c)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.legend(quantiles)


def plot_recall(y_pred: np.ndarray, y_act: np.ndarray, quantiles: list[float] = None, plt_mode: str = 'raw',
                dot_c: int = 1_000, delta: float = 0.05, delta_mode='absolute') -> None:
    """
    :param y_pred: sample of predicted values
    :param y_act: sample of actual values
    :param quantiles: quantiles list. Default: [0.05, 0.25, 0.50, 0.75, 0.95]
    :param plt_mode: plt_mode must be in ['raw', 'subtraction', 'division']
    :param dot_c:
    :param delta: slice radius
    :param delta_mode: 'absolute' or 'relative'
    """
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95] if quantiles is None else quantiles
    arg_sp = np.linspace(np.min(y_pred), np.max(y_pred), num=dot_c, endpoint=True)
    sample = np.vstack((y_pred, y_act))
    val_sp = np.vstack(tuple(np.quantile(get_slice(arg, sample, 1, delta, delta_mode), quantiles)
                             for arg in arg_sp))

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

    color_map = 'rgbk'
    line_styles = ['-', '--', '-.', ':']
    styles = (itertools.product(line_styles, color_map))
    colors = itertools.cycle(''.join(reversed(style)) for style in styles)
    for i, (q, c) in enumerate(zip(quantiles, colors)):
        plt.plot(arg_sp, val_sp[:, i], c)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend(quantiles)


def test():
    x = np.linspace(0, 2, 1000)
    y_a = x ** 2 + 3
    y_p = y_a * 1.5

    plot_precision(y_p, y_a, delta=0.05, plt_mode='raw')
    plt.show()
    plot_recall(y_p, y_a, delta=0.05, plt_mode='raw')
    plt.show()


if __name__ == '__main__':
    test()
