import timeit
import typing

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from using_kde.precision_recall import plot_precision as kde_precision, plot_recall as kde_recall
from using_fastkde.precision_recall import plot_precision as fast_precision, plot_recall as fast_recall
from naive_implementation.precision_recall import plot_precision as naive_precision, plot_recall as naive_recall


def sample(func: typing.Callable, y_pred: np.array, y_act: np.array, quantiles: list[float], rep_c: int = 10):
    variables = {"func": func, "y_pred": y_pred, "y_act": y_act, "quantiles": quantiles}
    results = np.zeros(rep_c)
    for i in range(rep_c):
        results[i] = timeit.timeit(lambda :func(y_pred, y_act, quantiles=quantiles), number=1)
    return np.mean(results), np.std(results)


def warm_up(func: typing.Callable):
    size = 25
    y_act = np.linspace(0, 1, num=size)
    y_pred = np.linspace(0, 1, num=size) + stats.norm.rvs(loc=0.0, scale=0.3, size=size)
    quantiles = [0.05, 0.5, 0.95]
    sample(func, y_pred, y_act, quantiles, rep_c=1)


def write_rec(size: int, mean: float, std: float, path: str):
    with open(path, "a") as f:
        f.write(f"{size}\t{mean * 1000:.2f}\t{std * 1000:.2f}\n")


def write_init(path: str, method: str, units: typing.Literal['s', 'ms', 'us', 'ns']):
    with open(path, "w") as f:
        f.write(f"{method}\n")
        f.write(f"size\tmean({units})\tstd({units})\n")


def main():
    rep_c = 10

    write_init(r"bench_res/fast_precision.txt", method="fast_precision", units="ms")
    write_init(r"bench_res/naive_precision.txt", method="naive_precision", units="ms")
    write_init(r"bench_res/kde_precision.txt", method="kde_precision", units="ms")

    warm_up(fast_precision)
    warm_up(kde_precision)
    warm_up(naive_precision)

    steps_c = 15
    for step, size in enumerate(map(int, np.linspace(25, 1_000, endpoint=True, num=steps_c))):
        y_act = np.linspace(0, 1, num=size)
        y_pred = np.linspace(0, 1, num=size) + stats.norm.rvs(loc=0.0, scale=0.3, size=size)
        print(f"----- {size=}\tstep {step+1}\\{steps_c} -----")
        mean, std = sample(fast_precision, y_pred, y_act, [0.05, 0.5, 0.95], rep_c=rep_c*10)
        print(f"\tFast_precision:   {mean * 1000:.2f} +/- {std * 1000:.2f} ms")
        write_rec(size, mean, std, r"bench_res/fast_precision.txt")
        mean, std = sample(naive_precision, y_pred, y_act, [0.05, 0.5, 0.95], rep_c=rep_c*10)
        print(f"\tNative_precision: {mean * 1000:.2f} +/- {std * 1000:.2f} ms")
        write_rec(size, mean, std, r"bench_res/naive_precision.txt")
        mean, std = sample(kde_precision, y_pred, y_act, [0.05, 0.5, 0.95], rep_c=rep_c)
        print(f"\tKDE_precision:    {mean * 1000:.2f} +/- {std * 1000:.2f} ms")
        write_rec(size, mean, std, r"bench_res/kde_precision.txt")


if __name__ == '__main__':
    main()
