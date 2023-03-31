import typing
import numpy as np
import scipy
from matplotlib import pyplot as plt
import timeit
import cProfile
import pstats


def make_samples(size: int, count: int):
    return [
        (np.linspace(0, 1, num=size), np.linspace(0, 1, num=size) + np.random.normal(0.0, 0.1, size))
        for _ in range(count)
    ]


def timeit_function(func: typing.Callable[[np.array, np.array], None], sample: typing.Tuple[np.array, np.array], rep_c: int) -> float:
    assert rep_c > 0

    t = timeit.timeit(lambda: func(*sample), number=rep_c)
    return t / rep_c


def bench_precision():
    from using_kde.precision_recall import plot_precision

    target = plot_precision
    size_sp = np.logspace(1, 3, num=3, endpoint=True, base=10, dtype=int)
    timeit_reps = 3
    loop_reps = 5

    for size in size_sp:
        samples = make_samples(size, loop_reps)
        results = np.array([
            timeit_function(target, sample, timeit_reps)
            for sample in samples
        ])
        print(f"{size=:4d}\tmu={np.mean(results):.3f}\tsigma={np.std(results):.3f}")


def profile_precision(size: int, o_path: str = None):
    from using_kde.precision_recall import plot_precision
    # target = plot_precision
    target = lambda y_act, y_pred: plot_precision(y_act, y_pred, quantiles=[0.003, 0.05, 0.32, 0.5, 0.68, 0.95, 0.997])
    sample, = make_samples(size, 1)

    with cProfile.Profile() as prof:
        target(*sample)

    st = pstats.Stats(prof)
    st.sort_stats(pstats.SortKey.TIME)
    if o_path is None:
        st.print_stats()
    else:
        st.dump_stats(o_path)


def main():
    # bench_precision()
    profile_precision(100, "stats_arg_delta_bad.prof")


if __name__ == '__main__':
    main()
