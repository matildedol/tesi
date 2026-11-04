"""
Experiments of tex/final/tesi.pdf, Sections 2 to 4.

Target function: sine with num_crossings crossings on [0, 10].
Model: DecisionTreeClassifier with depths 1, ..., floor(4 log2 n).
Noise level: percentage of flipped labels. For bernoulli noise p = 1 - level/100,
for gaussian noise sigma is such that the average flip probability
Phi(-|f(x)|/sigma) over the support equals level/100.

how to use:
    python experiments.py --curves gap bound_union bound_radmass --noise_level 5 20 40
"""

import argparse
import math
from typing import Literal, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp
from scipy.stats import binom, norm
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed

Curve = Literal["gap", "test_err", "train_err", "bound_union", "bound_radmass", "bound_noise",
                "train_bound_union", "train_bound_radmass", "train_bound_noise"]
CURVES = ["gap", "test_err", "train_err", "bound_union", "bound_radmass", "bound_noise",
          "train_bound_union", "train_bound_radmass", "train_bound_noise"]
BOUNDS = ["bound_union", "bound_radmass", "bound_noise"]
# curves computed from the T training sets: drawn with a ± 1 std band (dots only if no bound)
EMPIRICAL = ["gap", "test_err", "train_err", "train_bound_union", "train_bound_radmass", "train_bound_noise"]
CURVE_NAMES = {
    "gap": "Gap",
    "test_err": "Test Error",
    "train_err": "Train Error",
    "bound_union": "Union Bound",
    "bound_radmass": "Rademacher Bound",
    "bound_noise": "Noise Bound",
    "train_bound_union": "Train Error + Union Bound",
    "train_bound_radmass": "Train Error + Rademacher Bound",
    "train_bound_noise": "Train Error + Noise Bound",
}
# symbol of each curve in the legend
CURVE_SYMBOLS = {
    "gap": r"$\mathcal{G}(d)$",
    "test_err": r"$\mathcal{T}(d)$",
    "train_err": r"$\mathcal{L}(d)$",
    "bound_union": r"$\mathcal{B}_{Union}$",
    "bound_radmass": r"$\mathcal{B}_{RadMass}$",
    "bound_noise": r"$\mathcal{B}_{Noise}$",
    "train_bound_union": r"$\mathcal{L}(d) + \mathcal{B}_{Union}$",
    "train_bound_radmass": r"$\mathcal{L}(d) + \mathcal{B}_{RadMass}$",
    "train_bound_noise": r"$\mathcal{L}(d) + \mathcal{B}_{Noise}$",
}
# colormap of each curve, one shade per noise level; train error + bound has the colormap of the bound
CMAPS = {
    "gap": "Blues",
    "test_err": "Purples",
    "train_err": "Reds",
    "bound_union": "Greens",
    # oranges ending in a bright red, to stand apart from the pinks of the noise bound
    "bound_radmass": LinearSegmentedColormap.from_list(
        "radmass", [(0, "#fff5eb"), (0.45, "#fdae6b"), (0.7, "#f16913"), (0.95, "#e8000b"), (1, "#b30000")]),
    "bound_noise": "RdPu",
}

L, U = 0.0, 10.0
# grid used to approximate averages over the uniform support of N points
MAX_GRID = 1_000_000


# DATA

def sine(x, m, u=U):
    # m crossings in the interval [0, u]
    c = (np.pi * (m + 1)) / u
    return 5 * np.sin(c * x)

def sample_x(n, N, rng, l=L, u=U):
    """n points uniform on the N evenly spaced points of (l, u), same support as base.ipynb"""
    i = rng.integers(1, N + 1, size=n)
    return l + (u - l) * i / (N + 1)

def sample_y(x, f, noise_type, rng, sigma=None, p=None):
    if noise_type == "gaussian":
        e = sigma * rng.standard_normal(size=len(x))
        return np.where(f(x) + e >= 0, 1, -1)
    b = rng.choice(np.array([1, -1]), size=len(x), p=[p, 1 - p])
    return np.where(f(x) >= 0, 1, -1) * b

def support_grid(N, l=L, u=U):
    return np.linspace(l, u, min(N, MAX_GRID) + 2)[1:-1]

def flip_prob(x, f, noise_type, sigma=None, p=None):
    """P(y != sign(f(x)) | x)"""
    if noise_type == "bernoulli":
        return np.full(len(x), 1 - p)
    if sigma == 0:
        return np.zeros(len(x))
    return norm.cdf(-np.abs(f(x)) / sigma)

def sigma_for_level(f, level, N):
    """sigma such that gaussian noise flips on average a fraction level/100 of the labels"""
    if level == 0:
        return 0.0
    x = support_grid(N)
    return brentq(lambda s: norm.cdf(-np.abs(f(x)) / s).mean() - level / 100, 1e-6, 1e6)


# BOUNDS

def log_card_H(depth, N):
    """log |H_d| = log(2 sum_{j=0}^{min(2^d-1, N-1)} binom(N-1, j)), computed in floats"""
    k = min(2 ** int(depth) - 1, N - 1)
    if k == N - 1:
        return N * math.log(2)
    if k >= (N - 1) / 2 - 10 * math.sqrt(N):
        # sum = 2^(N-1) P(Bin(N-1, 1/2) <= k), not in the tail here
        return N * math.log(2) + binom.logcdf(k, N - 1, 0.5)
    # terms grow with j up to k, so the last ones dominate the sum
    j = np.arange(max(0, k + 1 - 1_000_000), k + 1)
    log_terms = gammaln(N) - gammaln(j + 1) - gammaln(N - j)
    return math.log(2) + logsumexp(log_terms)

def union_bound(depth, n, N, C=1):
    return C * 2 * np.sqrt(log_card_H(depth, N) / (2 * n))

def radem_massart_bound(depth, n, N, C=1):
    return C * min(2 * np.sqrt(2 * log_card_H(depth, N) / n), 2 * np.sqrt(2 * np.log(2)))

def noise_approxerr_bound(depth, n, R_star_d, N, C=1):
    L_ = (math.log(4) + log_card_H(depth, N) + 1) / n
    term_1 = 4 * np.sqrt(R_star_d * L_) + 20 * L_
    M = ((2 * n + 2) * np.log(2) + 1) / n
    term_2 = 4 * np.sqrt(R_star_d * M) + 20 * M
    return C * min(term_1, term_2)

def dp_R_star_d(f, depths, N, noise_type, sigma=None, p=None):
    """
    inf_{h in H_d} R(h) for every d in depths, as dp_R_star_d in base.ipynb.
    the flip probability q(x) can depend on x (gaussian noise): since q < 1/2,
    an optimal tree still splits only at the sign changes of f, so the dp over
    the contiguous blocks of sign(f) stays exact, with block costs sum of q
    (predict the true sign) and sum of 1 - q (predict the other one).
    """
    x = support_grid(N)
    G = len(x)
    true_labels = np.where(f(x) >= 0, 1, -1)
    q = flip_prob(x, f, noise_type, sigma=sigma, p=p)

    change_points = np.where(np.diff(true_labels) != 0)[0] + 1
    starts = np.r_[0, change_points]
    ends = np.r_[change_points, G]
    run_labels = true_labels[starts]
    M = len(run_labels)
    max_K = min(2 ** int(np.max(depths)), M)

    # cost[i, j]: risk on block i predicting labels[j]
    labels = np.array([-1, 1])
    cost = np.empty((M, 2))
    for i in range(M):
        q_sum = q[starts[i]:ends[i]].sum() / G
        w = (ends[i] - starts[i]) / G
        for j, pred in enumerate(labels):
            cost[i, j] = q_sum if pred == run_labels[i] else w - q_sum

    # dp[k, j]: min risk so far with exactly k intervals, last one predicting labels[j]
    dp = np.full((max_K + 1, 2), np.inf)
    dp[1, :] = cost[0, :]
    for i in range(1, M):
        new = np.full_like(dp, np.inf)
        for k in range(1, min(i + 1, max_K) + 1):
            for j in range(2):
                switch = dp[k - 1, 1 - j] if k > 1 else np.inf
                new[k, j] = cost[i, j] + min(dp[k, j], switch)
        dp = new

    at_most_k_risk = np.minimum.accumulate(np.min(dp, axis=1)[1:])
    return np.array([at_most_k_risk[min(2 ** int(d), M) - 1] for d in depths])


# EXPERIMENT

def fit_all_depths(x, y, x_test, y_test, depths):
    """train and test error of the trees of every depth fitted on one training set"""
    train_errs, test_errs = np.empty(len(depths)), np.empty(len(depths))
    for a, d in enumerate(depths):
        clf = DecisionTreeClassifier(max_depth=int(d), random_state=42)
        clf.fit(x.reshape(-1, 1), y)
        train_errs[a] = 1 - clf.score(x.reshape(-1, 1), y)
        test_errs[a] = 1 - clf.score(x_test.reshape(-1, 1), y_test)
    return train_errs, test_errs

def main_loop(x_test, y_test, n, T, N, f, depths, noise_type, rng, sigma=None, p=None, n_jobs=1):
    """E[hat R], E[R], E[R - hat R] over T training sets, the same T sets at every depth"""
    # data sampled here, not in the workers, so the result doesn't depend on n_jobs
    xs = [sample_x(n, N, rng) for _ in range(T)]
    ys = [sample_y(x, f, noise_type, rng, sigma=sigma, p=p) for x in xs]
    errs = Parallel(n_jobs=n_jobs)(
        delayed(fit_all_depths)(x, y, x_test, y_test, depths) for x, y in zip(xs, ys))
    # shape (depths, T)
    train_errs = np.array([e[0] for e in errs]).T
    test_errs = np.array([e[1] for e in errs]).T
    errs = {"train_err": train_errs, "test_err": test_errs, "gap": test_errs - train_errs}
    # mean and std across the T samples, for every depth
    return ({c: e.mean(axis=1) for c, e in errs.items()},
            {c: e.std(axis=1) for c, e in errs.items()})

def join_and(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

def fmt_level(level):
    return f"{level:g}%"

def experiment(curves: Sequence[Curve] = ("gap", "bound_union"),
               train_size=500, test_size=200_000, supp_size=1_000_000_000, num_iter=100,
               noise_type: Literal["gaussian", "bernoulli"] = "gaussian", noise_level=(20,),
               num_crossing=10, C_union=0.06, C_radmass=1.0, C_noise=1.0, seed=4, n_jobs=1):
    """returns the figure with the chosen curves for every noise level (percentage of flipped labels)"""
    curves = list(dict.fromkeys(curves))
    for c in curves:
        if c not in CURVES:
            raise ValueError(f"unknown curve {c!r}, choose among {CURVES}")
    noise_level = list(noise_level)

    rng = np.random.default_rng(seed=seed)
    f = lambda x: sine(x, m=num_crossing)
    n, N = train_size, supp_size
    depths = np.arange(1, 31)

    emp = [c for c in curves if c in EMPIRICAL]
    # bounds to compute, also when only train error + bound is plotted
    needed = {c.removeprefix("train_") for c in curves}
    # one shade of the curve color per noise level, lighter = less noise
    ranks = np.argsort(np.argsort(noise_level))
    shades = np.linspace(0.45, 0.95, len(noise_level)) if len(noise_level) > 1 else [0.75]

    fig, ax = plt.subplots(figsize=(10, 6))
    for level, rank in zip(noise_level, ranks):
        if noise_type == "gaussian":
            sigma, p = sigma_for_level(f, level, N), None
            print(f"noise {fmt_level(level)}: sigma = {sigma:.3f}")
        else:
            sigma, p = None, 1 - level / 100
            print(f"noise {fmt_level(level)}: p = {p:.3f}")

        values = {}
        if emp:
            x_test = sample_x(test_size, N, rng)
            y_test = sample_y(x_test, f, noise_type, rng, sigma=sigma, p=p)
            means, stds = main_loop(x_test, y_test, n, num_iter, N, f, depths, noise_type, rng,
                                    sigma=sigma, p=p, n_jobs=n_jobs)
            values.update(means)
            print(f"  gap at d = {depths[-1]}: {means['gap'][-1]:.4f} ± {stds['gap'][-1]:.4f}")
        if "bound_union" in needed:
            values["bound_union"] = np.array([union_bound(d, n, N, C=C_union) for d in depths])
        if "bound_radmass" in needed:
            values["bound_radmass"] = np.array([radem_massart_bound(d, n, N, C=C_radmass) for d in depths])
        if "bound_noise" in needed:
            R_star_d = dp_R_star_d(f, depths, N, noise_type, sigma=sigma, p=p)
            values["bound_noise"] = np.array([noise_approxerr_bound(d, n, r, N, C=C_noise)
                                              for d, r in zip(depths, R_star_d)])
        for b in BOUNDS:
            if "train_" + b in curves:
                # the bound is deterministic, the randomness comes only from the train error
                values["train_" + b] = values["train_err"] + values[b]
                stds["train_" + b] = stds["train_err"]

        for c in curves:
            label = CURVE_SYMBOLS[c] if len(noise_level) == 1 else f"{CURVE_SYMBOLS[c]}, {fmt_level(level)}"
            color = plt.get_cmap(CMAPS[c.removeprefix("train_") if "bound" in c else c])(shades[rank])
            ax.plot(depths, values[c], color=color, linewidth=2,
                    marker=None if "bound" in c else "o", markersize=5, label=label,
                    clip_on=False)
            if c in EMPIRICAL:
                # band of ± 1 std across the T training sets
                ax.fill_between(depths, values[c] - stds[c], values[c] + stds[c],
                                color=color, alpha=0.15, linewidth=0)

    vlines = [(np.log2(n), r"$\log_2 n$")]
    if "bound_union" in needed:
        vlines.append((np.log2(N), r"$\log_2 N$"))
    for xv, name in vlines:
        ax.axvline(xv, color="gray", linestyle="--", linewidth=1.2)
        ax.text(xv, 1.01, name, color="gray", ha="center", va="bottom", fontsize=13,
                transform=ax.get_xaxis_transform())

    title = join_and([CURVE_NAMES[c] for c in curves]) + " as depth increases"
    if not emp and "bound_noise" not in needed:
        # no curve depends on the noise: no noise in the title
        pass
    elif len(noise_level) == 1:
        title += f", with noise at {fmt_level(noise_level[0])}"
    else:
        title += ", with noise at " + ", ".join(fmt_level(l) for l in noise_level)
    ax.set_title(title, pad=24, fontsize=16)
    # axes start exactly at (1, 0)
    ax.set_xlim(depths[0], depths[-1])
    ax.set_xticks(np.r_[depths[0], np.arange(5, depths[-1] + 1, 5)])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("d", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def best_risk_decomposition(supp_size=1_000_000_000, noise_type: Literal["gaussian", "bernoulli"] = "gaussian",
                            noise_level=(20,), num_crossing=10):
    """best risk R*_{H_d} = Bayes risk R* + approximation error A_{H_d}, one panel per noise level"""
    noise_level = sorted(noise_level, reverse=True)
    f = lambda x: sine(x, m=num_crossing)
    N = supp_size
    depths = np.arange(1, 31)

    fig, axes = plt.subplots(len(noise_level), 1, figsize=(10, 4.5 * len(noise_level)),
                             sharex=True, sharey=True, squeeze=False)
    for ax, level in zip(axes[:, 0], noise_level):
        if noise_type == "gaussian":
            sigma, p = sigma_for_level(f, level, N), None
        else:
            sigma, p = None, 1 - level / 100
        R_star_d = dp_R_star_d(f, depths, N, noise_type, sigma=sigma, p=p)
        # the Bayes classifier predicts sign(f): its risk is the average flip probability
        R_star = flip_prob(support_grid(N), f, noise_type, sigma=sigma, p=p).mean()
        print(f"noise {fmt_level(level)}: R* = {R_star:.4f}, R*_Hd at d = {depths[-1]}: {R_star_d[-1]:.4f}")

        ax.plot(depths, R_star_d, color=plt.get_cmap("RdPu")(0.75), linewidth=2, marker="o", markersize=5,
                label=r"$R^*_{\mathcal{H}_d}$", clip_on=False)
        ax.axhline(R_star, color="black", linestyle="--", linewidth=1.5, label=r"$R^*$")
        ax.plot(depths, R_star_d - R_star, color=plt.get_cmap("Greys")(0.55), linewidth=2, marker="o",
                markersize=5, label=r"$A_{\mathcal{H}_d} = R^*_{\mathcal{H}_d} - R^*$", clip_on=False)
        ax.set_title(f"Noise at {fmt_level(level)}", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_xlim(depths[0], depths[-1])
    axes[0, 0].set_xticks(np.r_[depths[0], np.arange(5, depths[-1] + 1, 5)])
    axes[0, 0].set_ylim(bottom=0)
    axes[-1, 0].set_xlabel("d", fontsize=14)
    fig.suptitle("Best Risk, Bayes Risk and Approximation Error as depth increases", fontsize=16)
    fig.tight_layout()
    return fig


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curves", nargs="+", choices=CURVES, default=["gap", "bound_union"])
    ap.add_argument("--train_size", type=int, default=500)
    ap.add_argument("--test_size", type=int, default=200_000)
    ap.add_argument("--supp_size", type=int, default=1_000_000_000)
    ap.add_argument("--num_iter", type=int, default=100)
    ap.add_argument("--noise_type", choices=["gaussian", "bernoulli"], default="gaussian")
    ap.add_argument("--noise_level", type=float, nargs="+", default=[20],
                    help="percentage of flipped labels, one or more")
    ap.add_argument("--num_crossing", type=int, default=10)
    ap.add_argument("--C_union", type=float, default=0.06)
    ap.add_argument("--C_radmass", type=float, default=1.0)
    ap.add_argument("--C_noise", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--n_jobs", type=int, default=-1,
                    help="cores used to fit the trees, -1 = all of them")
    ap.add_argument("--out", default=None, help="where to save the figure, not saved if omitted")
    ap.add_argument("--show", action="store_true",
                    help="also open the figure in a window, always done if --out is omitted")
    ap.add_argument("--best_risk", action="store_true",
                    help="plot best risk, Bayes risk and approximation error instead of --curves")
    return ap.parse_args()

if __name__ == "__main__":
    args = vars(parse_args())
    out, show = args.pop("out"), args.pop("show")
    if args.pop("best_risk"):
        fig = best_risk_decomposition(args["supp_size"], args["noise_type"], args["noise_level"],
                                      args["num_crossing"])
    else:
        fig = experiment(**args)
    if out is not None:
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"saved {out}")
    if show or out is None:
        plt.show()
