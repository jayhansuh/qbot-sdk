from typing import Callable, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def correlation(
    source_series: pd.Series,
    target_series: pd.Series,
    shifts: range = range(-24, 25),
) -> pd.DataFrame:
    # Calculate correlations for different time shifts
    # Create shifted versions of target prices for all shifts at once
    target_shifts = pd.DataFrame(
        {
            s: target_series.shift(s)[s:-s] if s > 0 else target_series[abs(s) :]
            for s in shifts
        }
    )

    # Create aligned source prices for all shifts
    source_aligned = pd.DataFrame(
        {
            s: (
                source_series[s:]
                if s >= 0
                else source_series.shift(abs(s))[abs(s) : -abs(s)]
            )
            for s in shifts
        }
    )

    # Calculate correlations
    correlations = source_aligned.corrwith(target_shifts)
    return correlations


def correlation_plot(
    x: pd.Series,
    y: pd.Series,
    shifts: range = range(-150, 150),
    x_label: str = "x",
    y_label: str = "y",
) -> pd.DataFrame:
    correl_data = (
        correlation(x, y, shifts),
        correlation(x.diff(), y, shifts),
        correlation(x.diff(), y.diff(), shifts),
    )
    labels = (
        f"corr({x_label}(t), {y_label}(t+t_0))",
        f"dcorr({x_label}(t), {y_label}(t+t_0))",
        f"corr(d{x_label}(t), d{y_label}(t+t_0))",
    )
    for i, corr in enumerate(correl_data):
        plt.plot(shifts, corr, label=labels[i])
    plt.legend()
    plt.show()

    return correl_data, labels


def pairplot_scan(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_shifts: range = range(40, 300, 120),
    y_shifts: range = range(-30, -300, -90),
    hue_func: Callable = lambda df: df[f"close_pct_change_40"] < -0.085,
) -> pd.DataFrame:
    sns.set_theme(style="ticks")
    # new_df = pd.DataFrame(df.index, columns=['timestamp'])
    new_df = df.copy()

    x_vars = []
    for i in x_shifts:
        new_df[f"{x_col}_pct_change_{i}"] = df[x_col].pct_change(periods=i)
        x_vars.append(f"{x_col}_pct_change_{i}")

    y_vars = []
    for i in y_shifts:
        new_df[f"{y_col}_pct_change_{i}"] = df[y_col].pct_change(periods=i)
        y_vars.append(f"{y_col}_pct_change_{i}")

    new_df["hue"] = hue_func(new_df)

    # Different columns for x-axis and y-axis
    g = sns.pairplot(data=new_df, x_vars=x_vars, y_vars=y_vars, hue="hue")
    g.figure.suptitle("Rectangular PairPlot (different x_vars & y_vars)", y=1.02)
    sns.despine()

    return new_df


def calculate_alpha_beta_gamma(
    y: pd.Series, x: pd.Series
) -> Tuple[float, float, float]:
    # return_y =
    #   alpha + beta * return_x
    #   + gamma * return_x + noise

    # Calculate percentage changes and drop NaN values
    return_y = y.pct_change().dropna()
    return_x = x.pct_change().dropna()

    # Ensure alignment of the two series
    data = pd.concat([return_y, return_x], axis=1).dropna()
    return_y_aligned = data.iloc[:, 0]
    return_x_aligned = data.iloc[:, 1]

    # Calculate beta (slope) using covariance and variance
    beta = return_y_aligned.cov(return_x_aligned) / return_x_aligned.var()

    # Calculate alpha (intercept)
    alpha = return_y_aligned.mean() - beta * return_x_aligned.mean()

    # Calculate residuals
    residuals = return_y_aligned - (alpha + beta * return_x_aligned)

    # Calculate gamma as the standard deviation of residuals
    gamma = residuals.std()

    return alpha, beta, gamma
