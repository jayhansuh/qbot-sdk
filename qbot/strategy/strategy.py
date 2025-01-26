import pickle
from abc import ABC
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from qbot.data.data_field import DataField
from qbot.strategy.utils import calculate_alpha_beta_gamma


class Strategy(ABC):
    def __init__(self, name: str, hyper_params: Optional[Dict] = None, interval: str = "1h"):
        """
        hyper_params: dictionary of hyperparameters
        required_fields: list of field names
        lookback_rows: number of rows to look back for each field
        """
        self.name = name
        self.hyper_params = hyper_params or {}
        self.interval = interval
        self._compute_df = pd.DataFrame()
        self.weight_func = None
        self.reference_index = None

    def __setitem__(self, *args, **kwargs):
        self._compute_df.__setitem__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self._compute_df.__getitem__(*args, **kwargs)

    @property
    def loc(self):
        return self._compute_df.loc

    @property
    def iloc(self):
        return self._compute_df.iloc

    def get_weight_cols(self):
        return [col for col in self._compute_df.columns if col.endswith("_weight")]

    def eval_weight(self, data_field: DataField):
        # Clear all weight columns
        for col in self.get_weight_cols():
            self._compute_df.drop(columns=[col], inplace=True)

        # Make sure timestamp is set
        if "timestamp" not in self._compute_df.columns:
            if data_field.timestamp is not None:
                self._compute_df["timestamp"] = data_field.timestamp
            else:
                self._compute_df["timestamp"] = data_field["BTCUSDT"]["timestamp"]

        self.weight_func(self, data_field)

        # Make sure reference_index is set(default to BTCUSDT_close)
        if self.reference_index is None:
            # Any close column is set as reference_index
            for col in self._compute_df.columns:
                if col.endswith("_close"):
                    self.reference_index = col
                    break
            if self.reference_index is None:
                self.reference_index = "BTCUSDT_close"
                self._compute_df["BTCUSDT_close"] = data_field["BTCUSDT"]["close"]
        else:
            if self.reference_index not in self._compute_df.columns:
                ticker, field = self.reference_index.split("_")
                ticker, field = ticker.upper(), field.lower()
                self._compute_df[self.reference_index] = data_field[ticker][field]

        # Make sure all {ticker}_close are set when {ticker}_weight is set
        weight_cols = self.get_weight_cols()
        close_cols = [col.replace("_weight", "_close") for col in weight_cols]
        for col in close_cols:
            if col not in self._compute_df.columns:
                ticker, field = col.split("_")
                ticker, field = ticker.upper(), field.lower()
                self._compute_df[col] = data_field[ticker][field]

        # Check causality
        for col in weight_cols:
            last_val = self._compute_df[col].iloc[-1]
            if last_val is None or pd.isna(last_val) or np.isnan(last_val):
                raise ValueError(f"Causality violation: column {col} depends on future data")

    def save_weight_func(self):
        with open("weight_func.pkl", "wb") as f:
            pickle.dump(self.weight_func, f)

    def load_weight_func(self):
        with open("weight_func.pkl", "rb") as f:
            self.weight_func = pickle.load(f)

    def eval_asset(self, commission_rate=0.00045):
        weight_cols = self.get_weight_cols()

        self._compute_df["asset_gain_factor"] = 1.0  # gamma
        self._compute_df["trading_volume"] = 0.0

        w_shift = {}
        close_pctchange = {}

        for wcol in weight_cols:
            w_shift[wcol] = self._compute_df[wcol].ffill().shift(1).fillna(0.0)
            ccol = wcol.replace("_weight", "_close")
            close_pctchange[wcol] = self._compute_df[ccol].ffill().pct_change().fillna(0.0)
            self._compute_df["asset_gain_factor"] += w_shift[wcol] * close_pctchange[wcol]

        for wcol in weight_cols:
            trading_volume = self._compute_df[wcol].ffill()
            trading_volume -= w_shift[wcol] * (close_pctchange[wcol] + 1.0) / self._compute_df["asset_gain_factor"]
            self._compute_df["trading_volume"] += trading_volume.fillna(0.0).abs()

        asset_pctchange = self._compute_df["asset_gain_factor"]
        asset_pctchange *= 1.0 - self._compute_df["trading_volume"] * commission_rate
        self._compute_df["asset_value"] = asset_pctchange.cumprod()
        self._compute_df.drop(columns=["asset_gain_factor", "trading_volume"], inplace=True)

        return self._compute_df[["asset_value"] + weight_cols]

    def eval_perf(self, EVAL_PERIOD=1):
        perf_dict = {}
        perf_str = ""

        # YEAR_DIV_NUM
        if self.interval == "1M":
            YEAR_DIV_NUM = 12 / EVAL_PERIOD
        else:
            YEAR_DIV_NUM = pd.Timedelta(days=365) / pd.Timedelta(self.interval) / EVAL_PERIOD
        perf_dict["duration"] = len(self._compute_df) / YEAR_DIV_NUM

        # Print strategy statistics
        total_return = self._compute_df["asset_value"].iloc[-1] / self._compute_df["asset_value"].iloc[0] - 1
        annual_return = (1.0 + total_return) ** (YEAR_DIV_NUM / len(self._compute_df)) - 1
        perf_dict["total_return"] = total_return
        perf_dict["annual_return"] = annual_return

        reference_return = (
            self._compute_df[self.reference_index].iloc[-1] / self._compute_df[self.reference_index].iloc[0] - 1
        )
        annual_reference_return = (1.0 + reference_return) ** (YEAR_DIV_NUM / len(self._compute_df)) - 1
        perf_dict["reference_return"] = reference_return
        perf_dict["annual_reference_return"] = annual_reference_return

        mdd = (self._compute_df["asset_value"].div(self._compute_df["asset_value"].cummax()) - 1).min()
        perf_dict["mdd"] = mdd

        # Calculate alpha, beta, and gamma
        alpha, beta, gamma = calculate_alpha_beta_gamma(
            self._compute_df[self.reference_index],
            self._compute_df["asset_value"],
            periods=EVAL_PERIOD,
        )
        annual_alpha = (1.0 + alpha) ** YEAR_DIV_NUM - 1.0
        annual_gamma = gamma * np.sqrt(YEAR_DIV_NUM)

        perf_dict["annual_alpha"] = annual_alpha
        perf_dict["beta"] = beta
        perf_dict["annual_gamma"] = annual_gamma

        # Calculate Sharpe Ratio
        asset_return = self._compute_df["asset_value"].apply(np.log).diff(periods=EVAL_PERIOD)
        sharpe_ratio = asset_return.mean() / asset_return.std()
        annual_sharpe_ratio = sharpe_ratio * np.sqrt(YEAR_DIV_NUM)
        perf_dict["annual_sharpe_ratio"] = annual_sharpe_ratio

        # Calculate Sortino Ratio
        sortino_ratio = asset_return.mean() / asset_return[asset_return < 0].std()
        annual_sortino_ratio = sortino_ratio * np.sqrt(YEAR_DIV_NUM)
        perf_dict["annual_sortino_ratio"] = annual_sortino_ratio

        # Reference Sortino Ratio
        ref_return = self._compute_df[self.reference_index].apply(np.log).diff(periods=EVAL_PERIOD)
        ref_sortino_ratio = ref_return.mean() / ref_return[ref_return < 0].std()
        annual_ref_sortino_ratio = ref_sortino_ratio * np.sqrt(YEAR_DIV_NUM)
        perf_dict["annual_ref_sortino_ratio"] = annual_ref_sortino_ratio

        perf_str += f"Total Return: {total_return * 100:.2f}%\t\tARR: {annual_return * 100:.2f}%\n"
        perf_str += f"{self.reference_index.split('_', 1)[0]} Return: {reference_return * 100:.2f}%\t\tARR: {annual_reference_return * 100:.2f}%\n"
        perf_str += f"Annual Ref Return * Beta:\tARR: {(beta if beta else 1.0) * annual_reference_return * 100:.2f}%\n"
        perf_str += "-----------------------------------------\n"
        perf_str += f"annual alpha: {annual_alpha * 100:.2f}%, beta: {beta:.2f}, annual gamma: {annual_gamma:.2f}\n"
        perf_str += f"Duration: {perf_dict['duration']:.2f} years\t\tMax Drawdown: {mdd * 100:.2f}%\n"
        perf_str += "-----------------------------------------\n"
        perf_str += f"Annual Sharpe Ratio:     \t{annual_sharpe_ratio:.2f}\n"
        perf_str += f"Annual Sortino Ratio:    \t{annual_sortino_ratio:.2f}\n"
        perf_str += f"Annual Ref Sortino Ratio:\t{annual_ref_sortino_ratio:.2f}\n"

        return perf_dict, perf_str

    def plot_perf(self):
        weight_cols = self.get_weight_cols()
        cutoff_num = 0
        while cutoff_num < len(self._compute_df):
            # Check if all weights are set
            if all(self._compute_df[col].iloc[cutoff_num] is not None for col in weight_cols):
                break
            cutoff_num += 1
        timestamp_cutoff = self._compute_df["timestamp"].iloc[cutoff_num:]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

        ax1.set_title(self.name + " - " + datetime.now().strftime("%Y-%m-%d"))
        for col in self._compute_df.columns:
            if col == "timestamp" or col == "asset_value" or col.endswith("_weight") or col.endswith("_close"):
                continue
            # normalize column
            cutoff_series = self._compute_df[col].iloc[cutoff_num:]
            max_val = cutoff_series.max()
            min_val = cutoff_series.min()
            label = col + f" ({min_val:.2f}, {max_val:.2f})"
            cutoff_series = (cutoff_series - min_val) / (max_val - min_val)
            ax1.plot(timestamp_cutoff, cutoff_series, label=label, alpha=0.5)
        ax1.legend()
        ax1.set_yticks([0, 1], labels=["min", "max"])
        ax1.xaxis.set_major_locator(plt.MaxNLocator(7))
        # Draw reference and asset value on left axis of ax2
        ax2.set_yscale("log")

        # Plot asset value with thick solid line
        cutoff_series = self._compute_df["asset_value"].iloc[cutoff_num:]
        max_val = cutoff_series.max()
        min_val = cutoff_series.min()
        label = "asset_value" + f" ({min_val:.2f}, {max_val:.2f})"
        ax2.plot(
            timestamp_cutoff,
            cutoff_series / cutoff_series.iloc[0],
            label=label,
            linewidth=2,
            color="lightcoral",
        )

        # Plot reference index with dashed line
        cutoff_series = self._compute_df[self.reference_index].iloc[cutoff_num:]
        max_val = cutoff_series.max()
        min_val = cutoff_series.min()
        label = self.reference_index + f" ({min_val:.2f}, {max_val:.2f})"
        ax2.plot(
            timestamp_cutoff,
            cutoff_series / cutoff_series.iloc[0],
            label=label,
            linestyle="-",
            linewidth=2,
            color="teal",
            alpha=0.8,
        )

        # Plot other price series with thin transparent lines
        for col in weight_cols:
            col = col.replace("_weight", "_close")
            if col == self.reference_index:
                continue
            ax2.plot(
                timestamp_cutoff,
                self._compute_df[col].iloc[cutoff_num:] / self._compute_df[col].iloc[cutoff_num],
                label=col,
                alpha=0.5,
                linewidth=1,
            )
        ax2.legend()
        ax2.xaxis.set_major_locator(plt.MaxNLocator(7))

        # Draw weights on right axis
        ax3 = ax2.twinx()
        for col in weight_cols:
            cutoff_series = self._compute_df[col].iloc[cutoff_num:]
            ax3.plot(timestamp_cutoff, cutoff_series, label=col, alpha=0.5)
        ax3.hlines(
            y=0,
            xmin=timestamp_cutoff.min(),
            xmax=timestamp_cutoff.max(),
            color="slategrey",
            linestyle="--",
            alpha=0.5,
        )
        ax3.legend()
        fig.tight_layout()

        return fig, ax1, ax2, ax3
