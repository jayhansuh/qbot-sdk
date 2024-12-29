import pickle
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd

from qbot.data.data_field import DataField
from qbot.data.data_io import Symbol
from qbot.strategy.utils import calculate_alpha_beta_gamma


class Strategy(ABC):
    def __init__(
        self, name: str, hyper_params: Optional[Dict] = None, interval: str = "1h"
    ):
        """
        hyper_params: dictionary of hyperparameters
        required_fields: list of field names
        lookback_rows: number of rows to look back for each field
        """
        self.name = name
        self.hyper_params = hyper_params or {}
        self.interval = interval
        self._compute_df = pd.DataFrame()
        self._asset_df = pd.DataFrame()
        self.evaluate_func = None
        self.reference_index = None

    def __setitem__(self, *args, **kwargs):
        self._compute_df.__setitem__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self._compute_df.__getitem__(*args, **kwargs)

    def evaluate(self, data_field: DataField):
        self.evaluate_func(self, data_field)

    def save_evaluate_func(self):
        with open("evaluate_func.pkl", "wb") as f:
            pickle.dump(self.evaluate_func, f)

    def load_evaluate_func(self):
        with open("evaluate_func.pkl", "rb") as f:
            self.evaluate_func = pickle.load(f)

    def back_testing(self):

        # check if reference_index is set
        if self.reference_index is None:
            raise ValueError("reference_index is not set")
        self._asset_df = pd.DataFrame(
            self._compute_df["timestamp"], columns=["timestamp"]
        )
        self._asset_df["ln_reference_index"] = np.log(
            self._compute_df[self.reference_index]
        )
        self._asset_df["ln_reference_index"] -= self._asset_df[
            "ln_reference_index"
        ].iloc[0]

        weight_cols = [
            col for col in self._compute_df.columns if col.endswith("_weight")
        ]
        close_cols = [col.replace("_weight", "_close") for col in weight_cols]
        for col in close_cols:
            if col not in self._compute_df.columns:
                raise ValueError(f"Column {col} not found in _compute_df")

        self._asset_df["return"] = 0.0
        for wcol, ccol in zip(weight_cols, close_cols):
            self._asset_df[wcol] = self._compute_df[wcol].shift(1).fillna(0.0)
            self._asset_df[ccol] = self._compute_df[ccol].pct_change().fillna(0.0)
            self._asset_df["return"] += self._asset_df[wcol] * self._asset_df[ccol]
        self._asset_df["asset_value"] = (1.0 + self._asset_df["return"]).cumprod()
        self._asset_df["ln_asset_value"] = np.log(self._asset_df["asset_value"])

        return self._asset_df

    def display_result(self):
        weight_cols = [col for col in self._asset_df.columns if col.endswith("_weight")]
        self._asset_df[weight_cols + ["ln_reference_index", "ln_asset_value"]].plot()

        # Print strategy statistics
        total_return = self._asset_df["asset_value"].iloc[-1] - 1
        print(f"Total Return: {total_return * 100:.2f}%")
        mdd = (
            self._asset_df["asset_value"].div(self._asset_df["asset_value"].cummax())
            - 1
        ).min()
        print(f"Max Drawdown: {mdd * 100:.2f}%")
        
        alpha, beta, gamma = calculate_alpha_beta_gamma(self._asset_df['asset_value'], np.exp(self._asset_df['ln_reference_index']))
        if self.interval == '1h':
            annual_alpha = (1+alpha)**(24*365)-1

        print(f"annual alpha: {annual_alpha * 100:.2f}%, beta: {beta:.2f}, gamma: {gamma:.2f}")
        reference_return = np.exp(self._asset_df["ln_reference_index"].iloc[-1]) - 1
        print(f"Reference {self.reference_index} Return: {reference_return:.2f}%")
        print(
            f"Strategy Return * Beta: {(beta if beta else 1.) * reference_return:.2f}%"
        )
        print(
            f"Sharpe Ratio: {self._asset_df['return'].mean() / self._asset_df['return'].std() * np.sqrt(365*24):.2f}"
        )
        adjusted_score = self._asset_df["ln_asset_value"].iloc[-1]
        adjusted_score -= np.log(
            1.0
            + (beta if beta else 1.0) * (self._asset_df["ln_reference_index"].iloc[-1])
        )
        adjusted_score *= 1.0 + mdd
        # adjusted_score = 1 - np.exp(-1.*adjusted_score)
        print(f"Score: {adjusted_score:.2f}")
