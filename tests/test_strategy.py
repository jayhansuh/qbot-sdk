import numpy as np
import pandas as pd
import pytest

from qbot.data.data_field import DataField
from qbot.strategy.strategy import Strategy

from .conftest import check_binance_availability


def test_strategy_initialization():
    strategy = Strategy(
        name="test_strategy", hyper_params={"param1": 1.0}, interval="1h"
    )
    assert strategy.name == "test_strategy"
    assert strategy.hyper_params == {"param1": 1.0}
    assert strategy.interval == "1h"


@pytest.mark.skipif(
    check_binance_availability(),
    reason="Binance API not available or restricted",
)
def test_strategy_evaluation():
    strategy = Strategy(name="test_strategy")

    def evaluate_func(strategy, data_field):
        strategy["timestamp"] = data_field["BTCUSDT"]["timestamp"]
        strategy["BTCUSDT_close"] = data_field["BTCUSDT"]["close"]
        strategy["BTCUSDT_weight"] = 1.0

    strategy.evaluate_func = evaluate_func

    data_field = DataField(
        interval="1h",
        start_time="2024-01-01",
        end_time="2024-01-02",
        ticker_list=["BTCUSDT"],
    )

    strategy.evaluate(data_field)
    assert "BTCUSDT_weight" in strategy._compute_df.columns
    assert "timestamp" in strategy._compute_df.columns


@pytest.mark.skipif(
    check_binance_availability(),
    reason="Binance API not available or restricted",
)
def test_strategy_backtest():
    strategy = Strategy(name="test_strategy")

    def evaluate_func(strategy, data_field):
        strategy["timestamp"] = data_field["BTCUSDT"]["timestamp"]
        strategy["BTCUSDT_close"] = data_field["BTCUSDT"]["close"]
        strategy["BTCUSDT_weight"] = 1.0

    strategy.evaluate_func = evaluate_func
    strategy.reference_index = "BTCUSDT_close"

    data_field = DataField(
        interval="1h",
        start_time="2024-01-01",
        end_time="2024-01-02",
        ticker_list=["BTCUSDT"],
    )

    strategy.evaluate(data_field)
    asset_df = strategy.back_testing()

    assert "asset_value" in asset_df.columns
    assert "ln_asset_value" in asset_df.columns
