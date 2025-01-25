import pytest

from qbot.data.data_field import DataField
from qbot.strategy.strategy import Strategy

from .conftest import check_binance_availability


def test_strategy_initialization():
    strategy = Strategy(name="test_strategy", hyper_params={"param1": 1.0}, interval="1h")
    assert strategy.name == "test_strategy"
    assert strategy.hyper_params == {"param1": 1.0}
    assert strategy.interval == "1h"


@pytest.mark.skipif(
    check_binance_availability(),
    reason="Binance API not available or restricted",
)
def test_strategy_evaluation():
    strategy = Strategy(name="test_strategy")

    def weight_func(strategy, data_field):
        strategy["BTCUSDT_weight"] = 1.0

    strategy.weight_func = weight_func

    data_field = DataField(
        interval="1h",
        start_time="2024-01-01",
        end_time="2024-01-02",
        ticker_list=["BTCUSDT"],
    )

    strategy.eval_weight(data_field)
    assert "BTCUSDT_weight" in strategy._compute_df.columns
    assert "timestamp" in strategy._compute_df.columns
    assert "BTCUSDT_close" in strategy._compute_df.columns


@pytest.mark.skipif(
    check_binance_availability(),
    reason="Binance API not available or restricted",
)
def test_strategy_eval_asset():
    strategy = Strategy(name="test_strategy")

    def weight_func(strategy, data_field):

        strategy.reference_index = "BTCUSDT_close"
        strategy["timestamp"] = data_field["BTCUSDT"]["timestamp"]
        strategy["BTCUSDT_close"] = data_field["BTCUSDT"]["close"]
        strategy["BTCUSDT_weight"] = 1.0

    strategy.weight_func = weight_func

    data_field = DataField(
        interval="1h",
        start_time="2024-01-01",
        end_time="2024-01-02",
        ticker_list=["BTCUSDT"],
    )

    strategy.eval_weight(data_field)
    strategy.eval_asset()

    assert "asset_value" in strategy._compute_df.columns


@pytest.mark.skipif(
    check_binance_availability(),
    reason="Binance API not available or restricted",
)
def test_strategy_eval_perf():

    strategy = Strategy(name="3ETHUSDT_weight")

    def weight_func(strategy, data_field):

        strategy.reference_index = "ETHUSDT_close"
        strategy["ETHUSDT_weight"] = 3.0

    strategy.weight_func = weight_func

    data_field = DataField(
        start_time="2024-01-01",
        end_time="2024-01-02",
    )

    strategy.eval_weight(data_field)
    strategy.eval_asset()
    perf_dict, perf_str = strategy.eval_perf()

    assert perf_str.startswith("Total Return:")
    assert abs(perf_dict["beta"] - 3.0) < 1e-6

    def weight_func(strategy, data_field):

        strategy.reference_index = "BTCUSDT_close"
        strategy["BTCUSDT_weight"] = 1

    strategy.weight_func = weight_func
    strategy.eval_weight(data_field)
    strategy.eval_asset()
    perf_dict, perf_str = strategy.eval_perf()

    assert abs(perf_dict["annual_alpha"] - 0.0) < 1e-6
    assert abs(perf_dict["beta"] - 1.0) < 1e-6
    assert abs(perf_dict["annual_gamma"] - 0.0) < 1e-6
    assert abs(perf_dict["annual_sortino_ratio"] - perf_dict["annual_ref_sortino_ratio"]) < 1e-6


@pytest.mark.skipif(
    check_binance_availability(),
    reason="Binance API not available or restricted",
)
def test_strategy_causality():

    strategy = Strategy(name="future_data")

    def weight_func(strategy, data_field):
        strategy.reference_index = "BTCUSDT_close"
        strategy["BTCUSDT_weight"] = -100 * data_field["BTCUSDT"]["close"].pct_change(-1)

    strategy.weight_func = weight_func

    data_field = DataField(
        start_time="2024-01-01",
        end_time="2024-01-02",
    )

    try:
        strategy.eval_weight(data_field)
        assert False
    except Exception as e:
        assert str(e).startswith("Causality violation")
