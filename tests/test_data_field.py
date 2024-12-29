from datetime import datetime

import pandas as pd
import pytest

from qbot.data.data_field import DataField

from .conftest import check_binance_availability


@pytest.mark.skipif(
    check_binance_availability(),
    reason="Binance API not available or restricted",
)
def test_data_field_initialization():
    df = DataField(
        interval="1h",
        start_time="2024-01-01",
        end_time="2024-01-02",
        ticker_list=["BTCUSDT"],
    )
    assert isinstance(df["BTCUSDT"], pd.DataFrame)
    assert "timestamp" in df["BTCUSDT"].columns
    assert "close" in df["BTCUSDT"].columns


@pytest.mark.skipif(
    check_binance_availability(),
    reason="Binance API not available or restricted",
)
def test_data_field_wildcard():
    df = DataField(
        interval="1h",
        start_time="2024-01-01",
        end_time="2024-01-02",
        ticker_list=["BTCUSDT", "ETHUSDT"],
    )
    result = df["*USDT"]
    assert len(result) == 2
    assert all(isinstance(d, pd.DataFrame) for d in result)


@pytest.mark.skipif(
    check_binance_availability(),
    reason="Binance API not available or restricted",
)
def test_data_field_add_ticker():
    df = DataField(interval="1h", start_time="2024-01-01", end_time="2024-01-02")
    df.add_ticker("BTCUSDT")
    assert isinstance(df["BTCUSDT"], pd.DataFrame)

    df.add_ticker(["ETHUSDT", "XRPUSDT"])
    assert len(df["*USDT"]) == 3
