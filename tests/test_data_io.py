from datetime import datetime

import pandas as pd
import pytest

from qbot.data.data_io import EXCHANGE_METADATA, Symbol, TimeRange

from .conftest import check_binance_availability


def test_symbol_initialization():
    symbol = Symbol(
        exchange="BINANCE", interval="1h", market="SPOT", raw_symbol="BTCUSDT"
    )
    assert str(symbol) == "BINANCE_1h_SPOT_BTCUSDT"
    assert symbol.exchange == "BINANCE"
    assert symbol.interval == "1h"
    assert symbol.market == "SPOT"
    assert symbol.raw_symbol == "BTCUSDT"


def test_symbol_parse_str():
    symbol = Symbol.parse_str("BTCUSDT")
    assert str(symbol) == "BINANCE_1h_SPOT_BTCUSDT"

    symbol = Symbol.parse_str("BINANCE_1h_SPOT_BTCUSDT")
    assert str(symbol) == "BINANCE_1h_SPOT_BTCUSDT"


def test_time_range():
    tr = TimeRange(start_date="2024-01-01", end_date="2024-03-01")
    assert isinstance(tr.start_date, pd.Timestamp)
    assert isinstance(tr.end_date, pd.Timestamp)
    assert str(tr) == "2024-01-01_2024-03-01"


@pytest.mark.skipif(
    check_binance_availability(),
    reason="Binance API not available or restricted",
)
def test_exchange_metadata_validation():
    symbol = Symbol(
        exchange="BINANCE", interval="1h", market="SPOT", raw_symbol="BTCUSDT"
    )
    symbol._check_exchange_metadata()  # Should not raise error

    with pytest.raises(ValueError):
        Symbol(
            exchange="INVALID", interval="1h", market="SPOT", raw_symbol="BTCUSDT"
        )._check_exchange_metadata()
