import pandas as pd
import pytest

from qbot.data.data_io import SEP, Symbol, TimeRange

from .conftest import check_binance_availability


def test_symbol_initialization():

    # Test symbol initialization
    symbol = Symbol("btc")
    assert str(symbol) == f"BINANCE{SEP}1h{SEP}SPOT{SEP}BTCUSDT"
    assert symbol.exchange == "BINANCE"
    assert symbol.interval == "1h"
    assert symbol.market == "SPOT"
    assert symbol.raw_symbol == "BTCUSDT"

    symbol = Symbol("BTCUSDT_PERP")
    assert str(symbol) == f"BINANCE{SEP}1h{SEP}COIN-M{SEP}BTCUSDT_PERP"
    assert symbol.exchange == "BINANCE"
    assert symbol.interval == "1h"
    assert symbol.market == "COIN-M"
    assert symbol.raw_symbol == "BTCUSDT_PERP"

    symbol = Symbol("USDT-M__BTC")
    assert str(symbol) == f"BINANCE{SEP}1h{SEP}USDT-M{SEP}BTCUSDT"
    assert symbol.exchange == "BINANCE"
    assert symbol.interval == "1h"
    assert symbol.market == "USDT-M"
    assert symbol.raw_symbol == "BTCUSDT"

    symbol = Symbol("BINANCE__1M__COIN-M__BTCUSDT")
    assert str(symbol) == f"BINANCE{SEP}1M{SEP}COIN-M{SEP}BTCUSDT"
    assert symbol.exchange == "BINANCE"
    assert symbol.interval == "1M"
    assert symbol.market == "COIN-M"
    assert symbol.raw_symbol == "BTCUSDT"

    symbol = Symbol("upbit__1m__spot__usdtkrw")
    assert str(symbol) == f"UPBIT{SEP}1m{SEP}SPOT{SEP}USDTKRW"
    assert symbol.exchange == "UPBIT"
    assert symbol.interval == "1m"
    assert symbol.market == "SPOT"
    assert symbol.raw_symbol == "USDTKRW"


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
    symbol = Symbol(f"BINANCE{SEP}1h{SEP}SPOT{SEP}BTCUSDT")
    symbol._check_exchange_metadata()  # Should not raise error

    with pytest.raises(ValueError):
        Symbol(f"INVALID{SEP}1h{SEP}SPOT{SEP}BTCUSDT")._check_exchange_metadata()
