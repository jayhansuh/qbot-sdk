from datetime import datetime, timedelta

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


@pytest.mark.skipif(
    check_binance_availability(),
    reason="Binance API not available or restricted",
)
def test_data_field_getitem():
    # SPOT data
    # df = DataField(interval="1h", start_time="2024-01-01", end_time="2024-01-02")
    df = DataField(
        interval="1h",
        start_time="2024-01-01",
        end_time="2024-01-02",
        ticker_list=["BTCUSDT", "ETH"],
    )
    # df['BTCUSDT'].iloc[0]
    # timestamp                       2024-01-01 00:00:00
    # open                                       42283.58
    # high                                       42554.57
    # low                                        42261.02
    # close                                      42475.23
    # volume                                   1271.68108
    # close_time                            1704070799999
    # quote_asset_volume                  53957248.973789
    # number_of_trades                              47134
    # taker_buy_base_asset_volume               682.57581
    # taker_buy_quote_asset_volume        28957416.819645
    # ignore                                            0
    # Name: 0, dtype: object
    assert df["BTCUSDT"].iloc[0]["timestamp"] == pd.Timestamp("2024-01-01 00:00:00")
    assert df["BTCUSDT"].iloc[0]["open"] == 42283.58
    assert df["BTCUSDT"].iloc[0]["high"] == 42554.57
    assert df["BTCUSDT"].iloc[0]["low"] == 42261.02
    assert df["BTCUSDT"].iloc[0]["close"] == 42475.23
    assert df["BTCUSDT"].iloc[0]["volume"] == 1271.68108
    assert df["BTCUSDT"].iloc[0]["close_time"] == 1704070799999
    assert df["BTCUSDT"].iloc[0]["quote_asset_volume"] == 53957248.973789
    assert df["BTCUSDT"].iloc[0]["number_of_trades"] == 47134
    assert df["BTCUSDT"].iloc[0]["taker_buy_base_asset_volume"] == 682.57581
    assert df["BTCUSDT"].iloc[0]["taker_buy_quote_asset_volume"] == 28957416.819645
    assert df["BTCUSDT"].iloc[0]["ignore"] == 0


# TO DO
# def test_data_field_getitem_perp():
#     # Add df['BTCUSD_PERP']
#     # Add df['USDT-M__BTCUSDT']
#     today = datetime.now()
#     day_ago = today - timedelta(days=1)
#     df = DataField(interval="1h", start_time=day_ago, end_time=today)
#     assert df["BTCUSD_PERP"].iloc[0]["timestamp"]
#     assert df["USDT-M__BTCUSDT"].iloc[0]["timestamp"]
