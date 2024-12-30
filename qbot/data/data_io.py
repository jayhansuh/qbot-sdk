import os
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Any, Optional, Tuple, Union

import pandas as pd  # type: ignore

EXCHANGE_METADATA = {
    # TODO:
    # Add more exchanges
    # Make it dynamically updated
    "BINANCE": {
        "INTERVAL_LIST": [
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "12h",
            "1d",
            "3d",
            "1w",
            "1M",
        ],
        "MARKET_LIST": [
            "SPOT",
            "USDT-M",
            "COIN-M",
        ],
        "SYMBOL_LIST": [
            "ETHUSDT",
            "BTCUSDT",
            "SOLUSDT",
            "ADAUSDT",
            "DOGEUSDT",
            "XRPUSDT",
            "LTCUSDT",
            "BCHUSDT",
            "EOSUSDT",
            "ETCUSDT",
            "LINKUSDT",
            "XLMUSDT",
            "XMRUSDT",
            "XRPUSDT",
            "XTZUSDT",
            "ZECUSDT",
        ],
    }
}


@dataclass
class TimeRange:
    start_date: datetime
    end_date: datetime = datetime.now()

    def __post_init__(self) -> None:
        if not isinstance(self.start_date, datetime):
            self.start_date = pd.to_datetime(self.start_date)
        if self.end_date:
            if not isinstance(self.end_date, datetime):
                self.end_date = pd.to_datetime(self.end_date)

    def __str__(self) -> str:
        return f'{self.start_date.strftime("%Y-%m-%d")}_{self.end_date.strftime("%Y-%m-%d")}'

    def to_milliseconds(self) -> Tuple[int, int]:
        start_timestamp_ms = int(self.start_date.timestamp() * 1000)
        end_timestamp_ms = int((self.end_date or datetime.now()).timestamp() * 1000)
        return start_timestamp_ms, end_timestamp_ms

    def to_UTC_str(self) -> Tuple[str, str]:
        start_utc_str = self.start_date.strftime("%Y-%m-%d")
        end_utc_str = self.end_date.strftime("%Y-%m-%d")
        return start_utc_str, end_utc_str


class Symbol:
    def __init__(
        self,
        exchange: str = "BINANCE",
        interval: str = "1h",
        market: str = "SPOT",
        raw_symbol: str = "BTCUSDT",
    ):
        self.exchange = exchange
        self.interval = interval
        self.market = market
        self.raw_symbol = raw_symbol

    @staticmethod
    def parse_str(ticker: str, interval: str = "1h") -> Any:
        if "_" in ticker:
            exchange, _interval, market, raw_symbol = ticker.split("_")
            if _interval != interval:
                raise ValueError(f"Interval {_interval} does not match {interval}")
            return Symbol(
                exchange=exchange,
                interval=interval,
                market=market,
                raw_symbol=raw_symbol,
            )
        else:
            if not (
                ticker.endswith("USDT")
                or ticker.endswith("USDC")
                or ticker.endswith("KRW")
            ):
                print(f"Adding USDT to ticker: {ticker}")
                ticker += "USDT"
            return Symbol(
                exchange="BINANCE", interval=interval, market="SPOT", raw_symbol=ticker
            )

    def __str__(self) -> str:
        if not hasattr(self, "symbol_str"):
            self.symbol_str = (
                f"{self.exchange}_{self.interval}_{self.market}_{self.raw_symbol}"
            )
        return self.symbol_str

    def get_local_folder(self) -> str:
        """Get the local folder for the symbol"""
        if not hasattr(self, "local_folder"):
            base_dir = os.environ.get(
                "QBOT_DATA_DIR", os.path.dirname(os.path.abspath(__file__))
            )
            self.local_folder = os.path.join(base_dir, str(self))
        return self.local_folder

    def set(self, symbol_str: str) -> None:
        self.exchange, self.interval, self.market, self.raw_symbol = symbol_str.split(
            "_"
        )

    def _check_exchange_metadata(self) -> None:
        if self.exchange not in EXCHANGE_METADATA:
            raise ValueError(f"Exchange {self.exchange} not supported")
        # if self.interval not in EXCHANGE_METADATA[self.exchange]["INTERVAL_LIST"]:
        #     raise ValueError(
        #         f"Interval {self.interval} not supported for exchange {self.exchange}"
        #     )
        if self.market not in EXCHANGE_METADATA[self.exchange]["MARKET_LIST"]:
            raise ValueError(
                f"Market {self.market} not supported for exchange {self.exchange}"
            )
        # if self.raw_symbol not in EXCHANGE_METADATA[self.exchange]["SYMBOL_LIST"]:
        #     raise ValueError(
        #         f"Symbol {self.raw_symbol} not supported for exchange {self.exchange}"
        #     )

    def _fill_missing_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        pd_timedelta = pd.Timedelta(self.interval)
        index = 0
        missing_timestamps = []
        while index < len(df) - 1:
            timestamp = df["timestamp"].iloc[index] + pd_timedelta
            while timestamp < df["timestamp"].iloc[index + 1]:
                # insert missing timestamp null row
                missing_timestamps.append(timestamp)
                timestamp += pd_timedelta
            index += 1
        df = pd.concat(
            [df, pd.DataFrame(missing_timestamps, columns=["timestamp"])],
            ignore_index=True,
        )
        df.sort_values(by="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        if missing_timestamps:
            print(
                f"Warning: {len(missing_timestamps)} missing timestamps found and filled with NA"
            )
            print(missing_timestamps)
        return df

    def _check_timestamp(
        self, df: pd.DataFrame, time_range: Union[TimeRange, None] = None
    ) -> None:

        if "timestamp" not in df.columns:
            raise ValueError(f"Timestamp column not found in DataFrame")
        # timestamp should be in ascending order
        if not df["timestamp"].is_monotonic_increasing:
            raise ValueError(
                f"Timestamp is not in ascending order: {df['timestamp'].min()} - {df['timestamp'].max()}"
            )
        # timestamp should be unique
        if df["timestamp"].duplicated().any():
            raise ValueError(
                f"Timestamp is not unique: {df['timestamp'][df['timestamp'].duplicated()]}"
            )

        # Check if the timestamp is in the correct time range
        if time_range:
            if df["timestamp"].iloc[0] != time_range.start_date:
                raise ValueError(
                    f"Timestamp is not in the correct time range: {df['timestamp'].iloc[0]} - {time_range.start_date}"
                )
            if df["timestamp"].iloc[-1] != time_range.end_date:
                raise ValueError(
                    f"Timestamp is not in the correct time range: {df['timestamp'].iloc[-1]} - {time_range.end_date}"
                )

        # Check if the timestamp is in the correct interval
        if self.interval == "1M":
            # # 1M is a special case, we need to check if the one month is next month
            # next_month_series = df['timestamp'].apply(lambda x: x + pd.Timedelta(days=x.days_in_month)).shift(1)
            # if not (next_month_series[1:] == df['timestamp'][:-1]).all():
            #     raise ValueError(f"Timestamp is not next month: {df['timestamp'].min()} - {df['timestamp'].max()}")
            raise NotImplementedError(f"Checking timestamp for 1M is not implemented")
        else:
            pd_timedelta = pd.Timedelta(self.interval)
            diff_series = df["timestamp"].diff()[1:]
            if not (diff_series == pd_timedelta).all():
                # print row with diff_series != pd_timedelta
                for i in range(1, len(df)):
                    if df["timestamp"].diff()[i] != pd_timedelta:
                        print(df.iloc[max(0, i - 10) : min(len(df), i + 10)])
                raise ValueError(
                    f"Timestamp delta is not {pd_timedelta}: {diff_series.value_counts()}"
                )

    def fetch_data(
        self,
        start_datetime: datetime,
        end_datetime: Optional[datetime] = None,
        client: Any = None,
    ) -> pd.DataFrame:
        """Fetch data from the exchange"""

        time_range = TimeRange(start_date=start_datetime, end_date=end_datetime)

        self._check_exchange_metadata()

        if self.exchange != "BINANCE":
            raise ValueError(f"Exchange {self.exchange} not supported")
        if not client:
            from binance.client import Client

            client = Client()

        # Fetch new data based on market type
        from binance.enums import HistoricalKlinesType

        binance_client_get_hist_func = client.get_historical_klines
        if self.market == "SPOT":
            klines_type = HistoricalKlinesType.SPOT
        elif self.market == "USDT-M":
            klines_type = HistoricalKlinesType.FUTURES
        elif self.market == "COIN-M":
            klines_type = HistoricalKlinesType.FUTURES_COIN
        else:
            raise ValueError(f"Invalid market: {symbol.market}")

        start_ts, end_ts = time_range.to_milliseconds()
        klines = binance_client_get_hist_func(
            symbol=self.raw_symbol,
            interval=self.interval.lower(),
            start_str=start_ts,
            end_str=end_ts,
            klines_type=klines_type,
        )
        # Convert new data to DataFrame
        df = pd.DataFrame(
            klines,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )

        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = self._fill_missing_timestamp(df)
        self._check_timestamp(df, time_range)

        # Convert to float, keeping NA values
        float_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ]
        df[float_cols] = df[float_cols].apply(pd.to_numeric, errors="coerce")

        # Convert to int, keeping NA values
        int_cols = ["close_time", "number_of_trades", "ignore"]
        df[int_cols] = (
            df[int_cols].apply(pd.to_numeric, errors="coerce").astype("Int64")
        )

        return df

    def _get_filename(self, time_range: TimeRange) -> str:
        return os.path.join(self.get_local_folder(), str(time_range) + ".parquet")

    def save_data(
        self,
        df: pd.DataFrame,
        start_datetime: datetime = None,
        end_datetime: datetime = None,
    ) -> None:
        # Save to parquet
        time_range = TimeRange(
            start_date=start_datetime if start_datetime else df["timestamp"].iloc[0],
            end_date=end_datetime if end_datetime else df["timestamp"].iloc[-1],
        )
        self._check_timestamp(df, time_range=time_range)
        filename = self._get_filename(time_range)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_parquet(filename)

    def load_data(
        self,
        start_datetime: datetime,
        end_datetime: datetime = None,
        client: Any = None,
    ) -> pd.DataFrame:
        self._check_exchange_metadata()
        # Get the local filename
        time_range = TimeRange(start_date=start_datetime, end_date=end_datetime)
        filename = self._get_filename(time_range)
        if os.path.exists(filename):
            print(f"Loading data from {filename}")
            df = pd.read_parquet(filename)
        else:
            print(f"Fetching data from {start_datetime} to {end_datetime}")
            df = self.fetch_data(
                start_datetime=start_datetime, end_datetime=end_datetime, client=client
            )
            print(f"Saving data to {filename}")
            self.save_data(df, start_datetime=start_datetime, end_datetime=end_datetime)
        return df


if __name__ == "__main__":
    symbol = Symbol(
        exchange="BINANCE", interval="1h", market="SPOT", raw_symbol="BTCUSDT"
    )
    print(symbol)
    if "BINANCE_1h_SPOT_BTCUSDT" != str(symbol):
        raise ValueError(f"Symbol string mismatch: {str(symbol)}")
    print(symbol.get_local_folder())
    symbol.set("BINANCE_1h_SPOT_BTCUSDT")
    if not symbol.get_local_folder().endswith("qbot-sdk/data/BINANCE_1h_SPOT_BTCUSDT"):
        raise ValueError(f"Local folder mismatch: {symbol.get_local_folder()}")

    # Test saving data
    df = symbol.fetch_data(start_datetime="2024-01-01", end_datetime="2024-03-01")
    symbol.save_data(df, start_datetime="2024-01-01", end_datetime="2024-03-01")
    if not os.path.exists("BINANCE_1h_SPOT_BTCUSDT/2024-01-01_2024-03-01.parquet"):
        raise ValueError(
            f"Local parquet file does not exist: {symbol.get_local_folder()}/2024-01-01_2024-03-01.parquet"
        )

    # Test loading data
    df = symbol.load_data(start_datetime="2023-01-01", end_datetime="2023-03-01")
    if df.empty:
        raise ValueError(f"Loaded data is empty: {df.head()}")
    print(df.head(), df.tail())
