import fnmatch
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, List, Tuple, Union

import pandas as pd
from binance.client import Client

from qbot.data.data_io import Symbol


class DataField:
    def __init__(
        self,
        binance_client: Client = None,
        interval: str = "1h",
        start_time: datetime = None,
        end_time: datetime = datetime.now(),
        ticker_list: List[str] = [],
        min_update_interval: timedelta = None,
    ):
        self.binance_client = binance_client or Client()
        self.interval = interval
        self.end_time = end_time
        self.start_time = start_time or (end_time - timedelta(hours=300))
        self.timestamp = None
        self.last_updated = None
        self.min_update_interval = min_update_interval
        self.df_dict = {}
        self.symbol_list = []
        self.add_ticker(ticker_list)

    def _symbol_init(self, ticker: str) -> Symbol:
        symbol = Symbol(ticker)
        symbol.interval = self.interval
        return symbol

    def _set_symbol_data(self, symbol: Symbol) -> None:
        symbol_str = str(symbol)
        self.df_dict[symbol_str] = symbol.load_data(
            start_datetime=self.start_time,
            end_datetime=self.end_time,
            client=self.binance_client,
        )

        new_timestamp = self.df_dict[symbol_str]["timestamp"]
        if self.timestamp is not None:
            # Check if the new timestamp is the same as the old timestamp
            if not self.timestamp.equals(new_timestamp):
                print("Warning: Timestamp mismatch in a data field")
        self.timestamp = new_timestamp

    def update(self, symbol_list: List[Any], max_workers: int = 10) -> None:
        if len(symbol_list) == 0:
            return
        if max_workers > len(symbol_list):
            # print(f"Warning: max_workers is greater than the number of symbols to update. Setting max_workers to {len(symbol_list)}")
            max_workers = len(symbol_list)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self._set_symbol_data, symbol_list)

    def add_ticker(self, ticker: Union[str, List[str]], max_workers: int = 10) -> None:
        if isinstance(ticker, str):
            new_symbol = self._symbol_init(ticker)
            self.symbol_list.append(new_symbol)
            self._set_symbol_data(new_symbol)
        elif isinstance(ticker, list):
            new_symbol_list = [self._symbol_init(t) for t in ticker]
            self.symbol_list.extend(new_symbol_list)
            self.update(new_symbol_list, max_workers=max_workers)
        else:
            raise ValueError(f"Invalid ticker: {ticker}")

    def __getitem__(
        self, key: Union[Symbol, str, List[any], int]
    ) -> Union[Tuple[str, pd.DataFrame], List[Tuple[str, pd.DataFrame]]]:

        if isinstance(key, str):

            if key in self.df_dict:
                return self.df_dict[key]

            key = str(self._symbol_init(key))
            if "*" in key or "?" in key or "[" in key or "{" in key:
                fnmatch_list = [
                    k for k in self.df_dict.keys() if fnmatch.fnmatch(k, key)
                ]
                # return [(k, self.df_dict[k]) for k in fnmatch_list]
                return [self.df_dict[k] for k in fnmatch_list]

            if key not in self.df_dict:
                self.add_ticker(key)
            # return (key, self.df_dict[key])
            return self.df_dict[key]

        elif isinstance(key, list) or isinstance(key, tuple):
            # return [(k, self.df_dict[k]) for k in key]
            return [self.df_dict[k] for k in key]
        elif isinstance(key, int):
            # return (self.symbol_list[key], self.df_dict[self.symbol_list[key]])
            return self.df_dict[self.symbol_list[key]]
        elif isinstance(key, object):
            return self.__getitem__(str(key))
        else:
            raise ValueError(f"Invalid key: {key}")


if __name__ == "__main__":
    df = DataField(
        interval="1h",
        # start_time=datetime(2024, 1, 1), # This causes timezone error, need to be updated
        start_time="2024-01-01",
        end_time="2024-06-01",
        ticker_list=["BTCUSDT", "ETHUSDT", "XRPUSDT"],
    )
    print(df["*"])
