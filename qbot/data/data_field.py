import fnmatch
from datetime import datetime, timedelta
from typing import List, Tuple, Union

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
        self.last_updated = None
        self.min_update_interval = min_update_interval
        self.df_dict = {}
        self.symbol_list = []
        self.add_ticker(ticker_list)

    def add_ticker(self, ticker: Union[str, List[str]]) -> None:
        if isinstance(ticker, str):
            new_symbol = Symbol.parse_str(ticker, interval=self.interval)
            self.df_dict[str(new_symbol)] = new_symbol.load_data(
                start_datetime=self.start_time,
                end_datetime=self.end_time,
                client=self.binance_client,
            )
            self.symbol_list.append(new_symbol)
        elif isinstance(ticker, list):
            for t in ticker:
                self.add_ticker(t)
        else:
            raise ValueError(f"Invalid ticker: {ticker}")

    def __getitem__(
        self, key: Union[Symbol, str, List[any], int]
    ) -> Union[Tuple[str, pd.DataFrame], List[Tuple[str, pd.DataFrame]]]:

        if isinstance(key, str):

            if key in self.df_dict:
                return self.df_dict[key]

            key = str(Symbol.parse_str(key, interval=self.interval))
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

        elif isinstance(key, Symbol):
            return self.__getitem__(str(key))
        elif isinstance(key, list) or isinstance(key, tuple):
            # return [(k, self.df_dict[k]) for k in key]
            return [self.df_dict[k] for k in key]
        elif isinstance(key, int):
            # return (self.symbol_list[key], self.df_dict[self.symbol_list[key]])
            return self.df_dict[self.symbol_list[key]]
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
