import pandas as pd
import yfinance as yf
from stockstats import wrap
from typing import Annotated
import os
from .config import get_config
from .utils import is_cache_stale
from .exceptions import DataFetchError


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
    ):
        config = get_config()

        today_date = pd.Timestamp.today()
        curr_date_dt = pd.to_datetime(curr_date)

        end_date = today_date
        start_date = today_date - pd.DateOffset(years=15)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Ensure cache directory exists
        os.makedirs(config["data_cache_dir"], exist_ok=True)

        data_file = os.path.join(
            config["data_cache_dir"],
            f"{symbol}-YFin-data-{start_date_str}-{end_date_str}.csv",
        )

        cache_ttl = config.get("cache_ttl_hours", 24)
        timeout = config.get("request_timeout", 30)

        try:
            if os.path.exists(data_file) and not is_cache_stale(data_file, cache_ttl):
                data = pd.read_csv(data_file)
                data["Date"] = pd.to_datetime(data["Date"])
            else:
                data = yf.download(
                    symbol,
                    start=start_date_str,
                    end=end_date_str,
                    multi_level_index=False,
                    progress=False,
                    auto_adjust=True,
                    timeout=timeout,
                )
                if data.empty:
                    raise DataFetchError(f"No stock data returned for {symbol}")
                data = data.reset_index()
                data.to_csv(data_file, index=False)

            df = wrap(data)
            df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            curr_date_str = curr_date_dt.strftime("%Y-%m-%d")

            df[indicator]  # trigger stockstats to calculate the indicator
            matching_rows = df[df["Date"].str.startswith(curr_date_str)]

            if not matching_rows.empty:
                indicator_value = matching_rows[indicator].values[0]
                return indicator_value
            else:
                return "N/A: Not a trading day (weekend or holiday)"
        except DataFetchError:
            raise
        except Exception as e:
            raise DataFetchError(f"Failed to fetch indicator '{indicator}' for {symbol}: {e}") from e
