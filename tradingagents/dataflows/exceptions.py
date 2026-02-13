"""Custom exceptions for the dataflows layer."""


class DataFetchError(Exception):
    """Raised when data fetching fails after all retries/fallbacks."""
    pass


class DataFetchTimeout(DataFetchError):
    """Raised when a data fetch operation times out."""
    pass


class NoDataFoundError(DataFetchError):
    """Raised when a query returns no data."""
    pass
