"""Tests for data vendor routing and configuration."""

import pytest
from tradingagents.dataflows.interface import (
    get_category_for_method,
    get_vendor,
    VENDOR_METHODS,
    TOOLS_CATEGORIES,
    VENDOR_LIST,
)
from tradingagents.dataflows.config import set_config, get_config, initialize_config
from tradingagents.dataflows.exceptions import DataFetchError, DataFetchTimeout, NoDataFoundError


class TestCategoryMapping:
    def test_stock_data_in_core_apis(self):
        assert get_category_for_method("get_stock_data") == "core_stock_apis"

    def test_indicators_in_technical(self):
        assert get_category_for_method("get_indicators") == "technical_indicators"

    def test_fundamentals_in_fundamental_data(self):
        assert get_category_for_method("get_fundamentals") == "fundamental_data"

    def test_news_in_news_data(self):
        assert get_category_for_method("get_news") == "news_data"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_category_for_method("nonexistent_method")


class TestVendorMethods:
    def test_all_methods_have_at_least_one_vendor(self):
        for method, vendors in VENDOR_METHODS.items():
            assert len(vendors) > 0, f"{method} has no vendor implementations"

    def test_all_category_methods_are_in_vendor_methods(self):
        for category, info in TOOLS_CATEGORIES.items():
            for method in info["tools"]:
                assert method in VENDOR_METHODS, f"{method} from {category} not in VENDOR_METHODS"

    def test_vendor_list_matches_implementations(self):
        for method, vendors in VENDOR_METHODS.items():
            for vendor in vendors:
                assert vendor in VENDOR_LIST, f"Vendor '{vendor}' for {method} not in VENDOR_LIST"


class TestVendorResolution:
    def test_default_vendor_is_yfinance(self, test_config):
        set_config(test_config)
        vendor = get_vendor("core_stock_apis")
        assert vendor == "yfinance"

    def test_tool_level_override_takes_precedence(self, test_config):
        test_config["tool_vendors"] = {"get_stock_data": "alpha_vantage"}
        set_config(test_config)
        vendor = get_vendor("core_stock_apis", method="get_stock_data")
        assert vendor == "alpha_vantage"

    def test_category_fallback_when_no_tool_override(self, test_config):
        test_config["data_vendors"]["fundamental_data"] = "alpha_vantage"
        test_config["tool_vendors"] = {}
        set_config(test_config)
        vendor = get_vendor("fundamental_data", method="get_fundamentals")
        assert vendor == "alpha_vantage"


class TestExceptions:
    def test_data_fetch_error_is_exception(self):
        assert issubclass(DataFetchError, Exception)

    def test_data_fetch_timeout_is_subclass(self):
        assert issubclass(DataFetchTimeout, DataFetchError)

    def test_no_data_found_is_subclass(self):
        assert issubclass(NoDataFoundError, DataFetchError)

    def test_exception_messages(self):
        e = DataFetchError("test error")
        assert str(e) == "test error"

    def test_exception_chaining(self):
        try:
            try:
                raise ValueError("original")
            except ValueError as ve:
                raise DataFetchError("wrapped") from ve
        except DataFetchError as e:
            assert str(e) == "wrapped"
            assert isinstance(e.__cause__, ValueError)
