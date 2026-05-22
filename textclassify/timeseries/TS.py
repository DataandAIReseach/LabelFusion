"""
TS - Time Series Data Loader and Manager

Loads stock price data for the FNSPID dataset and provides it in a format
ready for TSEmbedder and the timeseries branch of the fusion model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime


class TS:
    """
    Time Series data loader and manager for stock prices.
    
    Loads historical stock price data and provides it in the format
    expected by the timeseries embedding pipeline.
    
    Attributes:
        data_dir: Directory containing stock price CSV files
        stock_symbols: List of stock ticker symbols (e.g., ["AAPL", "MSFT", ...])
        data: Dict mapping stock symbol to DataFrame with price history
        date_column: Name of the date column in the CSV files
        price_column: Name of the price column to use (close, open, high, low, adj_close)
    
    Example:
        >>> ts = TS(
        ...     data_dir="./data/stock_prices",
        ...     stock_symbols=["AAPL", "MSFT", "GOOG"],
        ...     price_column="close"
        ... )
        >>> ts.load_all()
        >>> series_dict = ts.get_series_for_date("2023-01-15", window_days=100)
        >>> # series_dict = {"AAPL": [prices...], "MSFT": [prices...], ...}
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        stock_symbols: List[str],
        date_column: str = "Date",
        price_column: str = "Close",
    ):
        """
        Initialize TS data loader.
        
        Args:
            data_dir: Directory containing CSV files (one per stock)
            stock_symbols: List of stock ticker symbols
            date_column: Name of the date column in CSV files
            price_column: Price column to use ("Close", "Open", "High", "Low", "Adj Close")
        """
        self.data_dir = Path(data_dir)
        self.stock_symbols = stock_symbols
        self.date_column = date_column
        self.price_column = price_column
        self.data: Dict[str, pd.DataFrame] = {}
        
    def load_stock(self, symbol: str) -> pd.DataFrame:
        """
        Load price data for a single stock from CSV.
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            
        Returns:
            DataFrame with Date and price columns
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
        """
        csv_path = self.data_dir / f"{symbol}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Stock data not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Ensure date column is datetime
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        # Sort by date
        df = df.sort_values(self.date_column).reset_index(drop=True)
        
        return df
    
    def load_all(self) -> None:
        """Load price data for all stock symbols."""
        for symbol in self.stock_symbols:
            self.data[symbol] = self.load_stock(symbol)
            
    def get_series_for_date(
        self,
        target_date: Union[str, datetime],
        window_days: int = 100,
        stocks: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get price series ending at target_date for specified stocks.
        
        Args:
            target_date: Date for which to retrieve historical prices
            window_days: Number of days of history to include
            stocks: List of stock symbols (default: all loaded stocks)
            
        Returns:
            Dict mapping stock symbol to 1-D numpy array of prices
            
        Example:
            >>> series = ts.get_series_for_date("2023-06-15", window_days=100)
            >>> series["AAPL"].shape
            (100,)
        """
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        stocks = stocks or self.stock_symbols
        series_dict = {}
        
        for symbol in stocks:
            if symbol not in self.data:
                raise ValueError(f"Stock {symbol} not loaded. Call load_all() first.")
            
            df = self.data[symbol]
            
            # Filter to dates <= target_date
            mask = df[self.date_column] <= target_date
            df_filtered = df[mask]
            
            # Take last window_days rows
            prices = df_filtered[self.price_column].tail(window_days).values
            
            series_dict[symbol] = prices
        
        return series_dict
    
    def get_series_batch(
        self,
        dates: List[Union[str, datetime]],
        window_days: int = 100,
        stocks: Optional[List[str]] = None,
    ) -> Dict[str, List[np.ndarray]]:
        """
        Get price series for multiple dates (batch processing).
        
        Args:
            dates: List of dates for which to retrieve historical prices
            window_days: Number of days of history to include
            stocks: List of stock symbols (default: all loaded stocks)
            
        Returns:
            Dict mapping stock symbol to list of 1-D numpy arrays (one per date)
            
        Example:
            >>> dates = ["2023-01-15", "2023-02-20", "2023-03-10"]
            >>> series_batch = ts.get_series_batch(dates, window_days=100)
            >>> series_batch["AAPL"][0].shape  # First date
            (100,)
            >>> len(series_batch["AAPL"])  # Number of dates
            3
        """
        stocks = stocks or self.stock_symbols
        
        # Initialize dict with empty lists
        series_batch = {symbol: [] for symbol in stocks}
        
        # For each date, get series and append to lists
        for date in dates:
            series_dict = self.get_series_for_date(date, window_days, stocks)
            for symbol in stocks:
                series_batch[symbol].append(series_dict[symbol])
        
        return series_batch
    
    def get_date_range(self, symbol: str) -> tuple[datetime, datetime]:
        """
        Get the date range available for a stock.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Tuple of (earliest_date, latest_date)
        """
        if symbol not in self.data:
            raise ValueError(f"Stock {symbol} not loaded. Call load_all() first.")
        
        df = self.data[symbol]
        return (
            df[self.date_column].min(),
            df[self.date_column].max(),
        )
    
    def __repr__(self) -> str:
        loaded = len(self.data)
        total = len(self.stock_symbols)
        return f"TS(stocks={total}, loaded={loaded}, data_dir='{self.data_dir}')"


# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage with FNSPID dataset
    stocks = ["AAPL", "AMD", "BRK", "DIS", "GOOG", "MSFT", "NVDA", "TSLA", "WMT", "XOM"]
    
    # This would work if you have the actual data
    # ts = TS(
    #     data_dir="./data/stock_prices",
    #     stock_symbols=stocks,
    #     price_column="Close"
    # )
    # ts.load_all()
    # 
    # # Get series for a single date
    # series = ts.get_series_for_date("2023-06-15", window_days=100)
    # print(f"AAPL series shape: {series['AAPL'].shape}")
    # 
    # # Get batch for multiple dates
    # dates = ["2023-01-15", "2023-02-20", "2023-03-10"]
    # batch = ts.get_series_batch(dates, window_days=100)
    # print(f"Batch size: {len(batch['AAPL'])}")
    
    print("TS class defined - see docstrings for usage examples")