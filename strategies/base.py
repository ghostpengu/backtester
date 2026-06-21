from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from portfolio import Order, Portfolio


class BaseStrategy(ABC):
    @abstractmethod
    def generate_orders(
        self,
        timestamp: pd.Timestamp,
        row: pd.Series,
        portfolio: Portfolio,
    ) -> list[Order]:
        raise NotImplementedError
