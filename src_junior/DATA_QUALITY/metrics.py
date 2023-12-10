"""Metrics."""
from typing import Any, Dict, Union, List
from dataclasses import dataclass
import datetime

import numpy as np
import pandas as pd
import pyspark.sql as ps

from pyspark.sql.functions import col, isnan, to_date

@dataclass
class Metric:
    """Base class for Metric"""

    def __call__(self, df: Union[pd.DataFrame, ps.DataFrame]) -> Dict[str, Any]:
        if isinstance(df, pd.DataFrame):
            return self._call_pandas(df)

        if isinstance(df, ps.DataFrame):
            return self._call_pyspark(df)

        msg = (
            f"Not supported type of arg 'df': {type(df)}. "
            "Supported types: pandas.DataFrame, "
            "pyspark.sql.dataframe.DataFrame"
        )
        raise NotImplementedError(msg)

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        return {}


@dataclass
class CountTotal(Metric):
    """Total number of rows in DataFrame"""

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {"total": len(df)}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        return {"total": df.count()}


@dataclass
class CountZeros(Metric):
    """Number of zeros in choosen column"""

    column: str

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] == 0)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        k = df.filter(col(self.column) == 0).count()
        return {"total": n, "count": k, "delta": k / n}

@dataclass
class CountNull(Metric):
    """Number of empty values in choosen columns"""
    columns: List[str]
    aggregation: str = "any"  # either "all", or "any"
    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        # No. of records
        n = len(df)
        if self.aggregation == "all":
            k = sum(df[self.columns].isnull().apply(all, axis=1))
        elif self.aggregation == "any":
            k = sum(df[self.columns].isnull().apply(any, axis=1))
        else:
            raise ValueError("Given aggregation doesn't exist.")

        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        if self.aggregation == "all":
            mask_all_empty = df.na.drop(how="all", subset=self.columns)
            k = n - mask_all_empty.count()
        elif self.aggregation == "any":
            mask_any_empty = df.na.drop(how="any", subset=self.columns)
            k = n - mask_any_empty.count()
        else:
            raise ValueError("Given aggregation doesn't exist.")

        return {"total": n, "count": k, "delta": k / n}

@dataclass
class CountDuplicates(Metric):
    """Number of duplicates in choosen columns"""
    columns: List[str]
    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        # No. of duplicates
        k = df[self.columns].duplicated().sum()
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        k = df.groupBy(self.columns).count().filter(col("count") > 1).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountValue(Metric):
    """Number of values in choosen column"""

    column: str
    value: Union[str, int, float]

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = df[self.column].to_list().count(self.value)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        k = df.filter(col(self.column) == self.value).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowValue(Metric):
    """Number of values below threshold"""

    column: str
    value: float
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if self.strict:
            k = sum(df[self.column] < self.value)
        else:
            k = sum(df[self.column] <= self.value)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        if self.strict:
            k = df.filter(~isnan(self.column) & col(self.column).isNotNull())\
                .filter(col(self.column) < self.value).count()
        else:
            k = df.filter(~isnan(self.column) & col(self.column).isNotNull())\
                .filter(col(self.column) <= self.value).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowColumn(Metric):
    """Count how often column X below Y"""

    column_x: str
    column_y: str
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if self.strict:
            k = sum(df[self.column_x] < df[self.column_y])
        else:
            k = sum(df[self.column_x] <= df[self.column_y])
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        # dropping null and nan values due to PySpark shticks
        #df_dropna = df.na.drop(subset=[self.column_x, self.column_y])
        if self.strict:
            k = df.filter(~isnan(self.column_x) & col(self.column_x).isNotNull())\
                .filter(~isnan(self.column_y) & col(self.column_y).isNotNull())\
                .filter(col(self.column_x) < col(self.column_y)).count()
        else:
            k = df.filter(~isnan(self.column_x) & col(self.column_x).isNotNull())\
                .filter(~isnan(self.column_y) & col(self.column_y).isNotNull())\
                .filter(col(self.column_x) <= col(self.column_y)).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountRatioBelow(Metric):
    """Count how often X / Y below Z"""

    column_x: str
    column_y: str
    column_z: str
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if self.strict:
            k = sum(df[self.column_x]/df[self.column_y] < df[self.column_z])
        else:
            k = sum(df[self.column_x]/df[self.column_y] <= df[self.column_z])
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        # dropping null and nan values due to PySpark shticks
        #df_dropna = df.na.drop(subset=[self.column_x, self.column_y])
        if self.strict:
            k = df.filter(~isnan(self.column_x) & col(self.column_x).isNotNull())\
                .filter(~isnan(self.column_y) & col(self.column_y).isNotNull())\
                .filter(~isnan(self.column_z) & col(self.column_z).isNotNull())\
                .filter(col(self.column_x)/col(self.column_y) < col(self.column_z)).count()
        else:
            k = df.filter(~isnan(self.column_x) & col(self.column_x).isNotNull())\
                .filter(~isnan(self.column_y) & col(self.column_y).isNotNull())\
                .filter(~isnan(self.column_z) & col(self.column_z).isNotNull())\
                .filter(col(self.column_x)/col(self.column_y) <= col(self.column_z)).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountCB(Metric):
    """Calculate lower/upper bounds for N%-confidence interval"""

    column: str
    conf: float = 0.95

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        # conf interval (as quantiles [(1-conf)/2; (1+conf)/2])
        lcb = np.quantile(df[self.column], (1-self.conf)/2)
        ucb = np.quantile(df[self.column], (1+self.conf)/2)
        return {"lcb": lcb, "ucb": ucb}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        lcb = np.quantile(df.select(self.column).collect(), (1-self.conf)/2)
        ucb = np.quantile(df.select(self.column).collect(), (1+self.conf)/2)
        return {"lcb": lcb, "ucb": ucb}


@dataclass
class CountLag(Metric):
    """A lag between latest date and today"""

    column: str
    fmt: str = "%Y-%m-%d"

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        # today
        a = datetime.date.today()
        # latest date
        b = pd.to_datetime(df[self.column], format=self.fmt).dt.date.max()
        # lag
        lag = abs((a - b).days)
        return {"today": str(a), "last_day": str(b), "lag": lag}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        # today
        a = datetime.date.today()
        # latest date
        b = df.select(to_date(col(self.column))\
                        .alias("Date")).selectExpr("max(Date)").first()[0]
        b = datetime.datetime.strptime(str(b), self.fmt).date()
        # lag
        lag = abs((a - b).days)
        return {"today": str(a), "last_day": str(b), "lag": lag}
