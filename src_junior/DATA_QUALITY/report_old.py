"""DQ Report."""

from typing import Dict, List, Tuple, Union, Callable
from dataclasses import dataclass
from user_input.metrics import Metric

import pandas as pd

def catch(func, *args, **kwargs):
    """Handling exceptions"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return e


def memoize(func: Callable) -> Callable:
    """
    Memoize function
    """
    cache = {}
    def wrapper(*args, **kwargs) -> object:
        """
        The wrapper collect all the agrument values 
        and the return value of a function
        """
        # gathering positional and keyword agrs
        # (priorly resolving 'unhashable type: list' issue)
        key = str(args) + str(kwargs)

        # if the current arguments are new,
        # calculate the function cache them
        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]
    return wrapper

LimitType = Dict[str, Tuple[float, float]]
CheckType = Tuple[str, Metric, LimitType]

@dataclass
class Report:
    """DQ report class."""

    checklist: List[CheckType]
    engine: str = "pandas"

    @memoize
    def fit(self, tables: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report."""
        self.report_ = {}
        report = self.report_

        # Check if engine supported
        if self.engine != "pandas":
            raise NotImplementedError("Only pandas API currently supported!")

        t_names = sorted(list(set(tables.keys())))
        #t_names = [t_names[-1]]+t_names[:-1]
        #t_names = ['big_table', 'sales', 'views']
        report['title'] = f'DQ Report for tables {t_names}'
        result = pd.DataFrame(columns=['table_name', 'metric', 'limits',
                                       'values', 'status', 'error'])
        for t_name, table in tables.items():
            # a report for a table
            table_result = pd.DataFrame()
            # leaving only metrics for the table
            table_checklist = list(filter(lambda tup: tup[0] == t_name,
                                          self.checklist))
            # metrics, the limits and metric values for a table
            metrics = [str(tup[1]) for tup in table_checklist]
            limits = [tup[2] for tup in table_checklist]
            # (here we also handle exceptions)
            values = [catch(func=lambda: tup[1](table))
                      for tup in table_checklist]
            # specific values to check
            values_to_check = [(vals.keys() & lims.keys()) if isinstance(vals, dict) else {}
                               for vals, lims in zip(values, limits)]
            values_to_check = list(list(vals) for vals in values_to_check)
            # checking if metrics are passed
            status, error = [], []
            for i in range(len(metrics)):
                stat, err = '.', ''
                # if there was an exception calculating a metric
                if isinstance(values[i], Exception):
                    stat, err = 'E', type(values[i]).__name__
                # otherwise, compare with the boundaries
                else:
                    for value in values_to_check[i]:
                        bounds = limits[i][value]
                        # in case out of the boundaries, fail
                        if not(values[i][value] >= bounds[0] and\
                           values[i][value] <= bounds[1]):
                            stat = 'F'
                status.append(stat)
                error.append(err)

            # gathering stats
            table_result['table_name'] = [t_name for _ in range(len(metrics))]
            table_result['metric'] = metrics
            table_result['limits'] = [str(limit) for limit in limits]
            table_result['values'] = [value if isinstance(value, dict) else {}
                                      for value in values]
            table_result['status'] = status
            table_result['error'] = error

            # aggregate into the general stats
            result = pd.concat([result, table_result], axis=0)

        report['result'] = result

        # numerical stats
        report['passed'], report['passed_pct'] =\
            result['status'].to_list().count('.'),\
            result['status'].to_list().count('.')/len(result) * 100
        report['failed'], report['failed_pct'] =\
            result['status'].to_list().count('F'),\
            result['status'].to_list().count('F')/len(result) * 100
        report['errors'], report['errors_pct'] =\
            result['status'].to_list().count('E'),\
            result['status'].to_list().count('E')/len(result) * 100

        report['total'] = len(result)
        return report

    def to_str(self) -> None:
        """Convert report to string format."""
        report = self.report_

        msg = (
            "This Report instance is not fitted yet. "
            "Call 'fit' before using this method."
        )

        assert isinstance(report, dict), msg

        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.max_colwidth", 20)
        pd.set_option("display.width", 1000)

        return (
            f"{report['title']}\n\n"
            f"{report['result']}\n\n"
            f"Passed: {report['passed']} ({report['passed_pct']}%)\n"
            f"Failed: {report['failed']} ({report['failed_pct']}%)\n"
            f"Errors: {report['errors']} ({report['errors_pct']}%)\n"
            "\n"
            f"Total: {report['total']}"
        )
