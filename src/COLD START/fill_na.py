'''Modules'''
import pandas as pd
import numpy as np

def fillna_with_mean(
    df: pd.DataFrame, target: str, group: str
) -> pd.DataFrame:
    '''Imputing missing values with product categories means'''
    new_df = df.copy()
    # calculate group means by grouping the df
    means = new_df.groupby(new_df[group]).transform('mean')[target]
    # replacing nans with those means wrt the corresponding group
    # not forgetting to round them down
    new_df[target].fillna(means.apply(np.floor), inplace=True)

    return new_df
