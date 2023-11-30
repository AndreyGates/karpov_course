'''MODULES'''
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

def elasticity_sku(item: pd.DataFrame) -> float:
    '''
    Calculates the elasticity of a single item
    (R2 coefficient between prices and quantity)
    '''
    # extracting the independent (price) and dependent (quantity) vars
    price, log_qty = item['price'].values[:, np.newaxis], np.log1p(item['qty'].values)

    # fitting to a linear regression model
    model = LinearRegression()
    model.fit(price, log_qty)
    log_qty_pred = model.predict(price)

    # r2 coefficient
    return r2_score(log_qty, log_qty_pred)


def elasticity_df(df: pd.DataFrame) -> pd.DataFrame:
    '''Calculates the elasticity of items'''
    # applying the calculations for each item
    elasticities = df.copy().groupby('sku').apply(elasticity_sku).reset_index(drop=False)
    elasticities.columns = ['sku', 'elasticity']

    return elasticities
