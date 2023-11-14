'Modules'
import pandas as pd
import numpy as np

def limit_gmv(df: pd.DataFrame) -> pd.DataFrame:
    '''Postprocesses model Gross Merchandise Volume (GMV) 
       predictions upon the stock selling limits'''
    df_copy = df.copy()

    # take into account the predicted stock cannot be greater than the actual stock
    pred_stock = np.floor(df_copy['gmv'] / df_copy['price'])
    post_pred_stock = np.minimum(pred_stock, df_copy['stock'])

    # actualizing the predictions
    df_copy['gmv'] = post_pred_stock * df_copy['price']

    return df_copy
