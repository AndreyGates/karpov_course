'''Modules'''
import numpy as np
import pandas as pd

def compare_prices(row):
    '''Comparing base and competitors' prices'''
    base, comp = row['base_price'], row['comp_price']
    # if there is a comp price
    # and base-comp difference is greater than 20%, leave the base
    if comp is None or (abs(comp-base)/base >= 0.2):
        return base
    return comp

def agg_comp_price(X: pd.DataFrame) -> pd.DataFrame:
    '''
    Aggregating competitors' prices and 
    establishing a new price
    '''
    X = X.copy()
    X.replace({'avg': 'mean', 'med': 'median'}, inplace=True)

    # iterating over items with their aggregation function
    comp_prices = []
    for (sku_agg, group) in X.groupby(['sku', 'agg']):
        # if there is no comp price
        if -1 in group['rank'].to_list():
            group['comp_price'] = None
            comp_prices.append(None)
            continue

        # if it's an usual agg func
        if sku_agg[1] != 'rnk':
            group = group.agg({"comp_price": sku_agg[1]})
            comp_prices.append(group['comp_price'])
        # if it's ranking, choose the price of the highest-ranked comp
        else:
            group = group[group['rank'] == 0]
            comp_prices.append(group['comp_price'].to_list()[0])

    ### creating a grouped df
    grouped = X.groupby(['sku', 'agg', 'base_price']).count()
    # saving new comp prices
    grouped['comp_price'] = comp_prices
    grouped = grouped.drop(['rank'], axis=1).reset_index(drop=False).replace({np.nan: None})
    # setting new prices
    grouped['new_price'] = grouped.apply(compare_prices, axis=1)

    grouped.replace({'mean': 'avg', 'median': 'med'}, inplace=True)

    return grouped
