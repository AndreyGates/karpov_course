"""Calculating attribution metrics"""
from typing import List
import pandas as pd

# marketing channels
CHANNELS = ['social_media', 'mobile_ads', 'bloggers', 'context_ads']

def convert_into_user_paths(events: pd.DataFrame) -> pd.DataFrame:
    """Converts a table of events into a table of user paths"""
    events['event'] = events.apply(lambda row:\
                                    [row['week'], row['channel'], row['is_purchased'], row['gmv']],\
                                    axis=1)
    events_trunc = events.drop(['channel', 'is_purchased', 'gmv'], axis=1)\
        .sort_values(['user_id', 'week']).reset_index(drop=True)

    grouped_events = events_trunc.groupby('user_id')['event'].agg(list).reset_index()
    grouped_events.columns = ['user_id', 'user_path']

    user_paths = grouped_events.iloc[:, :]
    return user_paths

def parse_user_path(user_path: pd.Series) -> pd.DataFrame:
    '''
    Parses a user path towards a puchase (with convertion into a purchase)
    by aggregating all the user's steps.

    Basically, returns the info about all a user's purchases
    '''
    user_id, user_path = user_path

    purchases = []
    current_purchase = []
    # do the splitting of user purchases
    for event in user_path:
        current_purchase.append(event)
        # the point where a purchase was made
        if event[2] == 1:
            # split the current part and add to the split parts list
            purchases.append(current_purchase)
            current_purchase = []

    parsed_purchases = []
    # collecting info about the purchase path in one place
    for purchase in purchases:
        channels = [event[1] for event in purchase]
        week, total_gmv = purchase[-1][0], purchase[-1][3]
        parsed_purchases.append({'week': week, 'channels': channels, 'total_gmv': total_gmv})

    # styling the final report about the user's purchases
    parsed_purchases = pd.DataFrame(parsed_purchases)
    parsed_purchases['user_id'] = user_id
    parsed_purchases[CHANNELS] = [0] * len(CHANNELS)
    parsed_purchases = parsed_purchases[['week', 'user_id', 'channels', *CHANNELS, 'total_gmv']]

    return parsed_purchases

def collect_all_purchases(user_paths: pd.DataFrame) -> pd.DataFrame:
    """Collect purchases of all users into one dataframe"""
    purchase_report: pd.DataFrame = pd.concat([parse_user_path(user_path.tolist())
                                    for _, user_path in user_paths.iterrows()], axis=0)

    return purchase_report.sort_values(['week', 'user_id']).reset_index(drop=True)


def last_touch_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate last touch attribution"""
    # step 1: generate the overall purchase report
    user_paths = convert_into_user_paths(events)
    purchase_report = collect_all_purchases(user_paths)
    # step 2: calculate the traffic attribution
    for index, row in purchase_report.iterrows():
        purchase_report.at[index, row['channels'][-1]] = row['total_gmv']

    # type casting
    purchase_report[CHANNELS + ['total_gmv']] =\
        purchase_report[CHANNELS + ['total_gmv']].astype(int)

    # karpov cracking jokes
    purchase_report['channels'] = purchase_report\
        .apply(lambda row: [row['week'], row['channels'][-1], 1, row['total_gmv']], axis=1)
    purchase_report.rename(columns={'channels': 'event'}, inplace=True)

    attribution = purchase_report.copy()#.drop(['channels'], axis=1)
    return attribution

def first_touch_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate first touch attribution"""
    # step 1: generate the overall purchase report
    user_paths = convert_into_user_paths(events)
    purchase_report = collect_all_purchases(user_paths)
    # step 2: calculate the traffic attribution
    for index, row in purchase_report.iterrows():
        purchase_report.at[index, row['channels'][0]] = row['total_gmv']

    # type casting
    purchase_report[CHANNELS + ['total_gmv']] =\
        purchase_report[CHANNELS + ['total_gmv']].astype(int)

    attribution = purchase_report.drop(['channels'], axis=1)
    return attribution

def linear_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate linear attribution"""
    # step 1: generate the overall purchase report
    user_paths = convert_into_user_paths(events)
    purchase_report = collect_all_purchases(user_paths)
    # step 2: calculate the traffic attribution
    for index, row in purchase_report.iterrows():
        for channel in row['channels']:
            purchase_report.at[index, channel] += row['total_gmv'] / len(row['channels'])

    # type casting
    purchase_report[CHANNELS + ['total_gmv']] =\
        purchase_report[CHANNELS + ['total_gmv']].round(2).astype(float)

    attribution = purchase_report.drop(['channels'], axis=1)
    return attribution

def u_shaped_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate U-Shaped attribution"""
    # step 0: generate channel weights
    def u_shaped_weights(channels: List[str]) -> List[float]:
        """Generate weights for used channel to calculate U-shaped attribution"""
        if len(channels) >= 3:
            weights = [0.4] + [0.2 / (len(channels)-2)] * (len(channels)-2) + [0.4]
        elif len(channels) == 2:
            weights = [0.5, 0.5]
        else:
            weights = [1.0]
        return weights

    # step 1: generate the overall purchase report
    user_paths = convert_into_user_paths(events)
    purchase_report = collect_all_purchases(user_paths)
    # step 2: calculate the traffic attribution
    for index, row in purchase_report.iterrows():
        for i, channel in enumerate(row['channels']):
            purchase_report.at[index, channel] +=\
                u_shaped_weights(row['channels'])[i] * row['total_gmv']

    # type casting
    purchase_report[CHANNELS + ['total_gmv']] =\
        purchase_report[CHANNELS + ['total_gmv']].round(2).astype(float)

    attribution = purchase_report.drop(['channels'], axis=1)
    return attribution

def roi(attribution: pd.DataFrame, ad_costs: pd.DataFrame) -> pd.DataFrame:
    """Calculate ROI"""
    roi_matrix = ad_costs.copy()
    # calculating GMV for each channel
    roi_matrix['gmv'] = 0
    for index, row in roi_matrix.iterrows():
        roi_matrix.at[index, 'gmv'] = round(sum(attribution[row['channel']]))

    # calculating ROI for each channel
    roi_matrix['roi%'] = 100 * (roi_matrix['gmv']-roi_matrix['costs']) / roi_matrix['costs']

    # type casting
    roi_matrix['gmv'] = roi_matrix['gmv'].astype(float)
    roi_matrix['roi%'] = roi_matrix['roi%'].round()

    roi_matrix = roi_matrix[['channel', 'gmv', 'costs', 'roi%']]
    return roi_matrix

'''if __name__ == '__main__':
    events = pd.read_csv('src_junior/MMM/events.csv')
    costs = pd.read_csv('src_junior/MMM/ad_costs.csv')
    attribution = last_touch_attribution(events)
    print(attribution)'''
