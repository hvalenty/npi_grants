import pandas as pd
import numpy as np
import jarowinkler

class FeatureExtractor():
    def __init__(self):
       pass

    def features(self, 
                 grantees: pd.DataFrame, 
                 providers: pd.DataFrame) -> pd.DataFrame:
        """Compute distance features from a pair of dataframes"""
        cols_to_lowercase = ['forename', 'city', 'state']
        for col in cols_to_lowercase:
            grantees[col] = grantees[col].str.lower()

        # Training data
        comb = pd.concat([grantees.add_suffix('_g'), providers.add_suffix('_p')], axis=1)

        # Testing data
        
        comb['jw_dist_forename'] = comb.apply(lambda row: jw_dist(row['forename_g'],
                                                                  row['forename_p']), 
                                                                  axis=1)  # Force row-by-row
        comb['set_dist_forename'] = comb.apply(lambda row: set_dist(row['forename_g'],
                                                                  row['forename_p']), 
                                                                  axis=1)  # Force row-by-row
        
        comb['jw_dist_city'] = comb.apply(lambda row: jw_dist(row['city_g'],
                                                                  row['city_p']), 
                                                                  axis=1)  # Force row-by-row
        comb['set_dist_city'] = comb.apply(lambda row: set_dist(row['city_g'],
                                                                  row['city_p']), 
                                                                  axis=1)  # Force row-by-row
        
        comb['jw_dist_state'] = comb.apply(lambda row: jw_dist(row['state_g'],
                                                                  row['state_p']), 
                                                                  axis=1)  # Force row-by-row
        comb['set_dist_state'] = comb.apply(lambda row: set_dist(row['state_g'],
                                                                  row['state_p']), 
                                                                  axis=1)  # Force row-by-row
        
        return comb[['jw_dist_forename',
                     'set_dist_forename',
                     'jw_dist_city',
                     'set_dist_city',
                     'jw_dist_state',
                     'set_dist_state']]
        

def jw_dist(v1: str, v2: str) -> float:
    if isinstance(v1, str) and isinstance(v2, str):
        return jarowinkler.jarowinkler_similarity(v1, v2)
    else:
        return np.nan


def set_dist(v1: str, v2: str) -> float:
    if isinstance(v1, str) and isinstance(v2, str):
        v1 = set(v1.split(' '))
        v2 = set(v2.split(' '))
        return len(v1.intersection(v2))/min(len(v1), len(v2))
    return np.nan