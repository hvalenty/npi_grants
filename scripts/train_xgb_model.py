import pandas as pd

from npi_grants.sql import db


def create_training_data_features_labels():
    '''Load known training data, extract values 
    from the database and convert to features'''
    training = pd.read_csv('data/likely_grantee_provider_mtches.csv')
    grantee_ids = ', '.join(training['g_id'])
    query = f'''SELECT TRIM(LOWER(last_name)) AS last_name, 
                        TRIM(LOWER(forename)),
                        TRIM(LOWER(city)),
                        TRIM(LOWER(state))
                FROM grants
                '''
    
#more here