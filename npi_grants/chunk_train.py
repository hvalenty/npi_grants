import pandas as pd
import sqlite3
import sqlalchemy
import model_chain.model_features as model_features
import model_chain.entity_resolution_model as entity_resolution_model

'''load in blocks of data from the database and runs the trained model on the output
1. iterate through all data
2. block, O(n^2) -> (c*n): c= block size -- for this block by last name
3. make output usable
4. solve duplications (de-dup)
'''

# connect to sql database
engine = sqlalchemy.create_engine('sqlite:///data/double_grants_npi.db')
connector = engine.connect()

def read_bridged():
    '''Read in joined tables and group by last names'''
    df = pd.read_sql('''
SELECT
    application_id,
  	npi.npi,
  	city,
  	grants.state AS g_state,
    npi.state AS n_state,
  	grants.forename AS g_forename,
  	npi.forename AS n_forename,
  	grants.last_name AS g_last_name,
  	npi.last_name AS n_last_name
FROM
    grants
INNER JOIN npi ON npi.last_name = grants.last_name;''', connector)
    #group dataframe by the lastnames
    #df = df.groupby(['g_last_name'])
    return df

def unique_names(df):
    '''draw unique name from dataframe'''
    unique = df['g_last_name'].unique()
    name1 = unique[1]
    chunk = df.loc[df['g_last_name'] == str(name1)]

    return chunk

def name_distances(df):
        '''compute distances between grants and npi firstnames using jw'''
        df['forename_jw_dist'] = df.apply(
        lambda row: model_features.jw_dist(row['g_forename'], row['n_forename']),
        axis=1)
    
        return df.sort_values(by='forename_jw_dist')


def apply_model(df):
    '''apply the model to the dataframe'''
    erm = entity_resolution_model.EntityResolutionModel().predict(df, proba=True)
    df.loc[df['proba'] > 0.5]
    return erm



if __name__ == '__main__':
    data = read_bridged()
    print(apply_model(name_distances(unique_names(data))))

