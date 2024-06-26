import pandas as pd
import db as db
import model_features as model_features
import sqlite3
import sqlalchemy

query = '''
    CREATE TABLE IF NOT EXISTS npi (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lastname VARCHAR(100) NOT NULL,
        forename VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    '''

conn = sqlite3.connect('data/double_grants_npi.db')
cursor = conn.cursor()
cursor.execute(query)
cursor.close()
engine = sqlalchemy.create_engine('sqlite:///data/double_grants_npi.db')
connector = engine.connect()


def sample_last_names():
    '''Get a sample of last names from both databases'''
    df = pd.read_sql('''SELECT DISTINCT gr.last_name
                        FROM grants gr
                        INNER JOIN npi pr
                            ON gr.last_name = pr.last_name
                        LIMIT 100;''', connector) # Used to be db.sql_eng(), but fixing con error
    df = df.loc[~df['last_name'].str.contains("'")]
    return df


def get_probable_matches():
    '''Get set of likely matches between grantee/grant and 
    provider/npi. Use distances to estimate likely matches.'''
    sample = sample_last_names()
    sample['last_name'] = "'" + sample['last_name'] + "'"
    names = ', '.join(sample['last_name'])

    #pull from sql database for grants matches
    #clunky but "add_prefix" was breaking
    query = f'''SELECT id AS g_id, forename AS g_forename, last_name, 
                organization AS g_org, city AS g_city, state AS g_state, country AS g_country
                FROM grants
                WHERE last_name IN ({names})'''
    grantees = pd.read_sql(query, connector) # Used to be db.sql_eng(), but fixing con error

    #pull from sql db for npi matches
    query = f'''SELECT last_name, forename AS p_forename, address AS p_address, 
                state AS p_state, country AS p_country
                FROM npi
                WHERE last_name IN ({names})'''
    providers = pd.read_sql(query, connector) # Used to be db.sql_eng(), but fixing con error

    #merge the pulls from tables and compute jaro winkler distances
    comb = grantees.merge(providers, on='last_name')
    comb['forename_jw_dist'] = comb.apply(
        lambda row: model_features.jw_dist(row['g_forename'], row['p_forename']),
        axis=1)
    
    return comb.sort_values(by='forename_jw_dist')


if __name__ == '__main__':
    matches = get_probable_matches()
    matches.to_csv('data/likely_grantee_provider_matches.csv', index=False)