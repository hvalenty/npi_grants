import pandas as pd
import db
import model_features

def sample_last_names():
    '''Get a sample of last names from both databases'''
    df = pd.read_sql('''SELECT DISTINCT gr.last_name
                        FROM grants gr
                        INNER JOIN npi pr
                            ON gr.last_name = pr.last_name
                        LIMIT 100;''', db.sql())
    df = df.loc[~df['last_name'].str.contains("'")]
    return df


def get_probable_matches():
    '''Get set of likely matches between grantee/grant and 
    provider/npi. Use distances to estimate likely matches.'''
    sample = sample_last_names()
    sample['last_name'] = "'" + sample['last_name'] + "'"
    names = ', '.join(sample['last_name'])

    query = f'''SELECT id AS g_id, forename AS g_forename, last_name, 
                organization AS g_org, city AS g_city, state AS g_state, country AS g_country
                FROM grantee
                WHERE last_name IN ({names})'''
    grantees = pd.read_sql((query, db.sql()))


    query = f'''SELECT last_name, forename AS p_forename, address AS p_address, 
                state AS p_state, country AS p_country
                FROM npi
                WHERE last_name IN ({names})'''
    providers = pd.read_sql(query, db.sql())

    comb = grantees.merge(providers, on='last_name')
    comb['forename_jw_dist'] = comb.apply(
        lambda row: model_features.jw_dist(row['g_forename'], row['p_forename']),
        axis=1)
    
    return comb.sort_values(by='forename_jw_dist')


if __name__ == '__main__':
    matches = get_probable_matches()
    matches.to_csv('data/likely_grantee_provider_matches.csv', index=False)