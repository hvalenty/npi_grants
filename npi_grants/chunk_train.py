import pandas as pd
import sqlite3
import entity_resolution_model

'''load in blocks of data from the database and runs the trained model on the output
1. iterate through all data
2. block, O(n^2) -> (c*n): c= block size -- for this block by last name
3. make output usable
4. solve duplications (de-dup)
'''

def read_in_chunk(path: str):
    rows_per_chunk = 100
    db = pd.read_csv(path,
                 chunksize=rows_per_chunk)


def apply_model():
    entity_resolution_model.EntityResolutionModel()

#db = pd.read_sql(sqlite3:///data/double_grants_npi.db)



if __name__ == '__main__':
    print()
    path_in = "data/likely_grantee_provider_matches.csv"
    read_in_chunk(path_in)