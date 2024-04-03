import sqlalchemy
import sqlite3

def sql():
    engine = sqlalchemy.create_engine('sqlite:///data/double_grants_npi.db')
    return engine.connect()