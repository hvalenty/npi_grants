import sqlalchemy
import sqlite3


def create_db():
    """A database creation statement"""
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


def sql_eng():
    engine = sqlalchemy.create_engine('sqlite:///data/double_grants_npi.db')
    return engine.connect()