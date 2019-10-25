import os
import sqlalchemy

db_user            = os.environ.get('CLOUDSQL_USER')
db_password        = os.environ.get('CLOUDSQL_PASSWORD')
db_name            = os.environ.get('CLOUD_SQL_DATABASE_NAME')
db_connection_name = os.environ.get('CLOUDSQL_CONNECTION_NAME')

db = sqlalchemy.create_engine(
    sqlalchemy.engine.url.URL(
        drivername='mysql+pymysql',
        username=db_user,
        password=db_password,
        database=db_name,
        query={
            'unix_socket': '/cloudsql/{}'.format(db_connection_name)
        }
    ),
    pool_size=5,
    max_overflow=2,
    pool_timeout=30,  # 30 seconds
    pool_recycle=1800,  # 30 minutes
)

def filename_query(encoding):
    return """
        SELECT year, filename FROM (
            SELECT filename, year, char_length(filename) as l 
            FROM wdp_ds.audio    f
            JOIN wdp_ds.encoding e ON f.encoding = e.encoding
            WHERE e.encoding = {}
            ORDER BY l 
            LIMIT 1
        ) x;
    """.format(encoding)

def encoding_query(year = None):
    if year is None:
        return "SELECT * FROM encoding"
    else:
        return "SELECT * FROM encoding WHERE year = {}".format(year)

def pvl_query(encoding):
    return "SELECT * FROM pvl WHERE encoding = {}".format(encoding)

def run_query(query):
    with db.connect() as conn:
        rows = []
        for row in conn.execute(query).fetchall():
            rows.append(dict(row.items()))
        return rows

def encodings():
    return run_query(encoding_query(2011))

def pvl(encoding):
    return run_query(pvl_query(encoding))

def filename(encoding):    
    return ["{}/{}".format(row['year'], row['filename']) for row in run_query(filename_query(encoding))]
