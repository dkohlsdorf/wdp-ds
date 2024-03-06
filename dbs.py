import sqlite3
import os
import datetime

class QueryHistory:
    
    def __init__(self, file='query.db'):
        self.file=file
        if not os.path.exists(file):
            conn = sqlite3.connect(file)       
            cur  = conn.cursor()             
            cur.execute("""
                CREATE TABLE query_history (
                     query_string text,
                     query_file text,
                     date text
                )
            """)            
            conn.commit()
            conn.close()
        
    def insert(self, query, file=None):
        conn = sqlite3.connect(self.file)       
        cur  = conn.cursor()             

        date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        cur.execute("INSERT INTO query_history VALUES (?, ?, ?)", (query, file, date))
        conn.commit()
        conn.close()

    def get(self, n=None):
        conn = sqlite3.connect(self.file)               
        cur  = conn.cursor()         
        filter_n = f"LIMIT {n}" if n is not None else ""   
        cur.execute(f"""
            SELECT * 
            FROM query_history
            ORDER BY date DESC
            {filter_n}
        """)
        result = cur.fetchall()
        conn.commit()
        conn.close()        
        return result

    
class AlignmentDB:

    def __init__(self, file='alignment.db'):
        self.file=file
        if not os.path.exists(file):
            conn = sqlite3.connect(file)       
            cur  = conn.cursor()             
            cur.execute("""
                CREATE TABLE alignment_files (
                     project int,
                     fileid text,
                     status text,
                     date text
                )
            """)            
            conn.commit()
            conn.close()

    def get_all_projects(self):
        conn = sqlite3.connect(self.file)               
        cur  = conn.cursor()         
        cur.execute(f"""
            SELECT disctinct project 
            FROM alignment_files
            ORDER BY date
        """)
        result = cur.fetchall()
        conn.commit()
        conn.close()        
        return result

    def get_files(self, project):
        conn = sqlite3.connect(self.file)               
        cur  = conn.cursor()         
        cur.execute(f"""
            SELECT distinct fileid, status
            FROM alignment_files
            WHERE project = {project}
            ORDER BY date
        """)
        result = cur.fetchall()
        done = set([fileid for fileid, status in result if status == 'done'])
        files = [(fileid, status)
                 for fileid, status in result
                 if (status == 'done' or fileid not in done)]        
        conn.commit()
        conn.close()        
        return files
    
    def insert_file(self, project, fileid):    
        conn = sqlite3.connect(self.file)       
        cur  = conn.cursor()             

        date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        cur.execute("INSERT INTO alignment_files VALUES (?, ?, ?, ?)", (project, fileid, 'processing', date))
        conn.commit()
        conn.close()

    def finish_file(self, fileid):
        print(f"finish: {fileid}")
        conn = sqlite3.connect(self.file)       
        cur  = conn.cursor()             

        date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        cur.execute(f"""
          INSERT INTO alignment_files 
          SELECT 
             project,
             fileid,
             'done',
             date
          FROM alignment_files
          WHERE fileid="{fileid}" and status="processing"
        """)

        conn.commit()
        conn.close()

    
