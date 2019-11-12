import pandas as pd
import os
import sys
import re 

def stopwords(filename):
    words = []
    for line in open(filename):
        words.append(matching_keywords(line))
    return set(words)

def matching_keywords(strg):
    strg = re.sub("[^A-Za-z ?]+", ' ', strg)    
    strg = re.sub(" +", ' ', strg)    
    strg = strg.upper()
    strg = strg.strip()
    return strg

def filter_stopwords(strg, stop):
    return " ".join([cmp for cmp in strg.split(" ") if cmp not in stop])

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print('python ingest_annotations.py PATH')
    else:
        path = sys.argv[1]
        stop = stopwords('scripts/stopwords.txt')
        enc  = []
        pvl  = []
        for name in os.listdir(path):
            if name.endswith('.xlsx'):
                df = pd.read_excel('{}/{}'.format(path, name))
                if 'Enc' in name:
                    enc.append(df)
                if 'PVL' in name:
                    pvl.append(df)
                    
        pvl = pd.concat(pvl, sort = False).dropna(how='all')
        pvl.index = [i for i in range(0, len(pvl))]
        pvl['shotlog::AC'] = pvl['shotlog::AC'].fillna("?").apply(lambda x: x.upper())
        pvl['shotlog::ID'] = pvl['shotlog::ID'].fillna("NO")
        pvl['shotlog::timecode'] = pvl['shotlog::timecode'].fillna('00:00:00')
        pvl = pvl.fillna(method='ffill')
        pvl['shotlog::BEHcontext'] = pvl['shotlog::BEHcontext'].fillna('?').apply(
            lambda x: filter_stopwords(matching_keywords(x), stop))
        pvl['shotlog::BEHdescription'] = pvl['shotlog::BEHdescription'].apply(
            lambda x: filter_stopwords(matching_keywords(x), stop))    
        pvl[['Year', 'ENC #']] = pvl[['Year', 'ENC #']].astype(int)        
        
        enc = pd.concat(enc, sort = False)
        enc.index = [i for i in range(0, len(enc))]
        enc = enc[['YEAR', 'ENC #', 'BEH CAT', 'ACT LEVEL', 'KEY STENELLA', 'SP ID']]
        enc = enc.dropna(how='all')
        enc['KEY STENELLA'] = enc['KEY STENELLA'].apply(matching_keywords)
        enc['BEH CAT'] = enc['BEH CAT'].apply(matching_keywords)        
        enc['SP ID'] = enc['SP ID'].apply(matching_keywords)        
        enc[['YEAR', 'ENC #']] = enc[['YEAR', 'ENC #']].astype(int)
        
        enc.to_csv('{}/encodings.csv'.format(path), header=None)
        pvl.to_csv('{}/pvl.csv'.format(path), header=None)
