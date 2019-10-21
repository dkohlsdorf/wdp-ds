import pandas as pd
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print('python ingest_annotations.py PATH')
    else:
        path = sys.argv[1]
        enc  = []
        pvl  = []
        for name in os.listdir(path):
            if name.endswith('.xlsx'):
                df = pd.read_excel('{}/{}'.format(path, name))
                if 'Enc' in name:
                    enc.append(df)
                if 'PVL' in name:
                    pvl.append(df)

        pvl = pd.concat(pvl, sort = True).dropna(how='all')
        pvl['shotlog::ID'] = pvl['shotlog::ID'].fillna("NO")
        pvl['shotlog::timecode'] = pvl['shotlog::timecode'].fillna('00:00:00')
        pvl = pvl.fillna(method='ffill')
        pvl['shotlog::BEHcontext'] = pvl['shotlog::BEHcontext'].fillna('?')

        enc = pd.concat(enc, sort = True)
        enc = enc[['YEAR', 'ENC #', 'BEH CAT', 'ACT LEVEL', 'KEY STENELLA', 'SP ID']]
        enc = enc.dropna(how='all')

        enc.to_csv('{}/encodings.csv'.format(path))
        pvl.to_csv('{}/pvl.csv'.format(path))
