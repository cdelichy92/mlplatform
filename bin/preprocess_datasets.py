import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('../mlplatform/datasets/imdb_tr.csv', encoding='latin1')
    df[['polarity', 'text']].to_csv('../mlplatform/datasets/imdb_trf.tsv', encoding='utf-8', sep='\t',
            header=False, index=False)
