import os
import pickle as pkl
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.append(str((Path(__file__).parent.parent))) 
path1 = os.path.dirname(sys.path[0])
sys.path.append(path1)

from utils import remove_space_before_newline, remove_comments, remove_empty_lines, remove_space_after_newline, get_dir, \
    processed_dir, cache_dir, dfmp,train_val_test_split_df
import utils.glove as glove
import utils.word2vec as word2vec
from utils.git import allfunc, c2dhelper


def cleaned_code(func_code):
    func_code = remove_empty_lines(func_code)
    func_code = remove_comments(func_code)
    func_code = remove_space_after_newline(func_code)

    return func_code
    pass


def cleaned_dataset(data, dataset):
    '''df = cleaned_dataset(df, dataset=dataset)'''
    print('Data shape:', data.shape)
    print('Data columns:', data.columns)
    print('Cleaning Code...')
    data['func_before'] = data['func_before'].apply(lambda x: cleaned_code(x))
    data['func_after'] = data['func_after'].apply(lambda x: cleaned_code(x))
    data = data[~data['func_before'].duplicated(keep=False)]
    print('Removing (func_before == func_after) for vulnerable function...')
    data = data[(data['vul'] == 0) | (data['vul'] == 1 & (data['func_before'] != data['func_after']))]
    print('Data shape:', data.shape)

    print('Cleaning Code Done!')

    # Save codediffs: 
    data = data.reset_index(drop=True).reset_index().rename(columns={'index': '_id'})
    data['dataset'] = dataset
    cols = ["func_before", "func_after", "_id", "dataset"]
    dfmp(data, c2dhelper, columns=cols, ordr=False, cs=300, workers=32)
    data["info"] = dfmp(data, allfunc, cs=500, workers=32)
    data = pd.concat([data, pd.json_normalize(data["info"])], axis=1)
    return data

def bigvul_minimal(data,sample=False):
    keepcols = [
        "_id",
        "CVE ID",
        "CWE ID",
        "Vulnerability Classification",
        "commit_id",
        "commit_message",
        "func_after",
        "func_before",
        "vul",
        "dataset",
        "id",
        "partition",
        "removed",
        "added",
        "diff",
        "project",
        "info",
    ]
    savedir = get_dir(cache_dir() / "minimal_datasets")
    print(savedir)
    df_savedir = savedir / f"minimal_bigvul_before_{sample}.pq" 

    #dataframe
    data[keepcols].to_parquet(
        df_savedir,
        object_encoding="json",
        index=0,
        compression="gzip",
        engine="fastparquet",
    )
    data = pd.read_parquet(
        savedir / f"minimal_bigvul_before_{sample}.pq", engine="fastparquet"
    ).dropna() 
    return data

def clean_abnormal_func(data):
    print('clean_abnormal_func...')
    # Remove functions with abnormal ending (no } or ;)
    # c++ class ending with }; 
    data = data[
        ~data.apply(
            lambda x: x.func_before.strip()[-1] != "}"
            and x.func_before.strip()[-1] != ";", 
            axis=1,
        )
    ]
    # Remove functions with abnormal ending (ending with ");")
    data = data[~data.func_before.apply(lambda x: x[-2:] == ");")]
    print('clean_abnormal_func done')
    print(data.info())
    return data

def mix_patch(df):
    df['mix'] = False
    origin_df = df.copy()
    print(df.shape)
    vul_df = df[df.vul == 1]
    func_after = vul_df['func_after']
    pat_id = vul_df['_id'] + 190000
    pat = vul_df.copy()
    assert len(pat) == len(func_after)
    pat['func_before'] = func_after
    pat['vul'] = 0
    pat['mix'] = True
    pat['_id'] = pat_id
    df = pd.concat([origin_df, pat])
    return df


def prepare_glove(dataset='devign'):
    # generate GloVe
    glove.generate_glove(dataset, sample=False)


def prepare_w2v(dataset='devign'):
    # generate GloVe
    word2vec.generate_w2v(dataset, sample=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()
    dataset='bigvul'

    cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
    if not cache_path.exists():
        filename ="MSR_data_cleaned.csv"
        data_path=cache_dir() /'data'/ dataset/filename
        print(data_path)
        df = pd.read_csv(data_path)
        print(df.info())
        dfv = df[df.vul==1]
        print(f"original dataset len(df)={len(df)} and len(dfv)={len(dfv)}")
        df = df.rename(columns={"Unnamed: 0": "id"})
        df = cleaned_dataset(df, dataset=dataset) # step1-clean
        df = clean_abnormal_func(df)  # step1-clean
        df = train_val_test_split_df(df, idcol='_id', labelcol='vul') 
        print(df.info())
        df.to_pickle(cache_path)

    pass
