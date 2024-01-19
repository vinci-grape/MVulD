import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_dir, processed_dir, full_run_joern, dfmp, storage_dir, cache_dir, train_val_test_split_df, \
    data_dir
from scripts.process_dataset import cleaned_dataset, mix_patch


def preprocess(row):
    """Parallelise svdj functions.

    Example:
    df = svdd.bigvul()
    row = df.iloc[180189]  # PAPER EXAMPLE
    row = df.iloc[177860]  # EDGE CASE 1
    preprocess(row)
    """
    savedir_before = get_dir(processed_dir() / row["dataset"] / "before")
    savedir_after = get_dir(processed_dir() / row["dataset"] / "after")
    # Write C Files
    fpath1 = savedir_before / f"{row['_id']}.c"

    # Run Joern on "before" code
    if not os.path.exists(f"{fpath1}.edges.json"):
        with open(fpath1, "w") as f:
            f.write(row["func_before"])
        full_run_joern(fpath1, verbose=3)

    # Run Joern on "after" code
    if row['vul'] == 1 and len(row["func_after"]) > 2:
        fpath2 = savedir_after / f"{row['_id']}.c"

        if not os.path.exists(f"{fpath2}.edges.json"):
            with open(fpath2, "w") as f:
                f.write(row["func_after"])
            full_run_joern(fpath2, verbose=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)

    args = parser.parse_args()
    print(args)
    dataset = args.dataset

    # SETUP
    NUM_JOBS = 100

    # Read Data
    cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
    
    if not cache_path.exists(): 
        if dataset == 'bigvul_mix': 
            df = pd.read_pickle(f'{cache_dir()}/data/bigvul/bigvul_cleaned.pkl')
        else:
            df = pd.read_pickle(f'{data_dir()}/{dataset}/{dataset}.pkl')
            df = cleaned_dataset(df, dataset=dataset)
            df = train_val_test_split_df(df, idcol='_id', labelcol='vul')

        if args.dataset == 'bigvul_mix':
            df = mix_patch(df)
        get_dir(cache_path.parent)
        df.to_pickle(cache_path) 
    else:
        df = pd.read_pickle(cache_path)

    print('Data shape:', df.shape)
    print('Data columns:', df.columns)
    df = df.iloc[::-1]
    splits = np.array_split(df, NUM_JOBS)