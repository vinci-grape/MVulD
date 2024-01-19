import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent))
path1 = os.path.dirname(sys.path[0])
sys.path.append(path1)
from utils import get_dir, processed_dir, full_run_joern, dfmp, storage_dir, cache_dir, train_val_test_split_df, \
    data_dir
from scripts.process_dataset import cleaned_dataset, mix_patch

# SETUP
NUM_JOBS = 100


def processJoern(row): 
    """Parallelise svdj functions.

    Example:
    df = svdd.bigvul()
    row = df.iloc[180189]  # PAPER EXAMPLE
    row = df.iloc[177860]  # EDGE CASE 1
    preprocess(row)
    """
    savedir_before = get_dir(processed_dir() / row["dataset"] /"func_before")

    # Write C Files
    fpath1 = savedir_before / f"{row['_id']}.c" 
    with open(fpath1, "w") as f:
        f.write(row["func_before"]) 
        
    if not os.path.exists(f"{fpath1}.edges.json"):
        full_run_joern(fpath1, verbose=3)


if __name__ == "__main__":
    
    savedir = cache_dir() / "minimal_datasets"/"minimal_bigvul_before_False.pq"
    df = pd.read_parquet(
        savedir, engine="fastparquet"
    )
    print(df.info())

    dfmp(df, processJoern, ordr=False, workers=8) 

   

   