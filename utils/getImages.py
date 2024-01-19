import os
import json
import pickle as pkl
import sys
path1 = os.path.dirname(sys.path[0])
sys.path.append(path1)
from PIL import Image

from utils import get_dir, processed_dir, cache_dir, dfmp,train_val_test_split_df
import pandas as pd

import sastvd as svd
import sastvd.helpers.joern as svdj
from pathlib import Path
from glob import glob


def itempath(_id):
        """Get itempath path from item id."""
        return processed_dir() / f"bigvul/func_before/{_id}.c" 

def check_validity(_id):
    """Check whether sample with id=_id has node/edges.

    Example:
    _id = 1320
    with open(str(svd.processed_dir() / f"bigvul/before/{_id}.c") + ".nodes.json", "r") as f:
        nodes = json.load(f)
    """
    valid = 0
    try:
        with open(str(itempath(_id)) + ".nodes.json", "r") as f:
            nodes = json.load(f)
            lineNums = set()
            for n in nodes:
                if "lineNumber" in n.keys():
                    lineNums.add(n["lineNumber"])
                    if len(lineNums) > 1:
                        valid = 1
                        break
            if valid == 0:
                return False
        with open(str(itempath(_id)) + ".edges.json", "r") as f:
            edges = json.load(f)
            edge_set = set([i[2] for i in edges])
            if "REACHING_DEF" not in edge_set and "CDG" not in edge_set: 
                return False
            return True
    except Exception as E:
        print(E, str(itempath(_id)))
        return False

def filterDataAndSplit(df):
    df = df[df.apply(lambda x: len(x.func_before.splitlines()) <100, axis=1)].copy()
    dfv=df[df.vul==1]
    print(len(dfv)/len(df)) # 0.04 
    print(df["partition"].value_counts()) 
    return df

def checkDataAfterJoern(df):
    finished = [
        int(Path(i).name.split(".")[0])
        for i in glob(str(processed_dir() / "bigvul/func_before/*nodes*")) 
    ]
    
    df = df[df._id.isin(finished)] 
    print("-------------checkData----------------")
    print(df.info()) # 177905  
    df["valid"] = svd.dfmp(
        df, check_validity, "_id", desc="Validate Samples: "
    )
    df_novalid = df[df.valid==False]
    df = df[df.valid]
    df = train_val_test_split_df(df, idcol='_id', labelcol='vul')
    return df

def rebalanceData(df): 
    '''
    1.undersample the number of non-vulnerable samples to produce an approximately balanced dataset
    2.the test and validation set is left in the original imbalanced ratio.
    '''
    train_df = df[df.partition == "train"] 
    train_df_vul = train_df[train_df.vul == 1]
    train_df_nonvul = train_df[train_df.vul == 0].sample(len(train_df_vul), random_state=0) # undersample
    train_balance_df = pd.concat([train_df_vul, train_df_nonvul])
    print(len(train_df_vul)/len(train_df)) # 0.0407
    val_df = df[df.partition == "valid"]
    val_df_vul = val_df[val_df.vul == 1]
    print(len(val_df_vul)/len(val_df)) # 0.0407
    test_df=df[df.partition == "test"]
    test_df_vul = test_df[test_df.vul == 1]
    print(len(test_df_vul)/len(test_df))
    return train_balance_df,val_df,test_df
  

def ne_groupnodes(n, e): # 
    """Group nodes with same line number."""
    nl = n[n.lineNumber != ""].copy()
    nl.lineNumber = nl.lineNumber.astype(int)   
    nl = nl.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
    nl = nl.groupby("lineNumber").head(1)
    el = e.copy()
    el.innode = el.line_in
    el.outnode = el.line_out
    nl.id = nl.lineNumber
    nl = svdj.drop_lone_nodes(nl, el)
    el = el.drop_duplicates(subset=["innode", "outnode", "etype"])
    el = el[el.innode.apply(lambda x: isinstance(x, float))]
    el = el[el.outnode.apply(lambda x: isinstance(x, float))]
    el.innode = el.innode.astype(int)
    el.outnode = el.outnode.astype(int)
    return nl, el

etype_map = {
    'AST': 0,
    'CDG': 1,
    'REACHING_DEF': 2,
    'CFG': 3,
    'EVAL_TYPE': 4,
    'REF': 5
}
def feature_extraction(_id, graph_type="all", return_nodes=False):
    """Extract graph feature (basic).
    _id = svddc.BigVulDataset.itempath(177775)
    return_nodes arg is used to get the node information (for empirical evaluation).
    """
    # Get Graph
    n, e = svdj.get_node_edges(_id) # 
    n, e = ne_groupnodes(n, e) # 
    if return_nodes:
        return n

    e = svdj.rdg(e, graph_type.split("+")[0]) 
    n = svdj.drop_lone_nodes(n, e) 
    cnt, ntypes = svdj.count_labels(n) 
    n["ntype"] = (ntypes)

    # draw picture 
    dot=svdj.plot_graph_node_edge_df(n, e) 

    # Map line numbers to indexing
    n = n.reset_index(drop=True).reset_index()
    iddict = pd.Series(n.index.values, index=n.id).to_dict()
    e.innode = e.innode.map(iddict)
    e.outnode = e.outnode.map(iddict)

    # Map edge types
    etypes = e.etype.tolist()
    d = dict([(y, x) for x, y in enumerate(sorted(set(etypes)))])
    etypes = [d[i] for i in etypes] # 
    edges_type = [etype_2_id(t) for t in e.etype.tolist()]

    return dot,n.code.tolist(), n.id.tolist(), e.innode.tolist(), e.outnode.tolist(), etypes

def etype_2_id(etype):
    return etype_map[etype]

def getGraphs(row,graph_type="ast"):
    
    id = row['_id']
    my_id = itempath(id) 
    print(my_id)

    sample_dot,code, lineno, ei, eo, et=feature_extraction(my_id,graph_type=graph_type)
    print(code)
    print(lineno)
    

    vul_label="clean"
    if(row["vul"]==1): 
        vul_label = "buggy"

    img_path=get_dir(processed_dir() / row["dataset"])/graph_type/f"{row['partition']}"/vul_label
    sample_dot.render(filename=f"{id}",directory=img_path,view = False, format='png')
    
    if not os.path.exists(f"{img_path}/{id}.png"):
        sample_dot.render(filename=f"{id}",directory=img_path,view = False, format='png')
    else:
        svd.debug("Already save imgs.")


if __name__ == "__main__":

    dataset = 'bigvul'
    cache_path3 = cache_dir() / 'data' / dataset / f'{dataset}_cleaned3.pkl'
    if not cache_path3.exists():
        savedir = cache_dir() / "minimal_datasets"/"minimal_bigvul_before_False.pq"
        df = pd.read_parquet(
            savedir, engine="fastparquet"
        )
        print(df.info())
        print("================checkDataAfterJoern")
        df = checkDataAfterJoern(df)
        print(df.info()) 

        cache_path2 = cache_dir() / 'data' / dataset / f'{dataset}_cleaned2.pkl'
        df.to_pickle(cache_path2)

        df = filterDataAndSplit(df)
        print(df.info())
        df.to_pickle(cache_path3) 
    else:
        df = pd.read_pickle(cache_path3)

    balanced_df_path = cache_dir() / "data/bigvul/bigvul_cleaned_balanced3.pkl"
    
    if not balanced_df_path.exists():
        train_df = df[df.partition == "train"] 
        print(f"train_df len=:{len(train_df)}")
        train_balance_df,valid_df,test_df = rebalanceData(df)
        print(f"train_balance_df =:{len(train_balance_df)}")
        print(f"valid_df len =:{len(valid_df)}")
        print(f"test_df len = :{len(test_df)}")
        df = pd.concat([train_balance_df,valid_df,test_df])
        print(df.info())
        print(df["partition"].value_counts())
        print(df._id.nunique())
        df.to_pickle(balanced_df_path)
    else:
        df = pd.read_pickle(balanced_df_path)
        train_df = df[df.partition == "train"]
        valid_df = df[df.partition == "valid"]
        test_df = df[df.partition == "test"]

    svd.dfmp(valid_df, getGraphs, ordr=False, workers=8)
    svd.dfmp(test_df, getGraphs, ordr=False, workers=8)
    svd.dfmp(train_balance_df, getGraphs, ordr=False, workers=8)