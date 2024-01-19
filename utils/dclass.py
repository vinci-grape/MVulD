import json
import pandas as pd
from utils import processed_dir

class BigVulDataset:
    """Represent BigVul as graph dataset."""
    DATASET = None

    def __init__(self, df, dataset="bigvul", partition="all", vulonly=False, sample=-1):
        """Init class."""
        BigVulDataset.DATASET = dataset
        if dataset == 'bigvul_mix':
            self.df = df[df.mix == False]
            self.mix = df[df.mix == True]
        else:
            self.df = df
            self.mix = None
        self.partition = partition
        if self.partition != 'all':
            self.df = self.df[self.df.partition == partition]

        if dataset == 'bigvul_mix':
            self.mix = self.mix[self.mix.partition == partition]
            self.mix = self.mix[self.mix._id.isin(self.finished)]
            print('df', self.df.shape)
            print('mix', self.mix.shape)

        # Small sample (for debugging):
        if sample > 0:
            self.df = self.df.sample(sample, random_state=0)

        # Filter only vulnerable
        if vulonly:
            self.df = self.df[self.df.vul == 1]
        # Mix patches
        if dataset == 'bigvul_mix' and partition == 'train':
            self.df = pd.concat([self.df, self.mix])


    def itempath(_id): 
        """Get itempath path from item id."""
        return processed_dir() / f"{BigVulDataset.DATASET}/func_before/{_id}.c"

    def after_itempath(_id): 
        """Get itempath path from item id."""
        return processed_dir() / f"{BigVulDataset.DATASET}/func_after/{_id}.c"
    
    def check_validity(_id):
        """Check whether sample with id=_id has node/edges."""
        valid = 0
        try:
            with open(str(BigVulDataset.itempath(_id)) + ".nodes.json", "r") as f:
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
            with open(str(BigVulDataset.itempath(_id)) + ".edges.json", "r") as f:
                edges = json.load(f)
                edge_set = set([i[2] for i in edges])
                if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                    return False
                return True
        except Exception as E:
            print(E, str(BigVulDataset.itempath(_id)))
            return False

    def stats(self):
        """Print dataset stats."""
        print(self.df.groupby(["partition", "vul"]).count()[["_id"]])

    def get_vul_label(self, _id):
        """Obtain vulnerable or not."""
        df = self.df[self.df._id == _id]
        label = df.vul.item()
        return label

    def __getitem__(self, idx):
        """Must override."""
        return self.df.iloc[idx].to_dict()

    def __len__(self):
        """Get length of dataset."""
        return len(self.df)

    def __repr__(self):
        """Override representation."""
        vulnperc = round(len(self.df[self.df.vul == 1]) / len(self), 3)
        return f"BigVulDataset(partition={self.partition}, samples={len(self)}, vulnperc={vulnperc})"
