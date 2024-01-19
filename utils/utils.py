from pathlib import Path
import inspect
from datetime import datetime
import subprocess
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import re
from sklearn.model_selection import train_test_split
import hashlib
import random
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc
)
import torch
import torch.nn.functional as F


# regex to remove empty lines
def remove_empty_lines(text):
    return re.sub(r'^$\n', '', text, flags=re.MULTILINE)


# regex to remove comments from a file
def remove_comments(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


# regex to remove space before newLine character
def remove_space_before_newline(text):
    return re.sub(r'\s+$', '', text, flags=re.MULTILINE)


# regex to remove space after newLine character
def remove_space_after_newline(text):
    return re.sub(r'^\s+', '', text, flags=re.MULTILINE)


def get_dir(path) -> Path:
    """Get path, if exists. If not, create it."""
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def external_dir() -> Path:
    """Get storage external path."""
    path = storage_dir() / "external"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def project_dir() -> Path:
    """Get project path."""
    return Path(__file__).parent.parent


def processed_dir() -> Path:
    """Get storage processed path."""
    path = storage_dir() / "processed"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def result_dir() -> Path:
    """Get result path."""
    path = storage_dir() / "results"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def storage_dir() -> Path:
    """Get storage path."""
    return Path(__file__).parent.parent.parent / "storage"


def data_dir() -> Path:
    """Get dataset path."""
    path = storage_dir() / "data"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def cache_dir() -> Path:
    """Get storage cache path."""
    path = storage_dir() / "cache"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def debug(msg, noheader=False, sep="\t"):
    """Print to console with debug information."""
    caller = inspect.stack()[1]
    file_name = caller.filename
    ln = caller.lineno
    now = datetime.now()
    time = now.strftime("%m/%d/%Y - %H:%M:%S")
    if noheader:
        print("\t\x1b[94m{}\x1b[0m".format(msg), end="")
        return
    print(
        '\x1b[40m[{}] File "{}", line {}\x1b[0m\n\x1b[94m{}\x1b[0m'.format(
            time, file_name, ln, msg
        )
    )


def subprocess_cmd(command: str, verbose: int = 0, force_shell: bool = False):
    """Run command line process.

    Example:
    subprocess_cmd('echo a; echo b', verbose=1)
    >>> a
    >>> b
    """
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output = process.communicate()
    if verbose > 1:
        debug(output[0].decode())
        debug(output[1].decode())
    return output


def hashstr(s):
    """Hash a string."""
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


def dfmp(df, function, columns=None, ordr=True, workers=6, cs=10, desc="Run: "):
    """Parallel apply function on dataframe.

    Example:
    def asdf(x):
        return xgpu

    dfmp(list(range(10)), asdf, ordr=False, workers=6, cs=1)
    """
    print(df)
    if isinstance(columns, str):
        items = df[columns].tolist()
    elif isinstance(columns, list):
        items = df[columns].to_dict("records")
    elif isinstance(df, pd.DataFrame):
        items = df.to_dict("records")
    elif isinstance(df, list):
        items = df
    else:
        raise ValueError("First argument of dfmp should be pd.DataFrame or list.")

    processed = []
    
    desc = f"({workers} Workers) {desc}"
    with Pool(processes=workers) as p:
        map_func = getattr(p, "imap" if ordr else "imap_unordered")
        for ret in tqdm(map_func(function, items, cs), total=len(items), desc=desc):
            processed.append(ret)
    return processed


def train_val_test_split_df(df, idcol='new_id', labelcol='vul'):
    """Add train/val/test column into dataframe."""
    print(df.shape, df.columns)

    X = df[idcol]
    y = df[labelcol]
    train_rat = 0.8
    val_rat = 0.1
    test_rat = 0.1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_rat, random_state=123456, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=test_rat / (test_rat + val_rat), random_state=123456, stratify=y_test
    )
    X_train = set(X_train)
    X_val = set(X_val)
    X_test = set(X_test)
    print(f'train:{len(X_train)}, valid:{len(X_val)}, test:{len(X_test)}')

    def path_to_label(path):
        if path in X_train:
            return "train"
        if path in X_val:
            return "valid"
        if path in X_test:
            return "test"

    df["partition"] = df[idcol].apply(path_to_label)
    return df


def gitsha():
    """Get current git commit sha for reproducibility."""
    return 'v1'


def get_run_id(args=None):
    """Generate run ID."""
    if not args:
        ID = datetime.now().strftime("%Y%m%d%H%M_{}".format(gitsha()))
        return ID
    ID = datetime.now().strftime(
        "%Y%m%d%H%M_{}_{}".format(
            gitsha(), "_".join([f"{v}" for _, v in vars(args).items()])
        )
    )
    return ID


def get_metrics(true, pred):
    """Get relevant metrics given true labels and logits."""
    metrics = {}

    metrics["acc"] = accuracy_score(true, pred)
    metrics["f1"] = f1_score(true, pred, zero_division=0)
    metrics["rec"] = recall_score(true, pred, zero_division=0)
    metrics["prec"] = precision_score(true, pred, zero_division=0)
    return metrics


def best_f1(true, pos_logits):
    """Find optimal threshold for F1 score.

    true = [1, 0, 0, 1]
    pos_logits = [0.27292988, 0.27282527, 0.7942509, 0.20574914]
    """
    precision, recall, thresholds = precision_recall_curve(true, pos_logits)
    thresh_scores = []
    for i in range(len(thresholds)):
        if precision[i] + recall[i] == 0:
            continue
        f1 = (2 * (precision[i] * recall[i])) / (precision[i] + recall[i])
        thresh = thresholds[i]
        thresh_scores.append([f1, thresh])
    thresh_scores = sorted(thresh_scores, reverse=True)
    thresh_scores = [i for i in thresh_scores if i[0] > 0]
    return thresh_scores[0][-1]


def get_metrics_logits(true, logits):
    """Call get_metrics with logits."""
    if not torch.is_tensor(true):
        true = torch.Tensor(true).long()
    if not torch.is_tensor(logits):
        logits = torch.Tensor(logits)
    # print('true', true.shape, 'logits', logits.shape) # logits shape :[9629,1]
    loss = F.cross_entropy(logits, true).detach().cpu().item()
    true_oh = torch.nn.functional.one_hot(true).detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    sm_logits = torch.nn.functional.softmax(logits, dim=1)
    pos_logits = sm_logits[:, 1].detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()

    f1_threshold = 0.5
    pred = [1 if i > f1_threshold else 0 for i in pos_logits]

    try:
        roc_auc = roc_auc_score(true, logits[:, 1])
    except:
        roc_auc = -1
    try:
        pr_auc = average_precision_score(true_oh, logits)
    except:
        pr_auc = -1
    ret = get_metrics(true, pred)
    ret["roc_auc"] = roc_auc
    ret["pr_auc"] = pr_auc
    ret["pr_auc_pos"] = average_precision_score(true, logits[:, 1]) #这个才是真正的prauc值
    ret["loss"] = loss
    ret["f1_threshold"] = f1_threshold
    precision, recall, _ = precision_recall_curve(true, logits[:, 1])
    auc_score = auc(recall, precision)
    # print('cal from precision_recall_curve',auc_score)
    return ret


def get_metrics_probs_bce(true, probs, logits):
    # train_mets = get_metrics_probs_bce(labels, probs, logits)
    """Call get_metrics with probs."""
    if not torch.is_tensor(true):
        true = torch.Tensor(true).long()
    if not torch.is_tensor(logits):
        logits = torch.Tensor(logits)
    loss = F.binary_cross_entropy(probs, true).detach().cpu().item()

    true = true.long()
    true_oh = torch.nn.functional.one_hot(true).detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    # sm_probs = pro
    probs = probs.detach().cpu().numpy()
    logits = logits.squeeze(-1).detach().cpu().numpy()
    f1_threshold = 0.5
    pred = [1 if i > f1_threshold else 0 for i in probs]

    try:
        roc_auc = roc_auc_score(true, logits)
    except:
        roc_auc = -1
    pr_auc = average_precision_score(true, logits)
    # print(pr_auc)
    try:
        pr_auc = average_precision_score(true, probs) 
    except:
        pr_auc = -1
    ret = get_metrics(true, pred)
    ret["roc_auc"] = roc_auc
    ret["pr_auc"] = pr_auc
    ret["pr_auc_pos"] = average_precision_score(true, logits) # 为什么计算三次？？
    ret["loss"] = loss
    ret["f1_threshold"] = f1_threshold
    return ret


def watch_subprocess_cmd(command: str, force_shell: bool = False):
    """Run subprocess and monitor output. Used for debugging purposes."""
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    # Poll process for new output until finished
    noheader = False
    while True:
        nextline = process.stdout.readline()
        if nextline == b"" and process.poll() is not None:
            break
        debug(nextline.decode(), noheader=noheader)
        noheader = True


def tokenize(s):
    """Tokenise according to IVDetect.

    Tests:
    s = "FooBar fooBar foo bar_blub23/x~y'z"
    """
    spec_char = re.compile(r"[^a-zA-Z0-9\s]")
    camelcase = re.compile(r".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    spec_split = re.split(spec_char, s)
    space_split = " ".join(spec_split).split()

    def camel_case_split(identifier):
        return [i.group(0) for i in re.finditer(camelcase, identifier)]

    camel_split = [i for j in [camel_case_split(i) for i in space_split] for i in j]
    remove_single = [i for i in camel_split if len(i) > 1]
    return " ".join(remove_single)


def tokenize_lines(s):
    r"""Tokenise according to IVDetect by splitlines.

    Example:
    s = "line1a line1b\nline2a asdf\nf f f f f\na"
    """
    slines = s.splitlines()
    lines = []
    for sline in slines:
        tokline = tokenize(sline)
        if len(tokline) > 0:
            lines.append(tokline)
    return lines


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


if __name__ == '__main__':
    debug('asd')
