from __future__ import absolute_import, division, print_function

import argparse
from PIL import Image
import pickle
import logging
import os
import random
import json
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from dataset import BigVulDatasetDevign
from dgl.dataloading import GraphDataLoader

from torch.utils.data import DataLoader, Dataset, SequentialSampler
from utils.early_stopping import EarlyStopping

from model import Model
from transformers import (get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer, AutoImageProcessor, Swinv2Model)

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score

no_deprecation_warning=True
logger = logging.getLogger(__name__)
early_stopping = EarlyStopping()
lines = dict()

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 image,
                 label,
                 index
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.image = image
        self.label = label
        self.index = index


def convert_examples_to_features(js, tokenizer, args):
    """convert examples to token ids"""
    code = js['code']
    code_tokens = tokenizer.tokenize(code)[:args.block_size-4]
    source_tokens = [tokenizer.cls_token,"<encoder_only>",tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length

    return InputFeatures(source_tokens, source_ids, js['image'], js['label'], js['index'])


class MultiDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, image_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip()
                js = json.loads(line)
                image = torch.load(f"{image_path}/{js['index']}.pt")
                js["image"] = image
                data.append(js)
        for js in tqdm(data):
            self.examples.append(convert_examples_to_features(js, tokenizer, args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("image: {}".format(example.image))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        input_ids = self.examples[i].input_ids
        image = self.examples[i].image
        label = self.examples[i].label
        index = self.examples[i].index
        return (torch.tensor(input_ids), image, torch.tensor(label), torch.tensor(index))


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
def train(args, train_dataloader, train_graph_dataloader, eval_dataloader, eval_graph_dataloader, model):
    """ Train the model """    
    args.max_steps = args.num_train_epochs * len(train_dataloader)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)
    
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu )
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", args.max_steps)

    losses, best_f1 = [], 0
    
    model.zero_grad()
    for idx in range(args.num_train_epochs): 
        for step, batch in tqdm(enumerate(zip(train_dataloader, train_graph_dataloader)), total=len(train_dataloader)):
            codes = batch[0][0].to(args.device)
            images = batch[0][1].to(args.device)
            labels = batch[0][2].to(args.device)
            indices = batch[0][3].to(args.device)
            graphes = batch[1].to(args.device)
            
            model.train()
            loss, _, = model(codes, images, graphes, indices, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # print('loss: {:.2f}'.format(loss.item()), end='\r')
            losses.append(loss.item())

            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(np.mean(losses[-100:]),4)))

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  
            torch.cuda.empty_cache()
            
        results, eval_loss = evaluate(args, eval_dataloader, eval_graph_dataloader, model)
        
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))                    
        
        if results['f1'] > best_f1:
            best_f1 = results['f1']
            logger.info("  "+"*"*20)  
            logger.info("  Best f1:%s",round(best_f1,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-f1'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            output_dir = os.path.join(output_dir, 'model.bin')
            model_to_save = model.module if hasattr(model,'module') else model
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

        early_stopping(eval_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def evaluate(args, eval_dataloader, eval_graph_dataloader, model):
    """ Evaluate the model """

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]
    labels=[]
    vecs=[]
    indices=[]

    for batch in tqdm(zip(eval_dataloader, eval_graph_dataloader), total=len(eval_dataloader)):
        code = batch[0][0].to(args.device)
        image = batch[0][1].to(args.device)
        label = batch[0][2].to(args.device)
        index = batch[0][3].to(args.device)
        graph = batch[1].to(args.device)
        
        with torch.no_grad():
            lm_loss, logit, vec = model(code, image, graph, index, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
            vecs.append(vec.cpu().numpy())
            indices += index.cpu().numpy().tolist()
        nb_eval_steps += 1

    logits = np.concatenate(logits,0)
    labels = np.concatenate(labels,0)
    vecs = np.concatenate(vecs,0)
    preds = logits[:, 1] > 0.5

    np.save("./test_all.npy", {"index": indices, "vec": vecs, "pred": preds, "label": labels})
    
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)
    pr_auc = average_precision_score(labels, logits[:, 1])
    roc_auc = roc_auc_score(labels, logits[:, 1])
    results = {
        "acc": float(acc),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc)
    }
    return results, eval_loss


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a jsonl file).")    
    parser.add_argument("--image_path", default=None, type=str,
                        help="The input image data file.")    
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test on the dev set.")     
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--text', type=bool, default=True,
                        help="Whether to fuse text feature")     
    parser.add_argument('--image', type=bool, default=True,
                        help="Whether to fuse image feature")     
    parser.add_argument('--graph', type=bool, default=True,
                        help="Whether to fuse graph feature")     
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")   
    
          
    # Print arguments
    args = parser.parse_args()
    configs = json.load(open('./config.json'))
    for item in configs:
        args.__dict__[item] = configs[item]
    
    # Set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)
    
    # Build unixcoder model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path)
    
    # Build swimv2 model
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    swimv2_model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    
    model = Model(model, swimv2_model, image_processor, config, args)
    logger.info("Training/evaluation parameters %s", args)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  
            
    # Training     
    if args.do_train:
        df = pd.read_pickle("./dataset/bigvul.pkl")

        train_dataset = MultiDataset(tokenizer, args, args.train_data_file, args.image_path)
        train_df = df[df["partition"] == "train"]
        train_ds = BigVulDatasetDevign(df=train_df)
        train_sampler = SequentialSampler(train_ds)
        train_graph_dataloader = GraphDataLoader(train_ds, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=6)
        
        train_sampler = SequentialSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=6)

        eval_dataset = MultiDataset(tokenizer, args, args.eval_data_file, args.image_path)
        eval_df = df[df.partition == 'valid']
        eval_ds = BigVulDatasetDevign(df=eval_df)
        eval_sampler = SequentialSampler(eval_ds)
        eval_graph_dataloader = GraphDataLoader(eval_ds, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=6)
        
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=6)

        train(args, train_dataloader, train_graph_dataloader, eval_dataloader, eval_graph_dataloader, model)
        
    # Testing          
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))      

        df = pd.read_pickle("./dataset/bigvul.pkl")

        test_dataset = MultiDataset(tokenizer, args, args.test_data_file, args.image_path)
        test_df = df[df.partition == 'test']
        test_ds = BigVulDatasetDevign(df=test_df)
        test_sampler = SequentialSampler(test_ds)
        test_graph_dataloader = GraphDataLoader(test_ds, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=6, pin_memory=True)
    
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=6, pin_memory=True)

        result, _ = evaluate(args, test_dataloader, test_graph_dataloader, model)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key]*100 if "map" in key else result[key],4)))       
    

if __name__ == "__main__":
    main()