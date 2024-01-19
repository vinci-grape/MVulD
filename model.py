import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
from PIL import Image
from dgl.nn.pytorch import GatedGraphConv
import dgl
import logging
import numpy as np
import time


logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, code_encoder, image_encoder, image_processor, config, args):
        super(Model, self).__init__()
        self.code_encoder = code_encoder
        self.image_encoder = image_encoder
        self.image_processor = image_processor
        self.config = config
        self.args = args
        
        # Graph 
        self.ggnn = GatedGraphConv(in_feats=args.input_dim, out_feats=args.output_dim,
                                   n_steps=args.num_steps, n_etypes=args.max_edge_types)
        
        self.concat_dim = args.input_dim + args.output_dim
        self.conv_l1_for_concat = nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = nn.MaxPool1d(2, stride=2)
        
        # info
        self.relation = np.load("./dataset/relation.npy", allow_pickle=True).tolist()
        self.line = np.load("./dataset/line.npy", allow_pickle=True).tolist()
        
        self.mlp1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlp2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, 2)
    
    
    def get_unixcoder_vec(self, source_ids):
        mask = source_ids.ne(self.config.pad_token_id)
        out = self.code_encoder(source_ids, attention_mask=mask.unsqueeze(1) * mask.unsqueeze(2), output_hidden_states=True)
        token_embeddings = out[0]
        code_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)  # averege
        return code_embeddings, token_embeddings


    def get_image_vec(self, image):
        return self.image_encoder(image).pooler_output


    def get_graph_vec(self, g):
        features = g.ndata['_WORD2VEC']
        edge_types = g.edata["_ETYPE"]
        outputs = self.ggnn(g, features, edge_types)
    
        g.ndata['GGNNOUTPUT'] = outputs

        x_i, h_i = self.unbatch_features(g)
        x_i = torch.stack(x_i)
        h_i = torch.stack(h_i)

        c_i = torch.cat((h_i, x_i), dim=-1)

        Z_1 = self.maxpool1_for_concat(
            F.relu(
                self.conv_l1_for_concat(c_i.transpose(1, 2))
            )
        )
        Z_2 = self.maxpool2_for_concat(
            F.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)

        return Z_2.sum(1)
    
    
    def unbatch_features(self, g):
        x_i = []
        h_i = []
        max_len = -1
        for g_i in dgl.unbatch(g):
            x_i.append(g_i.ndata['_WORD2VEC']) 
            h_i.append(g_i.ndata['GGNNOUTPUT'])
            max_len = max(g_i.number_of_nodes(), max_len)
        for i, (v, k) in enumerate(zip(x_i, h_i)):
            x_i[i] = torch.cat(
                (v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad,
                                device=v.device)), dim=0)
            h_i[i] = torch.cat(
                (k, torch.zeros(size=(max_len - k.size(0), *(k.shape[1:])), requires_grad=k.requires_grad,
                                device=k.device)), dim=0)
        return x_i, h_i
    
    
    def forward(self, input_ids, image, graph, index, labels=None):
        code_vec, token_vec = self.get_unixcoder_vec(input_ids)
        image_vec = image
        graph_vec = self.get_graph_vec(graph)
        local_vec = []
        for i in range(code_vec.shape[0]):
            lines = self.line[index.cpu().numpy()[i]]
            line_vecs = []
            block_vecs = []
            weight = []
            for idx in range(len(lines)):
                if os.path.exists(f"{self.args.image_path}/{index[i]}_{idx+1}.png") and str(idx+1) in self.relation[str(index.cpu().numpy()[i])]:
                    block_vec = torch.load(f"{self.args.image_path}/{index[i]}_{idx+1}.pt").to(self.args.device)
                    block_vecs.append(block_vec)
                    token_embeddings = token_vec[i][lines[idx][0]: lines[idx][1], :]
                    line_vec = torch.mean(token_embeddings, dim=0) # averege
                    line_vecs.append(line_vec)
                    weight.append(float(self.relation[str(index.cpu().numpy()[i])][str(idx+1)]))

            # degree
            if block_vecs:
                weighted_vec = (torch.stack(line_vecs) + torch.stack(block_vecs)) * torch.tensor(weight, dtype=torch.float32).view(-1, 1).to(self.args.device)
                sum_vec = (torch.sum(weighted_vec, dim=0) / weighted_vec.shape[0])
            else:
                sum_vec = torch.zeros([768]).to(self.args.device)
            local_vec.append(sum_vec)
         
        fusion_vec = torch.mul(code_vec, graph_vec)
        fusion_vec = torch.mul(fusion_vec, self.mlp1(image_vec))
        fusion_vec = torch.mul(fusion_vec, self.mlp2(torch.stack(local_vec)))

        logits = self.classifier(fusion_vec)
        prob = nn.functional.softmax(logits, dim=-1)
        
        # ce_loss + bind_loss
        code_vec2, _ = self.get_unixcoder_vec(input_ids)
        logits2 = self.classifier(code_vec2)
        ce_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits2, labels)) / 2
        kl_loss = compute_kl_loss(logits, logits2)
        loss = ce_loss + 0.1 * kl_loss
        
        return loss, prob


def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # Choose whether to use function "sum" and "mean" depending on task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss