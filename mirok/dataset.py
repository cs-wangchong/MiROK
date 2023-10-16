#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from typing import List
from torch.utils.data import Dataset as TorchDataset, DataLoader

from mirok.seq import Seq
from mirok.corpus import PAD, SOS, EOS

class Dataset(TorchDataset):
    def __init__(self, token2idx, type2idx, tag2idx, seqs:List[Seq], is_training=False, depth=1, max_len=200):
        super(Dataset, self).__init__()
        self.token2idx = token2idx
        self.type2idx = type2idx
        self.tag2idx = tag2idx
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.idx2type = {idx: type for type, idx in self.type2idx.items()}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        self.seqs = seqs
        self.token_seqs = []
        self.type_seqs = []
        self.tag_seqs = []
        for seq in seqs:
            # logging.info( seq.token_type_seq)
            # logging.info(self.type2idx)
            self.token_seqs.append([self.token2idx[token] for token in seq.token_seq])
            self.type_seqs.append([self.type2idx[type] for type in seq.token_type_seq])
            self.tag_seqs.append([[self.tag2idx[tag] for tag in tag_seq] for tag_seq, _ in seq.tag_seqs])
           
        self.is_training = is_training
        self.depth = depth
        self.max_len = max_len

    def init_op_pairs(self, op1s, op2s):
        self.op1s = {self.token2idx[op1] for op1 in op1s}
        self.op2s = {self.token2idx[op2] for op2 in op2s}


    def __len__(self):
        return len(self.token_seqs)

    def __getitem__(self, idx):
        if self.is_training:
            return self.token_seqs[idx], self.type_seqs[idx], self.tag_seqs[idx]
        else:
            return self.token_seqs[idx], self.type_seqs[idx]

    def collate_fn(self, examples):
        if self.is_training:
            token_seqs, type_seqs, tag_seqs =  tuple(zip(*examples))
        else:
            token_seqs, type_seqs =  tuple(zip(*examples))
        
        max_seq_len = max(len(token_seq) for token_seq in token_seqs)
        max_seq_len = min(max_seq_len, self.max_len)
        token_seqs = [token_seq[:max_seq_len] + [PAD] * (max_seq_len - len(token_seq)) for token_seq in token_seqs]
        type_seqs = [type_seq[:max_seq_len] + [PAD] * (max_seq_len - len(type_seq)) for type_seq in type_seqs]
        token_seqs = torch.LongTensor(token_seqs)
        type_seqs = torch.LongTensor(type_seqs)

        if not self.is_training:
            return token_seqs, type_seqs

        tag_seqs = [
            [tag_seq[:max_seq_len] + [PAD] * (max_seq_len - len(tag_seq)) for tag_seq in inner_tag_seqs][:self.depth]
            for inner_tag_seqs in tag_seqs
        ]
        tag_seqs = [inner_tag_seqs + [[PAD] * max_seq_len] * (self.depth - len(inner_tag_seqs)) for inner_tag_seqs in tag_seqs]
        tag_seqs = torch.LongTensor(tag_seqs)
        return token_seqs, type_seqs, tag_seqs

    def loader(self, batch_size: int, shuffle=True, num_workers: int = 2) -> DataLoader:
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)

    def __repr__(self):
        return self.__class__.__name__