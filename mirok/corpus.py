#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging
import pickle
from collections import defaultdict
from typing import List
from functools import reduce
from pathos.multiprocessing import ProcessPool
import math
import time
import random

from tqdm import tqdm
from gensim.models.fasttext import FastTextKeyedVectors
from gensim.models.word2vec import Word2Vec
import numpy as np

from mirok.api import Tag
from mirok.seq import Seq
from mirok.seq_extractor import SeqExtractor

PAD = 0
PAD_WORD = "<pad>"
OOV = 1
OOV_WORD = "<oov>"
SOS = 2
SOS_WORD = "<sos>"
EOS = 3
EOS_WORD = "<eos>"
MSK = 4
MSK_WORD = "<mask>"


TAGS = [Tag.NONE, Tag.RES, Tag.OP1, Tag.OP2]

class Corpus:
    def __init__(self):
        self.seqs: List[Seq] = list()

    def build_vocabs(self):
        logging.info("start building vocabularies")
        self.idx2token = {PAD: PAD_WORD, OOV: OOV_WORD, SOS: SOS_WORD, EOS: EOS_WORD, MSK: MSK_WORD}
        self.token2idx = {PAD_WORD: PAD, OOV_WORD: OOV, SOS_WORD: SOS, EOS_WORD: EOS, MSK_WORD: MSK}
        self.idx2type = {PAD: PAD_WORD,}
        self.type2idx = {PAD_WORD: PAD,}
        self.idx2tag = {PAD: PAD_WORD,}
        self.tag2idx = {PAD_WORD: PAD,}

        for tag in TAGS:
            self.tag2idx[tag] = len(self.tag2idx)
            self.idx2tag[len(self.idx2tag)] = tag
        for seq in tqdm(self.seqs, ascii=True):
            for token in seq.token_seq:
                if token not in self.token2idx:
                    self.token2idx[token] = len(self.token2idx)
                    self.idx2token[len(self.idx2token)] = token
            for type in seq.token_type_seq:
                if type not in self.type2idx:
                    self.type2idx[type] = len(self.type2idx)
                    self.idx2type[len(self.idx2type)] = type
        logging.info(f"token vocabulary size: {len(self.token2idx)}")
        logging.info(f"type vocabulary size: {len(self.type2idx)}")
        logging.info(f"tag vocabulary size: {len(self.tag2idx)}")

    def build_inverted_index(self):
        logging.info("start building inverted index")
        self.inverted_index = defaultdict(set)
        for seq in tqdm(self.seqs, ascii=True):
            for token in seq.token_seq:
                self.inverted_index[token].add(seq)
        logging.info(f"inverted index size: {len(self.inverted_index)}")
    
    def get_seqs_by_ops(self, op1, op2):
        return self.inverted_index[op1] & self.inverted_index[op2]
    
    def get_seqs_by_res(self, res):
        return self.inverted_index[res]

    def get_seqs_by_tokens(self, *tokens):
        return reduce(lambda s1, s2: s1 & s2, [self.inverted_index[token] for token in tokens])

    def learn_word_emb(self):
        logging.info("start learning word embeddings")
        w2v = Word2Vec([seq.token_seq for seq in self.seqs], sg=1, hs=1, window=15, min_count=1)
        self.emb_size = w2v.wv.vector_size
        self.word_emb = np.zeros([len(self.token2idx), self.emb_size])
        for token, idx in self.token2idx.items():
            if token in {PAD_WORD, OOV_WORD, SOS_WORD, EOS_WORD, MSK_WORD}:
                continue
            self.word_emb[idx] = w2v.wv.get_vector(token)
        logging.info("finish learning word embeddings")

    def load_word_emb(self, emb_path):
        logging.info("start building word embeddings")
        emb: FastTextKeyedVectors = FastTextKeyedVectors.load(emb_path)
        self.emb_size = emb.vector_size
        self.word_emb = np.zeros([len(self.token2idx), self.emb_size])
        for token, idx in self.token2idx.items():
            if token == PAD_WORD:
                continue
            self.word_emb[idx] = emb.get_vector(token)
        logging.info("finish building word embeddings")

    @classmethod
    def init_from_sources(cls, codes, min_len=3, max_len=50, n_workers=6):
        logging.info("start processing classes")
        
        def extract(batch, batch_id, step=10000):
            seqs = list()
            extractor = SeqExtractor()
            counter = 0
            start_time = time.time()
            for clazz in batch:  
                for seq in extractor.extract(clazz, back_level=2):
                    if seq is not None and min_len <= len(seq.api_seq) <= max_len and len(seq.token_seq) > 0:
                        seqs.append(seq)
                counter += 1
                if counter % step == 0 or counter == len(batch):
                    logging.info(f"worker-{batch_id}: {counter}/{len(batch)}, time: {time.time() - start_time}s")
            return seqs

        if n_workers > 1:
            random.shuffle(codes)

        pool = ProcessPool(nodes=n_workers)
        batch_size = math.ceil(len(codes) / n_workers)
        procs = list()
        for batch_id in range(n_workers):
            procs.append(pool.apipe(extract, codes[batch_id*batch_size:(batch_id+1)*batch_size], batch_id+1))
        pool.close()
        pool.join()
        seqs = list()
        for proc in procs:
            seqs.extend(proc.get())

        corpus = cls()
        for seq in seqs:
            corpus.seqs.append(seq) 
        corpus.build_vocabs()
        corpus.build_inverted_index()
        # corpus.train_emb()
        return corpus

    def save(self, path):
        with open(path,'wb')  as f:
            pickle.dump(self, f)
      
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            corpus = pickle.load(f)
        return corpus
