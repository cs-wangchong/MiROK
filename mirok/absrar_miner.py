#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re
from collections import defaultdict

from typing import Set
import time
import logging
import random

from nltk.corpus import wordnet

from mirok.absrar_base import AbsRARBase
from mirok.seq import Seq
from mirok.corpus import Corpus
from mirok.dataset import Dataset
from mirok.absrar_model import ModelEnsemble
from mirok.config import Config
from mirok.java_stoplist import *


class AbsRARMiner:

    def __init__(
        self,
        absrar_base: AbsRARBase=None,
        corpus: Corpus=None,
    ):
        self.absrar_base = absrar_base
        self.corpus = corpus
    

    def _augment_op_pair(self, op1, op2):
        logging.info(f"try to augment operation pair by synsets: {(op1, op2)}")

        pairs = {(op1, op2)}
        for synset in wordnet.synsets(op1):
            if synset.pos() != "v":
                continue
            for synonym in synset.lemmas():
                lemma_name = synonym.name().replace("_", " ")
                antonyms = set(antonym.name().replace("_", " ") for antonym in synonym.antonyms() if antonym.synset().pos() == "v")
                pairs.update([(lemma_name, ant) for ant in antonyms])
        for synset in wordnet.synsets(op2):
            if synset.pos() != "v":
                continue
            for synonym in synset.lemmas():
                lemma_name = synonym.name().replace("_", " ")
                antonyms = set(antonym.name().replace("_", " ") for antonym in synonym.antonyms() if antonym.synset().pos() == "v")
                pairs.update([(ant, lemma_name) for ant in antonyms])
        logging.info(f"augmented pairs: {pairs}")
        return pairs
    
    def rule_based_mine(self, min_support=3, max_times=3):
        logging.info(f'start rule based mining, corpus size;{len(self.corpus.seqs)}  ')
        start = time.time()
        support_dict = defaultdict(int)

        augmented_op_pairs = dict()
        for op1, op2 in set((op1, op2) for _, op1, op2 in self.absrar_base):
            augmented_op_pairs[op1, op2] = self._augment_op_pair(op1, op2)

        for res, op1, op2 in self.absrar_base:
            for op1, op2 in augmented_op_pairs[op1, op2]:
                tokens = set(res.split()) | set(op1.split()) | set(op2.split())
                candidates: Set[Seq] = self.corpus.get_seqs_by_tokens(*tokens)
                logging.info(f"try to expand Abs-RAR: {(res, op1, op2)}, candidates: {len(candidates)}")
                for seq in candidates:
                    cand_res = seq.expand_res(res=res, op1=op1, op2=op2)
                    if not cand_res:
                        continue
                    double = "|".join(c * 2 for c in "abcdefghijklmnopqrstuvwxyz")
                    logging.info(f"expanded resource: {cand_res}")
                    cand_res = re.sub(r"^(\w |my |cur |all |new |old |%s )+(.*)" % double, r"\2", cand_res).strip()
                    logging.info(f"expanded resource after cleaning: {cand_res}")
                    if len(cand_res) == 1:
                        continue
                    if (cand_res, op1, op2) in self.absrar_base:
                        continue
                    if len(cand_res.split()) / len(res.split()) > max_times:
                        continue
                    support_dict[cand_res, op1, op2] += 1
        for (res, op1, op2), support in sorted(support_dict.items(), key=lambda item: item[-1], reverse=True):
            logging.info(f'new Abs-RAR: {(res, op1, op2)}, support {support}')
            if support >= min_support:
                self.absrar_base.add_abs_rar(res, op1, op2)
        logging.info("time: %.3fs" % (time.time() - start))
    
    def distant_annotate(self):
        logging.info(f'start distant-supervision annotation, corpus size;{len(self.corpus.seqs)}  ')
        start = time.time()
        for seq in self.corpus.seqs:
            seq.clear_tag()
        rar2seqs = defaultdict(set)
        
        super_rars = defaultdict(set)
        for res, op1, op2 in sorted(self.absrar_base, key=lambda t: len(t[0].split()), reverse=True):
            words = res.split()
            for l in range(1, len(words)):
                for b in range(0, len(words) - l):
                    e = b + l
                    phrase = " ".join(words[b:e])
                    if (phrase, op1, op2) in self.absrar_base:
                        super_rars[phrase, op1, op2].add((res, op1, op2))
        
        for res, op1, op2 in sorted(self.absrar_base, key=lambda t: len(t[0].split()), reverse=True):
            tokens = set(res.split()) | set(op1.split()) | set(op2.split())
            candidates: Set[Seq] = self.corpus.get_seqs_by_tokens(*tokens)
            logging.info(f"try to annotate Abs-RAR: {(res, op1, op2)}, candidate sequences: {len(candidates)}")
            for seq in candidates:
                for super_res, _, _ in super_rars[res, op1, op2]:
                    if seq in rar2seqs[super_res, op1, op2]:
                        break
                else:
                    prev_tag_seqs = len(seq.tag_seqs)
                    seq.annotate(res=res, op1=op1, op2=op2)
                    if len(seq.tag_seqs) - prev_tag_seqs > 0:
                        rar2seqs[res, op1, op2].add(seq)
        logging.info(f"valid Abs-RARs: {len(rar2seqs)}")
        for (res, op1, op2), seqs in sorted(rar2seqs.items(), key=lambda item: len(item[-1]), reverse=True):
            logging.info(f'Abs-RARs: {(res, op1, op2)}, sequences {len(seqs)}')
        logging.info("time: %.3fs" % (time.time() - start))
        return rar2seqs

    def generate_datasets(self, rar2seqs, depth=1, valid_ratio=0.1, oversampling_count=100, oversampling_times=30, max_token_num=300):
        op_pair2seq_num = defaultdict(int)
        for (res, op1, op2), seqs in rar2seqs.items():
            op_pair2seq_num[op1, op2] += len(seqs)
        num_max = max(op_pair2seq_num.values())
        oversampling_count = min([oversampling_count, num_max])
        training_seqs = []
        prediction_seqs = []
        oversampling_dict = dict()
        for seq in self.corpus.seqs:
            if len(seq.token_seq) > max_token_num:
                continue
            if seq.valid():
                # align sample num
                for res, op1, op2 in seq.get_abs_rars():
                    if (op1, op2) not in op_pair2seq_num:
                        continue
                    oversampling = min([(oversampling_count - 1) // op_pair2seq_num[op1, op2], oversampling_times - 1])
                    oversampling_dict[seq] = oversampling
                    training_seqs.append(seq)
            else:
                prediction_seqs.append(seq)
        random.shuffle(training_seqs)
        training_samples = len(training_seqs) - int(len(training_seqs) * valid_ratio)
        training_seqs, validating_seqs = training_seqs[:training_samples], training_seqs[training_samples:]
        for seq in list(training_seqs):
            training_seqs.extend([seq] * oversampling_dict[seq])
        training_set = Dataset(self.corpus.token2idx, self.corpus.type2idx, self.corpus.tag2idx, training_seqs, is_training=True, depth=depth, max_len=max_token_num)
        training_set.init_op_pairs(self.absrar_base.op1s, self.absrar_base.op2s)
        validation_set = Dataset(self.corpus.token2idx, self.corpus.type2idx, self.corpus.tag2idx, validating_seqs, is_training=True, depth=depth, max_len=max_token_num)
        validation_set.init_op_pairs(self.absrar_base.op1s, self.absrar_base.op2s)
        prediction_set = Dataset(self.corpus.token2idx, self.corpus.type2idx, self.corpus.tag2idx, prediction_seqs, is_training=False, depth=depth, max_len=max_token_num)
        prediction_set.init_op_pairs(self.absrar_base.op1s, self.absrar_base.op2s)
        logging.info(f"training dataset: {len(training_set)}, validating dataset: {len(validation_set)}, prediction dataset: {len(prediction_set)}")
        return training_set, validation_set, prediction_set

    def filter_abs_rars(self, rar2support, rar2conf, min_support=5, min_conf=0.8, max_res_op_num=4):
        logging.info(f"start filtering Abs-RARs")
        abs_rars = set()
        op1s = set()
        op2s = set()
        for _, op1, op2 in self.absrar_base:
            for _op1, _op2 in self._augment_op_pair(op1, op2):
                op1s.add(_op1)
                op2s.add(_op2)

        res_op_num_dict = defaultdict(int)
        for (res, op1, op2) in self.absrar_base:
            res_op_num_dict[res] = res_op_num_dict[res]+1

        for (res, op1, op2), count in sorted(rar2support.items(), key=lambda item: item[-1], reverse=True):
            logging.info(f"cand Abs-RAR: {(res, op1, op2, count)}")
            if count < min_support or rar2conf[(res, op1, op2)] < min_conf:
                continue
            if op1 == op2:
                continue
            if res_op_num_dict[res] >= max_res_op_num:
                continue
            if len(res)<2 or len(op1)<2 or len(op2)<2:
                continue
            double = "|".join(c * 2 for c in "abcdefghijklmnopqrstuvwxyz")
            res = re.sub(r"^(\w |my |cur |all |new |old |%s )+(.*)" % double, r"\2", res).strip()

            if res.startswith("m ") or res.startswith("my ") or res.startswith("tmp ") or res.startswith("temp "):
                continue
            if res.endswith(" util") or res.endswith("utils") or res.endswith("read") or res.endswith("write"):
                continue
            if res.endswith("ed") or op1.endswith("ed") or op2.endswith("ed"):
                continue
            if res.endswith("ing") or op1.endswith("ing") or op2.endswith("ing"):
                continue
            if op1 == "<init>" and op2 not in op2s:
                continue
            if res in INVALID_RES or op1 in INVALID_OP1 or op2 in INVALID_OP2:
                continue
            if len({res, op1, op2} & {'<none>', '<unknown>'}) > 0:
                continue
            if op1 in op2s or op2 in op1s:
                continue
            if res != 'lock' and (res in op1s or res in op2s or res == op1 or res == op2):
                continue
            tokens = set(res.split()) | set(op1.split()) | set(op2.split())
            candidates: Set[Seq] = self.corpus.get_seqs_by_tokens(*tokens)
            # logging.info(f"annotate abs_rar: {(res, op1, op2)}, candidates: {len(candidates)}")
            for seq in candidates:
                if seq.contains(res=res, op1=op1, op2=op2) == 1:
                    break
            else:
                continue
            abs_rars.add((res, op1, op2))
            op1s.add(op1)
            op2s.add(op2)
            res_op_num_dict[res] = res_op_num_dict[res]+1
      
        for seq in self.corpus.seqs:
            seq.clear_tag()
        copied_abs_rars = set(abs_rars)
        abs_rars = set()
        for res, op1, op2 in copied_abs_rars:
            abs_rars.add((res, op1, op2))
        logging.info(f'finish filtering Abs-RARs')
        return abs_rars
  
    def mine(self, config=Config()):
        logging.info(config)
        logging.info(self.absrar_base)
        n_empty_iters = 0
        iter_list = []
        for iter in range(config.iterations): 
            logging.info(f"\n\n=============================== ITERATION {iter+1} ==============================")
            original_size = len(self.absrar_base.abs_rars) 
            self.rule_based_mine()      
            rar2seqs = self.distant_annotate()
            training_set, validation_set, prediction_set = self.generate_datasets(rar2seqs, depth=config.depth, valid_ratio=config.valid_ratio, oversampling_count=config.oversampling_count, oversampling_times=config.oversampling_times)

            ensemble = ModelEnsemble(len(self.corpus.token2idx), len(self.corpus.type2idx), len(self.corpus.tag2idx), self.corpus.emb_size, config.dim_type_emb, config.dim_hidden, ensemble_num=config.ensemble_num, depth=config.depth, dropout=config.dropout)
            ensemble.set_word_emb(self.corpus.word_emb)

            new_abs_rars = set()
            for start_epoch in range(0, config.max_epochs, config.epoch_step):
                rar2support = defaultdict(int)
                rar2conf = defaultdict(float)
                res2support = defaultdict(int)
                op2support = defaultdict(int)
                logging.info(f"===================== TRAINING & PREDICTION: EPOCH {start_epoch + 1}-{start_epoch + config.epoch_step} ====================")
                ensemble.train(training_set, validation_set, n_epochs=config.epoch_step, lr=config.lr, weight_decay=config.weight_decay, start_epoch=start_epoch, batch_size=config.training_batch, early_stopping=config.early_stopping, device=config.device, penalty_coef=config.penalty_coef)
                predictions = ensemble.predict(prediction_set, batch_size=config.predicting_batch, device=config.device)
                for seq, pred in zip(prediction_set.seqs, predictions):
                    tag_seqs = [([self.corpus.idx2tag[p] for p in _pred], _conf) for _pred, _conf in pred]
                    seq.tag_seqs = tag_seqs
                abs_rars_with_conf = list()
                for seq in prediction_set.seqs:
                    abs_rars_with_conf.extend(seq.get_abs_rars_with_conf())
                for res, op1, op2, conf in abs_rars_with_conf:
                    rar2support[res, op1, op2] += 1
                    rar2conf[res, op1, op2] += conf
                    res2support[res] += 1
                    op2support[op1, op2] += 1
                for abs_rar, conf in rar2conf.items():
                    rar2conf[abs_rar] = conf / rar2support[abs_rar]
                logging.info(f"=============================== RESOURCE ==============================")
                for res, count in sorted(res2support.items(), key=lambda item: item[-1], reverse=True):
                    logging.info(f"resource: {res}, support: {count}")
                logging.info(f"=============================== OP PAIR ==============================")
                for pair, count in sorted(op2support.items(), key=lambda item: item[-1], reverse=True):
                    logging.info(f"operation pair: {pair}, support: {count}")
                logging.info(f"=============================== abs_rar ==============================")
                for (res, op1, op2), count in sorted(rar2support.items(), key=lambda item: item[-1], reverse=True):
                    logging.info(f"Abs-RAR: {(res, op1, op2)}, support: {count}, confidence: {rar2conf[(res, op1, op2)]}")
    
                abs_rars = self.filter_abs_rars(rar2support, rar2conf, min_support=config.min_support, min_conf=config.min_conf, max_res_op_num= config.max_res_op_num)

                logging.info(f"=============================== NEW abs_rar =============================")
                for abs_rar in abs_rars - self.absrar_base.abs_rars:
                    new_abs_rars.add(abs_rar)
                    logging.info(abs_rar)
                for res, op1, op2 in abs_rars:
                    self.absrar_base.add_abs_rar(res, op1, op2)
                logging.info(self.absrar_base)
                iter_list.append(len(self.absrar_base.abs_rars))
                # self.save_abs_rars(iteration=iter, token_seq=token_seq)
                if len(new_abs_rars) >= int(original_size * config.stride):
                    break
            if len(new_abs_rars) == 0:
                n_empty_iters += 1
            else:
                n_empty_iters = 0
            if n_empty_iters >= 1:
                break
        return iter_list
    
            

            
