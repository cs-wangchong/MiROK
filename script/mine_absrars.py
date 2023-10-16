#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging

from mirok.corpus import Corpus
from mirok.absrar_base import AbsRARBase
from mirok.config import Config
from mirok.absrar_miner import AbsRARMiner
from mirok.logging_util import init_log


if __name__ == '__main__':
    init_log("log/mine_abs_rar.log", logging.INFO)

    absrar_base = AbsRARBase.from_csv("resources/seed_abs_rars.csv")

    '''load corpus'''
    corpus: Corpus = Corpus.load("corpus.pkl")

    config = Config(
        iterations=20,
        depth=2,
        max_res_op_num=30,
        min_conf=0.8,
        min_support=3,
        activating_emb_epoch=101,
        penalty_coef=1.,
        training_batch=128,
        predicting_batch=512,
        ensemble_num=3,
        device="cuda"
    )
    miner = AbsRARMiner(absrar_base=absrar_base, corpus=corpus)
    miner.mine(config=config)
    miner.absrar_base.to_csv("abs_rars.csv")
    
    logging.info(f"number of Abs-RARs: {len(miner.absrar_base.abs_rars)}")
