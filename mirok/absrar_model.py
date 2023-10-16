#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import random
import numpy as np
import time
import tqdm
import logging
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from mirok.corpus import PAD
from mirok.api import Tag
from mirok.dataset import Dataset


def set_seed(seed):
    # Be warned that even with all these seeds, complete reproducibility cannot be guaranteed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    return


class Model(nn.Module):
    def __init__(
            self,
            n_word,
            n_type,
            n_tag,
            dim_word_emb,
            dim_type_emb,
            dim_hidden,
            lr=0.001,
            weight_decay=0,
            depth=1,
            dropout=0.5
        ):
        super(Model, self).__init__()
        self.depth = depth
        self.word_emb = nn.Embedding(n_word, dim_word_emb, padding_idx=PAD)
        self.type_emb = nn.Embedding(n_type, dim_type_emb, padding_idx=PAD)
        self.tag_emb = nn.Embedding(n_tag, dim_hidden, padding_idx=PAD)

        self.encoding_layer = nn.LSTM(dim_word_emb + dim_type_emb, dim_hidden // 2, bidirectional=True, batch_first=True)
        self.iterative_layer = nn.LSTM(dim_hidden, dim_hidden // 2, bidirectional=True, batch_first=True)
        self.fusion_layer = nn.Linear(dim_hidden, dim_hidden)
        self.tagging_layer = nn.Linear(dim_hidden, n_tag)
        self.dropout = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss()
        self.best_state = None
        
        # Set optimizer (Adam optimizer for now)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False)
        # Set learning rate scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=(), gamma=0.1)


    def set_word_emb(self, word_emb, freeze=True):
        word_emb = torch.FloatTensor(word_emb)
        self.word_emb = nn.Embedding.from_pretrained(word_emb, freeze=freeze, padding_idx=PAD)

    def activate_emb(self):
        self.word_emb.weight.requires_grad = True

    def freeze_emb(self):
        self.word_emb.weight.requires_grad = False

    def record_best_state(self):
        self.best_state = self.state_dict()

    def load_best_state(self):
        self.load_state_dict(self.best_state)

    def forward(self, tokens, types, mode="train"):
        all_depth_scores = []
        all_depth_hiddens = []
        non_pad_mask = tokens.ne(PAD).type(torch.float).unsqueeze(-1)
        tokens = self.word_emb(tokens)
        types = self.type_emb(types) 
        inp = torch.cat([tokens, types], -1) * non_pad_mask
        hiddens, _ = self.encoding_layer(inp)

        for d in range(self.depth):
            hiddens = hiddens * non_pad_mask
            hiddens, _ = self.iterative_layer(hiddens) 
            hiddens = self.dropout(hiddens)

            word_hiddens = hiddens
            
            if d != 0:
                greedy_tags = torch.argmax(scores, dim=-1)      
                tag_embeddings = self.tag_emb(greedy_tags)
                word_hiddens = word_hiddens + tag_embeddings
            
            word_hiddens = self.fusion_layer(word_hiddens)
            all_depth_hiddens.append(word_hiddens)
            scores = self.tagging_layer(word_hiddens)
            all_depth_scores.append(scores)

            if mode != 'train':
                predictions = torch.argmax(scores, dim=-1)
                if torch.sum(predictions) == 0:
                    break
        return all_depth_hiddens, all_depth_scores

    def calc_loss(self, all_depth_hiddens, all_depth_scores, labels, dataset, coef=.5, gamma=0.1):
        batch_size, num_words, _ = all_depth_scores[0].shape
        loss = 0
        for d, scores in enumerate(all_depth_scores):
            loss += self.loss(scores.reshape(batch_size * num_words, -1), labels[:, d, :].reshape(-1))
        # ############### constraints ###############
        all_depth_labels = torch.transpose(labels, 0, 1)
        pair_loss = 0
        for hiddens, labels in zip(all_depth_hiddens, all_depth_labels):
            batch_loss = 0
            for h, l in zip(hiddens, labels):
                op1 = l.eq(dataset.tag2idx[Tag.OP1]).nonzero(as_tuple=True)[0]
                op2 = l.eq(dataset.tag2idx[Tag.OP2]).nonzero(as_tuple=True)[0]
                if len(op1) == 0 or len(op2) == 0:
                    batch_loss += 1
                else:
                    batch_loss += F.cosine_similarity(h[op1[0]], h[op2[0]], dim=-1) * 0.5 + 0.5
            batch_loss /= batch_size
            pair_loss += batch_loss
        pair_loss = coef * pair_loss
        # logging.info(f"constraint loss: {pair_loss}")
        loss += pair_loss
        return loss


class ModelEnsemble:
    def __init__(
            self,
            n_word,
            n_type,
            n_tag,
            dim_word_emb,
            dim_type_emb,
            dim_hidden,
            ensemble_num=6,
            depth=1,
            dropout=.5
        ):
        self.models: List[Model] = [
            Model(n_word, n_type, n_tag, dim_word_emb, dim_type_emb, dim_hidden, depth, dropout)
            for _ in range(ensemble_num)
        ]

    def set_word_emb(self, word_emb, freeze=True):
        for model in self.models:
            model.set_word_emb(word_emb, freeze=freeze)

    def activate_emb(self):
        for model in self.models:
            model.activate_emb()

    def freeze_emb(self):
        for model in self.models:
            model.freeze_emb()

    def train(
            self,
            training_set: Dataset,
            validation_set: Dataset,
            n_epochs=150,
            lr=0.001,
            weight_decay=1e-5,
            start_epoch=0,
            batch_size=128,
            early_stopping=3,
            penalty_coef=0.5,
            n_jobs_dataloader: int = 0,
            device='cuda',
        ):
        optimizers: List[optim.Adam] = []
        schedulers: List[optim.lr_scheduler.MultiStepLR] = []
        for model in self.models:
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=(), gamma=0.1)
            optimizers.append(optimizer)
            schedulers.append(scheduler)
            
        # Get train data loader
        training_loader = training_set.loader(batch_size=batch_size, shuffle=True, num_workers=n_jobs_dataloader)
        validating_loader = validation_set.loader(batch_size=batch_size, shuffle=False, num_workers=n_jobs_dataloader)
        
        best_loss = float('inf')
        n_increasing_epochs = 0
        # Training
        logging.info('Starting training...')
        start_time = time.time()
        for epoch in range(start_epoch, start_epoch + n_epochs):
            losses = []
            eval_losses = []
            epoch_start_time = time.time()
            for model, optimizer, scheduler in zip(self.models, optimizers, schedulers):
                model.train()
                loss_epoch = 0.0
                n_batches = 0
                for tokens, types, labels in training_loader:
                    # logging.info(hash(model))
                    tokens = tokens.to(device)
                    types = types.to(device)
                    labels = labels.to(device)
                
                    # Zero the network parameter gradients
                    optimizer.zero_grad()

                    # Update network parameters via backpropagation: forward + backward + optimize
                    hiddens, outputs = model.forward(tokens, types, mode="train")
                    loss = model.calc_loss(hiddens, outputs, labels, training_set, coef=penalty_coef)

                    loss.backward()
                    optimizer.step()

                    loss_epoch += loss.item()
                    n_batches += 1
                scheduler.step()
                losses.append(loss_epoch / n_batches)
                
                model.eval()
                loss_epoch = 0.0
                n_batches = 0
                for tokens, types, labels in validating_loader:
                    # logging.info(hash(model))
                    tokens = tokens.to(device)
                    types = types.to(device)
                    labels = labels.to(device)
                    # Update network parameters via backpropagation: forward + backward + optimize
                    hiddens, outputs = model.forward(tokens, types, mode="train")
                    loss = model.calc_loss(hiddens, outputs, labels, validation_set, coef=penalty_coef)
                    # constraints
                    loss_epoch += loss.item()
                    n_batches += 1
                eval_losses.append(loss_epoch / n_batches)
            # log epoch statistics
            avg_loss = sum(eval_losses) / len(self.models)
            epoch_train_time = time.time() - epoch_start_time
            logging.info(('  Epoch {}\t Time: {:.3f}\t Loss: ' + ', '.join(['{:.8f}'] * len(self.models)))
                        .format(epoch + 1, epoch_train_time, *losses)  + '\tEval Loss: {:.8f}'.format(avg_loss))
            if avg_loss < best_loss:
                best_loss = avg_loss 
                n_increasing_epochs = 0
                for model in self.models:
                    model.record_best_state()
            else:
                n_increasing_epochs += 1
            
            if n_increasing_epochs >= early_stopping:
                break

        train_time = time.time() - start_time
        logging.info('Training time: %.3fs' % train_time)
        logging.info('Finished training.')

    
    def predict(self, dataset: Dataset, batch_size=128, n_jobs_dataloader=2, device="cuda"):
        for model in self.models:
            model.load_best_state()
            model.to(device)
        # Get train data loader
        prediction_loader = dataset.loader(batch_size=batch_size, shuffle=False, num_workers=n_jobs_dataloader)

        logging.info('Starting predicting...')
        start_time = time.time()
        for model in self.models:
            model.eval()
        final_predictions = []
        for tokens, types in tqdm.tqdm(prediction_loader, desc="Predicting ", ascii=True):
            tokens = tokens.to(device)
            types = types.to(device)
            outputs = [model.forward(tokens, types, mode="predict")[1] for model in self.models]

            non_pad_mask = tokens.ne(PAD).type(torch.float)
            lengths = torch.sum(non_pad_mask, dim=-1).long().cpu().numpy()
            
            avg_outputs = []
            for single_depth in zip(*outputs):
                avg = sum([torch.softmax(scores, dim=2) for scores in single_depth]) / len(self.models)
                avg_outputs.append(avg)
            all_depth_predictions = []
            for scores in avg_outputs:
                # logging.info(scores)
                max_probs, predictions = torch.max(scores, dim=2)
                max_log_probs = torch.log(max_probs)
                # logging.info(max_log_probs)
                sro_label_predictions = (predictions != PAD).float() * non_pad_mask
                log_probs_norm_ext_len = (max_log_probs * sro_label_predictions) / (sro_label_predictions.sum(dim=0) + 1)
                confidences = torch.exp(torch.sum(log_probs_norm_ext_len, dim=1))
                # logging.info(f'confidence is {confidences}')

                predictions = predictions.detach().cpu().numpy()
                confidences = confidences.detach().cpu().numpy()
                
                all_depth_predictions.append([(pred.tolist()[:len], conf) for pred, conf, len in zip(predictions, confidences, lengths)])
            
            for merged_predictions in zip(*all_depth_predictions):
                merged_predictions = [(pred, conf) for (pred, conf) in merged_predictions if PAD not in pred]
                final_predictions.append(merged_predictions)


        logging.info('Predicting time: %.3fs' % (time.time() - start_time))
        logging.info('Finished predicting.')
        logging.info(f'final_predictions size is {len(final_predictions)}')
        return final_predictions